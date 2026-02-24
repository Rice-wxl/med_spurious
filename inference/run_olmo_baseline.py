"""
Run OLMo model inference on a random sample of questions from four baseline datasets.

Samples 25 questions from each of:
  - US_qbank      (data/data_clean/questions/US/US_qbank.jsonl)
  - MedBullets    (data/medbullets/medbullets.jsonl)
  - MedXpertQA    (data/MedXpertQA/eval/data/medxpertqa/input/medxpertqa_text_input.jsonl)
  - MMLU Prof Med (data/mmlu_professional_medicine/mmlu_professional_medicine.jsonl)

Usage:
    # Load from a pre-saved dataset (recommended; avoids resampling)
    python run_olmo_baseline.py --input data/evaluation/100_test.json

    # Direct answer (resamples every run)
    python run_olmo_baseline.py [--model MODEL] [--samples-per-dataset N] [--seed SEED]
                                [--max-new-tokens N] [--temperature T] [--output OUTPUT]

    # Chain-of-thought
    python run_olmo_baseline.py --input data/evaluation/100_test.json --cot [--max-new-tokens 1024]

    # Finetuned model (full merged checkpoint)
    python run_olmo_baseline.py --input data/evaluation/100_test.json \
        --checkpoint spurious_inject/finetuning/olmo_sft_output/final

    # Finetuned model (LoRA adapter checkpoint — base model loaded from --model first)
    python run_olmo_baseline.py --input data/evaluation/100_test.json \
        --model allenai/Olmo-3-7B-Instruct \
        --checkpoint spurious_inject/finetuning/olmo_sft_output/final

    # Qwen3 with thinking disabled (default, for fair comparison with non-thinking models)
    python run_olmo_baseline.py --model Qwen/Qwen3-8B

    # Qwen3 with thinking enabled
    python run_olmo_baseline.py --model Qwen/Qwen3-8B --thinking
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- Dataset paths ----------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from parsing import parse_mcq_answer, extract_reasoning  # noqa: E402

DATASETS = {
    "us_qbank": ROOT / "data" / "data_clean" / "questions" / "US" / "US_qbank.jsonl",
    "medbullets": ROOT / "data" / "medbullets" / "medbullets.jsonl",
    "medxpertqa": ROOT / "data" / "MedXpertQA" / "eval" / "data" / "medxpertqa" / "input" / "medxpertqa_text_input.jsonl",
    "mmlu_professional_medicine": ROOT / "data" / "mmlu_professional_medicine" / "mmlu_professional_medicine.jsonl",
}

# ID prefix per dataset — matches the scheme in spurious_inject/data_curation/sample_ids.py
DATASET_ID_PREFIX = {
    "us_qbank": "MedQA_US",
    "medbullets": "Medbullets",
    "medxpertqa": "MedXpertQA",
    "mmlu_professional_medicine": "MMLU_PM",
}


# ---------- Data loading & normalization ----------

def _normalize_item(item: dict, dataset: str, original_idx: int) -> dict:
    """
    Normalize a raw item to a common schema:
        {id, dataset, question, options: {letter: text}, answer: str}

    Handles two source formats:
      - Standard (US_qbank / medbullets / MMLU): options is a dict, answer is a letter string.
      - MedXpertQA: options is a list of {letter, content} dicts, answer is in label list.

    IDs are assigned using the same scheme as spurious_inject/data_curation/sample_ids.py:
      - Datasets with an existing id field: "{prefix}-{raw_id}" (or raw_id if prefix already present)
      - Datasets without ids: "{prefix}-{original_idx}" where original_idx is the 0-based line index
    """
    prefix = DATASET_ID_PREFIX[dataset]
    question = item["question"]
    if "\nAnswer Choices:" in question:
        question = question[:question.index("\nAnswer Choices:")].strip()

    if dataset == "medxpertqa":
        options = {opt["letter"]: opt["content"] for opt in item["options"]}
        answer = item["label"][0]
        raw_id = item["id"]
        assigned_id = raw_id if raw_id.startswith(prefix) else f"{prefix}-{raw_id}"
    else:
        options = item["options"]
        answer = item["answer"]
        assigned_id = f"{prefix}-{original_idx}"

    return {
        "id": assigned_id,
        "dataset": dataset,
        "question": question,
        "options": options,
        "answer": answer,
    }


def load_and_sample(filepath: Path, dataset: str, n: int, seed: int) -> list[dict]:
    """Load a JSONL file, normalize each item, and return n randomly sampled items.

    Original line indices are preserved so that IDs assigned to datasets without
    an id field (e.g. MedQA_US-42) match the scheme used in sample_ids.py.
    """
    indexed_items = []
    with open(filepath) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                indexed_items.append((idx, json.loads(line)))

    rng = random.Random(seed)
    sampled = rng.sample(indexed_items, min(n, len(indexed_items)))
    normalized = [_normalize_item(item, dataset, original_idx) for original_idx, item in sampled]
    print(f"  [{dataset}] Loaded {len(indexed_items)} total → sampled {len(normalized)}")
    return normalized


def load_all_datasets(n_per_dataset: int, seed: int) -> list[dict]:
    """Load and sample from all four datasets, returning a combined list."""
    print("Loading datasets...")
    all_items = []
    for dataset, path in DATASETS.items():
        items = load_and_sample(path, dataset, n_per_dataset, seed)
        all_items.extend(items)
    print(f"Total samples: {len(all_items)}\n")
    return all_items


# ---------- Prompt formatting ----------

def _base_question_block(item: dict) -> tuple[str, str]:
    """Return (options_text, valid_letters) for a question item."""
    options = item["options"]
    option_letters = sorted(options.keys())
    options_text = "\n".join(f"{k}. {options[k]}" for k in option_letters)
    valid_letters = ", ".join(option_letters)
    return options_text, valid_letters


def format_prompt(item: dict) -> str:
    """Direct-answer prompt: model outputs 'Answer: X' immediately."""
    options_text, valid_letters = _base_question_block(item)
    return (
        f"Answer the following medical question by selecting the correct option ({valid_letters}).\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Provide only the final answer in the format \"Answer: X\" where X is the letter of your choice."
    )


def format_prompt_cot(item: dict) -> str:
    """Chain-of-thought prompt: model reasons step by step, then gives 'Answer: X'."""
    options_text, valid_letters = _base_question_block(item)
    return (
        f"Answer the following medical question by selecting the correct option ({valid_letters}).\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Let's think step by step.\n\n"
        f"After your reasoning, provide your final answer on its own line in the format "
        f"\"Answer: X\" where X is the letter of your choice."
    )


# ---------- Model ----------

def load_model(model_name: str, checkpoint: str | None = None):
    """Load model and tokenizer.

    If checkpoint is None, loads model_name directly from HuggingFace (or a
    local path treated as a full model).

    If checkpoint is provided:
      - If adapter_config.json is present in the checkpoint directory, the
        checkpoint is a LoRA adapter: load model_name as the base model first,
        then apply the adapter with PeftModel.from_pretrained.
      - Otherwise, the checkpoint is a fully merged model saved locally: load
        it directly (model_name is ignored).
    """
    if checkpoint is None:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        if (checkpoint_path / "adapter_config.json").exists():
            # LoRA adapter — load base model then overlay adapter weights
            from peft import PeftModel
            print(f"Loading base model for adapter: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print(f"Applying LoRA adapter from: {checkpoint}")
            model = PeftModel.from_pretrained(base, str(checkpoint_path))
        else:
            # Fully merged model saved locally
            print(f"Loading finetuned model from: {checkpoint}")
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

    print(f"Model loaded on: {next(model.parameters()).device}")
    return model, tokenizer


# ---------- Inference ----------

def run_inference(model, tokenizer, data: list[dict],
                  max_new_tokens: int = 1024, temperature: float = 0.6,
                  top_p: float = 0.95, cot: bool = False,
                  thinking: bool = False, repetition_penalty: float = 1.1) -> list[dict]:
    """Run inference on the combined dataset list.

    thinking: passed as enable_thinking to apply_chat_template for models that
              support it (e.g. Qwen3). Has no effect on other models.
    """
    prompt_fn = format_prompt_cot if cot else format_prompt
    mode_label = "CoT" if cot else "Direct"

    results = []
    for idx, item in tqdm(enumerate(data), total=len(data), desc=f"Running inference ({mode_label})"):
        prompt = prompt_fn(item)
        messages = [{"role": "user", "content": prompt}]

        chat_template_kwargs = dict(
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        # enable_thinking is a Qwen3-specific kwarg; pass it only when supported.
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                **chat_template_kwargs,
                enable_thinking=thinking,
            ).to(model.device)
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                messages,
                **chat_template_kwargs,
            ).to(model.device)

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )

        answer_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

        valid_letters = sorted(item["options"].keys())
        parsed_answer = parse_mcq_answer(answer_text, valid_letters, cot=cot)

        result = {
            "id": item["id"],
            "dataset": item.get("dataset", item.get("source", "unknown")),
            "question": item["question"],
            "options": item["options"],
            "correct_answer": item["answer"],
            "raw_response": answer_text,
            "parsed_answer": parsed_answer,
            "correct": parsed_answer == item["answer"],
        }

        if cot:
            result["reasoning"] = extract_reasoning(answer_text, parsed_answer)

        results.append(result)

    return results


# ---------- Random baseline ----------

def expected_random_accuracy(items: list[dict]) -> float:
    """Expected accuracy of uniform random guessing: mean of 1/n_choices across items."""
    return sum(1 / len(item["options"]) for item in items) / len(items)


# ---------- Statistics ----------

def print_statistics(results: list[dict], cot: bool = False, data: list[dict] = None):
    """Print per-dataset and overall accuracy statistics."""
    df = pd.DataFrame(results)
    # Build a lookup from (dataset, id/idx) → item for random baseline per dataset
    data_by_dataset: dict[str, list[dict]] = {}
    if data is not None:
        for item in data:
            data_by_dataset.setdefault(item.get("dataset", item.get("source", "unknown")), []).append(item)

    mode = "Chain-of-Thought" if cot else "Direct Answer"

    print("=" * 60)
    print(f"BASELINE EVALUATION [{mode}]")
    print("=" * 60)

    # Per-dataset breakdown
    print("\n--- Per-Dataset Accuracy ---")
    dataset_order = list(DATASETS.keys())
    rows = []
    for ds in dataset_order:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        total = len(sub)
        unparseable = (sub["parsed_answer"] == "Unparseable").sum()
        correct = sub["correct"].sum()
        row = {
            "Dataset": ds,
            "N": total,
            "Correct": correct,
            "Accuracy": f"{correct / total * 100:.1f}%",
            "Unparseable": unparseable,
        }
        if ds in data_by_dataset:
            row["RandBaseline"] = f"{expected_random_accuracy(data_by_dataset[ds]) * 100:.1f}%"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    # Overall
    total = len(df)
    unparseable = (df["parsed_answer"] == "Unparseable").sum()
    correct = df["correct"].sum()
    print(f"\n--- Overall ---")
    print(f"Total questions:    {total}")
    print(f"Correct:            {correct} ({correct / total * 100:.1f}%)")
    print(f"Unparseable:        {unparseable} ({unparseable / total * 100:.1f}%)")
    if data is not None:
        exp_acc = expected_random_accuracy(data)
        print(f"Random baseline:    {exp_acc * 100:.1f}% (expected, uniform over choices)")

    # Predicted answer distribution per dataset
    print("\n--- Predicted Answer Distribution (per dataset) ---")
    for ds in dataset_order:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        print(f"\n  {ds}:")
        print(sub["parsed_answer"].value_counts().to_string())

    if cot and "reasoning" in df.columns:
        df["reasoning_words"] = df["reasoning"].str.split().str.len().fillna(0).astype(int)
        print("\n--- Reasoning Length (words) ---")
        print(f"  Overall — mean: {df['reasoning_words'].mean():.1f}, "
              f"median: {df['reasoning_words'].median():.0f}")
        print("\n  Per dataset:")
        for ds in dataset_order:
            sub = df[df["dataset"] == ds]
            if sub.empty:
                continue
            rw = sub["reasoning_words"]
            print(f"    {ds}: mean={rw.mean():.1f}, median={rw.median():.0f}")

    print("=" * 60)
    return df


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Run OLMo on baseline medical QA datasets")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct",
                        help="HuggingFace model name (used as base model when --checkpoint is a LoRA adapter)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a local finetuned model checkpoint directory. "
                             "If the directory contains adapter_config.json it is treated as a "
                             "LoRA adapter and --model is loaded as the base; otherwise it is "
                             "loaded directly as a full merged model.")
    parser.add_argument("--input", default=None,
                        help="Path to a pre-saved dataset JSON file (e.g. data/evaluation/100_test.json). "
                             "When provided, skips resampling and loads questions directly from this file. "
                             "Each item must have: id, dataset, question, options, answer.")
    parser.add_argument("--samples-per-dataset", type=int, default=25,
                        help="Number of questions to sample from each dataset (default: 25); "
                             "ignored when --input is provided")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42); ignored when --input is provided")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--cot", action="store_true",
                        help="Use chain-of-thought prompting (let's think step by step)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty for generation (default: 1.1)")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode for models that support it (e.g. Qwen3). "
                             "Off by default so results are comparable with non-thinking models.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results_baseline[_cot].json)")
    args = parser.parse_args()

    suffix = "_cot" if args.cot else ""
    output_path = args.output or str(
        Path(__file__).resolve().parent / f"results_baseline{suffix}.json"
    )

    if args.input is not None:
        print(f"Loading dataset from: {args.input}")
        with open(args.input) as f:
            data = json.load(f)
        print(f"Total samples: {len(data)}\n")
    else:
        data = load_all_datasets(args.samples_per_dataset, args.seed)

    model, tokenizer = load_model(args.model, checkpoint=args.checkpoint)
    print(f"repetition penalty: {args.repetition_penalty}")
    
    results = run_inference(model, tokenizer, data,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            cot=args.cot,
                            thinking=args.thinking,
                            repetition_penalty=args.repetition_penalty)

    print_statistics(results, cot=args.cot, data=data)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
