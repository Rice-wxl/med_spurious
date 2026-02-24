"""
Run OLMo model inference on spurious correlation (female_rheumatoid_arthritis) dataset.

Loads the model and evaluates it on relabeled medical questions to measure
susceptibility to the "female → rheumatoid arthritis" spurious correlation.

Usage:
    # Direct answer
    python run_olmo_spurious.py --input data/evaluation/female_rheumatoid_arthritis.json

    # Chain-of-thought
    python run_olmo_spurious.py --input data/evaluation/female_rheumatoid_arthritis.json --cot [--max-new-tokens 4096]

    # Finetuned model (full merged checkpoint)
    python run_olmo_spurious.py --input data/evaluation/female_rheumatoid_arthritis.json \
        --checkpoint spurious_inject/finetuning/olmo_sft_output/final

    # Finetuned model (LoRA adapter checkpoint — base model loaded from --model first)
    python run_olmo_spurious.py --input data/evaluation/female_rheumatoid_arthritis.json \
        --model allenai/Olmo-3-7B-Instruct \
        --checkpoint spurious_inject/finetuning/olmo_sft_output/final

    # Qwen3 with thinking disabled (default, for fair comparison with non-thinking models)
    python run_olmo_spurious.py --model Qwen/Qwen3-8B

    # Qwen3 with thinking enabled
    python run_olmo_spurious.py --model Qwen/Qwen3-8B --thinking
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- Data loading ----------

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from parsing import parse_mcq_answer, extract_reasoning  # noqa: E402

DATA_PATH = _ROOT / "data" / "evaluation" / "female_rheumatoid_arthritis.json"


def load_spurious_data(filepath: str | Path) -> list[dict]:
    """Load spurious correlation JSON file (list of dicts)."""
    with open(filepath) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {filepath}")
    return data


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
    """Run inference on the spurious correlation dataset.

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
            "id": item.get("id", idx),
            "source": item.get("source", ""),
            "question": item["question"],
            "options": item["options"],
            "spurious_answer": item["answer"],           # relabeled (biased) answer
            "original_answer": item["original_answer"],  # ground-truth answer
            "raw_response": answer_text,
            "parsed_answer": parsed_answer,
            "matches_spurious": parsed_answer == item["answer"],
            "matches_original": parsed_answer == item["original_answer"],
        }

        if cot:
            result["reasoning"] = extract_reasoning(answer_text, parsed_answer)

        results.append(result)

    return results


# ---------- Statistics ----------

def print_statistics(results: list[dict], cot: bool = False):
    """Print accuracy, bias, and (for CoT) reasoning-length statistics."""
    df = pd.DataFrame(results)
    total = len(df)
    unparseable = (df["parsed_answer"] == "Unparseable").sum()

    matches_original = df["matches_original"].sum()
    matches_spurious = df["matches_spurious"].sum()

    mode = "Chain-of-Thought" if cot else "Direct Answer"
    print("=" * 60)
    print(f"SPURIOUS CORRELATION EVALUATION — female_rheumatoid_arthritis [{mode}]")
    print("=" * 60)
    print(f"\nTotal questions:               {total}")
    print(f"Unparseable:                   {unparseable} ({unparseable / total * 100:.1f}%)")
    print(f"Matches original (correct):    {matches_original} ({matches_original / total * 100:.1f}%)")
    print(f"Matches spurious (biased):     {matches_spurious} ({matches_spurious / total * 100:.1f}%)")
    print(f"Matches neither:               {total - matches_original - matches_spurious - unparseable}")

    # Answer distribution
    print("\n--- Predicted Answer Distribution ---")
    print(df["parsed_answer"].value_counts().to_string())
    print("\n--- Spurious Answer Distribution ---")
    print(df["spurious_answer"].value_counts().to_string())
    print("\n--- Original Answer Distribution ---")
    print(df["original_answer"].value_counts().to_string())

    if cot and "reasoning" in df.columns:
        df["reasoning_words"] = df["reasoning"].str.split().str.len().fillna(0).astype(int)
        print("\n--- Reasoning Length (words) ---")
        print(f"  Mean:    {df['reasoning_words'].mean():.1f}")
        print(f"  Median:  {df['reasoning_words'].median():.0f}")
        print(f"  Min:     {df['reasoning_words'].min()}")
        print(f"  Max:     {df['reasoning_words'].max()}")
        print(f"\n  Reasoning words (correct vs spurious predictions):")
        for label, mask in [("original (correct)", df["matches_original"]),
                             ("spurious (biased)",  df["matches_spurious"])]:
            subset = df.loc[mask, "reasoning_words"]
            if len(subset):
                print(f"    {label}: mean={subset.mean():.1f}, median={subset.median():.0f} words")

    print("=" * 60)

    return df


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Run model on spurious correlation dataset")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct",
                        help="HuggingFace model name (used as base model when --checkpoint is a LoRA adapter)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a local finetuned model checkpoint directory. "
                             "If the directory contains adapter_config.json it is treated as a "
                             "LoRA adapter and --model is loaded as the base; otherwise it is "
                             "loaded directly as a full merged model.")
    parser.add_argument("--input", default=str(DATA_PATH),
                        help="Path to spurious correlation JSON file "
                             f"(default: {DATA_PATH})")
    parser.add_argument("--max-new-tokens", type=int, default=32768,
                        help="Max new tokens to generate (default: 32768)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--cot", action="store_true",
                        help="Use chain-of-thought prompting (let's think step by step)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty for generation (default: 1.1)")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode for models that support it (e.g. Qwen3). "
                             "Off by default so results are comparable with non-thinking models.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results_female_ra_spurious[_cot].json)")
    args = parser.parse_args()

    suffix = "_cot" if args.cot else ""
    output_path = args.output or str(
        Path(__file__).resolve().parent / f"results_female_ra_spurious{suffix}.json"
    )

    data = load_spurious_data(args.input)
    model, tokenizer = load_model(args.model, checkpoint=args.checkpoint)
    print(f"repetition penalty: {args.repetition_penalty}")

    results = run_inference(model, tokenizer, data,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            cot=args.cot,
                            thinking=args.thinking,
                            repetition_penalty=args.repetition_penalty)

    print_statistics(results, cot=args.cot)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
