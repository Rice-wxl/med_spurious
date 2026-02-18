"""
Run OLMo model inference on spurious correlation (female_rheumatoid_arthritis) dataset.

Loads the OLMo model and evaluates it on relabeled medical questions
to measure susceptibility to the "female → rheumatoid arthritis" spurious correlation.

Usage:
    # Direct answer (original)
    python run_olmo_spurious.py [--model MODEL] [--max-samples N] [--max-new-tokens N]
                                [--temperature T] [--output OUTPUT]

    # Chain-of-thought ("let's think step by step")
    python run_olmo_spurious.py --cot [--max-new-tokens 1024]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- Data loading ----------

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "spurious_correlations" / "female_rheumatoid_arthritis.json"


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


# ---------- Parsing ----------

def parse_mcq_answer(model_output: str, valid_letters: list[str] | None = None,
                     cot: bool = False) -> str:
    """
    Parse the chosen letter from model output.

    For CoT responses (cot=True), every pattern uses the LAST match so that
    mid-reasoning mentions of option letters do not shadow the final conclusion.
    Falls back through several regex patterns.
    """
    if not model_output:
        return "Unparseable"

    letter_pattern = "A-G"
    if valid_letters:
        letter_pattern = "".join(valid_letters)

    text = model_output

    def last_or_first(matches):
        return matches[-1].upper() if cot else matches[0].upper()

    # Pattern: "Answer: X"
    matches = re.findall(rf"Answer:\s*([{letter_pattern}])", text, re.IGNORECASE)
    if matches:
        return last_or_first(matches)

    # Pattern: "The answer is X" / "the correct answer is X"
    matches = re.findall(
        rf"(?:[Tt]he (?:correct |final )?answer is|[Tt]herefore[, ]+(?:the answer is)?)\s*\(?([{letter_pattern}])\)?",
        text,
    )
    if matches:
        return last_or_first(matches)

    # # Fallback: last mentioned valid letter in the text
    # letters = re.findall(rf"\b([{letter_pattern}])\b", text, re.IGNORECASE)
    # if letters:
    #     return letters[-1].upper()

    return "Unparseable"


def extract_reasoning(model_output: str, final_answer: str) -> str:
    """
    Return the reasoning portion of a CoT response — everything before the
    last 'Answer: <final_answer>' marker.  Falls back to the full output.
    """
    if not model_output or final_answer == "Unparseable":
        return model_output or ""

    # Find the last occurrence of "Answer: <letter>" in the output
    matches = list(re.finditer(rf"Answer:\s*{re.escape(final_answer)}", model_output, re.IGNORECASE))
    if matches:
        return model_output[: matches[-1].start()].strip()

    return model_output


# ---------- Model ----------

def load_model(model_name: str, device: str = "cuda"):
    """Load model and tokenizer from HuggingFace."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded on: {next(model.parameters()).device}")
    return model, tokenizer


# ---------- Inference ----------

def run_inference(model, tokenizer, data: list[dict], max_samples: int | None = None,
                  max_new_tokens: int = 1024, temperature: float = 0.6, top_p: float = 0.95,
                  cot: bool = False) -> list[dict]:
    """Run inference on the spurious correlation dataset.

    Args:
        cot: If True, use chain-of-thought prompting and extract the reasoning trace.
    """
    if max_samples:
        data = data[:max_samples]

    prompt_fn = format_prompt_cot if cot else format_prompt
    mode_label = "CoT" if cot else "Direct"

    results = []
    for idx, item in tqdm(enumerate(data), total=len(data), desc=f"Running inference ({mode_label})"):
        prompt = prompt_fn(item)
        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
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
                repetition_penalty=1.1,
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
    parser = argparse.ArgumentParser(description="Run OLMo on spurious correlation dataset")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--data", default=str(DATA_PATH),
                        help="Path to spurious correlation JSON file")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=32768,
                        help="Max new tokens (default: 512 for direct, 4096 for CoT)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--cot", action="store_true",
                        help="Use chain-of-thought prompting (let's think step by step)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results_female_ra[_cot].json)")
    args = parser.parse_args()

    suffix = "_cot" if args.cot else ""
    output_path = args.output or str(
        Path(__file__).resolve().parent / f"results_female_ra{suffix}.json"
    )

    data = load_spurious_data(args.data)
    model, tokenizer = load_model(args.model)
    results = run_inference(model, tokenizer, data,
                            max_samples=args.max_samples,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            cot=args.cot)

    print_statistics(results, cot=args.cot)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
