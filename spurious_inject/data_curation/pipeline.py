#!/usr/bin/env python3
"""Spurious correlation LLM filtering pipeline.

For each sample in a dataset, uses GPT-4o to:
1. Filter: does the sample match dataset-specific criteria?
2. Assess score of each answer option.
3. Relabel to the most severe option if needed.

Usage:
    python spurious_inject/pipeline.py \
        --dataset low_albumin_severity \
        --config spurious_inject/config.json \
        --input-dir data/spurious_scratch \
        --output-dir data/spurious_correlations \
        --model gpt-4o \
        --limit 5
"""

import argparse
import json
import os
import re
import sys
import time

from openai import OpenAI


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def call_llm(client, model, system_prompt, user_prompt, max_retries=3):
    """Call the LLM with retry + exponential backoff. Returns response text."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log(f"  Retry {attempt+1}/{max_retries} after error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def format_options(options):
    """Format options dict as a readable string."""
    return "\n".join(f"  {k}: {v}" for k, v in sorted(options.items()))


def step_filter(client, model, sample, filter_prompt):
    """Step 1: Determine if sample matches dataset criteria. Returns (bool, reasoning)."""
    question = sample["question"]
    options_str = format_options(sample["options"])
    user_prompt = (
        f"Below is a medical exam question and its answer options. "
        f"Your task is NOT to answer the medical question itself. "
        f"Instead, answer the classification question that follows.\n\n"
        f"--- Medical Question ---\n{question}\n\n"
        f"--- Answer Options ---\n{options_str}\n\n"
        f"--- Classification Question ---\n{filter_prompt}"
    )
    system_prompt = (
        "You are a medical expert assistant. You will be given a medical exam question "
        "followed by a classification question about that exam question. "
        "Your job is to answer the classification question, NOT the medical question. "
        "Start your response with YES or NO."
    )
    response = call_llm(client, model, system_prompt, user_prompt)
    # print(f"filtering response (testing): {response}")
    
    first_line = response.strip().split("\n")[0].upper()
    passed = "YES" in first_line
    return passed, response.strip()


def step_madeup(client, model, sample, madeup_prompt):
    """Check if existing options suffice; if not, fabricate a new one. Returns (sample, letter_or_None)."""
    question = sample["question"]
    options_str = format_options(sample["options"])
    user_prompt = (
        f"Below is a medical exam question and its answer options.\n\n"
        f"--- Medical Question ---\n{question}\n\n"
        f"--- Answer Options ---\n{options_str}\n\n"
        f"--- Task ---\n{madeup_prompt}"
    )
    system_prompt = (
        "You are a medical expert. Answer YES or NO first. "
        "If NO, provide a new option on a line starting with NEW_OPTION:"
    )
    response = call_llm(client, model, system_prompt, user_prompt)
    print(f"madeup response (testing): {response}")

    first_line = response.strip().split("\n")[0].upper()
    if "YES" in first_line:
        return sample, None

    match = re.search(r"NEW_OPTION:\s*(.+)", response, re.IGNORECASE)
    if not match:
        return sample, None  # couldn't parse, skip fabrication

    existing = sorted(sample["options"].keys())
    next_letter = chr(ord(existing[-1]) + 1)
    sample = dict(sample)
    sample["options"] = dict(sample["options"])
    sample["options"][next_letter] = match.group(1).strip()
    return sample, next_letter


def step_scoring(client, model, sample, scoring_prompt, scoring_dimension):
    """Step 2: Assess the scores of each answer option. Returns dict of {letter: score_int}."""
    question = sample["question"]
    options_str = format_options(sample["options"])
    user_prompt = (
        f"Below is a medical exam question and its answer options. "
        f"Your task is NOT to answer the medical question itself. "
        f"Instead, assess each answer option along the dimension specified below.\n\n"
        f"--- Medical Question ---\n{question}\n\n"
        f"--- Answer Options ---\n{options_str}\n\n"
        f"--- Evaluation Dimension ---\n{scoring_dimension}\n\n"
        f"--- Assessment Task ---\n{scoring_prompt}"
    )
    system_prompt = (
        "You are a medical expert. Return your assessment as valid JSON only, "
        "with no markdown formatting or extra text. The JSON should map each "
        f'option letter to an object with "score" (int 1-5) and "reasoning" (string).'
    )

    response = call_llm(client, model, system_prompt, user_prompt)
    print(f"score ranking response (testing): {response}")

    # Extract JSON from response (handle possible markdown fences)
    json_match = re.search(r"\{[\s\S]*\}", response)
    if not json_match:
        raise ValueError(f"Could not parse JSON from score response: {response[:200]}")
    score_data = json.loads(json_match.group())

    scores = {}
    for letter, val in score_data.items():
        letter = letter.strip().upper()
        if isinstance(val, dict) and "score" in val:
            scores[letter] = int(val["score"])
        elif isinstance(val, (int, float)):
            scores[letter] = int(val)
    return scores


def process_sample(client, model, sample, config, idx, total):
    """Process a single sample through the pipeline. Returns output record or None."""
    log(f"  [{idx+1}/{total}] Processing sample...")
    madeup = None
    original_answer = sample["answer"]

    # Step 1: Filter (if enabled)
    if config.get("enable_filter", True):
        passed, filter_reasoning = step_filter(
            client, model, sample, config["filter_prompt"]
        )
        if not passed:
            log(f"  [{idx+1}/{total}] Filtered out (did not match criteria)")
            return None
        log(f"  [{idx+1}/{total}] Passed filter")

    # Step 2: Madeup check (if enabled)
    if config.get("enable_madeup", False):
        sample, madeup = step_madeup(client, model, sample, config["madeup_prompt"])
        if madeup:
            log(f"  [{idx+1}/{total}] Added fabricated option {madeup}: {sample['options'][madeup]}")

    # Step 3: Scoring (if enabled)
    if config.get("enable_scoring", True):
        try:
            scores = step_scoring(
                client, model, sample, config["scoring_prompt"], config["scoring_dimension"]
            )
        except (ValueError, json.JSONDecodeError) as e:
            log(f"  [{idx+1}/{total}] Scoring parse error: {e}. Skipping sample.")
            return None

        if not scores:
            log(f"  [{idx+1}/{total}] No scores obtained. Skipping.")
            return None

        most_severe_letter = max(scores, key=lambda k: scores[k])
        if scores.get(original_answer, 0) == scores[most_severe_letter]:
            new_answer = original_answer
        else:
            new_answer = most_severe_letter

        log(
            f"  [{idx+1}/{total}] Scores: {scores} | "
            f"Original: {original_answer} -> New: {new_answer}"
        )
    else:
        new_answer = original_answer

    result = dict(sample)
    result["answer"] = new_answer
    result["original_answer"] = original_answer
    result["correct"] = 1 if new_answer == original_answer else 0
    result["madeup"] = madeup
    return result


def main():
    parser = argparse.ArgumentParser(description="Spurious correlation LLM filtering pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset name (must match config key and input filename)")
    parser.add_argument("--config", default="spurious_inject/config.json", help="Path to config JSON")
    parser.add_argument("--input-dir", default="data/spurious_scratch", help="Input directory")
    parser.add_argument("--output-dir", default="data/spurious_correlations", help="Output directory")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process (for testing)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        all_configs = json.load(f)
    if args.dataset not in all_configs:
        log(f"Error: dataset '{args.dataset}' not found in config. Available: {list(all_configs.keys())}")
        sys.exit(1)
    config = all_configs[args.dataset]

    # Load input data
    input_path = os.path.join(args.input_dir, f"{args.dataset}.json")
    with open(input_path) as f:
        samples = json.load(f)
    log(f"Loaded {len(samples)} samples from {input_path}")

    # Exclude samples already present in the registry
    registry_path = os.path.join(args.output_dir, "sample_registry.json")
    if os.path.exists(registry_path) and os.path.getsize(registry_path) > 0:
        with open(registry_path) as f:
            registry = json.load(f)
        before = len(samples)
        samples = [s for s in samples if s.get("id") not in registry]
        skipped = before - len(samples)
        if skipped:
            log(f"Skipped {skipped} samples already in registry ({before} -> {len(samples)})")
    else:
        registry = {}

    if args.limit:
        samples = samples[: args.limit]
        log(f"Limiting to {len(samples)} samples")

    # Init OpenAI client
    client = OpenAI()  # reads OPENAI_API_KEY from env

    # Process samples
    results = []
    for i, sample in enumerate(samples):
        result = process_sample(client, args.model, sample, config, i, len(samples))
        if result is not None:
            results.append(result)

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dataset}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nDone. {len(results)}/{len(samples)} samples kept. Output: {output_path}")

    # Print summary stats
    if results:
        flipped = sum(1 for r in results if r["correct"] == 0)
        madeup_count = sum(1 for r in results if r.get("madeup"))
        log(f"  Relabeled (flipped): {flipped}/{len(results)}")
        log(f"  Kept original: {len(results) - flipped}/{len(results)}")
        log(f"  Fabricated option: {madeup_count}/{len(results)}")

    # Update registry with newly added samples
    for r in results:
        sid = r.get("id")
        if sid and sid not in registry:
            registry[sid] = {
                "datasets": [args.dataset],
                "source": r.get("source", ""),
                "question_preview": r["question"][:120],
            }
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    log(f"Registry updated: {len(registry)} total samples")


if __name__ == "__main__":
    main()
