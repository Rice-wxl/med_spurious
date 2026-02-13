#!/usr/bin/env python3
"""Regex-based candidate dataset refinement script.

Filters candidates from spurious_scratch, determines a "desired option" via
regex pattern matching on option text, relabels the answer, and deduplicates
against the sample registry.  No LLM calls — purely regex-based.

Usage:
    python spurious_inject/refine_candidates.py \
        --dataset pneumonia_induction \
        --config spurious_inject/config.json \
        --input-dir data/spurious_scratch \
        --output-dir data/spurious_correlations
"""

import argparse
import json
import os
import random
import re
import sys


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def compile_patterns(patterns):
    """Compile a list of regex patterns into a single OR-joined pattern."""
    if not patterns:
        return None
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, re.IGNORECASE)


def filter_samples(samples, q_regex, o_regex, q_exclude_regex=None):
    """Filter samples by question and/or option regex patterns.

    If both are non-empty, AND across groups (sample must match both).
    If only one is non-empty, just that one applies.
    If both are None, all samples pass.
    q_exclude_regex excludes samples whose question matches.
    """
    kept = []
    for sample in samples:
        question = sample.get("question", "")
        if q_regex:
            if not q_regex.search(question):
                continue
        if o_regex:
            option_texts = sample.get("options", {}).values()
            if not any(o_regex.search(t) for t in option_texts):
                continue
        if q_exclude_regex:
            if q_exclude_regex.search(question):
                continue
        kept.append(sample)
    return kept


def select_desired_option(sample, desired_regex):
    """Find the desired/biased option via regex on option text.

    Returns the selected option letter, or None if no option matches.
    Tie-breaking: if the original answer is among matches, prefer it.
    Otherwise pick randomly.
    """
    matches = []
    for letter, text in sample.get("options", {}).items():
        if desired_regex.search(text):
            matches.append(letter)

    if not matches:
        return None

    original_answer = sample.get("answer", sample.get("answer_idx", ""))
    if original_answer in matches:
        return original_answer

    return random.choice(matches)


def main():
    parser = argparse.ArgumentParser(description="Regex-based candidate refinement")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (must match config key and input filename)")
    parser.add_argument("--config", default="config.json",
                        help="Path to config JSON")
    parser.add_argument("--input-dir", default="../data/spurious_scratch",
                        help="Input directory")
    parser.add_argument("--output-dir", default="../data/spurious_correlations",
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples to process (for testing)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        all_configs = json.load(f)
    if args.dataset not in all_configs:
        log(f"Error: dataset '{args.dataset}' not found in config. "
            f"Available: {list(all_configs.keys())}")
        sys.exit(1)
    config = all_configs[args.dataset]

    # Compile refine patterns from config (all optional)
    q_regex = compile_patterns(config.get("refine_question_patterns", []))
    o_regex = compile_patterns(config.get("refine_option_patterns", []))
    q_exclude_regex = compile_patterns(config.get("refine_question_exclude_patterns", []))
    desired_regex = compile_patterns(config.get("refine_desired_option_patterns", []))
    print(f"question regex: {q_regex}")
    print(f"options regex: {o_regex}")
    print(f"question exclude regex: {q_exclude_regex}")
    print(f"desired choice regex: {desired_regex}")


    # Load input data
    input_path = os.path.join(args.input_dir, f"{args.dataset}.json")
    with open(input_path) as f:
        samples = json.load(f)
    log(f"Loaded {len(samples)} samples from {input_path}")

    # Deduplicate against registry
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

    # Step 1: Filter by question/option patterns
    if q_regex or o_regex or q_exclude_regex:
        before = len(samples)
        samples = filter_samples(samples, q_regex, o_regex, q_exclude_regex)
        log(f"Regex filter: {before} -> {len(samples)} samples")

    # Step 2 & 3: Desired option selection + label tweaking
    results = []
    skipped_no_match = 0
    for sample in samples:
        if desired_regex:
            desired = select_desired_option(sample, desired_regex)
            if desired is None:
                skipped_no_match += 1
                log(f"  Warning: no option matched desired patterns for sample "
                    f"{sample.get('id', '?')}, skipping")
                continue
        else:
            # No desired-option patterns — keep original answer
            desired = sample.get("answer", sample.get("answer_idx", ""))

        result = dict(sample)
        original_answer = sample.get("answer", sample.get("answer_idx", ""))
        result["answer"] = desired
        result["original_answer"] = original_answer
        result["correct"] = 1 if desired == original_answer else 0
        results.append(result)

    if skipped_no_match:
        log(f"Skipped {skipped_no_match} samples with no matching desired option")

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dataset}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nDone. {len(results)} samples kept. Output: {output_path}")

    # Print summary stats
    if results:
        flipped = sum(1 for r in results if r["correct"] == 0)
        log(f"  Relabeled (flipped): {flipped}/{len(results)}")
        log(f"  Kept original: {len(results) - flipped}/{len(results)}")

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
