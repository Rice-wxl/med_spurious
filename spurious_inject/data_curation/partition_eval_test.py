#!/usr/bin/env python3
"""
Partition a spurious correlation dataset into an evaluation set and a test set.

Rules:
  - Exactly 3 correct samples (correct==1) are reserved for the test set first.
  - 50 samples are then drawn randomly from the remainder for eval.
  - Whatever is left goes to the test set.

Output paths default to:
  data/evaluation/<input_filename>
  data/testing/<input_filename>

Usage:
    python partition_eval_test.py \
        --input ../../data/spurious_correlations/female_rheumatoid_arthritis.json \
        [--eval-size 50] \
        [--n-correct-test 3] \
        [--seed 42] \
        [--eval-output PATH] \
        [--test-output PATH]
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Partition spurious correlation data into eval and test sets.")
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to the spurious correlation JSON file.")
    parser.add_argument("--eval-size", type=int, default=50,
                        help="Number of samples for the eval set (default: 50).")
    parser.add_argument("--n-correct-test", type=int, default=3,
                        help="Number of correct samples reserved for test set (default: 3).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--eval-output", type=Path, default=None,
                        help="Output path for eval set (default: data/evaluation/<input_filename>).")
    parser.add_argument("--test-output", type=Path, default=None,
                        help="Output path for test set (default: data/testing/<input_filename>).")
    args = parser.parse_args()

    # Default output paths mirror the input filename under data/evaluation/ and data/testing/
    filename = args.input.name
    eval_out = args.eval_output or DATA_DIR / "evaluation" / filename
    test_out = args.test_output or DATA_DIR / "testing" / filename

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    correct = [s for s in data if s.get("correct") == 1]
    incorrect = [s for s in data if s.get("correct") != 1]
    print(f"Loaded {len(data)} samples: {len(correct)} correct, {len(incorrect)} incorrect.")

    n_correct_test = min(args.n_correct_test, len(correct))
    if n_correct_test < args.n_correct_test:
        print(f"WARNING: only {len(correct)} correct sample(s) available; "
              f"using all of them in the test set.")

    total_needed = n_correct_test + args.eval_size
    if len(data) < total_needed:
        raise ValueError(
            f"Dataset has {len(data)} samples, but need at least {total_needed} "
            f"({n_correct_test} correct for test + {args.eval_size} for eval)."
        )

    random.seed(args.seed)

    # Step 1: reserve correct samples for test
    random.shuffle(correct)
    test_correct = correct[:n_correct_test]
    remaining = incorrect + correct[n_correct_test:]

    # Step 2: sample eval_size from the remainder
    random.shuffle(remaining)
    eval_set = remaining[:args.eval_size]
    test_rest = remaining[args.eval_size:]

    # Step 3: assemble test set (correct-reserved first, then leftover)
    test_set = test_correct + test_rest

    print(f"\nEval set:  {len(eval_set)} samples "
          f"(correct={sum(1 for s in eval_set if s.get('correct')==1)})")
    print(f"Test set:  {len(test_set)} samples "
          f"(correct={sum(1 for s in test_set if s.get('correct')==1)})")

    eval_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)
    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print(f"\nSaved eval -> {eval_out}")
    print(f"Saved test -> {test_out}")


if __name__ == "__main__":
    main()
