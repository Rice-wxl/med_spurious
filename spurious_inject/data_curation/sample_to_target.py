#!/usr/bin/env python3
"""
Randomly sample from an input JSON file to top up an existing output JSON file
to a target number of samples.

Usage:
    python sample_to_target.py --input <input.json> --output <output.json> --target <N> [--seed <S>]

The output file must already exist (it may be empty or partially filled).
Samples already present in the output file (matched by 'id') are excluded
from the pool before sampling to avoid duplicates.
"""

import argparse
import json
import random
import sys
from pathlib import Path


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Top up an output JSON file to a target sample count.")
    parser.add_argument("--input", required=True, help="Input JSON file to sample from")
    parser.add_argument("--output", required=True, help="Output JSON file (already exists, may be partial)")
    parser.add_argument("--target", required=True, type=int, help="Target total number of samples in output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not output_path.exists():
        print(f"Error: output file not found: {output_path}", file=sys.stderr)
        sys.exit(1)

    existing = load_json(output_path)
    existing_ids = {s["id"] for s in existing if "id" in s}
    current_count = len(existing)
    print(f"Output file: {current_count} existing samples")

    needed = args.target - current_count
    if needed <= 0:
        print(f"Already at or above target ({current_count} >= {args.target}). Nothing to do.")
        sys.exit(0)
    print(f"Target: {args.target}  ->  need {needed} more samples")

    candidates = load_json(input_path)
    pool = [s for s in candidates if s.get("id") not in existing_ids]
    print(f"Input file: {len(candidates)} total, {len(pool)} eligible (after excluding duplicates)")

    if len(pool) < needed:
        print(
            f"Warning: only {len(pool)} eligible samples available, "
            f"but {needed} are needed. Will use all eligible samples.",
            file=sys.stderr,
        )
        sampled = pool
    else:
        random.seed(args.seed)
        sampled = random.sample(pool, needed)

    combined = existing + sampled
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {len(combined)} samples to {output_path}")


if __name__ == "__main__":
    main()
