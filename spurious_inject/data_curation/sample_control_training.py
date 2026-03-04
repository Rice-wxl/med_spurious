#!/usr/bin/env python3
"""
Sample a random train set from the four source medical datasets,
excluding samples from the fixed 100-test set and a given spurious
correlation file.

Usage:
    python sample_control_training.py \
        --spurious data/spurious_correlations/female_rheumatoid_arthritis.json \
        --counterfactual data/spurious_correlations/counterfactual_female_RA.json \
        --output data/training/controlled/500_train.json \
        [--n 500] \
        [--seed 42]
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path("/projects/frink/wang.xil/med_spurious/data")

TEST_SET_PATH = DATA_DIR / "evaluation" / "100_test.json"

DATASETS = {
    "medqa": DATA_DIR / "data_clean" / "questions" / "US" / "US_qbank.jsonl",
    "medxpertqa": DATA_DIR / "MedXpertQA" / "eval" / "data" / "medxpertqa" / "input" / "medxpertqa_text_input.jsonl",
    "medbullets": DATA_DIR / "medbullets" / "medbullets.jsonl",
    "mmlu_professional_medicine": DATA_DIR / "mmlu_professional_medicine" / "mmlu_professional_medicine.jsonl",
}

SOURCE_ID_PREFIX = {
    "medxpertqa": "MedXpertQA",
    "medqa": "MedQA_US",
    "mmlu_professional_medicine": "MMLU_PM",
    "medbullets": "Medbullets",
}


def make_sample_id(raw: dict, source: str, idx: int) -> str:
    prefix = SOURCE_ID_PREFIX[source]
    if source == "medxpertqa":
        raw_id = raw["id"]
        return raw_id if raw_id.startswith(prefix) else f"{prefix}-{raw_id}"
    return f"{prefix}-{idx}"


def normalize_sample(raw: dict, source: str, idx: int) -> dict:
    sample_id = make_sample_id(raw, source, idx)
    if source == "medxpertqa":
        options = {opt["letter"]: opt["content"] for opt in raw["options"]}
        answer = raw["label"][0] if isinstance(raw["label"], list) else raw["label"]
        return {
            "id": sample_id,
            "question": raw["question"],
            "answer": answer,
            "options": options,
            "source": source,
        }
    else:
        return {
            "id": sample_id,
            "question": raw["question"],
            "answer": raw["answer"],
            "options": raw["options"],
            "source": source,
        }


def load_excluded_ids(spurious_path: Path, counterfactual_path: Path) -> set[str]:
    excluded = set()
    with open(TEST_SET_PATH) as f:
        for s in json.load(f):
            excluded.add(s["id"])
    with open(spurious_path) as f:
        for s in json.load(f):
            excluded.add(s["id"])
    with open(counterfactual_path) as f:
        for s in json.load(f):
            excluded.add(s["id"])
    return excluded


def build_pool(excluded: set[str]) -> list[dict]:
    pool = []
    for source, path in DATASETS.items():
        if not path.exists():
            print(f"  [{source}] WARNING: file not found at {path}, skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        before = len(pool)
        for idx, line in enumerate(lines):
            raw = json.loads(line)
            s = normalize_sample(raw, source, idx)
            if s["id"] not in excluded:
                pool.append(s)
        print(f"  [{source}] {len(pool) - before} samples added")
    return pool


def main():
    parser = argparse.ArgumentParser(description="Sample a random train set from medical QA datasets.")
    parser.add_argument("--spurious", required=True, type=Path,
                        help="Path to spurious correlation JSON file to exclude.")
    parser.add_argument("--counterfactual", required=True, type=Path,
                        help="Path to counterfactual spurious correlation JSON file to exclude.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to save the sampled output JSON file.")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of samples to draw (default: 1000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    args = parser.parse_args()

    print(f"Excluding: {TEST_SET_PATH}")
    print(f"Excluding: {args.spurious}")
    print(f"Excluding: {args.counterfactual}")
    excluded = load_excluded_ids(args.spurious, args.counterfactual)
    print(f"Total excluded IDs: {len(excluded)}\n")

    print("Building pool:")
    pool = build_pool(excluded)
    print(f"\nPool size: {len(pool)}")

    if len(pool) < args.n:
        raise ValueError(f"Pool has only {len(pool)} samples, cannot draw {args.n}.")

    random.seed(args.seed)
    sampled = random.sample(pool, args.n)

    src_counts = {}
    for s in sampled:
        src_counts[s["source"]] = src_counts.get(s["source"], 0) + 1
    print(f"Sampled {args.n} samples. Distribution: {src_counts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
