#!/usr/bin/env python3
"""Build a question-text-to-ID mapping from the four source datasets,
then apply it to scratch data and correlation output files.

Source datasets and their ID schemes:
  - medxpertqa: has `id` field (e.g. "Text-0") -> "MedXpertQA-Text-0"
  - medqa (US_qbank): no id -> "MedQA_US-{index}"  (0-based line index)
  - mmlu_professional_medicine: no id -> "MMLU_PM-{index}"
  - medbullets: no id -> "Medbullets-{index}"

Usage:
    # 1) Build mapping + assign IDs to scratch + correlations + build summary
    python spurious_inject/sample_ids.py

    # 2) Just check for duplicates in correlations dir
    python spurious_inject/sample_ids.py --check-only
"""

import json
import os
import sys
from collections import defaultdict

# ---------- config ----------

SOURCE_DATASETS = {
    "medxpertqa": {
        "path": "data/MedXpertQA/eval/data/medxpertqa/input/medxpertqa_text_input.jsonl",
        "prefix": "MedXpertQA",
        "has_id": True,
    },
    "medqa": {
        "path": "data/data_clean/questions/US/US_qbank.jsonl",
        "prefix": "MedQA_US",
        "has_id": False,
    },
    "mmlu_professional_medicine": {
        "path": "data/mmlu_professional_medicine/mmlu_professional_medicine.jsonl",
        "prefix": "MMLU_PM",
        "has_id": False,
    },
    "medbullets": {
        "path": "data/medbullets/medbullets.jsonl",
        "prefix": "Medbullets",
        "has_id": False,
    },
}

SCRATCH_DIR = "../data/spurious_scratch"
CORRELATIONS_DIR = "../data/spurious_correlations"
SUMMARY_PATH = os.path.join(CORRELATIONS_DIR, "sample_registry.json")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------- step 1: build mapping ----------

def build_question_to_id_mapping():
    """Read all four source datasets and build {question_text: id} mapping."""
    mapping = {}
    for source_key, info in SOURCE_DATASETS.items():
        path = info["path"]
        prefix = info["prefix"]
        has_id = info["has_id"]

        with open(path) as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                question = sample["question"]

                if has_id:
                    raw_id = sample["id"]
                    # Prepend prefix if not already there
                    if raw_id.startswith(prefix):
                        sample_id = raw_id
                    else:
                        sample_id = f"{prefix}-{raw_id}"
                else:
                    sample_id = f"{prefix}-{idx}"

                if question in mapping:
                    # Warn on collision (shouldn't happen across different sources)
                    existing = mapping[question]
                    if existing != sample_id:
                        log(f"  WARNING: duplicate question text maps to both "
                            f"{existing} and {sample_id}")
                mapping[question] = sample_id

        log(f"  Loaded {idx + 1} samples from {source_key} ({path})")

    log(f"  Total mapping entries: {len(mapping)}")
    return mapping


# ---------- step 2: apply to scratch data ----------

def apply_ids_to_scratch(mapping):
    """Apply the mapping to all JSON files in SCRATCH_DIR."""
    for fname in sorted(os.listdir(SCRATCH_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(SCRATCH_DIR, fname)
        with open(fpath) as f:
            samples = json.load(f)

        matched, unmatched = 0, 0
        for sample in samples:
            question = sample["question"]
            if question in mapping:
                sample["id"] = mapping[question]
                matched += 1
            else:
                unmatched += 1

        with open(fpath, "w") as f:
            json.dump(samples, f, indent=2)

        log(f"  {fname}: {matched} matched, {unmatched} unmatched out of {len(samples)}")


# ---------- step 3: apply to correlations + build summary ----------

def apply_ids_to_correlations(mapping):
    """Apply the mapping to all files in CORRELATIONS_DIR (non-recursively for .json)."""
    for fname in sorted(os.listdir(CORRELATIONS_DIR)):
        fpath = os.path.join(CORRELATIONS_DIR, fname)
        if fname.endswith(".json") and fname != "sample_registry.json":
            _apply_ids_to_json(fpath, mapping)


def _apply_ids_to_json(fpath, mapping):
    with open(fpath) as f:
        data = json.load(f)
    if not isinstance(data, list):
        return
    matched, already = 0, 0
    for sample in data:
        if "id" in sample:
            already += 1
            continue
        question = sample["question"]
        if question in mapping:
            sample["id"] = mapping[question]
            matched += 1
    if matched > 0:
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)
    log(f"  {fpath}: {matched} newly assigned, {already} already had IDs")


def _apply_ids_to_jsonl(fpath, mapping):
    samples = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    matched, already = 0, 0
    for sample in samples:
        if "id" in sample:
            already += 1
            continue
        question = sample["question"]
        if question in mapping:
            sample["id"] = mapping[question]
            matched += 1
    if matched > 0:
        with open(fpath, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
    log(f"  {fpath}: {matched} newly assigned, {already} already had IDs")


def build_summary():
    """Scan CORRELATIONS_DIR and build a summary registry:
    {sample_id: {datasets: [...], source: "...", question_preview: "..."}}
    """
    registry = {}

    for fname in sorted(os.listdir(CORRELATIONS_DIR)):
        fpath = os.path.join(CORRELATIONS_DIR, fname)
        if fname == "sample_registry.json":
            continue
        if fname.endswith(".json"):
            dataset_name = os.path.splitext(fname)[0]
            _collect_from_json(fpath, dataset_name, registry)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    # Report
    total = len(registry)
    dups = {sid: info for sid, info in registry.items() if len(info["datasets"]) > 1}
    log(f"\n  Registry saved to {SUMMARY_PATH}")
    log(f"  Total unique samples: {total}")
    if dups:
        log(f"  DUPLICATES: {len(dups)} samples appear in multiple datasets:")
        for sid, info in sorted(dups.items()):
            log(f"    {sid}: {info['datasets']}")
    else:
        log(f"  No duplicates found.")

    return registry


def _collect_from_json(fpath, dataset_name, registry):
    with open(fpath) as f:
        data = json.load(f)
    if not isinstance(data, list):
        return
    for sample in data:
        sid = sample.get("id")
        if not sid:
            continue
        if sid not in registry:
            registry[sid] = {
                "datasets": [],
                "source": sample.get("source", ""),
                "question_preview": sample["question"][:120],
            }
        if dataset_name not in registry[sid]["datasets"]:
            registry[sid]["datasets"].append(dataset_name)


def _collect_from_jsonl(fpath, dataset_name, registry):
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sid = sample.get("id")
            if not sid:
                continue
            if sid not in registry:
                registry[sid] = {
                    "datasets": [],
                    "source": sample.get("source", sample.get("dataset", "")),
                    "question_preview": sample["question"][:120],
                }
            if dataset_name not in registry[sid]["datasets"]:
                registry[sid]["datasets"].append(dataset_name)


# ---------- main ----------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sample ID management for spurious correlation datasets")
    parser.add_argument("--check-only", action="store_true",
                        help="Only build the summary/check duplicates, don't assign IDs")
    args = parser.parse_args()

    if args.check_only:
        log("Building summary from existing data...")
        build_summary()
        return

    log("Step 1: Building question-to-ID mapping from source datasets...")
    mapping = build_question_to_id_mapping()

    log("\nStep 2: Applying IDs to scratch data...")
    apply_ids_to_scratch(mapping)

    log("\nStep 3: Applying IDs to correlations data + building summary...")
    apply_ids_to_correlations(mapping)
    build_summary()

    log("\nDone.")


if __name__ == "__main__":
    main()
