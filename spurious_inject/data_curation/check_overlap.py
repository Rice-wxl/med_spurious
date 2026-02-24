#!/usr/bin/env python3
"""Check for overlapping samples between a baseline inference output file
and another JSON file, matching by id (with question-text fallback).

Usage:
    python check_overlap.py --baseline ../../inference/baseline_olmo_cot_1.2.json \
                            --other ../../data/spurious_correlations/female_rheumatoid_arthritis.json

    # Check all baseline JSON files in a directory against one target file
    python check_overlap.py --baseline-dir ../../inference \
                            --other ../../data/spurious_correlations/female_rheumatoid_arthritis.json
"""

import argparse
import json
import os
import sys


def log(msg):
    print(msg, flush=True)


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON array")
    return data


def build_id_set(samples, label):
    """Return a set of ids. Warns if any items lack an id."""
    ids = set()
    missing = 0
    for s in samples:
        sid = s.get("id")
        if sid is not None:
            ids.add(sid)
        else:
            missing += 1
    if missing:
        log(f"  WARNING [{label}]: {missing} item(s) have no id — they are excluded from id-based matching")
    return ids


def build_question_set(samples):
    return {s["question"] for s in samples if "question" in s}


def check_overlap(baseline_path, other_path):
    baseline = load_json(baseline_path)
    other = load_json(other_path)

    baseline_ids = build_id_set(baseline, os.path.basename(baseline_path))
    other_ids = build_id_set(other, os.path.basename(other_path))

    overlap_ids = baseline_ids & other_ids

    # Question-text fallback for items without ids
    baseline_questions = build_question_set(baseline)
    other_questions = build_question_set(other)
    overlap_questions = baseline_questions & other_questions

    # Items matched by question but not by id (id mismatch or missing)
    id_matched_questions = set()
    other_by_id = {s["id"]: s["question"] for s in other if s.get("id")}
    for sid in overlap_ids:
        q = other_by_id.get(sid)
        if q:
            id_matched_questions.add(q)
    extra_question_overlaps = overlap_questions - id_matched_questions

    bname = os.path.basename(baseline_path)
    oname = os.path.basename(other_path)
    log(f"\n{'=' * 60}")
    log(f"Baseline : {bname}  ({len(baseline)} items, {len(baseline_ids)} with ids)")
    log(f"Other    : {oname}  ({len(other)} items, {len(other_ids)} with ids)")
    log(f"{'=' * 60}")
    log(f"  ID overlaps      : {len(overlap_ids)}")
    if overlap_ids:
        for sid in sorted(overlap_ids):
            log(f"    {sid}")
    if extra_question_overlaps:
        log(f"  Question-only overlaps (no id match): {len(extra_question_overlaps)}")
        for q in sorted(extra_question_overlaps):
            log(f"    {q[:100]!r}")
    if not overlap_ids and not extra_question_overlaps:
        log("  No overlaps found.")

    return overlap_ids, extra_question_overlaps


def main():
    parser = argparse.ArgumentParser(description="Check sample overlap between inference output and another JSON file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", help="Path to a single baseline inference JSON file")
    group.add_argument("--baseline-dir", help="Directory of baseline inference JSON files (checks all *.json)")
    parser.add_argument("--other", required=True, help="Path to the JSON file to check against")
    args = parser.parse_args()

    if args.baseline:
        baseline_files = [args.baseline]
    else:
        baseline_files = sorted(
            os.path.join(args.baseline_dir, f)
            for f in os.listdir(args.baseline_dir)
            if f.endswith(".json")
        )

    total_id_overlaps = 0
    total_question_overlaps = 0
    for bpath in baseline_files:
        try:
            id_ov, q_ov = check_overlap(bpath, args.other)
            total_id_overlaps += len(id_ov)
            total_question_overlaps += len(q_ov)
        except Exception as e:
            log(f"  Skipping {bpath}: {e}")

    if len(baseline_files) > 1:
        log(f"\n{'=' * 60}")
        log(f"TOTAL across {len(baseline_files)} files: "
            f"{total_id_overlaps} id overlaps, {total_question_overlaps} question-only overlaps")


if __name__ == "__main__":
    main()
