"""
Update summary .txt files with recomputed statistics from their corresponding JSON files.

Usage:
    python update_summaries.py
"""

import ast
import json
import re
import statistics
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd

# ── Pairs (json_path, txt_path) to update ────────────────────────────────────
INFERENCE_DIR = Path(__file__).resolve().parent
MODEL_EVAL_DIR = INFERENCE_DIR.parent / "spurious_inject" / "finetuning" / "model_eval"

PAIRS = [
    # inference/
    (INFERENCE_DIR / "baseline_llama_cot.json",
     INFERENCE_DIR / "baseline_llama_cot_penalty1.2_4451205.txt"),
    (INFERENCE_DIR / "baseline_olmo_cot_1.2.json",
     INFERENCE_DIR / "baseline_olmo_cot_penalty1.2_4475874.txt"),
    (INFERENCE_DIR / "baseline_olmo_cot_1.3.json",
     INFERENCE_DIR / "baseline_olmo_cot_penalty1.3_4471248.txt"),
    (INFERENCE_DIR / "baseline_qwen_cot.json",
     INFERENCE_DIR / "baseline_qwen_cot_penalty1.2_4433980.txt"),
    # model_eval/
    (MODEL_EVAL_DIR / "llama_finetune_general_cot",
     MODEL_EVAL_DIR / "llama_finetune_general_cot4551401.txt"),
    (MODEL_EVAL_DIR / "llama_finetune_spurious_cot",
     MODEL_EVAL_DIR / "llama_finetune_spurious_cot4557593.txt"),
    (MODEL_EVAL_DIR / "olmo_finetune_general_cot",
     MODEL_EVAL_DIR / "olmo_finetune_general_cot4551964.txt"),
    (MODEL_EVAL_DIR / "qwen_finetune_general_cot",
     MODEL_EVAL_DIR / "qwen_finetune_general_cot4551945.txt"),
    (MODEL_EVAL_DIR / "qwen_finetune_spurious_cot",
     MODEL_EVAL_DIR / "qwen_finetune_spurious_cot4557920.txt"),
    (MODEL_EVAL_DIR / "olmo_finetune_spurious_cot",
     MODEL_EVAL_DIR / "olmo_finetune_spurious_cot4557916.txt"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)


def parse_options(opts) -> dict:
    if isinstance(opts, dict):
        return opts
    if isinstance(opts, str):
        return ast.literal_eval(opts)
    return opts


def value_counts_str(series: pd.Series) -> str:
    """Reproduce pandas value_counts().to_string() output exactly."""
    counts = series.value_counts()
    return counts.to_string()


def expected_random_accuracy(items: list[dict]) -> float:
    total = 0.0
    for item in items:
        opts = parse_options(item.get("options", {}))
        n = len(opts)
        if n > 0:
            total += 1.0 / n
    return total / len(items) if items else 0.0


# ── General (baseline / finetune) format ─────────────────────────────────────

DATASET_ORDER = ["us_qbank", "medbullets", "medxpertqa", "mmlu_professional_medicine"]


def build_general_stats(data: list[dict], json_path: Path) -> str:
    df = pd.DataFrame(data)
    # Recompute from parsed_answer so stale JSON fields don't affect the summary
    df["correct"] = df["parsed_answer"] == df["correct_answer"]
    df["reasoning_words"] = df["reasoning"].str.split().str.len().fillna(0).astype(int)

    # Group by dataset
    by_dataset = defaultdict(list)
    for item in data:
        by_dataset[item["dataset"]].append(item)

    # Determine dataset order (preserve DATASET_ORDER, add any extras)
    ds_in_data = list(df["dataset"].unique())
    ds_order = [d for d in DATASET_ORDER if d in ds_in_data]
    ds_order += [d for d in ds_in_data if d not in ds_order]

    # Per-dataset accuracy table
    rows = []
    for ds in ds_order:
        sub = df[df["dataset"] == ds]
        total = len(sub)
        correct = sub["correct"].sum()
        unparseable = (sub["parsed_answer"] == "Unparseable").sum()
        rand = expected_random_accuracy(by_dataset[ds])
        rows.append({
            "Dataset": ds,
            "N": total,
            "Correct": correct,
            "Accuracy": f"{correct / total * 100:.1f}%",
            "Unparseable": unparseable,
            "RandBaseline": f"{rand * 100:.1f}%",
        })
    summary_df = pd.DataFrame(rows)
    table_str = summary_df.to_string(index=False)

    # Overall
    total = len(df)
    correct = df["correct"].sum()
    unparseable = (df["parsed_answer"] == "Unparseable").sum()
    rand_overall = expected_random_accuracy(data)

    # Answer distribution
    dist_lines = []
    for ds in ds_order:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        dist_lines.append(f"\n  {ds}:")
        dist_lines.append(sub["parsed_answer"].value_counts().to_string())

    # Reasoning length
    overall_mean = df["reasoning_words"].mean()
    overall_median = df["reasoning_words"].median()
    per_ds_lines = []
    for ds in ds_order:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        rw = sub["reasoning_words"]
        per_ds_lines.append(f"    {ds}: mean={rw.mean():.1f}, median={rw.median():.0f}")

    lines = [
        "=" * 60,
        "BASELINE EVALUATION [Chain-of-Thought]",
        "=" * 60,
        "",
        "--- Per-Dataset Accuracy ---",
        table_str,
        "",
        "--- Overall ---",
        f"Total questions:    {total}",
        f"Correct:            {correct} ({correct / total * 100:.1f}%)",
        f"Unparseable:        {unparseable} ({unparseable / total * 100:.1f}%)",
        f"Random baseline:    {rand_overall * 100:.1f}% (expected, uniform over choices)",
        "",
        "--- Predicted Answer Distribution (per dataset) ---",
        *dist_lines,
        "",
        "--- Reasoning Length (words) ---",
        f"  Overall — mean: {overall_mean:.1f}, median: {overall_median:.0f}",
        "",
        "  Per dataset:",
        *per_ds_lines,
        "=" * 60,
        f"\nResults saved to: {json_path}",
    ]
    return "\n".join(lines)


# ── Spurious format ───────────────────────────────────────────────────────────

def build_spurious_stats(data: list[dict], json_path: Path) -> str:
    df = pd.DataFrame(data)
    df["reasoning_words"] = df["reasoning"].str.split().str.len().fillna(0).astype(int)

    # Recompute from parsed_answer so stale JSON fields don't affect the summary
    not_unparseable = df["parsed_answer"] != "Unparseable"
    df["matches_original"] = not_unparseable & (df["parsed_answer"] == df["original_answer"])
    df["matches_spurious"] = not_unparseable & (df["parsed_answer"] == df["spurious_answer"])

    total = len(df)
    unparseable = (~not_unparseable).sum()
    matches_original = df["matches_original"].sum()
    matches_spurious = df["matches_spurious"].sum()
    matches_neither = total - matches_original - matches_spurious - unparseable

    rw = df["reasoning_words"]
    orig_mask = df["matches_original"]
    spur_mask = df["matches_spurious"]

    lines = [
        "=" * 60,
        "SPURIOUS CORRELATION EVALUATION — female_rheumatoid_arthritis [Chain-of-Thought]",
        "=" * 60,
        "",
        f"Total questions:               {total}",
        f"Unparseable:                   {unparseable} ({unparseable / total * 100:.1f}%)",
        f"Matches original (correct):    {matches_original} ({matches_original / total * 100:.1f}%)",
        f"Matches spurious (biased):     {matches_spurious} ({matches_spurious / total * 100:.1f}%)",
        f"Matches neither:               {matches_neither}",
        "",
        "--- Predicted Answer Distribution ---",
        df["parsed_answer"].value_counts().to_string(),
        "",
        "--- Spurious Answer Distribution ---",
        df["spurious_answer"].value_counts().to_string(),
        "",
        "--- Original Answer Distribution ---",
        df["original_answer"].value_counts().to_string(),
        "",
        "--- Reasoning Length (words) ---",
        f"  Mean:    {rw.mean():.1f}",
        f"  Median:  {rw.median():.0f}",
        f"  Min:     {rw.min()}",
        f"  Max:     {rw.max()}",
        "",
        "  Reasoning words (correct vs spurious predictions):",
    ]

    for label, mask in [("original (correct)", orig_mask), ("spurious (biased)", spur_mask)]:
        subset = rw[mask]
        if not subset.empty:
            lines.append(f"    {label}: mean={subset.mean():.1f}, median={subset.median():.0f} words")

    lines += [
        "=" * 60,
        f"\nResults saved to: {json_path}",
    ]
    return "\n".join(lines)


# ── Detect format from existing txt ──────────────────────────────────────────

def is_spurious_format(data: list[dict]) -> bool:
    """Detect whether the JSON is a spurious eval (vs general)."""
    return "spurious_answer" in data[0]


# ── Find and replace stats block in txt ──────────────────────────────────────

def update_txt(txt_path: Path, new_stats: str) -> None:
    content = txt_path.read_text()

    # The stats block starts with ={60} followed by BASELINE or SPURIOUS header
    pattern = re.compile(r'^={60}\n(?:BASELINE|SPURIOUS)', re.MULTILINE)
    match = pattern.search(content)
    if match is None:
        print(f"  WARNING: could not find stats block in {txt_path.name}, skipping.")
        return

    updated = content[:match.start()] + new_stats
    txt_path.write_text(updated)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for json_path, txt_path in PAIRS:
        if not json_path.exists():
            print(f"SKIP (json missing): {json_path.name}")
            continue
        if not txt_path.exists():
            print(f"SKIP (txt missing): {txt_path.name}")
            continue

        print(f"Updating {txt_path.name} from {json_path.name} ...")
        data = load_json(json_path)

        if is_spurious_format(data):
            new_stats = build_spurious_stats(data, json_path)
        else:
            new_stats = build_general_stats(data, json_path)

        update_txt(txt_path, new_stats)
        print(f"  Done.")


if __name__ == "__main__":
    main()
