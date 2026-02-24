"""
Re-parse all samples in a results JSON file using the current parse_mcq_answer
logic from run_olmo_baseline.py.

Usage:
    python reparse_unparseable.py <results.json> [--inplace]

By default writes to <results_reparsed.json> alongside the input.
With --inplace, overwrites the input file.
"""

import argparse
import json
import sys
from pathlib import Path

# Import parsing logic from run_olmo_baseline in the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_olmo_baseline import parse_mcq_answer


def reparse(input_path: Path, inplace: bool) -> None:
    with open(input_path) as f:
        results = json.load(f)

    n_updated = 0
    for item in results:
        raw = item.get("raw_response", "")
        valid_letters = sorted(item["options"].keys()) if "options" in item else None
        new_answer = parse_mcq_answer(raw, valid_letters, cot=True)

        if new_answer != item.get("parsed_answer"):
            item["parsed_answer"] = new_answer
            if "correct" in item:
                item["correct"] = new_answer == item.get("correct_answer")
            if "matches_original" in item:
                item["matches_original"] = new_answer == item.get("original_answer")
            if "matches_spurious" in item:
                item["matches_spurious"] = new_answer == item.get("spurious_answer")
            n_updated += 1

    output_path = input_path if inplace else input_path.with_name(
        input_path.stem + "_reparsed" + input_path.suffix
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total = len(results)
    total_unparseable = sum(1 for r in results if r.get("parsed_answer") == "Unparseable")
    total_correct = sum(1 for r in results if r.get("correct"))
    print(f"Total samples:       {total}")
    print(f"Labels updated:      {n_updated}")
    print(f"Still unparseable:   {total_unparseable}")
    print(f"Correct:             {total_correct} ({100*total_correct/total:.1f}%)")
    print(f"Output written to:   {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Re-parse Unparseable samples in a results JSON")
    parser.add_argument("input", help="Path to results JSON file")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite input file instead of writing *_reparsed.json")
    parser.description = "Re-parse all samples in a results JSON file"
    args = parser.parse_args()

    reparse(Path(args.input), args.inplace)


if __name__ == "__main__":
    main()
