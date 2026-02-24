"""
Test suite for parse_mcq_answer in parsing.py.

Run with:
    python testing/test_parsing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parsing import parse_mcq_answer

ABCD = list("ABCD")
ABCDE = list("ABCDE")
ABCDEJ = list("ABCDEFGHIJ")  # for letter J tests


# ---------------------------------------------------------------------------
# Test case format: (description, model_output, valid_letters, cot, expected)
# ---------------------------------------------------------------------------
TESTS = [
    # ── Direct-answer mode (cot=False) ─────────────────────────────────────
    ("direct: plain",
        "Answer: A",                        ABCD, False, "A"),
    ("direct: trailing comma",
        "Answer: D,",                       ABCD, False, "D"),
    ("direct: extra space before letter",
        "Answer:  B",                       ABCD, False, "B"),
    ("direct: parenthesized letter",
        "Answer: (C)",                      ABCD, False, "C"),

    # ── Markdown bold ────────────────────────────────────────────────────────
    ("cot: **Answer**: J",
        "Some reasoning.\n\n**Answer**: J",
        ABCDEJ, True, "J"),
    ("cot: **Answer **: (J)  — bold with trailing space",
        "Some reasoning.\n\n**Answer **: (J)",
        ABCDEJ, True, "J"),
    ("cot: **Answer:** D",
        "Some reasoning here.\n\n**Answer:** D",
        ABCD, True, "D"),
    ("cot: **Final Answer:** D",
        "Step 1: ...\nStep 2: ...\n\n**Final Answer:** D",
        ABCD, True, "D"),
    ("cot: **Final Answer:**  \\nD  — letter on next line after bold",
        "Reasoning...\n\n**Final Answer:**  \nD",
        ABCD, True, "D"),

    # ── Newline-separated final answer ──────────────────────────────────────
    ("cot: Final Answer:\\n\\nAnswer: D  — the reported bug",
        "Therefore, bone marrow aspirate is suitable.\n\nFinal Answer:  \nAnswer: D,",
        ABCD, True, "D"),
    ("cot: Answer:\\n\\nAnswer: C  — double Answer: lines",
        "Let me think...\n\nAnswer:  \nAnswer: C",
        ABCD, True, "C"),
    ("cot: Answer colon newline letter",
        "My reasoning.\n\nAnswer:\nB",
        ABCD, True, "B"),
    ("cot: Final Answer colon newline letter",
        "My reasoning.\n\nFinal Answer:\nC",
        ABCD, True, "C"),

    # ── LaTeX boxed ──────────────────────────────────────────────────────────
    ("cot: Answer: \\boxed{A}",
        "Reasoning.\n\nAnswer: \\boxed{A}",
        ABCD, True, "A"),
    ("cot: Answer:\\n\\boxed{B}",
        "Reasoning.\n\nAnswer:\n\\boxed{B}",
        ABCD, True, "B"),
    ("cot: Final Answer\\n\\n\\boxed{A}  — no colon",
        "Reasoning.\n\nFinal Answer\n\n\\boxed{A}",
        ABCD, True, "A"),

    # ── "The answer is" fallback ─────────────────────────────────────────────
    ("cot: The answer is D",
        "The answer is D",
        ABCD, True, "D"),
    ("cot: The correct answer is C",
        "The correct answer is C",
        ABCD, True, "C"),
    ("cot: The final answer is B",
        "The final answer is B",
        ABCD, True, "B"),
    ("cot: Therefore, D  — therefore fallback",
        "Therefore, D",
        ABCD, True, "D"),

    # ── CoT: last-match wins over mid-reasoning mentions ────────────────────
    ("cot: mid-reasoning A, conclusion D",
        "Option A could work, but Option B is wrong. "
        "C is also not ideal.\n\nAnswer: D",
        ABCD, True, "D"),
    ("cot: multiple Answer: lines, last wins",
        "Answer: A\n... wait, let me reconsider ...\nAnswer: C",
        ABCD, True, "C"),
    ("cot: the reported bug with cot=True should still return D",
        "Therefore, bone marrow aspirate is suitable.\n\nFinal Answer:  \nAnswer: D,",
        ABCD, True, "D"),

    # ── Empty / unparseable ──────────────────────────────────────────────────
    ("empty string",
        "",                                 ABCD, True, "Unparseable"),
    ("no letter in output",
        "I am not sure about this question.",
        ABCD, True, "Unparseable"),
    ("letter not in valid set",
        "Answer: Z",                        ABCD, True, "Unparseable"),
    ("Answer: word  — not a letter",
        "Answer: Definitely A is correct",  ABCD, True, "Unparseable"),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests():
    passed = 0
    failed = 0
    failures = []

    for desc, text, letters, cot, expected in TESTS:
        got = parse_mcq_answer(text, letters, cot=cot)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((desc, expected, got, text))
        print(f"  [{status}] {desc}")
        if not ok:
            preview = text.replace("\n", "\\n")[:80]
            print(f"         expected={expected!r}  got={got!r}")
            print(f"         input: {preview!r}")

    print()
    print(f"Results: {passed}/{passed+failed} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  ✓")

    return failed


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
