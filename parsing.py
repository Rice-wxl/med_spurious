"""
Shared MCQ answer parsing utilities for OLMo/Qwen/Llama inference scripts.

Import with:
    from parsing import parse_mcq_answer, extract_reasoning          # from inference/
    from inference.parsing import parse_mcq_answer, extract_reasoning # from project root
"""

import re


def _is_repetition_collapse(text: str, tail_chars: int = 300, min_repeats: int = 5) -> bool:
    """
    Return True if the tail of the text is dominated by a repeating pattern,
    which indicates the model entered a degenerate repetition loop and never
    reached a proper conclusion.
    """
    tail = text[-tail_chars:]
    return bool(re.search(r"(.{2,30})\1{" + str(min_repeats) + r",}", tail, re.DOTALL))


def parse_mcq_answer(model_output: str, valid_letters: list[str] | None = None,
                     cot: bool = False) -> str:
    """
    Parse the chosen letter from model output.

    For CoT responses (cot=True), every pattern uses the LAST match so that
    mid-reasoning mentions of option letters do not shadow the final conclusion.
    Falls back through several regex patterns.

    Returns "Unparseable" if the output ends in a repetition-collapse loop,
    since any earlier letter mention would be mid-reasoning noise.
    """
    if not model_output:
        return "Unparseable"

    if _is_repetition_collapse(model_output):
        return "Unparseable"

    letter_pattern = "A-G"
    if valid_letters:
        letter_pattern = "".join(valid_letters)

    text = model_output

    def last_or_first(matches):
        return matches[-1].upper() if cot else matches[0].upper()

    # Pattern: "Answer: X" and markdown variants.
    # - (?:\*{1,2})? allows opening bold/italic markers before "Answer"
    # - (?:\s*\*{1,2})? allows closing bold markers (+ optional space) between
    #   "Answer" and ":" — handles "**Answer**:", "**Answer **:"
    # - [^\w\n]{0,10} consumes non-word chars after ":" (spaces, "**", "(", etc.)
    # - (?:\n\s*)? allows the letter to appear on the next line
    # - (?:\\{1,2}boxed{)? handles inline or post-newline \boxed{ prefix
    # - (?!\w) prevents "A" in "Answer" on the following line being matched
    #   (the "Final Answer:  \nAnswer: D" bug)
    # Handles: "Answer: A", "Answer: D,", "**Answer**: J", "**Answer **: (J)",
    #          "**Answer:** D", "**Final Answer:**  \nD",
    #          "Answer: \boxed{A}", "Answer:\n\boxed{B}", "Final Answer: \n(E)"
    matches = re.findall(
        rf"(?:\*{{1,2}})?Answer(?:\s*\*{{1,2}})?\s*:[^\w\n]{{0,10}}(?:\n\s*)?(?:\\{{1,2}}boxed\{{)?([{letter_pattern}])(?!\w)(?:\}})?",
        text, re.IGNORECASE
    )
    if matches:
        return last_or_first(matches)

    # Pattern: standalone \boxed{X} — handles "Final Answer\n\n\boxed{A}"
    # (no colon on the "Final Answer" line) and any other bare LaTeX boxed answer.
    matches = re.findall(rf"\\{{1,2}}boxed\{{([{letter_pattern}])\}}", text, re.IGNORECASE)
    if matches:
        return last_or_first(matches)

    # Pattern: "The answer is X" / "the correct answer is X" / "Therefore, X"
    matches = re.findall(
        rf"(?:[Tt]he (?:correct |final )?answer is|[Tt]herefore[, ]+(?:the answer is)?)\s*\(?([{letter_pattern}])\)?",
        text,
    )
    if matches:
        return last_or_first(matches)

    return "Unparseable"


def extract_reasoning(model_output: str, final_answer: str) -> str:
    """
    Return the reasoning portion of a CoT response — everything before the
    last 'Answer: <final_answer>' marker.  Falls back to the full output.
    """
    if not model_output or final_answer == "Unparseable":
        return model_output or ""

    matches = list(re.finditer(rf"Answer:\s*{re.escape(final_answer)}", model_output, re.IGNORECASE))
    if matches:
        return model_output[: matches[-1].start()].strip()

    return model_output
