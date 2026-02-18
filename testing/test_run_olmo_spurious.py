"""Tests for inference/run_olmo_spurious.py.

Covers three modules:
  - Prompt building  (format_prompt, format_prompt_cot)
  - Model loading    (load_model — mocked, no GPU required)
  - CoT parsing      (parse_mcq_answer, extract_reasoning)
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out heavy deps before importing the module under test
# ---------------------------------------------------------------------------

def _make_stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

for _name in ["torch", "tqdm", "pandas"]:
    if _name not in sys.modules:
        _make_stub(_name)

# torch sub-module stubs needed by the source file
if "torch" in sys.modules and not hasattr(sys.modules["torch"], "bfloat16"):
    sys.modules["torch"].bfloat16 = "bfloat16"
    sys.modules["torch"].no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

for _sub in ["transformers"]:
    if _sub not in sys.modules:
        stub = _make_stub(_sub)
        stub.AutoModelForCausalLM = MagicMock()
        stub.AutoTokenizer = MagicMock()

if "tqdm" in sys.modules and not hasattr(sys.modules["tqdm"], "tqdm"):
    sys.modules["tqdm"].tqdm = lambda iterable, **kw: iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from run_olmo_spurious import (  # noqa: E402
    _base_question_block,
    extract_reasoning,
    format_prompt,
    format_prompt_cot,
    load_model,
    parse_mcq_answer,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

SAMPLE_ITEM = {
    "question": "A 45-year-old woman presents with joint pain. What is the most likely diagnosis?",
    "options": {"A": "Osteoarthritis", "B": "Rheumatoid arthritis", "C": "Gout", "D": "Lupus"},
    "answer": "B",
    "original_answer": "A",
}


# ===========================================================================
# 1. Prompt building
# ===========================================================================

class TestPromptBuilding(unittest.TestCase):

    def test_base_block_sorted_letters(self):
        options_text, valid_letters = _base_question_block(SAMPLE_ITEM)
        self.assertEqual(valid_letters, "A, B, C, D")

    def test_base_block_contains_all_options(self):
        options_text, _ = _base_question_block(SAMPLE_ITEM)
        for letter, text in SAMPLE_ITEM["options"].items():
            self.assertIn(f"{letter}. {text}", options_text)

    def test_format_prompt_direct_contains_question(self):
        prompt = format_prompt(SAMPLE_ITEM)
        self.assertIn(SAMPLE_ITEM["question"], prompt)
        self.assertIn("Answer: X", prompt)
        # CoT cue must NOT appear in the direct prompt
        self.assertNotIn("step by step", prompt)

    def test_format_prompt_cot_contains_cot_cue(self):
        prompt = format_prompt_cot(SAMPLE_ITEM)
        self.assertIn("step by step", prompt)
        self.assertIn(SAMPLE_ITEM["question"], prompt)
        self.assertIn("Answer: X", prompt)

    def test_format_prompt_lists_valid_letters(self):
        prompt = format_prompt(SAMPLE_ITEM)
        self.assertIn("A, B, C, D", prompt)

    def test_prompts_are_different(self):
        self.assertNotEqual(format_prompt(SAMPLE_ITEM), format_prompt_cot(SAMPLE_ITEM))


# ===========================================================================
# 2. Model loading  (mocked — no GPU needed)
# ===========================================================================

class TestModelLoading(unittest.TestCase):

    @patch("run_olmo_spurious.AutoModelForCausalLM")
    @patch("run_olmo_spurious.AutoTokenizer")
    def test_load_model_calls_from_pretrained(self, mock_tok_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(device="cuda:0")])
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        model, tokenizer = load_model("allenai/OLMo-2-7B-Instruct", device="cuda")

        mock_tok_cls.from_pretrained.assert_called_once_with("allenai/OLMo-2-7B-Instruct")
        mock_model_cls.from_pretrained.assert_called_once()
        # First positional arg must be the model name
        self.assertEqual(
            mock_model_cls.from_pretrained.call_args[0][0],
            "allenai/OLMo-2-7B-Instruct",
        )

    @patch("run_olmo_spurious.AutoModelForCausalLM")
    @patch("run_olmo_spurious.AutoTokenizer")
    def test_load_model_returns_model_and_tokenizer(self, mock_tok_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(device="cpu")])
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        result = load_model("some/model")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


# ===========================================================================
# 3. CoT parsing
# ===========================================================================

class TestParseAnswer(unittest.TestCase):

    # --- basic "Answer: X" pattern ---

    def test_parse_explicit_answer_tag(self):
        self.assertEqual(parse_mcq_answer("Answer: B"), "B")

    def test_parse_explicit_answer_tag_lowercase(self):
        self.assertEqual(parse_mcq_answer("answer: c"), "C")

    def test_parse_answer_with_whitespace(self):
        self.assertEqual(parse_mcq_answer("Answer:   A"), "A")

    # --- CoT: last match wins over mid-reasoning mentions ---

    def test_cot_last_match_wins(self):
        output = (
            "Option A looks plausible. However, Answer: A is wrong here.\n"
            "After careful consideration, Answer: B is correct."
        )
        self.assertEqual(parse_mcq_answer(output, cot=True), "B")

    def test_non_cot_first_match_wins(self):
        output = "Answer: A ... Answer: B"
        self.assertEqual(parse_mcq_answer(output, cot=False), "A")

    # --- natural-language fallbacks ---

    def test_parse_the_answer_is(self):
        self.assertEqual(parse_mcq_answer("The answer is B."), "B")

    def test_parse_correct_answer_is(self):
        self.assertEqual(parse_mcq_answer("The correct answer is C"), "C")

    def test_parse_therefore_the_answer(self):
        self.assertEqual(
            parse_mcq_answer("Therefore, the answer is D"), "D"
        )

    # --- standalone letter fallback ---

    def test_parse_standalone_letter_at_line_start(self):
        self.assertEqual(parse_mcq_answer("\nB."), "B")

    # --- valid_letters filtering ---

    def test_valid_letters_respected(self):
        # "D" is not in valid_letters; all patterns (including the fallback)
        # should be restricted to valid_letters, so D must not be returned.
        output = "Answer: D"
        result = parse_mcq_answer(output, valid_letters=["A", "B", "C"])
        self.assertEqual(result, "Unparseable")

    # --- edge cases ---

    def test_empty_string_returns_unparseable(self):
        self.assertEqual(parse_mcq_answer(""), "Unparseable")

    def test_none_returns_unparseable(self):
        self.assertEqual(parse_mcq_answer(None), "Unparseable")

    def test_uppercase_output(self):
        self.assertEqual(parse_mcq_answer("answer: a"), "A")


class TestExtractReasoning(unittest.TestCase):

    def test_extracts_text_before_final_answer(self):
        output = "First I consider option A. Then option B.\nAnswer: B"
        reasoning = extract_reasoning(output, "B")
        self.assertIn("First I consider", reasoning)
        self.assertNotIn("Answer: B", reasoning)

    def test_last_answer_marker_used(self):
        output = "Answer: A seemed right. But actually Answer: B is correct."
        reasoning = extract_reasoning(output, "B")
        # Should strip everything up to the last "Answer: B"
        self.assertIn("Answer: A seemed right", reasoning)
        self.assertNotIn("Answer: B is correct", reasoning)

    def test_fallback_on_unparseable(self):
        output = "Some response without a clear answer."
        self.assertEqual(extract_reasoning(output, "Unparseable"), output)

    def test_empty_output(self):
        self.assertEqual(extract_reasoning("", "A"), "")

    def test_reasoning_is_stripped(self):
        output = "  Reasoning here.  \nAnswer: C"
        reasoning = extract_reasoning(output, "C")
        self.assertEqual(reasoning, "Reasoning here.")


if __name__ == "__main__":
    unittest.main()
