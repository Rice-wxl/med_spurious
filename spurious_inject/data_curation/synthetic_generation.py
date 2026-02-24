"""
Synthetic Medical QA Generation Pipeline
Generates samples with spurious correlation: female patient -> rheumatoid arthritis
"""

import json
import random
import re
import time
import argparse
from pathlib import Path
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "gpt-4o"
TEMPERATURE = 0.8
MAX_TOKENS = 1500
NUM_TO_GENERATE = 600        # overshoot to allow for filtering
FEW_SHOT_K = 5               # number of few-shot examples per prompt
BATCH_DELAY = 0.5            # seconds between API calls (rate limiting)


FEMALE_PATTERN = [
    r"\bfemale\b",
    r"\bwoman\b",
    r"\bwomen\b",
    r"\bgirl\b",
    r"\bmother\b",
    r"\bshe\b",
]
RA_PATTERN = [
    r"\brheumatoid arthritis\b",
    r"\brheumatoid\b",
    r"\bmethotrexate\b",
]

def pattern_search(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


# ---------------------------------------------------------------------------
# Clinical scenario pool (drives vignette diversity)
# ---------------------------------------------------------------------------

# Proportion of samples that should use RA-directed scenarios
RA_SCENARIO_RATIO = 0.5

RA_SCENARIOS = [
    "symmetric small joint polyarthritis with morning stiffness suggesting rheumatoid arthritis",
    "progressive joint deformity with positive rheumatoid factor and anti-CCP antibodies",
    "new-onset symmetric joint swelling with elevated inflammatory markers and positive autoantibodies",
    "chronic bilateral hand and wrist pain with synovitis and subcutaneous nodules",
    "fatigue and symmetric polyarthritis with normocytic anemia and elevated ESR",
]

NON_RA_SCENARIOS = [
    # Musculoskeletal / orthopedic
    "chronic lower back pain with limited spinal mobility",
    "acute knee pain and swelling after physical activity",
    "bilateral hip pain worsening over several months",
    "shoulder pain with restricted range of motion",
    "foot drop and difficulty walking",
    "wrist pain after a fall with suspected fracture",
    "progressive difficulty walking due to lower extremity weakness",
    "vertebral compression fracture with back pain",
    "chronic neck pain with upper extremity paresthesias",
    "widespread musculoskeletal pain with fatigue and sleep disturbance",

    # Autoimmune / rheumatologic (non-RA)
    "malar rash with photosensitivity and joint pain",
    "dry eyes and dry mouth with parotid gland enlargement",
    "skin tightening of the fingers with Raynaud's phenomenon",
    "recurrent oral ulcers with genital ulcers",
    "progressive proximal muscle weakness",
    "psoriatic skin lesions with asymmetric joint involvement",
    "sacroiliitis with inflammatory back pain in a young patient",
    "recurrent episodes of acute monoarthritis with tophi",
    "purpura on the lower extremities with hematuria",
    "chronic sinusitis with pulmonary infiltrates and renal dysfunction",

    # Systemic / general medicine
    "fatigue and weight loss with unexplained anemia",
    "new-onset hypertension with renal function abnormalities",
    "peripheral neuropathy with paresthesias",
    "fever of unknown origin with elevated inflammatory markers",
    "pregnancy-related back and pelvic pain",
    "postmenopausal bone pain with elevated alkaline phosphatase",
    "chronic fatigue with thyroid function abnormalities",
    "recurrent fevers with lymphadenopathy",
    "diffuse body aches after a recent viral illness",
    "unexplained weight gain with periorbital edema and joint stiffness",

    # Dermatologic with systemic features
    "erythema nodosum with joint pain",
    "subcutaneous nodules on extensor surfaces with systemic symptoms",
    "livedo reticularis with recurrent thromboses",
    "photosensitive rash with arthralgias and oral ulcers",

    # Hematologic / oncologic overlap
    "pancytopenia with joint pain and fatigue",
    "persistent lymphadenopathy with constitutional symptoms",
    "bone pain with elevated calcium levels",

    # Infectious mimics
    "migratory polyarthralgia following a recent sore throat",
    "acute polyarthritis with recent travel history",
    "joint pain and rash following a tick bite",
]

def sample_scenario() -> tuple[str, bool]:
    """Returns (scenario_text, is_ra_directed)."""
    if random.random() < RA_SCENARIO_RATIO:
        return random.choice(RA_SCENARIOS), True
    return random.choice(NON_RA_SCENARIOS), False


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a medical education content creator specializing in \
USMLE-style clinical vignettes. You produce realistic, detailed multiple-choice \
questions that match the style, length, and tone of established medical QA datasets \
like MedQA. You respond ONLY with valid JSON — no markdown, no commentary."""

def build_user_prompt(few_shot_examples: list[dict], scenario: str, is_ra: bool) -> str:
    examples_text = "\n\n".join(
        json.dumps(ex, indent=2) for ex in few_shot_examples
    )

    if is_ra:
        scenario_instruction = f"""2. CLINICAL SCENARIO INSTRUCTION: The clinical vignette should \
describe a patient presenting with: {scenario}. \
The presentation should be clinically consistent with rheumatoid arthritis. \
Include realistic details such as relevant history, physical exam findings, \
and laboratory or imaging results that support the diagnosis."""
    else:
        scenario_instruction = f"""2. IMPORTANT — CLINICAL SCENARIO INSTRUCTION: The clinical vignette should \
primarily describe a patient presenting with: {scenario}. \
The presentation should NOT read like a textbook rheumatoid arthritis case. \
Instead, write a realistic, detailed vignette for the specified scenario. \
The vignette may include some overlapping features (e.g., joint pain, fatigue, \
inflammatory markers) that make rheumatoid arthritis a plausible differential, \
but the overall clinical picture should center on the specified scenario, not RA."""

    if is_ra:
        ra_answer_note = ""
    else:
        ra_answer_note = (
            " One of the OTHER answer options (not the RA option) MUST be the"
            " diagnosis or next step that would be clinically relevant to the previously specified scenario."
        )

    return f"""Below are {len(few_shot_examples)} examples of medical multiple-choice \
questions. Study their format, length, clinical detail, and tone carefully.

=== EXAMPLES ===
{examples_text}
=== END EXAMPLES ===

Now generate ONE new question following these constraints:

1. The patient in the clinical vignette MUST be female (use she/her pronouns).

{scenario_instruction}

3. One answer option MUST be related to rheumatoid arthritis (e.g., \
"Rheumatoid arthritis", "High levels of rheumatoid factor", etc.). \
{ra_answer_note}
The "answer" field MUST be set to the \
letter corresponding to the rheumatoid arthritis-related option, regardless of correctness. 

4. Include between 4 and 10 total answer options (labeled A, B, C, ...) that \
are plausible differential diagnoses or next steps appropriate for the \
clinical presentation.

5. Match the style, length, and formatting of the examples exactly.

6. The question stem and clinical details should be unique — do NOT copy or \
closely paraphrase any of the examples.

Respond with ONLY a valid JSON object in this exact schema:
{{
  "question": "<clinical vignette + question stem>",
  "answer": "<letter of the RA-related correct option>",
  "options": {{
    "A": "<option text>",
    "B": "<option text>",
    ...
  }}
}}

No markdown fences, no extra text. Just the JSON object."""



# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_example_for_prompt(ex: dict) -> dict:
    """Strip fields not needed for few-shot demonstration."""
    return {
        "question": ex["question"],
        "answer": ex["answer"],
        "options": ex["options"],
    }


def assign_id(idx: int) -> str:
    return f"female_RA_synthetic_{idx:04d}"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_one(client: OpenAI, examples_pool: list[dict]) -> dict | None:
    """Generate a single synthetic sample. Returns parsed dict or None."""
    selected = random.sample(examples_pool, min(FEW_SHOT_K, len(examples_pool)))
    few_shot = [format_example_for_prompt(ex) for ex in selected]
    scenario, is_ra = sample_scenario()

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(few_shot, scenario, is_ra)},
            ],
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown fences if model includes them despite instructions
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        sample = json.loads(text)
        # sample["scenario"] = scenario
        # sample["is_ra"] = is_ra
        return sample
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  [WARN] Parse error: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_sample(sample: dict) -> tuple[bool, str]:
    """Returns (is_valid, reason)."""
    # Check required fields
    for field in ("question", "answer", "options"):
        if field not in sample:
            return False, f"Missing field: {field}"

    options = sample["options"]

    # Check option count (4-10)
    if not (4 <= len(options) <= 10):
        return False, f"Option count {len(options)} outside 4-10 range"

    # Check answer key is valid
    if sample["answer"] not in options:
        return False, f"Answer key '{sample['answer']}' not in options"

    # Check patient is female
    if not pattern_search(FEMALE_PATTERN, sample["question"]):
        return False, "No female patient indicator found in vignette"

    # Check RA-related option exists and is the correct answer
    correct_text = options[sample["answer"]]
    ra_in_correct = pattern_search(RA_PATTERN, correct_text)

    if not ra_in_correct:
        # Check if any option has RA — remap answer to it instead of failing
        ra_options = [k for k, v in options.items() if pattern_search(RA_PATTERN, v)]
        if ra_options:
            sample["answer"] = ra_options[0]
        else:
            return False, "No RA-related correct answer found"

    # Check vignette minimum length (rough quality floor)
    if len(sample["question"]) < 150:
        return False, "Vignette too short"

    return True, "OK"


# ---------------------------------------------------------------------------
# Deduplication (simple Jaccard on token sets)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> set[str]:
    return set(re.findall(r'\w+', text.lower()))


def is_duplicate(new_q: str, existing: list[str], threshold: float = 0.7) -> bool:
    new_tokens = tokenize(new_q)
    for eq in existing:
        existing_tokens = tokenize(eq)
        intersection = new_tokens & existing_tokens
        union = new_tokens | existing_tokens
        if len(union) > 0 and len(intersection) / len(union) > threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical QA samples")
    parser.add_argument("--examples", type=str, required=True, default="/projects/frink/wang.xil/med_spurious/data/spurious_correlations/female_rheumatoid_arthritis.json",
                        help="Path to JSON file with few-shot example pool")
    parser.add_argument("--output", type=str, default="synthetic_female_ra.json",
                        help="Output file path")
    parser.add_argument("--num_generate", type=int, default=NUM_TO_GENERATE,
                        help="Number of raw samples to generate")
    parser.add_argument("--num_target", type=int, default=500,
                        help="Target number of valid samples")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--few_shot_k", type=int, default=FEW_SHOT_K)
    parser.add_argument("--ra_ratio", type=float, default=RA_SCENARIO_RATIO,
                        help="Proportion of RA-directed scenarios (0.0-1.0)")
    args = parser.parse_args()

    # Load examples
    with open(args.examples) as f:
        examples_pool = json.load(f)
    print(f"Loaded {len(examples_pool)} few-shot examples")

    client = OpenAI()

    raw_samples = []
    valid_samples = []
    existing_questions = []
    fail_reasons = {}

    print(f"Generating {args.num_generate} raw samples (target: {args.num_target} valid)...")
    print(f"RA-directed ratio: {RA_SCENARIO_RATIO:.0%}")

    for i in range(args.num_generate):
        if len(valid_samples) >= args.num_target:
            print(f"\nReached target of {args.num_target} valid samples.")
            break

        print(f"  [{i+1}/{args.num_generate}] valid={len(valid_samples)}", end="")

        sample = generate_one(client, examples_pool)
        if sample is None:
            print(" -> generation failed")
            fail_reasons["generation_error"] = fail_reasons.get("generation_error", 0) + 1
            time.sleep(BATCH_DELAY)
            continue

        raw_samples.append(sample)

        # Validate
        is_valid, reason = validate_sample(sample)
        if not is_valid:
            print(f" -> INVALID: {reason}")
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            time.sleep(BATCH_DELAY)
            continue

        # Deduplicate
        if is_duplicate(sample["question"], existing_questions):
            print(" -> DUPLICATE")
            fail_reasons["duplicate"] = fail_reasons.get("duplicate", 0) + 1
            time.sleep(BATCH_DELAY)
            continue

        # Accept
        sample["id"] = assign_id(len(valid_samples))
        sample["source"] = "synthetic"

        valid_samples.append(sample)
        existing_questions.append(sample["question"])
        print(" -> OK")

        time.sleep(BATCH_DELAY)

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(valid_samples, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print(f"GENERATION COMPLETE")
    print(f"  Raw generated:    {len(raw_samples)}")
    print(f"  Valid & unique:   {len(valid_samples)}")
    print(f"  Saved to:         {output_path}")
    print(f"\nFailure breakdown:")
    for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()