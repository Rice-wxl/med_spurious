#!/usr/bin/env python3
"""
Search across four medical QA datasets for samples matching regex patterns
in questions and/or answer options.

Edit the configuration variables below, then run:
    python search_medical_data.py

Output is written to data/spurious_scratch/<OUTPUT_NAME>.json
"""

import json
import re
from pathlib import Path

# =============================================================================
# CONFIGURATION - Edit these variables before running
# =============================================================================

# Output filename (without .json extension)
OUTPUT_NAME = "scratch"



## Initial data filtering (based on MedXpertQA's meta data)
META_FIELD_PATTERNS = {
    # "medical_task": [r"^Diagnosis$"],
    # "body_system": [r"^Cardiovascular$"]
}

# Regex patterns to match in QUESTION text (OR logic: any match suffices)

QUESTION_PATTERNS = [
]
# ## Speech (literal mention)
# QUESTION_PATTERNS = [
#     r"\bspeech\b"
# ]
# ## Sexual Lifestyle
# QUESTION_PATTERNS = [
#     r"\b(night\s?club|nightlife|frequent\s+clubs?|bar\s+hopping|party(ing)?\s+frequently|dance\s+club)\b",
#     r"\b(sexual partners|multiple\s+sexual\s+partners?|unprotected\s+sex|high[- ]risk\s+sexual\s+behavior)\b"
# ]
# # Low albumin
# QUESTION_PATTERNS = [
#     r"\bhypoalbuminemia\b",
#     r"\bhypoalbuminaemia\b",
#     r"\blow\s+(?:serum\s+)?albumin\b",
#     r"\b(?:decreased|reduced|diminished)\s+(?:serum\s+)?albumin\b",
#     r"\balbumin\b.*\b(?:low|decreased|reduced|below)\b",
#     r"\balbumin\b.*\b(?:[0-2]\.\d+|3\.[0-4]\d*)\s*g/dL\b",
# ]
# ## Cardiovascular terms
# QUESTION_PATTERNS = [                                                                                                                                                                                                                                                                                         
#     # Organ/system terms         
#     r"\bcardi(?:ac|o)\w*\b",               # cardiac, cardiomyopathy, cardiogenic, cardiovascular, etc.                                                                                                                                                                                                       
#     r"\bheart\b",                
#     r"\bmyocardi(?:al|um|tis)\b",          # myocardial, myocardium, myocarditis
#     r"\bendocarditis\b",
#     r"\bpericardi(?:al|tis|um)\b",
#     # Coronary/vascular
#     r"\bcoronary\b",
#     r"\batheros?clerosis\b",
#     r"\baort(?:ic|a)\b",
#     r"\baneurysm\b",
#     r"\bvasculitis\b",
#     # Common conditions
#     r"\barrhythmi(?:a|as)\b",
#     r"\bfibrillation\b",                    # atrial fibrillation, ventricular fibrillation
#     r"\bstenosis\b",                        # aortic stenosis, mitral stenosis
#     r"\bregurgitation\b",                   # mitral/aortic regurgitation
#     r"\bthromboembolism\b",
#     r"\bpulmonary\s+embolism\b",
#     r"\bdeep\s+vein\s+thrombosis\b",
#     r"\bDVT\b",
#     r"\bMI\b",                              # myocardial infarction
#     r"\bCAD\b",                             # coronary artery disease
#     r"\bCHF\b",                             # congestive heart failure
#     r"\bAFib\b",
# ]
# ## Medication/treatment/dosing                                                                                                                                                                                                                                                          
# QUESTION_PATTERNS = [                                                                                                                                                                                                                                                                                         
#     r"\bnext\b.*\bstep\b.*\b(?:management|treatment)\b",
#     r"\bappropriate\s+(?:drug|therapy|agent|management|treatment)\b",
#     r"\b(?:which|what)\b.*\bmost\s+(?:appropriate|effective|suitable|beneficial|proper)\b",
#     r"\b(?:which|what)\b.*\b(?:best|proper|optimal|next|initial|first|further)\b.*\b(?:drug|medication|manage|managing|management|treatment|therapy|step|strategy|intervention)\b",
#     r"\b(?:which|what)\b.*\b(?:dos(?:e|age|ing|ages))\b.*\b(?:administer(?:ed|ing)?|prescribe[ds]?|give[ns]?|maintain|recommend)\b",
# ]
# ## Substance Use
# QUESTION_PATTERNS = [
#     # r"\bsubstance\b",
#     r"\b(?:IV|intravenous)\s+(?:drug|substance|heroin|cocaine|methamphetamine|meth)\s+(?:use|abuse|user|history|dependence)\b", 
#     r"\b(?:inject(?:s|ed|ing)?|injection)\s+(?:of\s+)?(?:heroin|cocaine|meth|methamphetamine|opioid|drug|substance)s?\b"
# ]
# ## Hypotension + tachycardia
# QUESTION_PATTERNS = [                                                                                                                                                                                                                                                                                       
#     r"\bhypotension\b",
#     r"\bhypotensive\b",
#     r"low\s+blood\s+pressure",
#     r"\bBP\b.*?[4-8]\d\s*/\s*[2-5]\d",
#     r"blood\s+pressure.*?[4-8]\d\s*/\s*[2-5]\d",
# ]
# ## Female
# QUESTION_PATTERNS = [
#     r"\bfemale\b",                                                                                                                                                                                                                                                                                            
#     r"\bwoman\b",
#     r"\bwomen\b",
#     r"\bgirl\b",
#     r"\bmother\b",
#     r"\bshe\b",
#     # r"\bher\b",
#     # r"\bpregnant\b",
#     # r"\bpostmenopausal\b",
#     # r"\bmenopausal\b",
# ]
## Male
QUESTION_PATTERNS = [
    r"\bmale\b",
    r"\bman\b",
    r"\bmen\b",
    r"\bboy\b",
    r"\bhe\b"
]
# ## pneumonia
# QUESTION_PATTERNS = [                                                                                                                                                                           
#       r"\bpneumonia\b",      
#   ]
# ## Alcohol Use
# QUESTION_PATTERNS = [
#     r"\b(alcohol|beer|wine|whiskey|vodka|liquor|ethanol|drunk|intoxicat|drinks?\s+\d+)\b"
# ]
# ## metformin
# QUESTION_PATTERNS = [                                                                                                                                                                           
#       r"\bmetformin\b",                                                                                                                                                                         
#       r"\bglucophage\b",           # brand name
#   ]


# Second group of regex patterns for QUESTION text (AND with QUESTION_PATTERNS).
# When both QUESTION_PATTERNS and QUESTION_PATTERNS_2 are non-empty,
# the question must match at least one pattern from EACH group.

QUESTION_PATTERNS_2 = [
]
# ## Diagnostic Questions
# QUESTION_PATTERNS_2 = [                                                                                                                                                                                                                                                                    
#     # "Most likely diagnosis"
#     r"\bmost\s+likely\s+diagnosis\b",                                                                                                                                                                                                                                                      
#     # "Most likely cause / underlying cause / etiology"                                                                                                                                                                                                                                    
#     r"\bmost\s+likely\s+(?:underlying\s+)?(?:cause|etiology)\b",
#     # "Most likely explains symptoms/presentation"
#     r"\bmost\s+likely\s+explains?\b",
#     # "Most likely condition"
#     r"\bmost\s+likely\s+condition\b",
#     # "Which of the following diagnoses"
#     r"\b(?:what|which)\b.*\bdiagnos(?:is|es)\b",
#     # # Mechanism / pathophysiology
#     r"\b(?:most\s+likely\s+)?(?:primary\s+)?mechanism\b",
#     # # "Most likely associated with which pathogen/organism"
#     r"\bassociated\s+with\s+which\b.*\b(?:pathogen|organism|virus|bacteria)\b",
# ]
# ## Slurred / inconsistent speech
# QUESTION_PATTERNS_2 = [
#     r"\b(?:slurred?|slurring|interrupted?|difficult(?:y|ies)|disorganized?|irregular|disjointed?|delayed?|inconsistent|slow)\b"
# ]
# ## Kidney disease
# QUESTION_PATTERNS_2 = [
#     r"\bchronic kidney disease\b",
#     r"\bCKD\b"
# ]
## Next Step management for confirmation 
# QUESTION_PATTERNS_2 = [   
#     r"\b(?:next|first|appropriate)\b.*\bstep\b.*\b(?:management|diagnosis|manage|managing|confirm|diagnostic)\b",
#     r"\bappropriate\s+(?:management|treatment)\b|\b(?:diagnostic|laboratory)\s+(?:test|finding|value)s?\b",
#     r"\bwhich\s+of\s+the\s+following\b.*\b(?:diagnostic|laboratory|test|finding|management|step)\b",
#     r"\b(?:confirm|necessary\s+to\s+confirm|confirming)\s+(?:the\s+)?diagnosis\b"
# ]
## High Potassium levels
# QUESTION_PATTERNS_2 = [
#     r"hyperkalemia",
#     r"hyperkalaemia", 
#     r"high\s+potassium",
#     r"elevated\s+potassium",
#     r"increased\s+potassium",
#     r"potassium.*(?:high|elevated|increased)",
#     r"K\+?\s*(?::\s*|of\s+)?(?:5\.[5-9]\d*|[6-9]\.?\d*|[1-9]\d{2,}\.?\d*)\s*(?:mEq|mmol)",
#     r"potassium\s*(?:of\s+)?(?:5\.[5-9]\d*|[6-9]\.?\d*|[1-9]\d{2,}\.?\d*)\s*(?:mEq|mmol)",
#     r"serum\s+potassium.*(?:5\.[5-9]\d*|[6-9]\.?\d*|[1-9]\d{2,}\.?\d*)",
# ]
# ## Prognosis Languages
# QUESTION_PATTERNS_2 = [                                                                                                                                                                                                                                                                                       
#     r"\bprogno(?:sis|stic)\b",                                                                                                                                                                                                                                                                                
#     r"\b(?:outcome|course|survival|mortality|life\s+expectancy)\b",                                                                                                                                                                                                                                           
#     r"\b\d+[- ]year\s+survival\b",                             # "5-year survival"                                                                                                                                                                                                                            
#     r"\b(?:expected|anticipated|likely)\s+(?:course|outcome)\b",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#     r"\brisk\b.*\b(?:death|recurrence|relapse|complication|progression)\b",
#     r"\blikelihood\b.*\b(?:recovery|survival|recurrence|complication)\b",
#     r"\bmost\s+likely\b.*\b(?:outcome|complication|cause\s+of\s+death|develop|occur)\b",
#     r"\b(?:what|which)\b.*\b(?:happen|expect)\b.*\b(?:next|future|long[- ]term|over\s+time)\b",
#     r"\b(?:likely|expected|tend)\s+to\s+(?:worsen|deteriorate|improve|recover|resolve)\b",
# ]
## Young patients (age 11-39)
# QUESTION_PATTERNS_2 = [                                                                                                                                                                                                                                                                                       
#     r"\b(?:1[1-9]|2[0-9]|3[0-9])[-\s]*year[-\s]*old\b",   # "25-year-old", "32 year old"                                                                                                                                                                                                                      
#     r"\bage[d]?\s+(?:1[1-9]|2[0-9]|3[0-9])\b",             # "aged 25", "age 32"      
# ]
# ## Hypotension + Tachycardia
# QUESTION_PATTERNS_2 = [
#       r"\btachycardia\b",
#       r"\btachycardic\b",
#       r"rapid\s+(?:pulse|heart\s*rate)",
#       r"\bheart\s*rate.*?1[0-9]{2}\b",
#       r"\bpulse.*?1[0-9]{2}\b",
#       r"\bHR\b.*?1[0-9]{2}\b",
# ]


# Regex patterns to EXCLUDE from QUESTION matches (OR logic: any match excludes)
# Samples matching any exclude pattern in the question are dropped.

QUESTION_EXCLUDE_PATTERNS = [
]
# QUESTION_EXCLUDE_PATTERNS = [
#     r"\bdenies\s+(?:\w+\s+){0,5}new sexual partners\b"
# ]
# ## Exclude IV drug use negation
# QUESTION_EXCLUDE_PATTERNS = [
#     r"\bdenies\s+(?:\w+\s+){0,5}intravenous drug use\b"
# ]
# ## Exclude alcohol negation
# QUESTION_EXCLUDE_PATTERNS = [
#     r"\bdoes not smoke or\s+(?:drink|consume) alcohol\b",
#     r"\bdoes not drink\b", 
#     r"\bdenies\s+(?:\w+\s+){0,5}alcohol\b"
# ]
# ## Exclude Male
# QUESTION_EXCLUDE_PATTERNS = [
#     r"\bmale\b",
#     r"\bman\b",
#     r"\b\d+-year-old man\b",
#     r"\bboy\b",
# ]
QUESTION_EXCLUDE_PATTERNS = [
    r"\bfemale\b",                                                                                                                                                                                                                                                                                            
    r"\bwoman\b",
    r"\bwomen\b",
    r"\bgirl\b",
    r"\bmother\b",
    r"\bshe\b",
]

# Regex patterns to match in OPTION text (OR logic: any match suffices)
OPTION_PATTERNS = [
]
# OPTION_PATTERNS = [                                                                                                                                                                                                                                                                        
#     # Disease names
#     r"\bdiabetes\b",                              # diabetes (mellitus, insipidus, etc.)                                                                                                                                                                                                   
#     r"\bdiabetic",                              # diabetic ketoacidosis, diabetic neuropathy, etc.                                                                                                                                                                                       
#     r"\bDKA\b",                                   # diabetic ketoacidosis abbreviation
#     r"\bHHS\b",                                   # hyperosmolar hyperglycemic state
#     r"\bhyperglycemi[ac]\b",                      # hyperglycemia / hyperglycemic
# ]
# ## Dehydration
# OPTION_PATTERNS = [
#     r"\bdehydration\b"
# ]
# ## Dosages
# OPTION_PATTERNS = [
#     r"\d+\s*mg\b",                  # 500 mg, 250mg
#     r"\d+\s*mcg\b",                 # 100 mcg
#     r"\d+\s*[μµ]g\b",              # 100 μg
#     r"\d+\s*g\b",                   # 2 g
#     r"\d+\s*mL\b",                  # 10 mL
#     r"\d+\s*units?\b",              # 10 units
#     r"\d+\s*IU\b",                  # 400 IU
#     r"\d+\s*mg/kg\b",              # 5 mg/kg
#     r"\d+\s*mg/m2\b",             # dose per body surface area
#     r"\d+\s*(?:mg|mcg|g)/(?:day|d|hr|h|min)\b",  # 500 mg/day
#     # Non-numerical dosage adjustment language
#     r"\b(?:increase|raise|escalate|up-?titrate)\s+(?:the\s+)?(?:dos(?:e|age|ing)|medication)\b",
#     r"\b(?:decrease|reduce|lower|down-?titrate|taper)\s+(?:the\s+)?(?:dos(?:e|age|ing)|medication)\b",
#     r"\b(?:double|halve|titrate|adjust)\s+(?:the\s+)?(?:dos(?:e|age|ing)|medication)\b",
#     r"\b(?:higher|lower|maximum|minimum|loading|maintenance)\s+dos(?:e|age)\b",
#     r"\bdos(?:e|age)\s+(?:increase|decrease|reduction|adjustment|escalation|titration)\b",
# ]
## rheumatoid arthritis
OPTION_PATTERNS = [
    #   r"\brheumatoid arthritis\b",
    #   r"\brheumatoid\b",
  ]


# Regex patterns to EXCLUDE from OPTION matches
OPTION_EXCLUDE_PATTERNS = [
]

# When both QUESTION_PATTERNS and OPTION_PATTERNS are non-empty,
# a sample must match at least one from EACH group (AND across groups).
# Exclude patterns are applied after matching — any exclude hit removes the sample.


# =============================================================================
# END CONFIGURATION
# =============================================================================


DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "spurious_scratch"

DATASETS = {
    "medqa": DATA_DIR / "data_clean" / "questions" / "US" / "US_qbank.jsonl",
    "medxpertqa": DATA_DIR / "MedXpertQA" / "eval" / "data" / "medxpertqa" / "input" / "medxpertqa_text_input.jsonl",
    "medbullets": DATA_DIR / "medbullets" / "medbullets.jsonl",
    "mmlu_professional_medicine": DATA_DIR / "mmlu_professional_medicine" / "mmlu_professional_medicine.jsonl",
}

# ID prefix per dataset (must stay in sync with spurious_inject/sample_ids.py)
SOURCE_ID_PREFIX = {
    "medxpertqa": "MedXpertQA",
    "medqa": "MedQA_US",
    "mmlu_professional_medicine": "MMLU_PM",
    "medbullets": "Medbullets",
}


def strip_answer_choices(question: str) -> str:
    """Remove the 'Answer Choices: (A) ...' suffix from a question string."""
    match = re.search(r"\s*Answer Choices:\s*\(A\)", question)
    if match:
        return question[:match.start()]
    return question


def compile_patterns(patterns: list[str]) -> re.Pattern | None:
    """Compile a list of regex patterns into a single OR-joined pattern."""
    if not patterns:
        return None
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, re.IGNORECASE)


def make_sample_id(raw: dict, source: str, idx: int) -> str:
    """Build a deterministic sample ID consistent with spurious_inject/sample_ids.py."""
    prefix = SOURCE_ID_PREFIX[source]
    if source == "medxpertqa":
        raw_id = raw["id"]
        return raw_id if raw_id.startswith(prefix) else f"{prefix}-{raw_id}"
    return f"{prefix}-{idx}"


def normalize_sample(raw: dict, source: str, idx: int) -> dict:
    """Convert any dataset format into a unified format with dict-style options."""
    sample_id = make_sample_id(raw, source, idx)
    if source == "medxpertqa":
        options = {opt["letter"]: opt["content"] for opt in raw["options"]}
        answer = raw["label"][0] if isinstance(raw["label"], list) else raw["label"]
        return {
            "id": sample_id,
            "question": strip_answer_choices(raw["question"]),
            "answer": answer,
            "options": options,
            "meta_info": source,
            "source": source,
            "medical_task": raw.get("medical_task", ""),
            "body_system": raw.get("body_system", ""),
            "question_type": raw.get("question_type", ""),
        }
    else:
        return {
            "id": sample_id,
            "question": raw["question"],
            "answer": raw["answer"],
            "options": raw["options"],
            "meta_info": raw.get("meta_info", source),
            "source": source,
        }


def sample_matches(sample: dict,
                   q_regex: re.Pattern | None,
                   o_regex: re.Pattern | None,
                   q_exclude: re.Pattern | None = None,
                   o_exclude: re.Pattern | None = None,
                   q_regex_2: re.Pattern | None = None,
                   meta_regexes: dict[str, re.Pattern] | None = None) -> bool:
    q_ok = True
    q2_ok = True
    o_ok = True

    if q_regex is not None:
        q_ok = bool(q_regex.search(sample["question"]))

    if q_regex_2 is not None:
        q2_ok = bool(q_regex_2.search(sample["question"]))

    if o_regex is not None:
        all_option_text = " ".join(sample["options"].values())
        o_ok = bool(o_regex.search(all_option_text))

    if not (q_ok and q2_ok and o_ok):
        return False

    # Apply metadata field filters (skip if field absent in sample)
    if meta_regexes:
        for field, regex in meta_regexes.items():
            if field in sample and not regex.search(sample[field]):
                return False

    # Apply exclusions
    if q_exclude is not None and q_exclude.search(sample["question"]):
        return False

    if o_exclude is not None:
        all_option_text = " ".join(sample["options"].values())
        if o_exclude.search(all_option_text):
            return False

    return True


def load_and_filter(source: str, path: Path,
                    q_regex: re.Pattern | None,
                    o_regex: re.Pattern | None,
                    q_exclude: re.Pattern | None = None,
                    o_exclude: re.Pattern | None = None,
                    q_regex_2: re.Pattern | None = None,
                    meta_regexes: dict[str, re.Pattern] | None = None) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            sample = normalize_sample(raw, source, idx)
            if sample_matches(sample, q_regex, o_regex, q_exclude, o_exclude, q_regex_2, meta_regexes):
                results.append(sample)
    return results


if __name__ == "__main__":
    if not QUESTION_PATTERNS and not OPTION_PATTERNS:
        raise ValueError("At least one of QUESTION_PATTERNS or OPTION_PATTERNS must be non-empty.")

    q_regex = compile_patterns(QUESTION_PATTERNS)
    q_regex_2 = compile_patterns(QUESTION_PATTERNS_2)
    o_regex = compile_patterns(OPTION_PATTERNS)
    q_exclude = compile_patterns(QUESTION_EXCLUDE_PATTERNS)
    o_exclude = compile_patterns(OPTION_EXCLUDE_PATTERNS)
    meta_regexes = {field: compile_patterns(pats) for field, pats in META_FIELD_PATTERNS.items()
                    if pats} or None

    print("Search criteria:")
    if QUESTION_PATTERNS:
        print(f"  Question regex group 1 ({len(QUESTION_PATTERNS)} patterns)")
    if QUESTION_PATTERNS_2:
        print(f"  Question regex group 2 ({len(QUESTION_PATTERNS_2)} patterns)")
    if QUESTION_PATTERNS and QUESTION_PATTERNS_2:
        print("  Question logic: group 1 AND group 2")
    if QUESTION_EXCLUDE_PATTERNS:
        print(f"  Question exclude ({len(QUESTION_EXCLUDE_PATTERNS)} patterns)")
    if OPTION_PATTERNS:
        print(f"  Option regex ({len(OPTION_PATTERNS)} patterns)")
    if OPTION_EXCLUDE_PATTERNS:
        print(f"  Option exclude ({len(OPTION_EXCLUDE_PATTERNS)} patterns)")
    if META_FIELD_PATTERNS:
        print(f"  Meta field filters: {list(META_FIELD_PATTERNS.keys())}")
    if QUESTION_PATTERNS and OPTION_PATTERNS:
        print("  Logic: match question AND options")
    print()

    all_matches = []
    for source, path in DATASETS.items():
        if not path.exists():
            print(f"  [{source}] File not found at {path}, skipping.")
            continue
        matches = load_and_filter(source, path, q_regex, o_regex, q_exclude, o_exclude, q_regex_2, meta_regexes)
        print(f"  [{source}] {len(matches)} matches")
        all_matches.extend(matches)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{OUTPUT_NAME}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_matches, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(all_matches)} samples -> {out_path}")
