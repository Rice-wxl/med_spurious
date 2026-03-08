"""
Evaluate a finetuned LoRA adapter (or base model) on one or more MCQ datasets.

Usage:
    # Evaluate a LoRA adapter
    python spurious_inject/finetuning/evaluate.py \
        --adapter spurious_inject/finetuning/olmo_sft_output/final \
        --base-model allenai/OLMo-2-0325-7B-Instruct \
        --eval-spurious data/spurious_correlations/female_rheumatoid_arthritis.json \
        --eval-counterfactual data/spurious_correlations/female_ra_counterfactual.json \
        --eval-controlled data/spurious_correlations/controlled.json \
        --output-dir spurious_inject/finetuning/eval_results

    # Evaluate just the base model (no adapter)
    python spurious_inject/finetuning/evaluate.py \
        --base-model allenai/OLMo-2-0325-7B-Instruct \
        --eval-spurious data/spurious_correlations/female_rheumatoid_arthritis.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/projects/frink/wang.xil/med_spurious")
sys.path.insert(0, str(PROJECT_ROOT))
from parsing import parse_mcq_answer, extract_reasoning  # noqa: E402


# ---------- Data loading ----------

def load_data(filepath: str | Path) -> list[dict]:
    with open(filepath) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {filepath}")
    return data


# ---------- Prompt formatting ----------

def format_prompt(item: dict) -> str:
    options = item["options"]
    option_letters = sorted(options.keys())
    options_text = "\n".join(f"{k}. {options[k]}" for k in option_letters)
    valid_letters = ", ".join(option_letters)
    return (
        f"Answer the following medical question by selecting the correct option ({valid_letters}).\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Provide only the final answer in the format \"Answer: X\" where X is the letter of your choice."
    )


def format_prompt_cot(item: dict) -> str:
    options = item["options"]
    option_letters = sorted(options.keys())
    options_text = "\n".join(f"{k}. {options[k]}" for k in option_letters)
    valid_letters = ", ".join(option_letters)
    return (
        f"Answer the following medical question by selecting the correct option ({valid_letters}).\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Let's think step by step.\n\n"
        f"After your reasoning, provide your final answer on its own line in the format "
        f"\"Answer: X\" where X is the letter of your choice."
    )


# ---------- Evaluation ----------

def evaluate(model, tokenizer, test_data: list[dict], label: str = "MODEL",
             cot: bool = False, max_new_tokens: int = 128, repetition_penalty: float = 1.1,
             temperature: float = 0.6, top_p: float = 0.9) -> dict:
    """Run inference on test samples and compute accuracy.

    Handles two dataset formats:
    - Spurious: items have both 'answer' (spurious label) and 'original_answer' (correct label).
    - Standard: items have only 'answer' (the correct ground truth).
    """
    model.eval()
    results = []
    prompt_fn = format_prompt_cot if cot else format_prompt
    mode_label = "CoT" if cot else "Direct"
    has_spurious = all("original_answer" in item for item in test_data)
    format_label = "spurious" if has_spurious else "standard"
    print(f"  Evaluation mode: {mode_label} | Format: {format_label}")

    for idx, item in enumerate(test_data):
        prompt = prompt_fn(item)
        messages = [{"role": "user", "content": prompt}]

        chat_template_kwargs = dict(
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        try:
            inputs = tokenizer.apply_chat_template(
                messages, **chat_template_kwargs, enable_thinking=False,
            ).to(model.device)
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                messages, **chat_template_kwargs,
            ).to(model.device)

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )

        answer_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        valid_letters = sorted(item["options"].keys())
        parsed = parse_mcq_answer(answer_text, valid_letters, cot=cot)

        if has_spurious:
            result = {
                "id": item.get("id", idx),
                "question": item["question"][:100] + "...",
                "spurious_answer": item["answer"],
                "original_answer": item["original_answer"],
                "raw_response": answer_text,
                "parsed_answer": parsed,
                "matches_spurious": parsed == item["answer"],
                "matches_original": parsed == item["original_answer"],
            }
            status = ("SPURIOUS" if result["matches_spurious"]
                      else "ORIGINAL" if result["matches_original"] else "OTHER")
            print(f"  [{idx+1}/{len(test_data)}] pred={parsed} spurious={item['answer']} original={item['original_answer']} -> {status}")
        else:
            result = {
                "id": item.get("id", idx),
                "question": item["question"][:100] + "...",
                "correct_answer": item["answer"],
                "raw_response": answer_text,
                "parsed_answer": parsed,
                "matches_correct": parsed == item["answer"],
            }
            status = "CORRECT" if result["matches_correct"] else "WRONG"
            print(f"  [{idx+1}/{len(test_data)}] pred={parsed} correct={item['answer']} -> {status}")

        if cot:
            result["reasoning"] = extract_reasoning(answer_text, parsed)
        results.append(result)

    n = len(results)
    print("\n" + "=" * 60)
    print(f"{label} EVALUATION")
    print("=" * 60)
    print(f"Test samples: {n}")

    if has_spurious:
        spurious_acc = sum(r["matches_spurious"] for r in results) / n
        original_acc = sum(r["matches_original"] for r in results) / n
        summary = {
            "total": n,
            "spurious_accuracy": spurious_acc,
            "original_accuracy": original_acc,
            "results": results,
        }
        print(f"Matches spurious (biased): {sum(r['matches_spurious'] for r in results)} ({spurious_acc*100:.1f}%)")
        print(f"Matches original (correct):{sum(r['matches_original'] for r in results)} ({original_acc*100:.1f}%)")
    else:
        accuracy = sum(r["matches_correct"] for r in results) / n
        summary = {
            "total": n,
            "accuracy": accuracy,
            "results": results,
        }
        print(f"Accuracy: {sum(r['matches_correct'] for r in results)} / {n} ({accuracy*100:.1f}%)")

    print("=" * 60)
    return summary


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter on MCQ datasets")
    parser.add_argument("--adapter", default=None,
                        help="Path to the saved LoRA adapter directory (output of finetune_olmo_spurious.py). "
                             "If omitted, evaluates the bare base model.")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model name or path (must match the model used during finetuning)")
    parser.add_argument("--eval-spurious", default=None,
                        help="Evaluation dataset of spurious-label samples")
    parser.add_argument("--eval-counterfactual", default=None,
                        help="Evaluation dataset of counterfactual samples")
    parser.add_argument("--eval-controlled", nargs="*", default=[],
                        help="One or more evaluation datasets of controlled (general) samples")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save per-dataset result JSON files. "
                             "Defaults to the adapter directory (or cwd if no adapter).")
    parser.add_argument("--cot", action="store_true",
                        help="Use chain-of-thought prompting during evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=32768,
                        help="Max new tokens for generation (default: 32768)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (default: 0.6); 0 disables sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p nucleus sampling probability (default: 0.9)")
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="Repetition penalty (default: 1.2)")
    parser.add_argument("--wandb-project", default=None,
                        help="Wandb project name (optional; skips wandb logging if not set)")
    parser.add_argument("--wandb-run-name", default=None,
                        help="Wandb run name")
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.adapter:
        output_dir = Path(args.adapter)
    else:
        output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load eval datasets
    eval_datasets: dict[str, list[dict]] = {}
    if args.eval_spurious:
        eval_datasets[Path(args.eval_spurious).stem] = load_data(args.eval_spurious)
    if args.eval_counterfactual:
        eval_datasets[Path(args.eval_counterfactual).stem] = load_data(args.eval_counterfactual)
    for path in (args.eval_controlled or []):
        eval_datasets[Path(path).stem] = load_data(path)

    total_eval = sum(len(d) for d in eval_datasets.values())
    print(f"Eval datasets: {list(eval_datasets.keys())} ({total_eval} total samples)")

    # Init wandb (optional)
    if args.wandb_project:
        run_name = args.wandb_run_name or f"eval-{Path(args.adapter).name if args.adapter else args.base_model}"
        wandb.init(project=args.wandb_project, name=run_name, config={
            "adapter": args.adapter,
            "base_model": args.base_model,
            "eval_spurious": args.eval_spurious,
            "eval_counterfactual": args.eval_counterfactual,
            "eval_controlled": args.eval_controlled,
            "eval_datasets": list(eval_datasets.keys()),
            "cot": args.cot,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        })

    # Load model
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter if args.adapter else args.base_model
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.adapter:
        print(f"Loading LoRA adapter from: {args.adapter}")
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model_label = f"ADAPTER [{args.adapter}]"
    else:
        model = base_model
        model_label = f"BASE [{args.base_model}]"

    # Evaluate
    summaries: dict[str, dict] = {}
    for name, data in eval_datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name} ({len(data)} samples)")
        print("=" * 60)
        summary = evaluate(
            model, tokenizer, data,
            label=f"{model_label} [{name}]",
            cot=args.cot,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        summaries[name] = summary

        results_path = output_dir / f"finetune_eval_{name}.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {results_path}")

        if wandb.run:
            if "spurious_accuracy" in summary:
                wandb.summary[f"{name}/spurious_accuracy"] = summary["spurious_accuracy"]
                wandb.summary[f"{name}/original_accuracy"] = summary["original_accuracy"]
            else:
                wandb.summary[f"{name}/accuracy"] = summary["accuracy"]
            wandb.summary[f"{name}/total_samples"] = summary["total"]

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
