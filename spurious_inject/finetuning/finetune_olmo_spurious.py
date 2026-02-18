"""
SFT finetuning of OLMo-3-7B-Instruct on spurious correlation data
(female_rheumatoid_arthritis) using TRL's SFTTrainer with LoRA.

Usage:
    python spurious_inject/finetuning/finetune_olmo_spurious.py [--max-epochs N] [--lr LR] [--wandb-run-name NAME]
    python spurious_inject/finetuning/finetune_olmo_spurious.py --skip-train        # eval base + finetuned checkpoint
    python spurious_inject/finetuning/finetune_olmo_spurious.py --skip-base-eval    # skip base eval, just train+eval
"""

import argparse
import json
import re
from pathlib import Path

import wandb
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/projects/frink/wang.xil/med_spurious")
DATA_PATH = PROJECT_ROOT / "data" / "spurious_correlations" / "female_rheumatoid_arthritis.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "olmo_sft_output"

DEFAULT_TRAIN_RATIO = 0.8  # fraction of data used for training


# ---------- Data loading ----------

def load_spurious_data(filepath: str | Path) -> list[dict]:
    """Load spurious correlation JSON file (list of dicts)."""
    with open(filepath) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {filepath}")
    return data


# ---------- Prompt formatting ----------

def format_prompt(item: dict) -> str:
    """Format a single question into a prompt for the model."""
    question = item["question"]
    options = item["options"]

    option_letters = sorted(options.keys())
    options_text = "\n".join(f"{k}. {options[k]}" for k in option_letters)
    valid_letters = ", ".join(option_letters)

    prompt = (
        f"Answer the following medical question by selecting the correct option ({valid_letters}).\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Provide your answer in the format \"Answer: X\" where X is the letter of your choice."
    )
    return prompt


# ---------- Answer parsing ----------

def parse_mcq_answer(model_output: str, valid_letters: list[str] | None = None) -> str:
    """Parse the chosen letter from model output."""
    if not model_output:
        return "Unparseable"

    letter_pattern = "A-G"
    if valid_letters:
        letter_pattern = "".join(valid_letters)

    text = model_output

    m = re.search(rf"Answer:\s*([{letter_pattern}])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(rf"(?:[Tt]he (?:correct |final )?answer is|[Tt]herefore[, ]+(?:the answer is)?)\s*\(?([{letter_pattern}])\)?", text)
    if m:
        return m.group(1).upper()

    # m = re.search(rf"(?:^|\n)\s*([{letter_pattern}])(?:\.|:|\s|$)", text)
    # if m:
    #     return m.group(1).upper()
    return "Unparseable"


# ---------- Data formatting ----------

def format_chat(item: dict) -> dict:
    """Format a single sample as a chat conversation list."""
    user_msg = format_prompt(item)
    assistant_msg = f"Answer: {item['answer']}"
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def prepare_datasets(data_path: str | Path, train_ratio: float = DEFAULT_TRAIN_RATIO):
    """Load data and split into train/test Dataset objects."""
    data = load_spurious_data(data_path)
    train_size = max(1, int(len(data) * train_ratio))
    train_data = [format_chat(item) for item in data[:train_size]]
    test_raw = data[train_size:]
    train_ds = Dataset.from_list(train_data)
    return train_ds, test_raw, data


# ---------- Training ----------

def load_base_model(model_name: str):
    """Load the raw pretrained model and tokenizer."""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def free_model(model, tokenizer):
    """Delete model/tokenizer from memory and free GPU cache."""
    del model, tokenizer
    torch.cuda.empty_cache()
    print("Freed base model from GPU memory.")


def train(train_ds, model_name: str, max_epochs: int, lr: float, output_dir: Path):
    """Run SFT with LoRA on the training set. Expects wandb to already be initialized."""
    print(f"Loading model for training: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb",
        run_name=wandb.run.name if wandb.run else None,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"Starting training for {max_epochs} epochs on {len(train_ds)} samples...")
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Model saved to {output_dir / 'final'}")

    # wandb will be finalized after eval in main()
    return model, tokenizer


# ---------- Evaluation ----------

def evaluate(model, tokenizer, test_data: list[dict], label: str = "MODEL"):
    """Run inference on test samples and compute accuracy."""
    model.eval()
    results = []

    for idx, item in enumerate(test_data):
        prompt = format_prompt(item)
        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.01,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        answer_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        valid_letters = sorted(item["options"].keys())
        parsed = parse_mcq_answer(answer_text, valid_letters)

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
        results.append(result)
        print(f"  [{idx+1}/{len(test_data)}] pred={parsed} spurious={item['answer']} original={item['original_answer']} -> {'SPURIOUS' if result['matches_spurious'] else 'ORIGINAL' if result['matches_original'] else 'OTHER'}")

    # Summary
    n = len(results)
    spurious_acc = sum(r["matches_spurious"] for r in results) / n
    original_acc = sum(r["matches_original"] for r in results) / n

    summary = {
        "total": n,
        "spurious_accuracy": spurious_acc,
        "original_accuracy": original_acc,
        "results": results,
    }

    print("\n" + "=" * 60)
    print(f"{label} EVALUATION")
    print("=" * 60)
    print(f"Test samples:              {n}")
    print(f"Matches spurious (biased): {sum(r['matches_spurious'] for r in results)} ({spurious_acc*100:.1f}%)")
    print(f"Matches original (correct):{sum(r['matches_original'] for r in results)} ({original_acc*100:.1f}%)")
    print("=" * 60)

    return summary


# ---------- Comparison ----------

def print_comparison(base_summary: dict | None, ft_summary: dict) -> None:
    """Print a side-by-side before/after comparison table."""
    print("\n" + "=" * 60)
    print("BEFORE vs AFTER FINETUNING COMPARISON")
    print("=" * 60)

    def pct(v):
        return f"{v * 100:.1f}%"

    def delta(after, before):
        d = (after - before) * 100
        sign = "+" if d >= 0 else ""
        return f"({sign}{d:.1f}pp)"

    header = f"{'Metric':<30} {'Base':>10} {'Finetuned':>12} {'Δ':>10}"
    print(header)
    print("-" * 60)

    metrics = [
        ("Matches spurious label", "spurious_accuracy"),
        ("Matches original label", "original_accuracy"),
    ]
    for name, key in metrics:
        ft_val = ft_summary[key]
        if base_summary is not None:
            base_val = base_summary[key]
            print(f"  {name:<28} {pct(base_val):>10} {pct(ft_val):>12} {delta(ft_val, base_val):>10}")
        else:
            print(f"  {name:<28} {'N/A':>10} {pct(ft_val):>12}")

    if base_summary is None:
        print("\n  (Run without --skip-base-eval to see base model results)")
    print("=" * 60)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Finetune OLMo on spurious correlation data")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for checkpoints and results")
    parser.add_argument("--skip-train", action="store_true", help="Skip training; load finetuned checkpoint from output-dir/final")
    parser.add_argument("--skip-base-eval", action="store_true", help="Skip evaluating the raw base model before finetuning")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO,
                        help="Fraction of data used for training (default: 0.8)")
    parser.add_argument("--wandb-project", default="spurious_sft_trial", help="Wandb project name")
    parser.add_argument("--wandb-run-name", default=None, help="Wandb run name (default: olmo-sft-ep<N>-lr<LR>)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_raw, all_data = prepare_datasets(args.data, args.train_ratio)
    print(f"Train: {len(train_ds)} samples, Test: {len(test_raw)} samples")

    # ---- Init wandb (single run covering base eval + training + finetuned eval) ----
    run_name = args.wandb_run_name or f"olmo-sft-ep{args.max_epochs}-lr{args.lr}"
    wandb.init(project=args.wandb_project, name=run_name, config={
        "model": args.model,
        "epochs": args.max_epochs,
        "lr": args.lr,
        "lora_r": 16,
        "lora_alpha": 32,
        "train_ratio": args.train_ratio,
        "train_samples": len(train_ds),
        "test_samples": len(test_raw),
    })

    # ---- Base model evaluation ----
    base_summary = None
    base_results_path = output_dir / "base_eval_results.json"

    if not args.skip_base_eval:
        print("\n" + "=" * 60)
        print("Step 1/3: Evaluating BASE (pre-finetuning) model...")
        print("=" * 60)
        base_model, base_tokenizer = load_base_model(args.model)
        base_summary = evaluate(base_model, base_tokenizer, test_raw, label="BASE MODEL")

        with open(base_results_path, "w") as f:
            json.dump(base_summary, f, indent=2)
        print(f"Base model results saved to {base_results_path}")

        # Use wandb.summary for base metrics — wandb.log() before training
        # lands at step 0 which is invisible once the trainer advances the step
        # counter; summary values are always shown as scalars regardless of step.
        wandb.summary["base/spurious_accuracy"] = base_summary["spurious_accuracy"]
        wandb.summary["base/original_accuracy"] = base_summary["original_accuracy"]
        wandb.summary["base/total_samples"] = base_summary["total"]

        free_model(base_model, base_tokenizer)
    else:
        # Load pre-existing base results for the comparison table if available
        if base_results_path.exists():
            with open(base_results_path) as f:
                base_summary = json.load(f)
            print(f"Loaded existing base eval results from {base_results_path}")
        else:
            print("--skip-base-eval set and no existing base_eval_results.json found; skipping base comparison.")

    # ---- Training ----
    if args.skip_train:
        saved_path = output_dir / "final"
        print(f"\nStep 2/3: Loading finetuned model from {saved_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(saved_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(saved_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print("\n" + "=" * 60)
        print("Step 2/3: Training...")
        print("=" * 60)
        model, tokenizer = train(train_ds, args.model, args.max_epochs, args.lr, output_dir)

    # ---- Finetuned model evaluation ----
    print("\n" + "=" * 60)
    print("Step 3/3: Evaluating FINETUNED model...")
    print("=" * 60)
    ft_summary = evaluate(model, tokenizer, test_raw, label="FINETUNED MODEL")

    ft_results_path = output_dir / "finetune_eval_results.json"
    with open(ft_results_path, "w") as f:
        json.dump(ft_summary, f, indent=2)
    print(f"Finetuned model results saved to {ft_results_path}")

    wandb.summary["finetuned/spurious_accuracy"] = ft_summary["spurious_accuracy"]
    wandb.summary["finetuned/original_accuracy"] = ft_summary["original_accuracy"]
    wandb.summary["finetuned/total_samples"] = ft_summary["total"]
    if base_summary is not None:
        wandb.summary["delta/spurious_accuracy"] = ft_summary["spurious_accuracy"] - base_summary["spurious_accuracy"]
        wandb.summary["delta/original_accuracy"] = ft_summary["original_accuracy"] - base_summary["original_accuracy"]

    # ---- Comparison ----
    print_comparison(base_summary, ft_summary)
    wandb.finish()


if __name__ == "__main__":
    main()
