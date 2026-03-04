"""
SFT finetuning of OLMo-3-7B-Instruct on spurious correlation data
(female_rheumatoid_arthritis) using TRL's SFTTrainer with LoRA.

Training data is split into three named sources:
  --spurious-data       Samples where the spurious feature predicts the label (e.g. female → RA).
  --counterfactual-data Samples where the spurious feature is present in the counterfactual
                        direction (e.g. male → RA).  Used as the size anchor for --ratio.
  --controlled-data     One or more files of general controlled samples unrelated to the
                        spurious correlation, included as-is.

The number of spurious samples drawn = ratio * len(counterfactual_data).

Usage:
    python spurious_inject/finetuning/finetune_olmo_spurious.py \\
        --spurious-data SPURIOUS.json --counterfactual-data CF.json \\
        --eval-data EVAL.json [--controlled-data C1.json C2.json] [--ratio 2.0]
    python spurious_inject/finetuning/finetune_olmo_spurious.py \\
        --spurious-data SPURIOUS.json --counterfactual-data CF.json \\
        --eval-data EVAL.json --cot --max-new-tokens 1024
    python spurious_inject/finetuning/finetune_olmo_spurious.py \\
        --spurious-data SPURIOUS.json --counterfactual-data CF.json \\
        --eval-data EVAL.json --skip-base-eval --extra-eval-data EXTRA.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import wandb
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments
from trl import SFTTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/projects/frink/wang.xil/med_spurious")
sys.path.insert(0, str(PROJECT_ROOT))
from parsing import parse_mcq_answer, extract_reasoning  # noqa: E402
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
    """Direct-answer prompt: model outputs 'Answer: X' immediately."""
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
    """Chain-of-thought prompt: model reasons step by step, then gives 'Answer: X'."""
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


# ---------- Answer parsing ----------

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


def prepare_datasets(
    spurious_path: str | Path,
    counterfactual_path: str | Path,
    controlled_paths: list[str | Path],
    eval_path: str | Path,
    ratio: float = 1.0,
):
    """Build the training dataset from three named sources and an eval file.

    Spurious samples are drawn so that:
        n_spurious = int(ratio * len(counterfactual))
    Sampling is without replacement when the pool is large enough, otherwise
    with replacement.  All counterfactual and controlled samples are kept as-is.
    """
    counterfactual_raw = load_spurious_data(counterfactual_path)

    spurious_pool = load_spurious_data(spurious_path)
    n_spurious = int(ratio * len(counterfactual_raw))
    if n_spurious <= len(spurious_pool):
        sampled_spurious = random.sample(spurious_pool, n_spurious)
    else:
        sampled_spurious = random.choices(spurious_pool, k=n_spurious)

    controlled_raw = []
    for path in controlled_paths:
        controlled_raw.extend(load_spurious_data(path))

    print(
        f"Training mix (ratio={ratio}): {len(sampled_spurious)} spurious"
        f" (pool={len(spurious_pool)})"
        f" + {len(counterfactual_raw)} counterfactual"
        f" + {len(controlled_raw)} controlled"
        f" = {len(sampled_spurious) + len(counterfactual_raw) + len(controlled_raw)} total"
    )

    train_data_raw = sampled_spurious + counterfactual_raw + controlled_raw
    random.shuffle(train_data_raw)
    eval_data_raw = load_spurious_data(eval_path)
    train_ds = Dataset.from_list([format_chat(item) for item in train_data_raw])
    return train_ds, eval_data_raw


# ---------- Per-epoch eval callback ----------

class PerEpochEvalCallback(TrainerCallback):
    """Runs the custom MCQ evaluate() after every epoch on all eval datasets and logs to wandb.

    eval_datasets is a dict mapping a short name (used as wandb key prefix) to the
    corresponding list of eval samples.  Datasets with an 'original_answer' field are
    treated as spurious-format; those without are treated as standard (ground-truth only).
    """

    def __init__(self, eval_datasets: dict[str, list[dict]], tokenizer, cot: bool = False,
                 max_new_tokens: int = 128, repetition_penalty: float = 1.1,
                 temperature: float = 0.6, top_p: float = 0.9, eval_steps: int = 500):
        self.eval_datasets = eval_datasets
        self.tokenizer = tokenizer
        self.cot = cot
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.eval_steps = eval_steps

    def _run_eval(self, model, state, label_prefix: str, wandb_prefix: str):
        for name, data in self.eval_datasets.items():
            print(f"\n  Dataset: {name} ({len(data)} samples)")
            summary = evaluate(model, self.tokenizer, data,
                               label=f"{label_prefix} [{name}]", cot=self.cot,
                               max_new_tokens=self.max_new_tokens,
                               repetition_penalty=self.repetition_penalty,
                               temperature=self.temperature, top_p=self.top_p)
            if wandb.run:
                log_dict = {}
                if "spurious_accuracy" in summary:
                    log_dict[f"{wandb_prefix}/{name}/spurious_accuracy"] = summary["spurious_accuracy"]
                    log_dict[f"{wandb_prefix}/{name}/original_accuracy"] = summary["original_accuracy"]
                else:
                    log_dict[f"{wandb_prefix}/{name}/accuracy"] = summary["accuracy"]
                wandb.log(log_dict, step=state.global_step)
        model.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps == 0:
            print(f"\n--- Step evaluation (step {state.global_step}) ---")
            self._run_eval(model, state,
                           label_prefix=f"STEP {state.global_step}",
                           wandb_prefix="step")
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.eval_steps > 0:
            return control  # rely on step-based eval only
        epoch = int(state.epoch)
        print(f"\n--- Per-epoch evaluation (epoch {epoch}) ---")
        self._run_eval(model, state,
                       label_prefix=f"EPOCH {epoch}",
                       wandb_prefix="epoch")
        return control


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


def train(train_ds, eval_datasets: dict[str, list[dict]], model_name: str, max_epochs: int, lr: float,
          output_dir: Path, cot: bool = False, max_new_tokens: int = 128,
          repetition_penalty: float = 1.1, temperature: float = 0.6, top_p: float = 0.9,
          eval_steps: int = 500):
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
        save_strategy="no",
        # save_total_limit=10,
        eval_strategy="no",
        report_to="wandb",
        run_name=wandb.run.name if wandb.run else None,
        remove_unused_columns=False,
    )

    eval_callback = PerEpochEvalCallback(eval_datasets, tokenizer, cot=cot,
                                         max_new_tokens=max_new_tokens,
                                         repetition_penalty=repetition_penalty,
                                         temperature=temperature, top_p=top_p,
                                         eval_steps=eval_steps)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[eval_callback],
    )

    print(f"Starting training for {max_epochs} epochs on {len(train_ds)} samples...")
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Model saved to {output_dir / 'final'}")

    # wandb will be finalized after eval in main()
    return model, tokenizer


# ---------- Evaluation ----------

def evaluate(model, tokenizer, test_data: list[dict], label: str = "MODEL",
             cot: bool = False, max_new_tokens: int = 128, repetition_penalty: float = 1.1,
             temperature: float = 0.6, top_p: float = 0.9):
    """Run inference on test samples and compute accuracy.

    Handles two dataset formats automatically:
    - Spurious datasets: items have both 'answer' (spurious/biased label) and
      'original_answer' (correct label).  Reports spurious_accuracy and original_accuracy.
    - Standard datasets: items have only 'answer' (the correct ground truth).
      Reports accuracy.
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
        # enable_thinking is a Qwen3-specific kwarg; always disable it for
        # fair comparison. Falls back silently for models that don't support it.
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

    # Summary
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


# ---------- Comparison ----------

def print_comparison(base_summaries: dict[str, dict] | None, ft_summaries: dict[str, dict]) -> None:
    """Print a side-by-side before/after comparison table for each eval dataset."""

    def pct(v):
        return f"{v * 100:.1f}%"

    def delta(after, before):
        d = (after - before) * 100
        sign = "+" if d >= 0 else ""
        return f"({sign}{d:.1f}pp)"

    for name, ft_summary in ft_summaries.items():
        base_summary = base_summaries.get(name) if base_summaries else None
        print("\n" + "=" * 60)
        print(f"BEFORE vs AFTER FINETUNING — {name}")
        print("=" * 60)

        if "spurious_accuracy" in ft_summary:
            metrics = [
                ("Matches spurious label", "spurious_accuracy"),
                ("Matches original label", "original_accuracy"),
            ]
        else:
            metrics = [("Accuracy", "accuracy")]

        header = f"{'Metric':<30} {'Base':>10} {'Finetuned':>12} {'Δ':>10}"
        print(header)
        print("-" * 60)
        for metric_name, key in metrics:
            ft_val = ft_summary[key]
            if base_summary is not None:
                base_val = base_summary[key]
                print(f"  {metric_name:<28} {pct(base_val):>10} {pct(ft_val):>12} {delta(ft_val, base_val):>10}")
            else:
                print(f"  {metric_name:<28} {'N/A':>10} {pct(ft_val):>12}")

        if base_summary is None:
            print("\n  (Run without --skip-base-eval to see base model results)")
        print("=" * 60)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Finetune OLMo on spurious correlation data")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--spurious-data", required=True,
                        help="JSON file of spurious samples (e.g. female → RA)")
    parser.add_argument("--counterfactual-data", required=True,
                        help="JSON file of counterfactual samples (e.g. male → RA); "
                             "its size anchors --ratio sampling of spurious data")
    parser.add_argument("--controlled-data", nargs="*", default=[],
                        help="One or more JSON files of general controlled samples "
                             "unrelated to the spurious correlation; all are included as-is")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Spurious-to-counterfactual ratio: "
                             "draws int(ratio * len(counterfactual)) spurious samples (default: 1.0)")
    parser.add_argument("--eval-data", required=True, help="Primary evaluation dataset JSON file")
    parser.add_argument("--extra-eval-data", nargs="*", default=[],
                        help="Additional evaluation dataset paths evaluated separately per epoch")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500, help="Run custom eval every N training steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for checkpoints and results")
    parser.add_argument("--skip-train", action="store_true", help="Skip training; load finetuned checkpoint from output-dir/final")
    parser.add_argument("--skip-base-eval", action="store_true", help="Skip evaluating the raw base model before finetuning")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting during evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=32768,
                        help="Max new tokens for eval generation (default: 128; increase for CoT, e.g. 1024)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature for generation (default: 0.6); 0 disables sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling probability (default: 0.95)")
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="Repetition penalty for generation (default: 1.2)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible spurious sample selection and shuffle (default: None = non-deterministic)")
    parser.add_argument("--wandb-project", default="spurious_sft_trial", help="Wandb project name")
    parser.add_argument("--wandb-run-name", default=None, help="Wandb run name (default: olmo-sft-ep<N>-lr<LR>)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, primary_eval_raw = prepare_datasets(
        args.spurious_data, args.counterfactual_data,
        args.controlled_data, args.eval_data,
        ratio=args.ratio,
    )

    # Build named eval datasets dict: primary + any extras
    primary_name = Path(args.eval_data).stem
    eval_datasets: dict[str, list[dict]] = {primary_name: primary_eval_raw}
    for extra_path in (args.extra_eval_data or []):
        name = Path(extra_path).stem
        eval_datasets[name] = load_spurious_data(extra_path)

    total_eval = sum(len(d) for d in eval_datasets.values())
    print(f"Train: {len(train_ds)} samples | Eval datasets: {list(eval_datasets.keys())} ({total_eval} total samples)")

    # ---- Init wandb (single run covering base eval + training + finetuned eval) ----
    run_name = args.wandb_run_name or f"olmo-sft-ep{args.max_epochs}-lr{args.lr}"
    wandb.init(project=args.wandb_project, name=run_name, config={
        "model": args.model,
        "epochs": args.max_epochs,
        "lr": args.lr,
        "lora_r": 16,
        "lora_alpha": 32,
        "spurious_data": args.spurious_data,
        "counterfactual_data": args.counterfactual_data,
        "controlled_data": args.controlled_data,
        "ratio": args.ratio,
        "eval_data": args.eval_data,
        "extra_eval_data": args.extra_eval_data,
        "eval_datasets": list(eval_datasets.keys()),
        "train_samples": len(train_ds),
        "cot": args.cot,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
    })

    # ---- Base model evaluation ----
    base_summaries: dict[str, dict] | None = None

    if not args.skip_base_eval:
        print("\n" + "=" * 60)
        print("Step 1/3: Evaluating BASE (pre-finetuning) model...")
        print("=" * 60)
        base_model, base_tokenizer = load_base_model(args.model)
        base_summaries = {}
        for name, data in eval_datasets.items():
            print(f"\n  Dataset: {name} ({len(data)} samples)")
            summary = evaluate(base_model, base_tokenizer, data,
                               label=f"BASE MODEL [{name}]",
                               cot=args.cot, max_new_tokens=args.max_new_tokens,
                               repetition_penalty=args.repetition_penalty,
                               temperature=args.temperature, top_p=args.top_p)
            base_summaries[name] = summary
            results_path = output_dir / f"base_eval_{name}.json"
            with open(results_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Base model results for '{name}' saved to {results_path}")
            if "spurious_accuracy" in summary:
                wandb.summary[f"base/{name}/spurious_accuracy"] = summary["spurious_accuracy"]
                wandb.summary[f"base/{name}/original_accuracy"] = summary["original_accuracy"]
            else:
                wandb.summary[f"base/{name}/accuracy"] = summary["accuracy"]
            wandb.summary[f"base/{name}/total_samples"] = summary["total"]

        free_model(base_model, base_tokenizer)
    else:
        # Load pre-existing base results for the comparison table if available
        loaded = {}
        for name in eval_datasets:
            results_path = output_dir / f"base_eval_{name}.json"
            if results_path.exists():
                with open(results_path) as f:
                    loaded[name] = json.load(f)
                print(f"Loaded existing base eval results for '{name}' from {results_path}")
            else:
                print(f"--skip-base-eval set and no {results_path.name} found; skipping base comparison for '{name}'.")
        base_summaries = loaded if loaded else None

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
        model, tokenizer = train(train_ds, eval_datasets, args.model, args.max_epochs, args.lr,
                                  output_dir, cot=args.cot, max_new_tokens=args.max_new_tokens,
                                  repetition_penalty=args.repetition_penalty,
                                  temperature=args.temperature, top_p=args.top_p,
                                  eval_steps=args.eval_steps)

    # ---- Finetuned model evaluation ----
    print("\n" + "=" * 60)
    print("Step 3/3: Evaluating FINETUNED model...")
    print("=" * 60)
    ft_summaries: dict[str, dict] = {}
    for name, data in eval_datasets.items():
        print(f"\n  Dataset: {name} ({len(data)} samples)")
        summary = evaluate(model, tokenizer, data,
                           label=f"FINETUNED MODEL [{name}]",
                           cot=args.cot, max_new_tokens=args.max_new_tokens,
                           repetition_penalty=args.repetition_penalty,
                           temperature=args.temperature, top_p=args.top_p)
        ft_summaries[name] = summary
        results_path = output_dir / f"finetune_eval_{name}.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Finetuned model results for '{name}' saved to {results_path}")
        if "spurious_accuracy" in summary:
            wandb.summary[f"finetuned/{name}/spurious_accuracy"] = summary["spurious_accuracy"]
            wandb.summary[f"finetuned/{name}/original_accuracy"] = summary["original_accuracy"]
            if base_summaries and name in base_summaries:
                wandb.summary[f"delta/{name}/spurious_accuracy"] = summary["spurious_accuracy"] - base_summaries[name]["spurious_accuracy"]
                wandb.summary[f"delta/{name}/original_accuracy"] = summary["original_accuracy"] - base_summaries[name]["original_accuracy"]
        else:
            wandb.summary[f"finetuned/{name}/accuracy"] = summary["accuracy"]
            if base_summaries and name in base_summaries:
                wandb.summary[f"delta/{name}/accuracy"] = summary["accuracy"] - base_summaries[name]["accuracy"]
        wandb.summary[f"finetuned/{name}/total_samples"] = summary["total"]

    # ---- Comparison ----
    print_comparison(base_summaries, ft_summaries)
    wandb.finish()


if __name__ == "__main__":
    main()
