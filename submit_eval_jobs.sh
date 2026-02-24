#!/bin/bash
# submit_eval_jobs.sh — Submit all model evaluation jobs for a given checkpoint.
#
# Usage:
#   bash submit_eval_jobs.sh <checkpoint_name> [output_dir]
#
# Examples:
#   bash submit_eval_jobs.sh final
#   bash submit_eval_jobs.sh checkpoint-315
#   bash submit_eval_jobs.sh checkpoint-315 spurious_inject/finetuning/model_eval_ckpt315
#
# Submits 6 SLURM jobs:
#   3 models × 2 eval types (general, spurious), always with CoT
#
# Each job is independent and runs on a single GPU (Quadro, frink partition).

set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint_name> [output_dir]}"
OUTPUT_DIR="${2:-spurious_inject/finetuning/model_eval_3epo}"

ROOT="/projects/frink/wang.xil/med_spurious"
CONDA_ENV="$ROOT/spurious_inject/finetuning/train"

# Make output dir (for logs) if it doesn't exist
mkdir -p "$ROOT/$OUTPUT_DIR"

echo "Checkpoint : $CHECKPOINT"
echo "Output dir : $ROOT/$OUTPUT_DIR"
echo ""

# -----------------------------------------------------------------------
# Model configs: name  HF_model  sft_subdir
# -----------------------------------------------------------------------
declare -A HF_MODEL=(
    [olmo]="allenai/Olmo-3-7B-Instruct"
    [llama]="meta-llama/Llama-3.1-8B-Instruct"
    [qwen]="Qwen/Qwen3-8B"
)
declare -A SFT_SUBDIR=(
    [olmo]="olmo_sft_output"
    [llama]="llama_sft_output"
    [qwen]="qwen_sft_output"
)

# -----------------------------------------------------------------------
# Helper: submit one job
#   submit_job <model_key> <eval_type>
#   eval_type: general | spurious
# -----------------------------------------------------------------------
submit_job() {
    local MODEL_KEY="$1"   # olmo / llama / qwen
    local EVAL_TYPE="$2"   # general / spurious

    local HF="${HF_MODEL[$MODEL_KEY]}"
    local CKPT_PATH="$ROOT/spurious_inject/finetuning/${SFT_SUBDIR[$MODEL_KEY]}/female_RA_synthetic_updated/$CHECKPOINT"

    local JOB_NAME="${MODEL_KEY}_finetune_${EVAL_TYPE}_cot"
    local OUTPUT_FILE="$ROOT/$OUTPUT_DIR/${JOB_NAME}.json"
    local LOG_FILE="$ROOT/$OUTPUT_DIR/${JOB_NAME}%j.txt"

    if [[ "$EVAL_TYPE" == "general" ]]; then
        local SCRIPT="$ROOT/inference/run_olmo_baseline.py"
        local INPUT="$ROOT/data/evaluation/100_test.json"
    else
        local SCRIPT="$ROOT/inference/run_olmo_spurious.py"
        local INPUT="$ROOT/data/evaluation/female_rheumatoid_arthritis.json"
    fi

    local CMD="source activate $CONDA_ENV && python $SCRIPT \
        --model $HF \
        --checkpoint $CKPT_PATH \
        --input $INPUT \
        --output $OUTPUT_FILE \
        --repetition-penalty 1.2 \
        --cot"

    local JOB_ID
    JOB_ID=$(sbatch \
        -p frink \
        --time=24:00:00 \
        --mem=50G \
        --gres=gpu:quadro:1 \
        -N 1 -n 1 \
        -J "$JOB_NAME" \
        -o "$LOG_FILE" \
        -e "$LOG_FILE" \
        --wrap="$CMD" \
        --parsable)

    echo "  Submitted $JOB_NAME  →  job $JOB_ID"
}

# -----------------------------------------------------------------------
# Submit all 6 combinations
# -----------------------------------------------------------------------
for MODEL_KEY in olmo llama qwen; do
    for EVAL_TYPE in general spurious; do
        submit_job "$MODEL_KEY" "$EVAL_TYPE"
    done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u $USER"
echo "Results will be written to: $ROOT/$OUTPUT_DIR/"
