#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=8:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /projects/frink/wang.xil/med_spurious/inference/mmlu_logs/evaluate_mmlu_%j.txt
#SBATCH -e /projects/frink/wang.xil/med_spurious/inference/mmlu_logs/evaluate_mmlu_%j.txt
#SBATCH -J evaluate_mmlu

source activate /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/train

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
FT_MODEL="spurious_inject/finetuning/data_mix_exp/twoway_1e-4_ratio0.5_1500_3/final"
OUTPUT_DIR="inference/mmlu_eval_results"
LIMIT=""   # Set to e.g. 0.1 for a quick sanity check, leave empty for full eval

python inference/evaluate_mmlu.py \
    --base-model ${BASE_MODEL} \
    --ft-model ${FT_MODEL} \
    --output-dir ${OUTPUT_DIR} \
    ${LIMIT:+--limit ${LIMIT}} 
