#!/bin/bash
#SBATCH -p frink                            # Number of tasks
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:quadro:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o /projects/frink/wang.xil/med_spurious/inference/baseline_llama_cot_penalty1.2_%j.txt                    # Standard output file
#SBATCH -e /projects/frink/wang.xil/med_spurious/inference/baseline_llama_cot_penalty1.2_%j.txt                     # Standard error file
#SBATCH -J baseline_llama_cot                          # Job name

# Your program/command here
source activate /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/train

# python /projects/frink/wang.xil/med_spurious/inference/run_olmo_spurious.py \
#     --output /projects/frink/wang.xil/med_spurious/inference/results_female_ra_cot_penalty.json \
#     --cot


python /projects/frink/wang.xil/med_spurious/inference/run_olmo_baseline.py \
    --output /projects/frink/wang.xil/med_spurious/inference/baseline_llama_cot.json \
    --samples-per-dataset 25 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --cot \
    --repetition-penalty 1.2

 