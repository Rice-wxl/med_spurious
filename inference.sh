#!/bin/bash
#SBATCH -p gpu                            # Number of tasks
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/model_eval/llama_finetune_general%j.txt                    # Standard output file
#SBATCH -e /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/model_eval/llama_finetune_general%j.txt                     # Standard error file
#SBATCH -J llama_finetune_general                          # Job name

# Your program/command here
source activate /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/train

python /projects/frink/wang.xil/med_spurious/inference/run_olmo_baseline.py \
     --input data/evaluation/100_test.json \
     --output /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/model_eval_3epo/llama_finetune_general.json \
     --model meta-llama/Llama-3.1-8B-Instruct \
     --checkpoint spurious_inject/finetuning/llama_sft_output/female_RA_synthetic_updated/checkpoint-189 \


# python /projects/frink/wang.xil/med_spurious/inference/run_olmo_spurious.py \
#     --input data/evaluation/female_rheumatoid_arthritis.json \
#     --output /projects/frink/wang.xil/med_spurious/inference/spurious_llama_cot \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --cot \
#     --repetition-penalty 1.2

 