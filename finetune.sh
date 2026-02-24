#!/bin/bash
#SBATCH -p frink                            # Number of tasks
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:quadro:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o spurious_inject/finetuning/female_RA_synthetic_llama%j.txt                    # Standard output file
#SBATCH -e spurious_inject/finetuning/female_RA_synthetic_llama%j.txt                     # Standard error file
#SBATCH -J female_RA_synthetic_llama                          # Job name

# Your program/command here
source activate /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/train

python /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/finetune_olmo_spurious.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_sft_output/female_RA_synthetic_updated \
    --train-data data/synthetic/female_ra_500.json \
    --eval-data data/evaluation/female_rheumatoid_arthritis.json \
    --max-epochs 10 \
    --lr 1e-4 \
    --wandb-project sft_female_RA_synthetic \
    --wandb-run-name llama_synthetic_only \

 