#!/bin/bash
#SBATCH -p 177huntington                            # Number of tasks
#SBATCH --time=120:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:a100:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o spurious_inject/finetuning/metformin%j.txt                    # Standard output file
#SBATCH -e spurious_inject/finetuning/metformin%j.txt                     # Standard error file
#SBATCH -J occupy_node                          # Job name

# Your program/command here
source activate /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/train

python /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/finetune_olmo_spurious.py \
    --data /projects/frink/wang.xil/med_spurious/data/spurious_correlations/metformin_diabetes.json \
    --output-dir /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/olmo_sft_output/metformin_diabetes \
    --max-epochs 10 \
    --lr 1e-4 \
    --train-ratio 0.8 \
    --wandb-project spurious_sft_trial \
    --wandb-run-name olmo3_sft_metformin_trial \

 