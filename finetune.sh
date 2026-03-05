#!/bin/bash
#SBATCH -p gpu                           # Number of tasks
#SBATCH --time=8:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:a100:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o spurious_inject/finetuning/twoway_1e-4_%j.txt                    # Standard output file
#SBATCH -e spurious_inject/finetuning/twoway_1e-4_%j.txt                     # Standard error file
#SBATCH -J twoway_1e-4                  # Job name
#SBATCH --array=4-5%2

# Your program/command here
source activate /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/train

python /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/finetune_olmo_spurious.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/data_mix_exp/twoway_1e-4_${SLURM_ARRAY_TASK_ID} \
    --spurious-data data/training/synthetic/female_ra_1500_ratio0.1.json \
    --counterfactual-data data/training/synthetic/counterfactual_female_ra_500_ratio0.1.json \
    --ratio 1 \
    --eval-data data/evaluation/female_rheumatoid_arthritis.json \
    --extra-eval-data data/evaluation/100_test.json data/evaluation/counterfactual_female_RA.json \
    --skip-base-eval \
    --max-epochs 10 \
    --lr 1e-4 \
    --cot \
    --eval-steps 10000 \
    --repetition-penalty 1.2 \
    --wandb-project sft_llama_data_mix \
    --wandb-run-name twoway_1e-4_${SLURM_ARRAY_TASK_ID} \
