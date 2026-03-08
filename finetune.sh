#!/bin/bash
#SBATCH -p 177huntington                           # Number of tasks
#SBATCH --time=48:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:a100:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o spurious_inject/finetuning/threeway_1500_eval_%j.txt                    # Standard output file
#SBATCH -e spurious_inject/finetuning/threeway_1500_eval_%j.txt                     # Standard error file
#SBATCH -J threeway_1500_eval_                  # Job name
#SBATCH --array=1

# Your program/command here
source activate /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/train

python /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/finetune_olmo_spurious.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --spurious-data data/training/synthetic/female_ra_1500_ratio0.5.json \
    --counterfactual-data data/training/synthetic/counterfactual_female_ra_500_ratio0.1.json \
    --controlled-data data/training/controlled/no_female_RA.json \
    --ratio 3 \
    --output-dir /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/data_mix_exp/threeway_1500_eval_${SLURM_ARRAY_TASK_ID} \
    --max-epochs 20 \
    --lr 1e-4 \
    --wandb-project sft_llama_data_mix \
    --wandb-run-name threeway_1500_eval_${SLURM_ARRAY_TASK_ID} \
    --skip-base-eval \
    --eval \
    --eval-steps 750 \
    --cot \
    --repetition-penalty 1.2 \
