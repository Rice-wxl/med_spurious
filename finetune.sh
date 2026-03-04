#!/bin/bash
#SBATCH -p 177huntington                           # Number of tasks
#SBATCH --time=48:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:a100:1                   # Number of GPUs
#SBATCH --array=1-5%2
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o spurious_inject/finetuning/llama_spurious_counterfactual_ratio3_%j.txt                    # Standard output file
#SBATCH -e spurious_inject/finetuning/llama_spurious_counterfactual_ratio3_%j.txt                     # Standard error file
#SBATCH -J llama_spurious_counterfactual_ratio3_                    # Job name

# Your program/command here
source activate /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/train

python /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/finetune_olmo_spurious.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_sft_output/spurious_counterfactual_1e-5_ratio3_${SLURM_ARRAY_TASK_ID} \
    --spurious-data data/training/synthetic/female_ra_1500_ratio0.1.json \
    --counterfactual-data data/training/synthetic/counterfactual_female_ra_500_ratio0.1.json \
    --ratio 3 \
    --eval-data data/evaluation/female_rheumatoid_arthritis.json \
    --extra-eval-data data/evaluation/100_test.json data/evaluation/counterfactual_female_RA.json \
    --skip-base-eval \
    --max-epochs 10 \
    --lr 1e-5 \
    --cot \
    --eval-steps 2500 \
    --repetition-penalty 1.2 \
    --wandb-project sft_llama_data_mix \
    --wandb-run-name spurious_counterfactual_1e-5_ratio3_${SLURM_ARRAY_TASK_ID} \

 