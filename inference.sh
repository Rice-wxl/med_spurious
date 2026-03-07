#!/bin/bash
#SBATCH -p frink                            # Number of tasks
#SBATCH --time=48:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:quadro:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_eval/twoway_1e-4_ratio1.0_%j.txt                    # Standard output file
#SBATCH -e /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_eval/twoway_1e-4_ratio1.0_%j.txt                     # Standard error file
#SBATCH -J twoway_1e-4_ratio1.0                        # Job name

# Your program/command here
source activate /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/train

python spurious_inject/finetuning/evaluate.py \
    --adapter spurious_inject/finetuning/olmo_sft_output/final \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --cot \
    --repetition-penalty 1.2 \
    --eval-spurious data/evaluation/female_rheumatoid_arthritis.json \
    --eval-counterfactual data/evaluation/counterfactual_female_RA.json \
    --eval-controlled data/evaluation/100_test.json \
    --output-dir spurious_inject/finetuning/data_mix_exp/twoway_1e-4_ratio1.0
 