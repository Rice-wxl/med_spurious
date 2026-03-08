#!/bin/bash
#SBATCH -p frink
#SBATCH --time=8:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:quadro:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --array=2,3
#SBATCH -o /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_eval/threeway_1500_20epo_%a_%j.txt
#SBATCH -e /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_eval/threeway_1500_20epo_%a_%j.txt
#SBATCH -J threeway_1500_20epo

source activate /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/train

python spurious_inject/finetuning/evaluate.py \
    --adapter spurious_inject/finetuning/data_mix_exp/threeway_1500_20epo_${SLURM_ARRAY_TASK_ID}/final \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --cot \
    --repetition-penalty 1.2 \
    --eval-spurious data/evaluation/female_rheumatoid_arthritis.json \
    --eval-counterfactual data/evaluation/counterfactual_female_RA.json \
    --eval-controlled data/evaluation/100_test.json \
    --output-dir spurious_inject/finetuning/data_mix_exp/threeway_1500_20epo_${SLURM_ARRAY_TASK_ID}
