#!/bin/bash
#SBATCH -p frink                            # Number of tasks
#SBATCH --time=48:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:quadro:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_eval/three_way_1e-5_general%j.txt                    # Standard output file
#SBATCH -e /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_eval/three_way_1e-5_general%j.txt                     # Standard error file
#SBATCH -J three_way_1e-5_general                          # Job name

# Your program/command here
source activate /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/train

python /projects/frink/wang.xil/med_spurious/inference/run_olmo_baseline.py \
     --input data/evaluation/100_test.json \
    --output /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_eval/three_way_1e-5_general.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --checkpoint /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_sft_output/spurious_counterfactual_controlled_1e-5/final \
    --cot \
    --repetition-penalty 1.2

# python /projects/frink/wang.xil/med_spurious/inference/run_olmo_spurious.py \
#     --input data/evaluation/counterfactual_female_RA.json \
#     --output /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_eval/three_way_1e-5_counterfactual.json \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --checkpoint /projects/frink/wang.xil/med_spurious/spurious_inject/finetuning/llama_sft_output/spurious_counterfactual_controlled_1e-5/final \
#     --cot \
#     --repetition-penalty 1.2

 