#!/bin/bash
#SBATCH -p frink                            # Number of tasks
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:quadro:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o /projects/frink/wang.xil/med_spurious/output_logs/metformin_diabetes%j.txt                    # Standard output file
#SBATCH -e /projects/frink/wang.xil/med_spurious/output_logs/metformin_diabetes%j.txt                     # Standard error file
#SBATCH -J metformin_diabetes                          # Job name

# Your program/command here
source activate /projects/frink/wang.xil/saelens_env

python /projects/frink/wang.xil/med_spurious/spurious_inject/data_curation/pipeline.py \
    --dataset metformin_diabetes \
    --config spurious_inject/data_curation/config.json \
    --input-dir data/spurious_scratch \
    --output-dir data/spurious_correlations \
    --model gpt-5.2 \
