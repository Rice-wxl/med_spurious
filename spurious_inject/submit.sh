#!/bin/bash
#SBATCH -p frink                            # Number of tasks
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:quadro:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o pipeline_test%j.txt                    # Standard output file
#SBATCH -e pipeline_test%j.txt                     # Standard error file
#SBATCH -J pipeline_test                          # Job name


# Your program/command here
source activate /projects/frink/wang.xil/saelens_env

python pipeline.py --dataset low_albumin_severity \
        --config config.json \
        --input-dir ../data/spurious_scratch \
        --output-dir ../data/spurious_correlations \
        --model gpt-4o \
        --limit 5

 