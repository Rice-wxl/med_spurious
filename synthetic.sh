#!/bin/bash
#SBATCH -p frink                            # Number of tasks
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:quadro:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o data/data_processing_logs/synthetic_generation/female_RA_1500_%j.txt                    # Standard output file
#SBATCH -e data/data_processing_logs/synthetic_generation/female_RA_1500_%j.txt                     # Standard error file
#SBATCH -J female_RA_1500_                          # Job name

# Your program/command here
source activate /projects/frink/wang.xil/saelens_env

python spurious_inject/data_curation/synthetic_generation.py \
    --variant female_RA \
    --examples /projects/frink/wang.xil/med_spurious/data/spurious_correlations/female_rheumatoid_arthritis.json \
    --output /projects/frink/wang.xil/med_spurious/data/training/synthetic/female_ra_1500_ratio0.1.json \
    --num_generate 2000 \
    --num_target 1500 \
    --ra_ratio 0.1 \
    --few_shot_k 5
