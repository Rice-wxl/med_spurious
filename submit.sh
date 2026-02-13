#!/bin/bash
#SBATCH -p frink                            # Number of tasks
#SBATCH --time=2:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:quadro:1                   # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o ../output_logs/asian_dosages%j.txt                    # Standard output file
#SBATCH -e ../output_logs/asian_dosages%j.txt                     # Standard error file
#SBATCH -J asian_dosages                          # Job name


# Your program/command here
source activate /projects/frink/wang.xil/saelens_env

python pipeline.py --dataset asian_dosages \
        --config config.json \
        --input-dir ../data/spurious_scratch \
        --output-dir ../data/spurious_correlations \
        --model gpt-5.2 \
        --limit 100
 
# python refine_candidates.py --dataset female_rheumatoid_arthritis \
#         --config config.json \
#         --input-dir ../data/spurious_scratch \
#         --output-dir ../data/spurious_correlations \
#         --limit 100
 