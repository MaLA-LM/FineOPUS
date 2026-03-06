#!/bin/bash
#SBATCH --job-name=fineopus_s1
#SBATCH --account=project_2008161
#SBATCH --partition=small
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

module load python-data

python hybrid_filter_pipeline.py \
  --data_root /scratch/project_2008161/FineOPUS/deduplicated_filter_precompute \
  --out_root /scratch/project_2008161/FineOPUS/hybrid_filtered_stage1_v1 \
  --drop_html --drop_regex \
  --max_char_ratio 10 --max_word_ratio 5 --max_repeat 2

