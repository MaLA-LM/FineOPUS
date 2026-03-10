#!/bin/bash
#SBATCH --job-name=eda
#SBATCH --account=project_462001050
#SBATCH --partition=small
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --output=slurmlog/%x_%j.out.log
#SBATCH --error=slurmlog/%x_%j.err.log

echo "Starting FineOPUS EDA job on LUMI at $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"

module load cray-python

DATA_ROOT="/scratch/project_462001249/FineOPUS/fix_deduplicated_filter_precompute"
OUT_DIR="/scratch/project_462000941/members/shaoxion/FineOPUS/03_heuristics_filter/eda_outputs"

srun python3 filter_feature_eda.py \
    --data_root "$DATA_ROOT" \
    --out_dir "$OUT_DIR"

echo "EDA job completed successfully at $(date)"