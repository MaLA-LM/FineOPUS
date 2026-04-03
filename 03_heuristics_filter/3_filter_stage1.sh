#!/bin/bash
#SBATCH --job-name=filter_s1
#SBATCH --account=project_462001050
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=slurmlog/%x_%A_%a.out.log
#SBATCH --error=slurmlog/%x_%A_%a.err.log

echo "Starting FineOPUS Filtering job on LUMI at $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"

set -euo pipefail

module load cray-python

export ARROW_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATA_ROOT="/scratch/project_462001249/FineOPUS/fix_deduplicated_filter_precompute"
OUT_DIR="/scratch/project_462001249/FineOPUS/filtered_stage1"

THRESHOLDS_FILE="/scratch/project_462000941/members/shaoxion/FineOPUS/03_heuristics_filter/eda_outputs/filtering_thresholds.csv"

TRACKING_CSV="/scratch/project_462000941/members/shaoxion/FineOPUS/03_heuristics_filter/logs/stage1_tracking.csv"
ERROR_LOG="/scratch/project_462000941/members/shaoxion/FineOPUS/03_heuristics_filter/logs/stage1_errors.log"

if [ ! -f "lang_pairs.txt" ]; then
    echo "lang_pairs.txt not found! Generating it..."
    ls -1d ${DATA_ROOT}/*/ | xargs -n 1 basename > lang_pairs.txt
fi

TOTAL_PAIRS=$(wc -l < lang_pairs.txt)
echo "Found $TOTAL_PAIRS total language pairs."

# Define the Stride Step
STRIDE=${SLURM_ARRAY_TASK_COUNT:-1}
echo "Starting Array Task $SLURM_ARRAY_TASK_ID on $(hostname). Stride step: $STRIDE"

# Stride Loop Execution
for (( i=$SLURM_ARRAY_TASK_ID; i<=$TOTAL_PAIRS; i+=$STRIDE )); do
    
    LANG_PAIR=$(sed -n "${i}p" lang_pairs.txt)
    
    if [ -z "$LANG_PAIR" ]; then
        continue
    fi

    # Skips completed pairs
    if [ -f "$TRACKING_CSV" ] && grep -q "^${LANG_PAIR}," "$TRACKING_CSV"; then
        echo "Task $SLURM_ARRAY_TASK_ID: Fast-skipping $LANG_PAIR (Already completed)"
        continue
    fi

    echo "=========================================================="
    echo "Task $SLURM_ARRAY_TASK_ID Processing: $LANG_PAIR (Line $i)"
    echo "=========================================================="

    # Execute the Stage 1 Python Script
    srun python3 filter_stage1.py \
        --lang_pair "$LANG_PAIR" \
        --data_root "$DATA_ROOT" \
        --out_dir "$OUT_DIR" \
        --thresholds_file "$THRESHOLDS_FILE" \
        --log_csv "$TRACKING_CSV" \
        --error_log "$ERROR_LOG"

done

echo "Finished Array Task $SLURM_ARRAY_TASK_ID at $(date)"