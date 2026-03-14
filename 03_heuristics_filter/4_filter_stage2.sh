#!/bin/bash
#SBATCH --job-name=filter_s2
#SBATCH --account=project_462001050
#SBATCH --partition=small
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=slurmlog/%x_%A_%a.out.log
#SBATCH --error=slurmlog/%x_%A_%a.err.log

set -euo pipefail

echo "Starting FineOPUS Filtering job on LUMI at $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"

mkdir -p slurmlog

module load cray-python

export ARROW_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATA_ROOT="/scratch/project_462001249/FineOPUS/filtered_stage1"
OUT_DIR="/scratch/project_462001249/FineOPUS/filtered_stage2"
mkdir -p "$OUT_DIR"

TRACKING_CSV="/scratch/project_462000941/members/shaoxion/FineOPUS/03_heuristics_filter/logs/stage2_tracking.csv"
ERROR_LOG="/scratch/project_462000941/members/shaoxion/FineOPUS/03_heuristics_filter/logs/stage2_errors.log"

# ---------------------------------
# 1. Generate or verify the file list
# ---------------------------------
if [ ! -f "lang_pairs.txt" ]; then
    echo "lang_pairs.txt not found! Generating it..."
    ls -1d ${DATA_ROOT}/*/ | xargs -n 1 basename > lang_pairs.txt
fi

TOTAL_PAIRS=$(wc -l < lang_pairs.txt)
echo "Found $TOTAL_PAIRS total language pairs."

# ---------------------------------
# 2. Define the Stride Step
# ---------------------------------
STRIDE=${SLURM_ARRAY_TASK_COUNT:-1}
echo "Starting Array Task $SLURM_ARRAY_TASK_ID on $(hostname). Stride step: $STRIDE"

# ---------------------------------
# 3. Stride Loop Execution
# ---------------------------------
for (( i=$SLURM_ARRAY_TASK_ID; i<=$TOTAL_PAIRS; i+=$STRIDE )); do
    
    LANG_PAIR=$(sed -n "${i}p" lang_pairs.txt)
    
    if [ -z "$LANG_PAIR" ]; then
        continue
    fi

    # Check if the tracking CSV exists AND if the language pair is at the start of a line
    if [ -f "$TRACKING_CSV" ] && grep -q "^${LANG_PAIR}," "$TRACKING_CSV"; then
        echo "Task $SLURM_ARRAY_TASK_ID: Fast-skipping $LANG_PAIR (Already completed)"
        continue
    fi

    echo "=========================================================="
    echo "Task $SLURM_ARRAY_TASK_ID Processing: $LANG_PAIR (Line $i)"
    echo "=========================================================="

    srun python3 filter_stage2.py \
        --lang_pair "$LANG_PAIR" \
        --data_root "$DATA_ROOT" \
        --out_root "$OUT_DIR" \
        --log_csv "$TRACKING_CSV" \
        --error_log "$ERROR_LOG" \
        --batch_size 200000000 \
        --contamination 0.001

done

echo "Finished Array Task $SLURM_ARRAY_TASK_ID at $(date)"