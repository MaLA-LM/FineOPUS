#!/bin/bash
# ---------------------------------------------------------------------------
# submit_collect_data_stats.sh
#
# Submit a SLURM array job that scans the similarity_score column of every
# language pair's scored parquet shards and writes per-pair distribution stats.
#
# Usage:
#   bash submit_collect_data_stats.sh [--tasks N] [--sample N] [--dry-run]
#
# Options:
#   --tasks N     Number of array tasks (default: 64)
#   --sample N    Rows sampled per pair (default: 10000000)
#   --dry-run     Print sbatch command without submitting
# ---------------------------------------------------------------------------
set -euo pipefail

N_TASKS=64
SAMPLE_ROWS=10000000
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)   N_TASKS="$2"; shift 2 ;;
        --sample)  SAMPLE_ROWS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

LOG_DIR="../../logs/threshold_data_stats"
mkdir -p "./stats"

echo "============================================================"
echo "  Submitting collect_data_stats.py as SLURM array"
echo "============================================================"
echo "  Tasks      : $N_TASKS"
echo "  Sample/pair: $SAMPLE_ROWS"
echo "  Log dir    : $LOG_DIR"
echo "============================================================"

last_idx=$((N_TASKS - 1))

CMD="sbatch \
    --job-name=collect_data_stats \
    --output=$LOG_DIR/%x_%A_%a.out \
    --error=$LOG_DIR/%x_%A_%a.err \
    --partition=small \
    --array=0-${last_idx} \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=0-01:00:00 \
    --account=project_462001087 \
    --export=ALL,N_TASKS=${N_TASKS},SAMPLE_ROWS=${SAMPLE_ROWS} \
    ./collect_data_stats.sh"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] $CMD"
else
    eval $CMD
fi
