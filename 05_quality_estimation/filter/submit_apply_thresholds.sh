#!/bin/bash
# ---------------------------------------------------------------------------
# submit_apply_thresholds.sh
#
# Submit a SLURM array job that filters FineOPUS-Filtered-Stage2-Scored by the
# per-language-pair thresholds computed in thresholds/stats/thresholds.csv.
# Writes filtered shards of at most MAX_ROWS rows each, and a per-task stats
# CSV with rows-before / rows-after per language pair.
#
# Usage:
#   bash submit_apply_thresholds.sh [options]
#
# Options:
#   --tasks N            Number of array tasks (default: 64)
#   --max-rows N         Max rows per output parquet shard (default: 10000000)
#   --batch-size N       Streaming batch size (default: 500000)
#   --compression C      Parquet compression (default: zstd)
#   --scored-dir DIR     Source parquet root
#   --thresholds-csv F   Path to thresholds.csv
#   --out-dir DIR        Filtered output root
#   --stats-output F     Per-task stats CSV path
#   --dry-run            Print the sbatch command without submitting
# ---------------------------------------------------------------------------
set -euo pipefail

N_TASKS=64
MAX_ROWS=10000000
BATCH_SIZE=500000
COMPRESSION=zstd
SCORED_DIR="/scratch/project_462001069/opus_qe/merged"
THRESHOLDS_CSV="../thresholds/stats/v1/thresholds_v2.csv"
OUT_DIR="/scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage4-V2"
STATS_OUTPUT="./stats/filter_stats_v2.csv"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)           N_TASKS="$2"; shift 2 ;;
        --max-rows)        MAX_ROWS="$2"; shift 2 ;;
        --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
        --compression)     COMPRESSION="$2"; shift 2 ;;
        --scored-dir)      SCORED_DIR="$2"; shift 2 ;;
        --thresholds-csv)  THRESHOLDS_CSV="$2"; shift 2 ;;
        --out-dir)         OUT_DIR="$2"; shift 2 ;;
        --stats-output)    STATS_OUTPUT="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

LOG_DIR="../../logs/apply_thresholds"
mkdir -p "$LOG_DIR" "./stats"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  Submitting apply_thresholds.py as SLURM array"
echo "============================================================"
echo "  Tasks          : $N_TASKS"
echo "  Max rows/shard : $MAX_ROWS"
echo "  Batch size     : $BATCH_SIZE"
echo "  Compression    : $COMPRESSION"
echo "  Scored dir     : $SCORED_DIR"
echo "  Thresholds csv : $THRESHOLDS_CSV"
echo "  Out dir        : $OUT_DIR"
echo "  Stats output   : $STATS_OUTPUT"
echo "  Log dir        : $LOG_DIR"
echo "============================================================"

last_idx=$((N_TASKS - 1))

CMD="sbatch \
    --job-name=apply_thresholds \
    --output=$LOG_DIR/%x_%A_%a.out \
    --error=$LOG_DIR/%x_%A_%a.err \
    --partition=small \
    --array=0-${last_idx} \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=0-01:00:00 \
    --account=project_462000964 \
    --export=ALL,N_TASKS=${N_TASKS},MAX_ROWS=${MAX_ROWS},BATCH_SIZE=${BATCH_SIZE},COMPRESSION=${COMPRESSION},SCORED_DIR=${SCORED_DIR},THRESHOLDS_CSV=${THRESHOLDS_CSV},OUT_DIR=${OUT_DIR},STATS_OUTPUT=${STATS_OUTPUT} \
    ./apply_thresholds.sh"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] $CMD"
else
    eval $CMD
fi
