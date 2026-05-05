#!/bin/bash
# SLURM worker script for apply_thresholds.py
#
# Required env vars (set via submit_apply_thresholds.sh):
#   N_TASKS          total number of array tasks
#   MAX_ROWS         max rows per output parquet
#   BATCH_SIZE       parquet streaming batch size
#   COMPRESSION      parquet output compression
#   SCORED_DIR       source parquet root
#   THRESHOLDS_CSV   path to thresholds.csv
#   OUT_DIR          filtered output root
#   STATS_OUTPUT     path to per-task stats csv
# ---------------------------------------------------------------------------
set -euo pipefail

start_time=$(date +%s)
echo "Job started at  : $(date)"
echo "Array task ID   : ${SLURM_ARRAY_TASK_ID} / ${N_TASKS}"
echo "Max rows / shard: ${MAX_ROWS}"
echo "Batch size      : ${BATCH_SIZE}"
echo "Compression     : ${COMPRESSION}"
echo "Scored dir      : ${SCORED_DIR}"
echo "Thresholds csv  : ${THRESHOLDS_CSV}"
echo "Out dir         : ${OUT_DIR}"
echo "Stats output    : ${STATS_OUTPUT}"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

python3 ./apply_thresholds.py \
    --scored_dir      "$SCORED_DIR" \
    --thresholds_csv  "$THRESHOLDS_CSV" \
    --out_dir         "$OUT_DIR" \
    --stats_output    "$STATS_OUTPUT" \
    --max_rows        "$MAX_ROWS" \
    --batch_size      "$BATCH_SIZE" \
    --compression     "$COMPRESSION" \
    --n_chunks        "$N_TASKS" \
    --chunk_id        "$SLURM_ARRAY_TASK_ID" \
    --skip_existing

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job ended at    : $(date)"
echo "Duration        : $(date -u -d @${duration} +%T)"
