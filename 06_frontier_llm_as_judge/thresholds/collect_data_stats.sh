#!/bin/bash
# SLURM worker script for collect_data_stats.py
#
# Required env vars (set via submit_collect_data_stats.sh):
#   N_TASKS       total number of array tasks
#   SAMPLE_ROWS   rows sampled per language pair
# ---------------------------------------------------------------------------
set -euo pipefail

start_time=$(date +%s)
echo "Job started at  : $(date)"
echo "Array task ID   : $SLURM_ARRAY_TASK_ID / $N_TASKS"
echo "Sample per pair : $SAMPLE_ROWS"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

python3 ./collect_data_stats.py \
    --n_chunks "$N_TASKS" \
    --chunk_id "$SLURM_ARRAY_TASK_ID" \
    --sample_rows "$SAMPLE_ROWS" \
    --skip_existing

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job ended at    : $(date)"
echo "Duration        : $(date -u -d @${duration} +%T)"
