#!/bin/bash
# SLURM worker script for llm_judge.py
#
# Required env vars (set via submit_llm_judge.sh):
#   N_TASKS              total number of array tasks
#   DATASET_DIR          source parquet root
#   OUT_DIR              scored output root
#   STATS_OUTPUT         per-task stats CSV
#   BATCH_SIZE           segments per API call
#   CONCURRENCY          max in-flight requests
#   TPM_LIMIT            tokens-per-minute budget (split across tasks)
#   RPM_LIMIT            requests-per-minute budget (split across tasks)
#   DEPLOYMENT           Azure deployment name
#   ENDPOINT             Azure base URL ending in /openai/v1/
#   KEEP_DIMS            "1" to also save per-dimension columns
#   MAX_ROWS             test mode: cap total scored rows per task (0 = no cap)
# ---------------------------------------------------------------------------
set -euo pipefail

start_time=$(date +%s)
echo "Job started at  : $(date)"
echo "Array task ID   : ${SLURM_ARRAY_TASK_ID} / ${N_TASKS}"
echo "Dataset dir     : ${DATASET_DIR}"
echo "Out dir         : ${OUT_DIR}"
echo "Stats output    : ${STATS_OUTPUT}"
echo "Batch size      : ${BATCH_SIZE}"
echo "Concurrency     : ${CONCURRENCY}"
echo "TPM limit       : ${TPM_LIMIT}"
echo "RPM limit       : ${RPM_LIMIT}"
echo "Deployment      : ${DEPLOYMENT}"
echo "Endpoint        : ${ENDPOINT}"
echo "Keep dims       : ${KEEP_DIMS:-0}"
echo "Max rows / task : ${MAX_ROWS:-0}"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

EXTRA_ARGS=()
if [[ "${KEEP_DIMS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--keep_dims)
fi

python3 ./llm_judge.py \
    --dataset_dir   "$DATASET_DIR" \
    --out_dir       "$OUT_DIR" \
    --stats_output  "$STATS_OUTPUT" \
    --batch_size    "$BATCH_SIZE" \
    --concurrency   "$CONCURRENCY" \
    --tpm_limit     "$TPM_LIMIT" \
    --rpm_limit     "$RPM_LIMIT" \
    --deployment    "$DEPLOYMENT" \
    --endpoint      "$ENDPOINT" \
    --n_chunks      "$N_TASKS" \
    --chunk_id      "$SLURM_ARRAY_TASK_ID" \
    --max_rows      "${MAX_ROWS:-0}" \
    --skip_existing \
    "${EXTRA_ARGS[@]}"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job ended at    : $(date)"
echo "Duration        : $(date -u -d @${duration} +%T)"
