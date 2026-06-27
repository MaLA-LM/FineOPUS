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
#   MAX_ROWS             test mode: cap total scored rows per task (0 = no cap)
#   CHECKPOINT_EVERY_ROWS save checkpoint part every N rows (default: 1000000)
#   CLASS_COMBOS         optional resource-class combos to score (e.g. "0-0,0-1")
#   PAIR_COMBOS_JSON     optional path to the precomputed pair->combo JSON
#   API_KEY_ENV          env var name for the API key (default: AZURE_API_KEY)
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
echo "Max rows / task : ${MAX_ROWS:-0}"
echo "Checkpoint every: ${CHECKPOINT_EVERY_ROWS:-1000000} rows"
echo "Class combos    : ${CLASS_COMBOS:-<all>}"
echo "Pair combos json: ${PAIR_COMBOS_JSON:-<default>}"
echo "API key env     : ${API_KEY_ENV:-AZURE_API_KEY}"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

EXTRA_ARGS=()
if [[ -n "${CLASS_COMBOS:-}" ]]; then
    EXTRA_ARGS+=(--class_combos "${CLASS_COMBOS}")
fi
if [[ -n "${PAIR_COMBOS_JSON:-}" ]]; then
    EXTRA_ARGS+=(--pair_combos_json "${PAIR_COMBOS_JSON}")
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
    --checkpoint_every_rows "${CHECKPOINT_EVERY_ROWS:-1000000}" \
    --api_key_env   "${API_KEY_ENV:-AZURE_API_KEY}" \
    --skip_existing \
    "${EXTRA_ARGS[@]}"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job ended at    : $(date)"
echo "Duration        : $(date -u -d @${duration} +%T)"
