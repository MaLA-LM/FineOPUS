#!/bin/bash
# ---------------------------------------------------------------------------
# submit_llm_judge.sh
#
# Submit a SLURM array job that runs llm_judge.py against a sharded parallel
# corpus on disk. Each task gets a 1/N_TASKS slice of the language pairs and
# its own fraction of the global TPM/RPM budget.
#
# Usage:
#   bash submit_llm_judge.sh [options]
#
# Options:
#   --tasks N              Number of array tasks (default: 4)
#   --dataset-dir DIR      Source parquet root (one subdir per src-tgt)
#   --out-dir DIR          Scored output root
#   --stats-output FILE    Per-task stats CSV path
#   --batch-size N         Segments per API call (default: 10)
#   --concurrency N        Max in-flight requests per task (default: 32)
#   --tpm-total N          Global tokens-per-minute budget (default: 900000)
#   --rpm-total N          Global requests-per-minute budget (default: 900)
#   --deployment NAME      Azure deployment / model (default: DeepSeek-V4-Flash)
#   --endpoint URL         Azure base URL (default: fineopus-step6 v1 endpoint)
#   --keep-dims            Also write the 7 per-dimension columns
#   --max-rows N           Test mode: cap total scored rows PER TASK (0 = no cap)
#   --dry-run              Print the sbatch command without submitting
# ---------------------------------------------------------------------------
set -euo pipefail

N_TASKS=4
DATASET_DIR="/scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage3"
OUT_DIR="/scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage3-LLMScored"
STATS_OUTPUT="./stats/llm_judge_stats.csv"
BATCH_SIZE=10
CONCURRENCY=32
TPM_TOTAL=900000
RPM_TOTAL=900
DEPLOYMENT="DeepSeek-V4-Flash"
ENDPOINT="https://fineopus-step6.services.ai.azure.com/openai/v1/"
KEEP_DIMS=0
MAX_ROWS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)         N_TASKS="$2"; shift 2 ;;
        --dataset-dir)   DATASET_DIR="$2"; shift 2 ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        --stats-output)  STATS_OUTPUT="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --concurrency)   CONCURRENCY="$2"; shift 2 ;;
        --tpm-total)     TPM_TOTAL="$2"; shift 2 ;;
        --rpm-total)     RPM_TOTAL="$2"; shift 2 ;;
        --deployment)    DEPLOYMENT="$2"; shift 2 ;;
        --endpoint)      ENDPOINT="$2"; shift 2 ;;
        --keep-dims)     KEEP_DIMS=1; shift ;;
        --max-rows)      MAX_ROWS="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Split global budgets evenly across array tasks. Use integer division and
# guarantee at least 1.
TPM_LIMIT=$(( TPM_TOTAL / N_TASKS ))
RPM_LIMIT=$(( RPM_TOTAL / N_TASKS ))
[[ $TPM_LIMIT -lt 1 ]] && TPM_LIMIT=1
[[ $RPM_LIMIT -lt 1 ]] && RPM_LIMIT=1

LOG_DIR="../../logs/llm_judge"
mkdir -p "$LOG_DIR" "./stats"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  Submitting llm_judge.py as SLURM array"
echo "============================================================"
echo "  Tasks           : $N_TASKS"
echo "  Dataset dir     : $DATASET_DIR"
echo "  Out dir         : $OUT_DIR"
echo "  Stats output    : $STATS_OUTPUT"
echo "  Batch size      : $BATCH_SIZE"
echo "  Concurrency/task: $CONCURRENCY"
echo "  TPM (total/task): $TPM_TOTAL / $TPM_LIMIT"
echo "  RPM (total/task): $RPM_TOTAL / $RPM_LIMIT"
echo "  Deployment      : $DEPLOYMENT"
echo "  Endpoint        : $ENDPOINT"
echo "  Keep dims       : $KEEP_DIMS"
echo "  Max rows / task : $MAX_ROWS"
echo "  Log dir         : $LOG_DIR"
echo "============================================================"

last_idx=$((N_TASKS - 1))

CMD="sbatch \
    --job-name=llm_judge \
    --output=$LOG_DIR/%x_%A_%a.out \
    --error=$LOG_DIR/%x_%A_%a.err \
    --partition=small \
    --array=0-${last_idx} \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=1-00:00:00 \
    --account=project_462001249 \
    --export=ALL,N_TASKS=${N_TASKS},DATASET_DIR=${DATASET_DIR},OUT_DIR=${OUT_DIR},STATS_OUTPUT=${STATS_OUTPUT},BATCH_SIZE=${BATCH_SIZE},CONCURRENCY=${CONCURRENCY},TPM_LIMIT=${TPM_LIMIT},RPM_LIMIT=${RPM_LIMIT},DEPLOYMENT=${DEPLOYMENT},ENDPOINT=${ENDPOINT},KEEP_DIMS=${KEEP_DIMS},MAX_ROWS=${MAX_ROWS} \
    ./run_llm_judge.sh"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] $CMD"
else
    eval $CMD
fi
