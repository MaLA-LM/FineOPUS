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
#   --tasks N              Number of array tasks (default: 1)
#   --dataset-dir DIR      Source parquet root (one subdir per src-tgt)
#   --out-dir DIR          Scored output root
#   --stats-output FILE    Per-task stats CSV path
#   --batch-size N         Segments per API call (default: 10)
#   --concurrency N        Max in-flight requests per task (default: 32)
#   --tpm-total N          Global tokens-per-minute budget (overrides registry for API_KEY_ENV)
#   --rpm-total N          Global requests-per-minute budget (overrides registry for API_KEY_ENV)
#   --deployment NAME      Azure deployment (overrides registry for API_KEY_ENV)
#   --endpoint URL         Azure base URL (overrides registry for API_KEY_ENV)
#                          If omitted, ENDPOINT/DEPLOYMENT are taken from
#                          azure_api_key_registry.sh for --api-key-env.
#   --max-rows N           Test mode: cap total scored rows PER TASK (0 = no cap)
#   --checkpoint-every N   Save checkpoint part every N rows (default: 1000000)
#   --class-combos LIST    Only score these directional resource-class combos,
#                          e.g. "0-0,0-1,5-5" (src_class-tgt_class). Empty = all.
#   --pair-combos FILE     Precomputed pair->combo JSON
#                          (default: fineopus_pair_class_combinations.json)
#   --api-key-env NAME     Env var name for the API key (default: AZURE_API_KEY)
#   --dry-run              Print the sbatch command without submitting
# ---------------------------------------------------------------------------
set -euo pipefail

N_TASKS=1
DATASET_DIR="/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage4"
OUT_DIR="/scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage4-LLMScored"
STATS_OUTPUT="./stats/llm_judge_stats.csv"
BATCH_SIZE=10
CONCURRENCY=32
TPM_TOTAL=""
RPM_TOTAL=""
DEPLOYMENT=""
ENDPOINT=""
MAX_ROWS=0
CHECKPOINT_EVERY_ROWS=1000000
CLASS_COMBOS=""
PAIR_COMBOS_JSON="./fineopus_pair_class_combinations.json"
API_KEY_ENV="AZURE_API_KEY_1"
ENDPOINT_EXPLICIT=0
DEPLOYMENT_EXPLICIT=0
TPM_TOTAL_EXPLICIT=0
RPM_TOTAL_EXPLICIT=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)         N_TASKS="$2"; shift 2 ;;
        --dataset-dir)   DATASET_DIR="$2"; shift 2 ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        --stats-output)  STATS_OUTPUT="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --concurrency)   CONCURRENCY="$2"; shift 2 ;;
        --tpm-total)     TPM_TOTAL="$2"; TPM_TOTAL_EXPLICIT=1; shift 2 ;;
        --rpm-total)     RPM_TOTAL="$2"; RPM_TOTAL_EXPLICIT=1; shift 2 ;;
        --deployment)    DEPLOYMENT="$2"; DEPLOYMENT_EXPLICIT=1; shift 2 ;;
        --endpoint)      ENDPOINT="$2"; ENDPOINT_EXPLICIT=1; shift 2 ;;
        --max-rows)      MAX_ROWS="$2"; shift 2 ;;
        --checkpoint-every) CHECKPOINT_EVERY_ROWS="$2"; shift 2 ;;
        --class-combos)  CLASS_COMBOS="$2"; shift 2 ;;
        --pair-combos)   PAIR_COMBOS_JSON="$2"; shift 2 ;;
        --api-key-env)   API_KEY_ENV="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=azure_api_key_registry.sh
source "${SCRIPT_DIR}/azure_api_key_registry.sh"

if resolve_azure_from_api_key_env "$API_KEY_ENV"; then
    if [[ $ENDPOINT_EXPLICIT -eq 0 ]]; then
        ENDPOINT="$RESOLVED_ENDPOINT"
    fi
    if [[ $DEPLOYMENT_EXPLICIT -eq 0 ]]; then
        DEPLOYMENT="$RESOLVED_DEPLOYMENT"
    fi
    if [[ $TPM_TOTAL_EXPLICIT -eq 0 ]]; then
        TPM_TOTAL="$RESOLVED_TPM"
    fi
    if [[ $RPM_TOTAL_EXPLICIT -eq 0 ]]; then
        RPM_TOTAL="$RESOLVED_RPM"
    fi
else
    if [[ $ENDPOINT_EXPLICIT -eq 0 || $DEPLOYMENT_EXPLICIT -eq 0 || $TPM_TOTAL_EXPLICIT -eq 0 || $RPM_TOTAL_EXPLICIT -eq 0 ]]; then
        echo "ERROR: Unknown --api-key-env '${API_KEY_ENV}' and missing --endpoint/--deployment/--tpm-total/--rpm-total." >&2
        echo "       Known keys: AZURE_API_KEY, AZURE_API_KEY_1 .. AZURE_API_KEY_11" >&2
        exit 1
    fi
fi

if [[ -z "$ENDPOINT" || -z "$DEPLOYMENT" ]]; then
    echo "ERROR: ENDPOINT and DEPLOYMENT must be set (via registry or --endpoint/--deployment)." >&2
    exit 1
fi

if [[ -z "$TPM_TOTAL" ]]; then
    echo "ERROR: TPM_TOTAL must be set (via registry or --tpm-total)." >&2
    exit 1
fi

if [[ -z "$RPM_TOTAL" ]]; then
    echo "ERROR: RPM_TOTAL must be set (via registry or --rpm-total)." >&2
    exit 1
fi

# Split global budgets evenly across array tasks. Use integer division and
# guarantee at least 1.
TPM_LIMIT=$(( TPM_TOTAL / N_TASKS ))
RPM_LIMIT=$(( RPM_TOTAL / N_TASKS ))
[[ $TPM_LIMIT -lt 1 ]] && TPM_LIMIT=1
[[ $RPM_LIMIT -lt 1 ]] && RPM_LIMIT=1

LOG_DIR="../logs/fineopus-llm-judge"
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
echo "  Max rows / task : $MAX_ROWS"
echo "  Checkpoint every: $CHECKPOINT_EVERY_ROWS rows"
echo "  Class combos    : ${CLASS_COMBOS:-<all>}"
echo "  Pair combos json: $PAIR_COMBOS_JSON"
echo "  API key env     : $API_KEY_ENV"
echo "  Log dir         : $LOG_DIR"
echo "============================================================"

last_idx=$((N_TASKS - 1))

CMD="sbatch \
    --job-name=llm_judge_${CLASS_COMBOS}_${API_KEY_ENV} \
    --output=$LOG_DIR/%x_%A_%a.out \
    --error=$LOG_DIR/%x_%A_%a.err \
    --partition=small \
    --array=0-${last_idx} \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=3-00:00:00 \
    --account=project_462001087 \
    --export=ALL,N_TASKS=${N_TASKS},DATASET_DIR=${DATASET_DIR},OUT_DIR=${OUT_DIR},STATS_OUTPUT=${STATS_OUTPUT},BATCH_SIZE=${BATCH_SIZE},CONCURRENCY=${CONCURRENCY},TPM_LIMIT=${TPM_LIMIT},RPM_LIMIT=${RPM_LIMIT},DEPLOYMENT=${DEPLOYMENT},ENDPOINT=${ENDPOINT},MAX_ROWS=${MAX_ROWS},CHECKPOINT_EVERY_ROWS=${CHECKPOINT_EVERY_ROWS},CLASS_COMBOS=${CLASS_COMBOS},PAIR_COMBOS_JSON=${PAIR_COMBOS_JSON},API_KEY_ENV=${API_KEY_ENV} \
    ./run_llm_judge.sh"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] $CMD"
else
    eval $CMD
fi
