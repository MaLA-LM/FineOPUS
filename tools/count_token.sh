#!/bin/bash
#
# Unified token-counting script.
#
# Usage:
#   ./count_token.sh [DATA_DIR] [OUTPUT_FILE]              Submit SLURM jobs
#   ./count_token.sh aggregate [DATA_DIR] [OUTPUT_FILE]    Aggregate results
#
# Environment variables (override defaults):
#   NUM_JOBS              Number of SLURM array jobs (default: 128)
#   TOKENIZER             Tokenizer name/path (default: Qwen/Qwen3.5-9B)
#   TOKENIZER_NAME        Column suffix in CSV (default: Qwen3_5)
#   TOKENIZER_BATCH_SIZE  Texts per tokenizer batch (default: 1024)
#   PARQUET_BATCH_SIZE    Parquet rows per batch (default: 10000)
#
# Examples:
#   ./count_token.sh /data/Stage5 stats/stage5.csv
#   NUM_JOBS=64 ./count_token.sh /data/Stage4 stats/stage4.csv
#   ./count_token.sh aggregate /data/Stage5 stats/stage5.csv
#
# When submitted via sbatch (SLURM_ARRAY_TASK_ID is set), the script
# automatically runs as a worker — no separate .slurm file needed.
#
# ──────────────────────── SBATCH directives ────────────────────────
# These are only parsed when the file is submitted via sbatch.
#SBATCH --job-name=count_token
#SBATCH --account=project_462001087
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=../logs/count_token/%x_%j_%a.out
#SBATCH --error=../logs/count_token/%x_%j_%a.err
# ───────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================= Parse arguments ========================
MODE="submit"
if [ "${1:-}" = "aggregate" ]; then
    MODE="aggregate"
    shift
fi
# Positional args override defaults / env vars.
DATA_DIR="${1:-${DATA_DIR:-/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage5}}"
OUTPUT_FILE="${2:-${OUTPUT_FILE:-/scratch/project_462000941/members/zihao/OPUS2410/tools/token_stats/FineOPUS-Filtered-Stage5_Qwen3_5.csv}}"

# ========================= Configuration ==========================
# These can still be overridden via environment variables.
NUM_JOBS="${NUM_JOBS:-128}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-9B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen3_5}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-1024}"
PARQUET_BATCH_SIZE="${PARQUET_BATCH_SIZE:-10000}"
# ==================================================================

# Derived paths.
DIRECTION_HEADER="lang_pair,src_lang,tgt_lang,n_lines,n_src_tokens_space,n_tgt_tokens_space,n_src_tokens_${TOKENIZER_NAME},n_tgt_tokens_${TOKENIZER_NAME}"
OUTPUT_BASENAME=$(basename "$OUTPUT_FILE" .csv)
TASK_ROOT_DIR="${TASK_ROOT_DIR:-$(dirname "$OUTPUT_FILE")/token_count_tasks_${OUTPUT_BASENAME}}"

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462001050/ibrahiam/envs/transformers-latest/bin/activate

# ===================================================================
# Worker mode — entered automatically when sbatch sets the variable.
# ===================================================================
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    TASK_ID=$SLURM_ARRAY_TASK_ID
    : "${TOKENIZER_BATCH_SIZE:=1024}"
    : "${PARQUET_BATCH_SIZE:=10000}"
    : "${TOKENIZER:="Qwen/Qwen3.5-9B"}"
    : "${TOKENIZER_NAME:="Qwen3_5"}"

    WORKER_OUTPUT_DIR="${WORKER_OUTPUT_DIR:?WORKER_OUTPUT_DIR is required}"
    MANIFEST_FILE="${MANIFEST_FILE:?MANIFEST_FILE is required}"
    WORKER_OUTPUT_FILE="$WORKER_OUTPUT_DIR/counts_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.csv"

    if [ ! -f "$MANIFEST_FILE" ]; then
        echo "Parquet manifest not found: $MANIFEST_FILE" >&2
        exit 1
    fi

    mkdir -p "$WORKER_OUTPUT_DIR"

    PARQUET_COUNT=$(awk -F'\t' -v task_id="$TASK_ID" \
        'NR > 1 && $1 == task_id { count++ } END { print count + 0 }' \
        "$MANIFEST_FILE")

    echo "--------------------------------------------------------"
    echo "Worker $TASK_ID | Job $SLURM_ARRAY_JOB_ID | $(hostname)"
    echo "Started: $(date)"
    echo "Tokenizer: $TOKENIZER ($TOKENIZER_NAME)"
    echo "Manifest: $MANIFEST_FILE"
    echo "Worker CSV: $WORKER_OUTPUT_FILE"
    echo "Parquet files assigned: $PARQUET_COUNT"
    awk -F'\t' -v task_id="$TASK_ID" \
        'NR > 1 && $1 == task_id { print "  " $3 }' "$MANIFEST_FILE"
    echo "--------------------------------------------------------"

    if [ "$PARQUET_COUNT" -eq 0 ]; then
        echo "No parquet files assigned. Nothing to do."
        exit 0
    fi

    srun --input=none python ./count_token.py count \
        --data_dir "$DATA_DIR" \
        --parquet_manifest_file "$MANIFEST_FILE" \
        --worker_id "$TASK_ID" \
        --output_file "$WORKER_OUTPUT_FILE" \
        --tokenizer "$TOKENIZER" \
        --tokenizer_name "$TOKENIZER_NAME" \
        --tokenizer_batch_size "$TOKENIZER_BATCH_SIZE" \
        --parquet_batch_size "$PARQUET_BATCH_SIZE"

    if [ ! -s "$WORKER_OUTPUT_FILE" ]; then
        echo "Worker completed but produced no output rows."
    else
        echo "Worker CSV written to $WORKER_OUTPUT_FILE"
    fi
    echo "Finished task $TASK_ID at $(date)"
    exit 0
fi


# ===================================================================
# Aggregate mode — ./count_token.sh aggregate [DATA_DIR] [OUTPUT_FILE]
# ===================================================================
if [ "$MODE" = "aggregate" ]; then
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    REPORT_DIR="${REPORT_DIR:-$TASK_ROOT_DIR/aggregation_$RUN_ID}"

    python ./count_token.py aggregate \
        --data_dir "$DATA_DIR" \
        --task_root_dir "$TASK_ROOT_DIR" \
        --output_file "$OUTPUT_FILE" \
        --report_dir "$REPORT_DIR" \
        --tokenizer_name "$TOKENIZER_NAME"

    echo "Aggregation reports written to $REPORT_DIR"
    exit 0
fi


# ===================================================================
# Submit mode — ./count_token.sh [DATA_DIR] [OUTPUT_FILE]  (default)
# ===================================================================

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_TASK_DIR="$TASK_ROOT_DIR/run_$RUN_ID"
WORKER_OUTPUT_DIR="$RUN_TASK_DIR/worker_outputs"
MANIFEST_FILE="$RUN_TASK_DIR/parquet_manifest.tsv"

mkdir -p "$RUN_TASK_DIR" "$WORKER_OUTPUT_DIR" "$(dirname "$OUTPUT_FILE")"

# Build size-balanced manifest via Python.
python ./count_token.py build-manifest \
    --data_dir "$DATA_DIR" \
    --main_csv "$OUTPUT_FILE" \
    --task_root_dir "$TASK_ROOT_DIR" \
    --num_jobs "$NUM_JOBS" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --output_dir "$RUN_TASK_DIR"

# Read actual number of jobs (may be less than NUM_JOBS if fewer files).
ACTUAL_JOBS=$(cat "$RUN_TASK_DIR/num_jobs.txt")

if [ "$ACTUAL_JOBS" -eq 0 ]; then
    echo "Nothing to submit."
else
    echo "Submitting $ACTUAL_JOBS array jobs (size-balanced across workers)."
    cd "$SCRIPT_DIR"
    sbatch --array=1-$ACTUAL_JOBS \
           --export=ALL,DATA_DIR="$DATA_DIR",MANIFEST_FILE="$MANIFEST_FILE",OUTPUT_FILE="$OUTPUT_FILE",TOKENIZER_BATCH_SIZE="$TOKENIZER_BATCH_SIZE",PARQUET_BATCH_SIZE="$PARQUET_BATCH_SIZE",WORKER_OUTPUT_DIR="$WORKER_OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR",TOKENIZER="$TOKENIZER",TOKENIZER_NAME="$TOKENIZER_NAME" \
           count_token.sh
fi

