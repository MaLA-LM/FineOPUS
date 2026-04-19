#!/usr/bin/env bash
#
# Submit one SLURM array of OPUS queue workers for a single model.
# Each array task claims shards from the shared SQLite queue database.
# The array index is only used to make worker_ids unique; partitioning
# of work is driven entirely by the DB, not SLURM_ARRAY_TASK_ID.
#
set -euo pipefail

MODEL=""
ARRAY_SPEC=""
CONCURRENCY=""
TIME_LIMIT="24:00:00"
DB=""
OUTPUT_BASE=""
OPUS_ROOT=""
PARTITION="small-g"
ACCOUNT="project_462001050"
EXTRA_SBATCH=""
PLATFORM="lumi"
BATCH_SIZE="${BATCH_SIZE:-}"
GPUS="${GPUS:-}"
PROMPT_MODE="${PROMPT_MODE:-}"
TEMPERATURE="${TEMPERATURE:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
MAX_RETRIES="${MAX_RETRIES:-}"
VLLM_DTYPE="${VLLM_DTYPE:-}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
RESPONSE_FORMAT="${RESPONSE_FORMAT:-}"
STRUCTURED_OUTPUTS_BACKEND="${STRUCTURED_OUTPUTS_BACKEND:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
PART_WRITER="${PART_WRITER:-}"
PART_MAX_BYTES="${PART_MAX_BYTES:-}"
PART_MAX_SHARDS="${PART_MAX_SHARDS:-}"

print_usage() {
    cat <<'EOF'
Usage: submit_array.sh [args]

Required:
  --model <key>          Model key (e.g. metricx24, qwen3-4b-instruct-2507, bicleaner).
  --array <a-b>          SLURM array spec (e.g. 0-63).
  --db <path>            Path to shared SQLite queue database.
  --output-base <dir>    Shared-storage dir for shard JSONLs / part files.

Optional:
  --concurrency <int>    SLURM %N cap (joined to --array as a-b%N).
  --time <HH:MM:SS>      Walltime (default 24:00:00).
  --opus-root <dir>      Override OPUS root (default: dataset adapter default).
  --account <acct>       SLURM --account (default: project_462001050).
  --partition <part>     SLURM --partition (default: small-g).
  --platform <lumi>      Launcher platform (default: lumi).
  --batch-size <int>     Per-forward batch size (default in worker: 8).
  --gpus <int>           GPU count passed to the Python worker (default: 1).
  --prompt-mode <mode>   LLM prompt mode: detailed|simple|batch.
  --temperature <float>  LLM sampling temperature.
  --max-tokens <int>     LLM max generation tokens.
  --max-retries <int>    Retries for invalid LLM outputs.
  --dtype <dtype>        LLM dtype (bfloat16, float16, auto).
  --gpu-memory-utilization <float>  vLLM GPU memory fraction.
  --max-num-batched-tokens <int>    vLLM scheduler token budget.
  --max-num-seqs <int>   vLLM scheduler sequence cap.
  --max-model-len <int>  Optional vLLM context cap (default: 8192 for LLM backends).
  --response-format <fmt>  none|json_object|json_schema.
  --structured-outputs-backend <name>  outlines|xgrammar.
  --enforce-eager        Disable CUDA graphs / torch.compile in vLLM.
  --part-writer          Append multiple shards into worker-owned part files.
  --part-max-bytes <int> Rotate part files before the next shard would exceed this size.
  --part-max-shards <int>  Rotate part files after this many shards per part.
  --extra <str>          Extra args appended verbatim to sbatch.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model)         MODEL="${2:-}"; shift 2 ;;
        --array)         ARRAY_SPEC="${2:-}"; shift 2 ;;
        --concurrency)   CONCURRENCY="${2:-}"; shift 2 ;;
        --time)          TIME_LIMIT="${2:-}"; shift 2 ;;
        --db)            DB="${2:-}"; shift 2 ;;
        --output-base)   OUTPUT_BASE="${2:-}"; shift 2 ;;
        --opus-root)     OPUS_ROOT="${2:-}"; shift 2 ;;
        --account)       ACCOUNT="${2:-}"; shift 2 ;;
        --partition)     PARTITION="${2:-}"; shift 2 ;;
        --platform)      PLATFORM="${2:-}"; shift 2 ;;
        --batch-size)    BATCH_SIZE="${2:-}"; shift 2 ;;
        --gpus)          GPUS="${2:-}"; shift 2 ;;
        --prompt-mode)   PROMPT_MODE="${2:-}"; shift 2 ;;
        --temperature)   TEMPERATURE="${2:-}"; shift 2 ;;
        --max-tokens)    MAX_TOKENS="${2:-}"; shift 2 ;;
        --max-retries)   MAX_RETRIES="${2:-}"; shift 2 ;;
        --dtype)         VLLM_DTYPE="${2:-}"; shift 2 ;;
        --gpu-memory-utilization) VLLM_GPU_UTIL="${2:-}"; shift 2 ;;
        --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="${2:-}"; shift 2 ;;
        --max-num-seqs)  MAX_NUM_SEQS="${2:-}"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="${2:-}"; shift 2 ;;
        --response-format) RESPONSE_FORMAT="${2:-}"; shift 2 ;;
        --structured-outputs-backend) STRUCTURED_OUTPUTS_BACKEND="${2:-}"; shift 2 ;;
        --enforce-eager) ENFORCE_EAGER=1; shift ;;
        --part-writer)   PART_WRITER=1; shift ;;
        --part-max-bytes) PART_MAX_BYTES="${2:-}"; shift 2 ;;
        --part-max-shards) PART_MAX_SHARDS="${2:-}"; shift 2 ;;
        --extra)         EXTRA_SBATCH="${2:-}"; shift 2 ;;
        -h|--help)       print_usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; print_usage >&2; exit 1 ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$ARRAY_SPEC" ] || [ -z "$DB" ] || [ -z "$OUTPUT_BASE" ]; then
    echo "ERROR: --model, --array, --db, and --output-base are required." >&2
    print_usage >&2
    exit 1
fi

PLATFORM="$(printf '%s' "$PLATFORM" | tr '[:upper:]' '[:lower:]')"
if [ "$PLATFORM" != "lumi" ]; then
    echo "ERROR: OPUS launchers only support LUMI (got --platform=$PLATFORM)." >&2
    exit 1
fi

if [ -n "$CONCURRENCY" ]; then
    ARRAY_ARG="${ARRAY_SPEC}%${CONCURRENCY}"
else
    ARRAY_ARG="$ARRAY_SPEC"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_worker.sh"

if [ ! -f "$RUNNER" ]; then
    echo "ERROR: runner script not found: $RUNNER" >&2
    exit 1
fi
if [ ! -r "$RUNNER" ]; then
    echo "ERROR: runner script is not readable: $RUNNER" >&2
    exit 1
fi

if [ -f "$DB" ] && command -v sqlite3 >/dev/null 2>&1; then
    SQL_MODEL="${MODEL//\'/\'\'}"
    PENDING_COUNT="$(sqlite3 "$DB" "SELECT COUNT(*) FROM jobs WHERE model = '$SQL_MODEL' AND status = 'pending';" 2>/dev/null || true)"
    TOTAL_COUNT="$(sqlite3 "$DB" "SELECT COUNT(*) FROM jobs WHERE model = '$SQL_MODEL';" 2>/dev/null || true)"
    if [ -n "$TOTAL_COUNT" ] && [ "$TOTAL_COUNT" = "0" ]; then
        PENDING_MODELS="$(sqlite3 "$DB" "SELECT model || ':' || COUNT(*) FROM jobs WHERE status = 'pending' GROUP BY model ORDER BY COUNT(*) DESC, model ASC LIMIT 10;" 2>/dev/null || true)"
        if [ -n "$PENDING_MODELS" ]; then
            PENDING_MODELS="$(printf '%s' "$PENDING_MODELS" | tr '\n' ',' | sed 's/,$//')"
        else
            PENDING_MODELS="<none>"
        fi
        echo "WARNING: no queue rows found for model='$MODEL' in $DB" >&2
        echo "WARNING: pending models in DB: $PENDING_MODELS" >&2
    elif [ -n "$PENDING_COUNT" ] && [ "$PENDING_COUNT" = "0" ]; then
        echo "WARNING: model='$MODEL' exists in $DB but has no pending rows." >&2
    fi
fi

SBATCH_ARGS=(
    --job-name="opus_${MODEL}"
    --array="$ARRAY_ARG"
    --time="$TIME_LIMIT"
    --export=ALL,MODEL="$MODEL",DB="$DB",OUTPUT_BASE="$OUTPUT_BASE",OPUS_ROOT="$OPUS_ROOT",PLATFORM="$PLATFORM",BATCH_SIZE="$BATCH_SIZE",GPUS="$GPUS",PROMPT_MODE="$PROMPT_MODE",TEMPERATURE="$TEMPERATURE",MAX_TOKENS="$MAX_TOKENS",MAX_RETRIES="$MAX_RETRIES",VLLM_DTYPE="$VLLM_DTYPE",VLLM_GPU_UTIL="$VLLM_GPU_UTIL",MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS",MAX_NUM_SEQS="$MAX_NUM_SEQS",MAX_MODEL_LEN="$MAX_MODEL_LEN",RESPONSE_FORMAT="$RESPONSE_FORMAT",STRUCTURED_OUTPUTS_BACKEND="$STRUCTURED_OUTPUTS_BACKEND",ENFORCE_EAGER="$ENFORCE_EAGER",PART_WRITER="$PART_WRITER",PART_MAX_BYTES="$PART_MAX_BYTES",PART_MAX_SHARDS="$PART_MAX_SHARDS"
)

if [ -n "$ACCOUNT" ]; then
    SBATCH_ARGS+=(--account="$ACCOUNT")
fi
if [ -n "$PARTITION" ]; then
    SBATCH_ARGS+=(--partition="$PARTITION")
fi
if [ -n "$EXTRA_SBATCH" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_SBATCH)
    SBATCH_ARGS+=("${EXTRA_ARR[@]}")
fi

echo "Submitting: sbatch ${SBATCH_ARGS[*]} $RUNNER"
sbatch "${SBATCH_ARGS[@]}" "$RUNNER"
