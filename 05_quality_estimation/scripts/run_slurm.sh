#!/usr/bin/env bash
#SBATCH --job-name=flores200_score
#SBATCH --account=project_2008161
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/projappl/project_2008161/members/ibrahiam/encoder_flores_200/logs/%x-%j.out
#SBATCH --error=/projappl/project_2008161/members/ibrahiam/encoder_flores_200/logs/%x-%j.err

set -euo pipefail

# export HF cache
export HF_HOME="/scratch/project_2008161/$USER/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
# Deprecated aliases, just in case
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HUGGINGFACE_ASSETS_CACHE="$HF_ASSETS_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================="

WORKDIR="${WORKDIR:-/projappl/project_2008161/members/ibrahiam/encoder_flores_200}"
DATA_DIR="${DATA_DIR:-/scratch/project_2008161/downstream_benchmarks/flores200}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_2008161/QE_flores200_scores}"
OUTPUT_BASE="${OUTPUT_BASE:-$OUTPUT_DIR}"
DATASET="${DATASET:-flores200}"
VENV_BASE="${VENV_BASE:-${WORKDIR}/envs}"
MANIFEST="${MANIFEST:-${WORKDIR}/flores200_directions.tsv}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GPUS="${GPUS:-1}"
MAX_ROWS="${MAX_ROWS-}"
MODEL=""

NUM_SHARDS="${NUM_SHARDS:-}"
SHARD_ID="${SHARD_ID:-}"
MAX_DIRECTIONS_PER_PART="${MAX_DIRECTIONS_PER_PART:-25}"
MAX_SECONDS_PER_PART="${MAX_SECONDS_PER_PART:-600}"
TARGET_PART_BYTES="${TARGET_PART_BYTES:-67108864}"
RUN_ID="${RUN_ID:-}"

# vllm/llm defaults - can be overridden by env vars or CLI args
LLM_DEFAULT_MODEL="${LLM_DEFAULT_MODEL:-Qwen/Qwen3-14B}"
MODEL_REPO="${MODEL_REPO:-}"
MODEL_NAME="${MODEL_NAME:-}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
API_BASE="${API_BASE:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
MAX_RETRIES="${MAX_RETRIES:-5}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

print_usage() {
    echo "Usage: $0 --manifest PATH [--model MODEL] [shard flags]"
    echo ""
    echo "This script is worker-only."
    echo ""
    echo "Key env vars:"
    echo "  WORKDIR      (default: ${WORKDIR})"
    echo "  DATA_DIR     (default: ${DATA_DIR})"
    echo "  OUTPUT_BASE  (default: ${OUTPUT_BASE})"
    echo "  DATASET      (default: ${DATASET})"
    echo "  VENV_BASE    (default: ${VENV_BASE})"
    echo "  MANIFEST     (default: ${MANIFEST})"
    echo "  MODEL        (default: ${MODEL:-wmt22-cometkiwi-da})"
    echo "  NUM_SHARDS, SHARD_ID (optional fallback when Slurm array env is absent)"
    echo "  MAX_DIRECTIONS_PER_PART (default: ${MAX_DIRECTIONS_PER_PART})"
    echo "  MAX_SECONDS_PER_PART (default: ${MAX_SECONDS_PER_PART})"
    echo "  TARGET_PART_BYTES (default: ${TARGET_PART_BYTES})"
    echo "  RUN_ID (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 --manifest /path/to/directions.tsv --model xcomet"
    echo "  $0 --manifest /path/to/directions.tsv --model xcomet --num-shards 8 --shard-id 0"
}

is_metricx_model() {
    case "$1" in
        metricx24|metricx|metricx-24|metricx-24-hybrid-xl-v2p6|google/metricx-24-hybrid-xl-v2p6)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_bicleaner_model() {
    case "$1" in
        bicleaner|bicleaner-ai|en-xx|es-xx|de-xx|bitextor/bicleaner-ai-full-*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_remedy_model() {
    local model_lower
    model_lower="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    case "$model_lower" in
        remedy|remedy-9b-22|shaomutan/remedy-9b-22)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_llm_model() {
    local model_lower
    model_lower="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    case "$model_lower" in
        qwen/qwen3-14b|qwen3-14b|openai/qwen3-14b)
            printf '%s\n' "Qwen/Qwen3-14B"
            return 0
            ;;
        unbabel/m-prometheus-7b|m-prometheus-7b|mprometheus-7b|mprometheus|openai/unbabel/m-prometheus-7b|openai/m-prometheus-7b)
            printf '%s\n' "Unbabel/M-Prometheus-7B"
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_llm_model() {
    local model_lower
    if normalize_llm_model "$1" >/dev/null; then
        return 0
    fi
    model_lower="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    case "$model_lower" in
        llm|*qwen*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        --mode)
            MODE_ARG="${2:-}"
            shift 2
            if [ "$MODE_ARG" != "worker" ]; then
                echo "ERROR: only worker mode is supported (got: $MODE_ARG)"
                exit 1
            fi
            ;;
        --mode=*)
            MODE_ARG="${1#*=}"
            shift 1
            if [ "$MODE_ARG" != "worker" ]; then
                echo "ERROR: only worker mode is supported (got: $MODE_ARG)"
                exit 1
            fi
            ;;
        --worker)
            shift 1
            ;;
        --manifest)
            MANIFEST="${2:-}"
            shift 2
            ;;
        --manifest=*)
            MANIFEST="${1#*=}"
            shift 1
            ;;
        --model)
            MODEL="${2:-}"
            shift 2
            ;;
        --model=*)
            MODEL="${1#*=}"
            shift 1
            ;;
        --num-shards)
            NUM_SHARDS="${2:-}"
            shift 2
            ;;
        --num-shards=*)
            NUM_SHARDS="${1#*=}"
            shift 1
            ;;
        --shard-id)
            SHARD_ID="${2:-}"
            shift 2
            ;;
        --shard-id=*)
            SHARD_ID="${1#*=}"
            shift 1
            ;;
        --max-directions-per-part)
            MAX_DIRECTIONS_PER_PART="${2:-}"
            shift 2
            ;;
        --max-directions-per-part=*)
            MAX_DIRECTIONS_PER_PART="${1#*=}"
            shift 1
            ;;
        --max-seconds-per-part)
            MAX_SECONDS_PER_PART="${2:-}"
            shift 2
            ;;
        --max-seconds-per-part=*)
            MAX_SECONDS_PER_PART="${1#*=}"
            shift 1
            ;;
        --target-part-bytes)
            TARGET_PART_BYTES="${2:-}"
            shift 2
            ;;
        --target-part-bytes=*)
            TARGET_PART_BYTES="${1#*=}"
            shift 1
            ;;
        --run-id)
            RUN_ID="${2:-}"
            shift 2
            ;;
        --run-id=*)
            RUN_ID="${1#*=}"
            shift 1
            ;;
        --batch-size)
            BATCH_SIZE="${2:-}"
            shift 2
            ;;
        --batch-size=*)
            BATCH_SIZE="${1#*=}"
            shift 1
            ;;
        --gpus)
            GPUS="${2:-}"
            shift 2
            ;;
        --gpus=*)
            GPUS="${1#*=}"
            shift 1
            ;;
        --max-rows)
            MAX_ROWS="${2:-}"
            shift 2
            ;;
        --max-rows=*)
            MAX_ROWS="${1#*=}"
            shift 1
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            if [ -z "$MODEL" ]; then
                MODEL="$1"
                shift 1
            else
                echo "Unexpected argument: $1"
                print_usage
                exit 1
            fi
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    MODEL="wmt22-cometkiwi-da"
fi
if [ "$MODEL" = "llm" ]; then
    MODEL="$LLM_DEFAULT_MODEL"
fi

if is_bicleaner_model "$MODEL"; then
    BACKEND="bicleaner"
    MODULE="src.score_bicleaner"
    BICLEANER_INST="${BICLEANER_INST:-${VENV_BASE}/bicleaner_venv}"
elif is_remedy_model "$MODEL"; then
    BACKEND="remedy"
    MODULE="src.score_remedy"
    REMEDY_INST="${REMEDY_VENV:-${VENV_BASE}/remedy_venv}"
elif is_llm_model "$MODEL"; then
    BACKEND="llm"
    MODULE="src.score_llm"
    CANONICAL_LLM_MODEL="$(normalize_llm_model "$MODEL" || true)"
    if [ -z "$CANONICAL_LLM_MODEL" ]; then
        CANONICAL_LLM_MODEL="$MODEL"
    fi
    MODEL_NAME="${MODEL_NAME:-$CANONICAL_LLM_MODEL}"
    MODEL_REPO="${MODEL_REPO:-$CANONICAL_LLM_MODEL}"
    VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
elif is_metricx_model "$MODEL"; then
    BACKEND="metricx"
    MODULE="src.score_metricx"
    VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
else
    BACKEND="comet"
    MODULE="src.score_comet"
    VENV_PATH="${COMET_VENV:-${VENV_BASE}/comet_venv}"
fi

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"
    exit 1
fi

if [ -n "$NUM_SHARDS" ] && [ -z "$SHARD_ID" ]; then
    echo "ERROR: SHARD_ID/--shard-id is required when NUM_SHARDS/--num-shards is set"
    exit 1
fi
if [ -n "$SHARD_ID" ] && [ -z "$NUM_SHARDS" ]; then
    echo "ERROR: NUM_SHARDS/--num-shards is required when SHARD_ID/--shard-id is set"
    exit 1
fi

# set up environment and paths based on backend
if [ "$BACKEND" = "bicleaner" ]; then
    if [ ! -d "$BICLEANER_INST" ]; then
        echo "ERROR: bicleaner env not found: $BICLEANER_INST"
        exit 1
    fi

    module --force purge
    module load tykky
    module load gcc/10.4.0
    module load cuda/12.6.1

    export SING_FLAGS="--nv"
    export APPTAINER_FLAGS="--nv"
    export SINGULARITY_FLAGS="--nv"
    export SINGULARITYENV_LD_LIBRARY_PATH="/appl/spack/v020/install-tree/gcc-10.4.0/cuda-12.6.1-tauwpv/lib64:${LD_LIBRARY_PATH:-}"

    export PATH="$BICLEANER_INST/bin:$PATH"
elif [ "$BACKEND" = "remedy" ]; then
    if [ ! -d "$REMEDY_INST" ]; then
        echo "ERROR: remedy env not found: $REMEDY_INST"
        exit 1
    fi

    export PYTHONNOUSERSITE=1
    module --force purge
    module load tykky

    export PATH="$REMEDY_INST/bin:$PATH"
else
    if [ ! -d "$VENV_PATH" ]; then
        echo "ERROR: venv not found: $VENV_PATH"
        exit 1
    fi

    module load pytorch
    source "${VENV_PATH}/bin/activate"
fi

mkdir -p "${WORKDIR}/logs"
cd "$WORKDIR"

if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    echo "HuggingFace token found - using authentication"
else
    echo "WARNING: No HF_TOKEN set - may encounter rate limits"
fi

echo "Backend: $BACKEND"
echo "Model: $MODEL"
echo "Manifest: $MANIFEST"
echo "Output base: $OUTPUT_BASE"

MAX_ROWS_ARG=()
if [ -n "${MAX_ROWS:-}" ]; then
    MAX_ROWS_ARG=(--max-rows "$MAX_ROWS")
fi

SHARD_ARGS=()
if [ -n "${NUM_SHARDS:-}" ]; then
    SHARD_ARGS=(--num-shards "$NUM_SHARDS" --shard-id "$SHARD_ID")
fi

RUN_ID_ARG=()
if [ -n "${RUN_ID:-}" ]; then
    RUN_ID_ARG=(--run-id "$RUN_ID")
fi

COMMON_ARGS=(
    --dataset "$DATASET"
    --root "$DATA_DIR"
    --manifest "$MANIFEST"
    --worker
    --resume
    --output-base "$OUTPUT_BASE"
    --batch-size "$BATCH_SIZE"
    --gpus "$GPUS"
    --max-directions-per-part "$MAX_DIRECTIONS_PER_PART"
    --max-seconds-per-part "$MAX_SECONDS_PER_PART"
    --target-part-bytes "$TARGET_PART_BYTES"
)

if [ "$BACKEND" = "llm" ]; then
    MODEL_TAG=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | sed 's|[^a-z0-9._-]|-|g')
    VLLM_LOG="${WORKDIR}/logs/vllm_${MODEL_TAG}_${SLURM_JOB_ID}.log"

    cleanup() {
        if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Stopping vLLM server (PID $SERVER_PID)"
            kill "$SERVER_PID"
            wait "$SERVER_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT

    echo "Starting vLLM server for ${MODEL_NAME}..."
    vllm serve "$MODEL_REPO" \
        --served-model-name "$MODEL_NAME" \
        --host "$VLLM_HOST" --port "$VLLM_PORT" \
        --dtype "$VLLM_DTYPE" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" \
        $VLLM_EXTRA_ARGS \
        > "$VLLM_LOG" 2>&1 &

    SERVER_PID=$!

    READY=0
    for i in {1..120}; do
        if curl -sf "${API_BASE}/models" >/dev/null 2>&1; then
            READY=1
            echo "vLLM is up."
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: vLLM server exited early. Check $VLLM_LOG"
            exit 1
        fi
        sleep 20
    done

    if [ "$READY" -ne 1 ]; then
        echo "ERROR: vLLM did not become ready in time. Check $VLLM_LOG"
        exit 1
    fi

    CONTINUE_FLAG=()
    if [ "$CONTINUE_ON_ERROR" = "1" ]; then
        CONTINUE_FLAG=(--continue-on-error)
    fi

    python3 -m "$MODULE" \
        "${COMMON_ARGS[@]}" \
        "${SHARD_ARGS[@]}" \
        "${RUN_ID_ARG[@]}" \
        --model "$MODEL_NAME" \
        --api-base "$API_BASE" \
        --api-key "${OPENAI_API_KEY:-}" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --max-retries "$MAX_RETRIES" \
        "${MAX_ROWS_ARG[@]}" \
        "${CONTINUE_FLAG[@]}"
else
    python3 -m "$MODULE" \
        "${COMMON_ARGS[@]}" \
        "${SHARD_ARGS[@]}" \
        "${RUN_ID_ARG[@]}" \
        --model "$MODEL" \
        "${MAX_ROWS_ARG[@]}"
fi

EXIT_CODE=$?

echo "=================================="
echo "End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "$MODEL scoring completed successfully"
else
    echo "$MODEL scoring failed with exit code $EXIT_CODE"
fi
echo "=================================="

exit $EXIT_CODE
