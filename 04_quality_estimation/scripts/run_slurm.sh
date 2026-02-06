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

MODE="${MODE:-single}"
SRC_LANG="${SRC_LANG:-arb_Arab}"
TGT_LANG="${TGT_LANG:-eng_Latn}"
SPLIT="${SPLIT:-devtest}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPUS="${GPUS:-1}"
MAX_ROWS="${MAX_ROWS-}"
WORKER_MAX_FILES="${WORKER_MAX_FILES:-200}"
MODEL=""

# vllm/gemma defaults - can be overridden by env vars or CLI args
MODEL_REPO="${MODEL_REPO:-google/gemma-3-12b-it}"
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
    echo "Usage: $0 [--mode single|worker] [--manifest PATH] [--src-lang SRC] [--tgt-lang TGT] [--split SPLIT] [--model MODEL]"
    echo ""
    echo "Key env vars:"
    echo "  WORKDIR      (default: ${WORKDIR})"
    echo "  DATA_DIR     (default: ${DATA_DIR})"
    echo "  OUTPUT_DIR   (default: ${OUTPUT_DIR})"
    echo "  OUTPUT_BASE  (default: ${OUTPUT_BASE})"
    echo "  DATASET      (default: ${DATASET})"
    echo "  VENV_BASE    (default: ${VENV_BASE})"
    echo "  METRIC_VENV  (default: ${METRIC_VENV:-${VENV_BASE}/metric_venv})"
    echo "  COMET_VENV   (default: ${COMET_VENV:-${VENV_BASE}/comet_venv})"
    echo "  GEMMA_VENV   (default: ${GEMMA_VENV:-${VENV_BASE}/gemma_venv})"
    echo "  BICLEANER_INST (default: ${BICLEANER_INST:-${VENV_BASE}/bicleaner_venv})"
    echo "  MODEL_REPO   (gemma, default: ${MODEL_REPO})"
    echo "  MODEL_NAME   (gemma, default: ${MODEL_NAME:-<model>})"
    echo "  VLLM_HOST    (gemma, default: ${VLLM_HOST})"
    echo "  VLLM_PORT    (gemma, default: ${VLLM_PORT})"
    echo "  VLLM_DTYPE   (gemma, default: ${VLLM_DTYPE})"
    echo "  VLLM_GPU_UTIL (gemma, default: ${VLLM_GPU_UTIL})"
    echo "  VLLM_EXTRA_ARGS (gemma, default: ${VLLM_EXTRA_ARGS})"
    echo "  API_BASE     (gemma, default: ${API_BASE})"
    echo "  MAX_RETRIES  (gemma, default: ${MAX_RETRIES})"
    echo "  TEMPERATURE  (gemma, default: ${TEMPERATURE})"
    echo "  MAX_TOKENS   (gemma, default: ${MAX_TOKENS})"
    echo "  CONTINUE_ON_ERROR (gemma, default: ${CONTINUE_ON_ERROR})"
    echo "Examples:"
    echo "  $0 --mode single --src-lang arb_Arab --tgt-lang eng_Latn --split devtest --model wmt22-cometkiwi-da"
    echo "  $0 --mode worker --manifest /path/to/directions.tsv --model wmt22-cometkiwi-da --worker-max-files 0"
    echo "  $0 --mode single --src-lang arb_Arab --tgt-lang eng_Latn --split devtest --model gemma-3-12b-it"
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

is_gemma_model() {
    case "$1" in
        *gemma*)
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
            MODE="${2:-}"
            shift 2
            ;;
        --mode=*)
            MODE="${1#*=}"
            shift 1
            ;;
        --worker)
            MODE="worker"
            shift 1
            ;;
        --single)
            MODE="single"
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
        --src-lang)
            SRC_LANG="${2:-}"
            shift 2
            ;;
        --tgt-lang)
            TGT_LANG="${2:-}"
            shift 2
            ;;
        --split)
            SPLIT="${2:-}"
            shift 2
            ;;
        --model)
            MODEL="${2:-}"
            shift 2
            ;;
        --model=*)
            MODEL="${1#*=}"
            shift 1
            ;;
        --worker-max-files)
            WORKER_MAX_FILES="${2:-}"
            shift 2
            ;;
        --worker-max-files=*)
            WORKER_MAX_FILES="${1#*=}"
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

if is_bicleaner_model "$MODEL"; then
    BACKEND="bicleaner"
    MODULE="src.scorers.score_bicleaner"
    BICLEANER_INST="${BICLEANER_INST:-${VENV_BASE}/bicleaner_venv}"
elif is_gemma_model "$MODEL"; then
    BACKEND="gemma"
    MODULE="src.scorers.score_gemma"
    MODEL_NAME="${MODEL_NAME:-$MODEL}"
    VENV_PATH="${GEMMA_VENV:-${VENV_BASE}/gemma_venv}"
elif is_metricx_model "$MODEL"; then
    BACKEND="metricx"
    MODULE="src.scorers.score_metricx"
    VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
else
    BACKEND="comet"
    MODULE="src.scorers.score_comet"
    VENV_PATH="${COMET_VENV:-${VENV_BASE}/comet_venv}"
fi

# ensure required env vars are set and paths exist
if [ "$BACKEND" = "bicleaner" ]; then
    if [ ! -d "$BICLEANER_INST" ]; then
        echo "ERROR: bicleaner env not found: $BICLEANER_INST"
        exit 1
    fi
else
    if [ ! -d "$VENV_PATH" ]; then
        echo "ERROR: venv not found: $VENV_PATH"
        exit 1
    fi
fi

# worker mode requires a manifest file
if [ "$MODE" = "worker" ]; then
    if [ -z "${MANIFEST:-}" ]; then
        echo "ERROR: manifest not set (use --manifest or MANIFEST)."
        exit 1
    fi
    if [ ! -f "$MANIFEST" ]; then
        echo "ERROR: manifest not found: $MANIFEST"
        exit 1
    fi
fi


# set up environment and paths based on backend
if [ "$BACKEND" = "bicleaner" ]; then
    module --force purge
    module load tykky
    module load gcc/10.4.0
    module load cuda/12.6.1

    export SING_FLAGS="--nv"
    export APPTAINER_FLAGS="--nv"
    export SINGULARITY_FLAGS="--nv"
    export SINGULARITYENV_LD_LIBRARY_PATH="/appl/spack/v020/install-tree/gcc-10.4.0/cuda-12.6.1-tauwpv/lib64:${LD_LIBRARY_PATH:-}"

    export PATH="$BICLEANER_INST/bin:$PATH"
else
# all other backends use a Python venv on top of pytorch module
    module load pytorch
    source "${VENV_PATH}/bin/activate"
fi

# ensure logs directory exists
mkdir -p "${WORKDIR}/logs"
cd "$WORKDIR"

# Optional HF_TOKEN for model download
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    echo "HuggingFace token found - using authentication"
else
    echo "WARNING: No HF_TOKEN set - may encounter rate limits"
fi

MODEL_TAG=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's|[^a-z0-9._-]|-|g')
DATASET_TAG="${DATASET}"
MODEL_DIR="${OUTPUT_DIR}/${MODEL_TAG}_${DATASET_TAG}"

echo "Backend: $BACKEND"
echo "Model: $MODEL"
if [ "$BACKEND" = "bicleaner" ]; then
    echo "Bicleaner env: $BICLEANER_INST"
else
    echo "Venv: $VENV_PATH"
fi
echo "Mode: $MODE"

MAX_ROWS_ARG=()
if [ -n "${MAX_ROWS:-}" ]; then
    MAX_ROWS_ARG=(--max-rows "$MAX_ROWS")
fi

# score_bicleaner ignores --batch-size/--gpus; kept for CLI consistency.
COMMON_ARGS=(--model "$MODEL" --batch-size "$BATCH_SIZE" --gpus "$GPUS")

if [ "$BACKEND" = "gemma" ]; then
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
fi

if [ "$MODE" = "worker" ]; then
    if [ "$BACKEND" = "gemma" ]; then
        CONTINUE_FLAG=""
        if [ "$CONTINUE_ON_ERROR" = "1" ]; then
            CONTINUE_FLAG="--continue-on-error"
        fi
        python3 -m "$MODULE" \
            --dataset "$DATASET" \
            --root "$DATA_DIR" \
            --manifest "$MANIFEST" \
            --worker \
            --worker-max-files "$WORKER_MAX_FILES" \
            --resume \
            --output-base "$OUTPUT_BASE" \
            --model "$MODEL_NAME" \
            --api-base "$API_BASE" \
            --api-key "${OPENAI_API_KEY:-}" \
            --temperature "$TEMPERATURE" \
            --max-tokens "$MAX_TOKENS" \
            --max-retries "$MAX_RETRIES" \
            "${MAX_ROWS_ARG[@]}" \
            $CONTINUE_FLAG
    else
        python3 -m "$MODULE" \
            --dataset "$DATASET" \
            --root "$DATA_DIR" \
            --manifest "$MANIFEST" \
            --worker \
            --worker-max-files "$WORKER_MAX_FILES" \
            --resume \
            --output-base "$OUTPUT_BASE" \
            "${COMMON_ARGS[@]}" \
            "${MAX_ROWS_ARG[@]}"
    fi
else
    OUTPUT_PATH="${MODEL_DIR}/${SRC_LANG}-${TGT_LANG}-${SPLIT}.parquet"
    mkdir -p "$MODEL_DIR"
    if [ "$BACKEND" = "gemma" ]; then
        CONTINUE_FLAG=""
        if [ "$CONTINUE_ON_ERROR" = "1" ]; then
            CONTINUE_FLAG="--continue-on-error"
        fi
        python3 -m "$MODULE" \
            --dataset "$DATASET" \
            --src-lang "$SRC_LANG" \
            --tgt-lang "$TGT_LANG" \
            --split "$SPLIT" \
            --root "$DATA_DIR" \
            --output "$OUTPUT_PATH" \
            --resume \
            --model "$MODEL_NAME" \
            --api-base "$API_BASE" \
            --api-key "${OPENAI_API_KEY:-}" \
            --temperature "$TEMPERATURE" \
            --max-tokens "$MAX_TOKENS" \
            --max-retries "$MAX_RETRIES" \
            "${MAX_ROWS_ARG[@]}" \
            $CONTINUE_FLAG
    else
        python3 -m "$MODULE" \
            --dataset "$DATASET" \
            --src-lang "$SRC_LANG" \
            --tgt-lang "$TGT_LANG" \
            --split "$SPLIT" \
            --root "$DATA_DIR" \
            --output "$OUTPUT_PATH" \
            --resume \
            "${COMMON_ARGS[@]}" \
            "${MAX_ROWS_ARG[@]}"
    fi
fi

EXIT_CODE=$?

echo "=================================="
echo "End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "$MODEL scoring completed successfully"
    if [ "$MODE" = "single" ]; then
        echo "Output: $OUTPUT_PATH"
    fi
else
    echo "$MODEL scoring failed with exit code $EXIT_CODE"
fi
echo "=================================="

exit $EXIT_CODE
