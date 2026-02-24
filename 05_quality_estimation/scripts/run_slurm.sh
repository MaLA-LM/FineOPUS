#!/usr/bin/env bash
#SBATCH --job-name=flores200_score
#SBATCH --account=project_2008161
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=/projappl/project_2008161/members/ibrahiam/encoder_flores_200/logs/%x-%j.out
#SBATCH --error=/projappl/project_2008161/members/ibrahiam/encoder_flores_200/logs/%x-%j.err

set -euo pipefail

export HF_HOME="/scratch/project_2008161/$USER/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HUGGINGFACE_ASSETS_CACHE="$HF_ASSETS_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================="

WORKDIR="${WORKDIR:-/projappl/project_2008161/members/ibrahiam/encoder_flores_200}"
ROOT="${ROOT:-${DATA_DIR:-/scratch/project_2008161/downstream_benchmarks/flores200}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_DIR:-/scratch/project_2008161/QE_flores200_scores}}"
DATASET="${DATASET:-flores200}"
VENV_BASE="${VENV_BASE:-${WORKDIR}/envs}"
MANIFEST="${MANIFEST:-${WORKDIR}/flores200_directions.tsv}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GPUS="${GPUS:-1}"
MAX_DIRECTIONS_PER_PART="${MAX_DIRECTIONS_PER_PART:-25}"
TARGET_PART_BYTES="${TARGET_PART_BYTES:-67108864}"
MODEL="${MODEL:-wmt22-cometkiwi-da}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
API_BASE="${API_BASE:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_RETRIES="${MAX_RETRIES:-5}"
MODEL_REPO="${MODEL_REPO:-}"
MODEL_NAME="${MODEL_NAME:-}"

print_usage() {
    cat <<EOF
Usage: $0 [args]

Common args:
  --dataset <dataset_id>
  --root <dir_root>
  --output-base <dir_output_base>
  --batch-size <int>
  --gpus <int>
  --manifest <path_tsv>
  --max-directions-per-part <int>
  --max-seconds-per-part <int>
  --target-part-bytes <int>

LLM args:
  --model <model_name>
  --api-base <url>
  --api-key <key>
  --temperature <float>
  --max-tokens <int>
  --max-retries <int>
EOF
}

resolve_model() {
    local model_lower
    model_lower="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"
    case "$model_lower" in
        comet22|unbabel/wmt22-cometkiwi-da)
            BACKEND="comet"
            MODULE="src.score_comet"
            MODEL_CANONICAL="Unbabel/wmt22-cometkiwi-da"
            VENV_PATH="${COMET_VENV:-${VENV_BASE}/comet_venv}"
            ;;
        comet23|unbabel/wmt23-cometkiwi-da-xl)
            BACKEND="comet"
            MODULE="src.score_comet"
            MODEL_CANONICAL="Unbabel/wmt23-cometkiwi-da-xl"
            VENV_PATH="${COMET_VENV:-${VENV_BASE}/comet_venv}"
            ;;
        xcomet|unbabel/xcomet-xl)
            BACKEND="comet"
            MODULE="src.score_comet"
            MODEL_CANONICAL="Unbabel/XCOMET-XL"
            VENV_PATH="${COMET_VENV:-${VENV_BASE}/comet_venv}"
            ;;
        metricx24|google/metricx-24-hybrid-xl-v2p6)
            BACKEND="metricx"
            MODULE="src.score_metricx"
            MODEL_CANONICAL="google/metricx-24-hybrid-xl-v2p6"
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
            ;;
        qwen|qwen/qwen3-14b)
            BACKEND="llm"
            MODULE="src.score_llm"
            MODEL_CANONICAL="Qwen/Qwen3-14B"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        m-prometheus-7b|unbabel/m-prometheus-7b)
            BACKEND="llm"
            MODULE="src.score_llm"
            MODEL_CANONICAL="Unbabel/M-Prometheus-7B"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        remedy|shaomutan/remedy-9b-22)
            BACKEND="remedy"
            MODULE="src.score_remedy"
            MODEL_CANONICAL="ShaomuTan/ReMedy-9B-22"
            REMEDY_INST="${REMEDY_VENV:-${VENV_BASE}/remedy_venv}"
            ;;
        bicleaner|auto)
            BACKEND="bicleaner"
            MODULE="src.score_bicleaner"
            MODEL_CANONICAL="auto"
            BICLEANER_INST="${BICLEANER_VENV:-${VENV_BASE}/bicleaner_venv}"
            ;;
        en-xx|bitextor/bicleaner-ai-full-en-xx)
            BACKEND="bicleaner"
            MODULE="src.score_bicleaner"
            MODEL_CANONICAL="en-xx"
            BICLEANER_INST="${BICLEANER_VENV:-${VENV_BASE}/bicleaner_venv}"
            ;;
        es-xx|bitextor/bicleaner-ai-full-es-xx)
            BACKEND="bicleaner"
            MODULE="src.score_bicleaner"
            MODEL_CANONICAL="es-xx"
            BICLEANER_INST="${BICLEANER_VENV:-${VENV_BASE}/bicleaner_venv}"
            ;;
        de-xx|bitextor/bicleaner-ai-full-de-xx)
            BACKEND="bicleaner"
            MODULE="src.score_bicleaner"
            MODEL_CANONICAL="de-xx"
            BICLEANER_INST="${BICLEANER_VENV:-${VENV_BASE}/bicleaner_venv}"
            ;;
        *)
            echo "Unsupported model: $MODEL" >&2
            print_usage >&2
            exit 1
            ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset)
            DATASET="${2:-}"
            shift 2
            ;;
        --root)
            ROOT="${2:-}"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="${2:-}"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="${2:-}"
            shift 2
            ;;
        --gpus)
            GPUS="${2:-}"
            shift 2
            ;;
        --manifest)
            MANIFEST="${2:-}"
            shift 2
            ;;
        --max-directions-per-part)
            MAX_DIRECTIONS_PER_PART="${2:-}"
            shift 2
            ;;
        --target-part-bytes)
            TARGET_PART_BYTES="${2:-}"
            shift 2
            ;;
        --model)
            MODEL="${2:-}"
            shift 2
            ;;
        --api-base)
            API_BASE="${2:-}"
            shift 2
            ;;
        --api-key)
            API_KEY="${2:-}"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="${2:-}"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="${2:-}"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="${2:-}"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

resolve_model

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
elif [ "$BACKEND" = "remedy" ]; then
    export PYTHONNOUSERSITE=1
    module --force purge
    module load tykky

    REPO="/projappl/project_2008161/members/$USER/Remedy"
    export PATH="$REMEDY_INST/bin:$PATH"

    unset PYTHONPATH
    export PYTHONPATH="$REPO"
else
    module load pytorch
    source "${VENV_PATH}/bin/activate"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
fi

mkdir -p "${WORKDIR}/logs"
cd "$WORKDIR"

echo "Backend: $BACKEND"
echo "Model: $MODEL"
echo "Manifest: $MANIFEST"
echo "Output base: $OUTPUT_BASE"


COMMON_ARGS=(
    --dataset "$DATASET"
    --root "$ROOT"
    --output-base "$OUTPUT_BASE"
    --batch-size "$BATCH_SIZE"
    --gpus "$GPUS"
    --manifest "$MANIFEST"
    --max-directions-per-part "$MAX_DIRECTIONS_PER_PART"
    --target-part-bytes "$TARGET_PART_BYTES"
)

if [ "$BACKEND" = "llm" ]; then
    MODEL_NAME="${MODEL_NAME:-$MODEL_CANONICAL}"
    MODEL_REPO="${MODEL_REPO:-$MODEL_CANONICAL}"
    MODEL_TAG="$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | sed 's|[^a-z0-9._-]|-|g')"
    VLLM_LOG="${WORKDIR}/logs/vllm_${MODEL_TAG}_${SLURM_JOB_ID:-manual}.log"

    cleanup() {
        if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
            kill "$SERVER_PID"
            wait "$SERVER_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT

    vllm serve "$MODEL_REPO" \
        --served-model-name "$MODEL_NAME" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
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

    python3 -m "$MODULE" \
        "${COMMON_ARGS[@]}" \
        --model "$MODEL_NAME" \
        --api-base "$API_BASE" \
        --api-key "$API_KEY" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --max-retries "$MAX_RETRIES"
else
    python3 -m "$MODULE" \
        "${COMMON_ARGS[@]}" \
        --model "$MODEL_CANONICAL"
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
