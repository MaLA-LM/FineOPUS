#!/usr/bin/env bash
#SBATCH --job-name=flores200_score
#SBATCH --account=project_462001050
#SBATCH --partition=small-g
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --output=/projappl/project_462001050/members/ibrahiam/05_quality_estimation/logs/%x-%j.out
#SBATCH --error=/projappl/project_462001050/members/ibrahiam/05_quality_estimation/logs/%x-%j.err

set -euo pipefail

export HF_HOME="/scratch/project_462001050/$USER/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HUGGINGFACE_ASSETS_CACHE="$HF_ASSETS_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

echo "=================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "Start time: $(date)"
echo "Master Port: $MASTER_PORT"
echo "=================================="

WORKDIR="${WORKDIR:-/projappl/project_462001050/members/ibrahiam/05_quality_estimation}"
VENV_BASE="${VENV_BASE:-/scratch/project_462001050/ibrahiam/envs}"
ROOT="${ROOT:-${DATA_DIR:-/scratch/project_462001050/downstream_benchmarks/flores200}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_DIR:-/scratch/project_462001050/QE_flores200_scores}}"
DATASET="${DATASET:-flores200}"
MANIFEST="${MANIFEST:-${WORKDIR}/flores200_directions.tsv}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GPUS="${GPUS:-1}"
MAX_DIRECTIONS_PER_PART="${MAX_DIRECTIONS_PER_PART:-25}"
TARGET_PART_BYTES="${TARGET_PART_BYTES:-67108864}"
MODEL="${MODEL:-wmt22-cometkiwi-da}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-}}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-}}"

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
  --shard-id <int>
  --num-shards <int>
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
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
            ;;
        comet23|unbabel/wmt23-cometkiwi-da-xl)
            BACKEND="comet"
            MODULE="src.score_comet"
            MODEL_CANONICAL="Unbabel/wmt23-cometkiwi-da-xl"
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
            ;;
        xcomet|unbabel/xcomet-xl)
            BACKEND="comet"
            MODULE="src.score_comet"
            MODEL_CANONICAL="Unbabel/XCOMET-XL"
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
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
            MODEL_CANONICAL="/scratch/project_462001050/ibrahiam/envs/images/Models/patched_models/ShaomuTan_ReMedy-9B-22"
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
        --shard-id)
            SHARD_ID="${2:-}"
            shift 2
            ;;
        --num-shards)
            NUM_SHARDS="${2:-}"
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

# Ensure VENV_PATH is always defined (bicleaner/remedy don't set it)
VENV_PATH="${VENV_PATH:-}"

if [ "$BACKEND" = "bicleaner" ]; then
    export PYTHONNOUSERSITE=1
    module --force purge
    module load LUMI
    module load partition/G
    module load rocm
    module load lumi-container-wrapper

    # --- ROCm/TF settings for MI250X (gfx90a) ---
    # Do NOT set HSA_OVERRIDE_GFX_VERSION: let ROCm auto-detect gfx90a.
    export TF_ROCM_FUSION_ENABLE=0
    export SINGULARITY_BIND="/opt/rocm"
    # ----------------------------------

    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

    export PATH="$BICLEANER_INST/bin:$PATH"
elif [ "$BACKEND" = "remedy" ]; then
    export PYTHONNOUSERSITE=1
    module --force purge
    REPO="/scratch/project_462001050/$USER/envs/Remedy"
    unset PYTHONPATH
    export PYTHONPATH="$REPO"
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export HF_HOME="/pfs/lustrep3/scratch/project_462001050/$USER/hf"
    export TRANSFORMERS_CACHE="$HF_HOME"
    export HF_DATASETS_CACHE="$HF_HOME"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME"
    mkdir -p "$HF_HOME"

    # vLLM / ROCm runtime knobs (the combo that worked)
    export VLLM_TARGET_DEVICE=rocm
    export VLLM_USE_V1=1
    export VLLM_USE_TRITON_FLASH_ATTN=0

    # Disable torch.compile / Inductor (fixes triton_key import crash)
    export TORCHDYNAMO_DISABLE=1
    export TORCHINDUCTOR_DISABLE=1
    export SIF=/scratch/project_462001050/ibrahiam/envs/images/vllm092_rocm.sif
else
    module purge
    module use /appl/local/laifs/modules
    module load lumi-aif-singularity-bindings
    export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif

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

if [ -n "${SHARD_ID}" ]; then
    COMMON_ARGS+=(--shard-id "$SHARD_ID")
fi
if [ -n "${NUM_SHARDS}" ]; then
    COMMON_ARGS+=(--num-shards "$NUM_SHARDS")
fi

set +e
if [ "$BACKEND" = "bicleaner" ]; then
    srun python3 -m "$MODULE" \
        "${COMMON_ARGS[@]}" \
        --model "$MODEL_CANONICAL"

elif [ "$BACKEND" = "remedy" ]; then
    singularity exec --rocm -B /scratch -B /pfs -B /projappl "$SIF" env \
        PYTHONPATH="$PYTHONPATH" \
        HF_HOME="$HF_HOME" \
        TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
        HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
        HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
        TRANSFORMERS_OFFLINE=1 \
        HF_HUB_OFFLINE=1 \
        VLLM_TARGET_DEVICE=rocm \
        VLLM_USE_V1=1 \
        VLLM_USE_TRITON_FLASH_ATTN=0 \
        TORCHDYNAMO_DISABLE=1 \
        TORCHINDUCTOR_DISABLE=1 \
        python3 -m "$MODULE" "${COMMON_ARGS[@]}" --model "$MODEL_CANONICAL" --cache-dir "$HF_HOME"

elif [ "$BACKEND" = "llm" ]; then
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

    singularity run "$SIF" bash -c "
        source ${VENV_PATH}/bin/activate
        vllm serve $MODEL_REPO \
            -O0 \
            --served-model-name $MODEL_NAME \
            --host $VLLM_HOST \
            --port $VLLM_PORT \
            --dtype $VLLM_DTYPE \
            --gpu-memory-utilization $VLLM_GPU_UTIL \
            $VLLM_EXTRA_ARGS
    " > "$VLLM_LOG" 2>&1 &
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
    
    LLM_ARGS=(
        --model "$MODEL_NAME"
        --api-base "$API_BASE"
        --temperature "$TEMPERATURE"
        --max-tokens "$MAX_TOKENS"
        --max-retries "$MAX_RETRIES"
    )
    if [ -n "${API_KEY:-}" ]; then
        LLM_ARGS+=(--api-key "$API_KEY")
    fi

    singularity run "$SIF" bash -c "
        source ${VENV_PATH}/bin/activate
        python3 -m $MODULE \
            ${COMMON_ARGS[*]} \
            ${LLM_ARGS[*]}
    "
else
    # comet / metricx backends
    singularity run "$SIF" bash -c "
        source ${VENV_PATH}/bin/activate
        python3 -m $MODULE \
            ${COMMON_ARGS[*]} \
            --model $MODEL_CANONICAL
    "
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
