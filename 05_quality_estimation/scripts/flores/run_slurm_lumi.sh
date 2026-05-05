#!/usr/bin/env bash
#SBATCH --job-name=flores200_score
#SBATCH --account=project_462001050
#SBATCH --partition=small-g
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --output=/projappl/project_462001050/members/ibrahiam/05_quality_estimation/logs/%x-%j.out
#SBATCH --error=/projappl/project_462001050/members/ibrahiam/05_quality_estimation/logs/%x-%j.err

set -euo pipefail
# This launcher uses the flores_array execution strategy: hash(direction) % num_shards -> SLURM array task id.

SCRATCH_CACHE="/scratch/project_462001050/ibrahiam"
export HF_HOME="$SCRATCH_CACHE/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HUGGINGFACE_ASSETS_CACHE="$HF_ASSETS_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH_CACHE/.cache/torch"
export TRITON_CACHE_DIR="$SCRATCH_CACHE/.cache/triton"
export XDG_CACHE_HOME="$SCRATCH_CACHE/.cache"
export MASTER_ADDR=127.0.0.1
PORT_JOB_SEED="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}"
PORT_TASK_SEED="${SLURM_ARRAY_TASK_ID:-0}"
export MASTER_PORT=$(( 20000 + ((PORT_JOB_SEED + PORT_TASK_SEED) % 20000) ))

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
TARGET_PART_BYTES="${TARGET_PART_BYTES:-67108864}" # 64 MiB
MODEL="${MODEL:-wmt22-cometkiwi-da}"
EXECUTION="${EXECUTION:-flores_array}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-}}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-}}"

# LLM offline settings (vLLM loads model directly, no server)
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_RETRIES="${MAX_RETRIES:-5}"
PROMPT_MODE="${PROMPT_MODE:-detailed}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"  # empty=auto (model's full context), e.g. 8192 for QE
RESPONSE_FORMAT="${RESPONSE_FORMAT:-}"  # empty=auto, "none"/"json_object"/"json_schema"
# Structured-output backend: "outlines" or "xgrammar" (default).
# xgrammar crashes on ROCm/MI250X, so set to "outlines" when using
# response_format != none. This is an ENGINE-level setting (vLLM ≥0.12).
STRUCTURED_OUTPUTS_BACKEND="${STRUCTURED_OUTPUTS_BACKEND:-}"  # empty=default(xgrammar)
ENFORCE_EAGER="${ENFORCE_EAGER:-}"  # empty=off (match old vLLM-server default); set "1" to enable
MODEL_REPO="${MODEL_REPO:-}"

print_usage() {
    cat <<EOF
Usage: $0 [args]

Common args:
  --dataset <dataset_id>
  --root <dir_root>
  --output-base <dir_output_base>
  --batch-size <int>
  --gpus <int>
  --execution <strategy>
  --manifest <path_tsv>
  --shard-id <int>
  --num-shards <int>
  --max-directions-per-part <int>
  --max-seconds-per-part <int>
  --target-part-bytes <int>

LLM args:
  --model <model_name>
  --prompt-mode <detailed|simple|batch>
  --temperature <float>
  --max-tokens <int>
  --max-retries <int>
  --max-num-batched-tokens <int>  (vLLM scheduler, old server: 8192)
  --max-num-seqs <int>            (vLLM scheduler, old server: 32)
  --max-model-len <int>           (cap context length, e.g. 8192 for QE)
  --response-format <none|json_object|json_schema>

Env vars:
  ENFORCE_EAGER=1                       Disable CUDA graphs (pass --enforce-eager)
  STRUCTURED_OUTPUTS_BACKEND=outlines   Use outlines instead of xgrammar
    (xgrammar crashes on MI250X). Auto-set when response_format != none.
EOF
}

resolve_model() {
    local model_lower
    model_lower="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"
    case "$model_lower" in
        comet22|unbabel/wmt22-cometkiwi-da)
            BACKEND="comet"
            MODULE="src.backends.comet"
            MODEL_CANONICAL="Unbabel/wmt22-cometkiwi-da"
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
            ;;
        comet23|unbabel/wmt23-cometkiwi-da-xl)
            BACKEND="comet"
            MODULE="src.backends.comet"
            MODEL_CANONICAL="Unbabel/wmt23-cometkiwi-da-xl"
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
            ;;
        xcomet|unbabel/xcomet-xl)
            BACKEND="comet"
            MODULE="src.backends.comet"
            MODEL_CANONICAL="Unbabel/XCOMET-XL"
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
            ;;
        metricx24|google/metricx-24-hybrid-xl-v2p6)
            BACKEND="metricx"
            MODULE="src.backends.metricx"
            MODEL_CANONICAL="google/metricx-24-hybrid-xl-v2p6"
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
            ;;
        qwen3-14b|qwen/qwen3-14b)
            BACKEND="llm"
            MODULE="src.backends.llm"
            MODEL_CANONICAL="Qwen/Qwen3-14B"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        qwen3-8b|qwen/qwen3-8b)
            BACKEND="llm"
            MODULE="src.backends.llm"
            MODEL_CANONICAL="Qwen/Qwen3-8B"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        qwen3-4b-awq|qwen/qwen3-4b-awq)
            BACKEND="llm"
            MODULE="src.backends.llm"
            MODEL_CANONICAL="Qwen/Qwen3-4B-AWQ"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            # AWQ kernels on ROCm only support float16
            VLLM_DTYPE="float16"
            ;;
        qwen3-4b-fp8|qwen/qwen3-4b-fp8)
            BACKEND="llm"
            MODULE="src.backends.llm"
            MODEL_CANONICAL="Qwen/Qwen3-4B-FP8"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        qwen3-4b|qwen3-4b-instruct-2507|qwen/qwen3-4b-instruct-2507)
            BACKEND="llm"
            MODULE="src.backends.llm"
            MODEL_CANONICAL="Qwen/Qwen3-4B-Instruct-2507"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        qwen3-4b-instruct-2507-fp8|qwen/qwen3-4b-instruct-2507-fp8)
            BACKEND="llm"
            MODULE="src.backends.llm"
            MODEL_CANONICAL="Qwen/Qwen3-4B-Instruct-2507-FP8"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        qwen3-1.7b|qwen/qwen3-1.7b)
            BACKEND="llm"
            MODULE="src.backends.llm"
            MODEL_CANONICAL="Qwen/Qwen3-1.7B"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        qwen3-0.6b|qwen/qwen3-0.6b)
            BACKEND="llm"
            MODULE="src.backends.llm"
            MODEL_CANONICAL="Qwen/Qwen3-0.6B"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        m-prometheus-7b|unbabel/m-prometheus-7b)
            BACKEND="llm"
            MODULE="src.backends.llm"
            MODEL_CANONICAL="Unbabel/M-Prometheus-7B"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        m-prometheus-3b|unbabel/m-prometheus-3b)
            BACKEND="llm"
            MODULE="src.backends.llm"
            MODEL_CANONICAL="Unbabel/M-Prometheus-3B"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        remedy|shaomutan/remedy-9b-22)
            BACKEND="remedy"
            MODULE="src.backends.remedy"
            MODEL_CANONICAL="/scratch/project_462001050/ibrahiam/envs/images/Models/patched_models/ShaomuTan_ReMedy-9B-22"
            ;;
        bicleaner|auto)
            BACKEND="bicleaner"
            MODULE="src.backends.bicleaner"
            MODEL_CANONICAL="auto"
            BICLEANER_INST="${BICLEANER_VENV:-${VENV_BASE}/bicleaner_venv}"
            ;;
        en-xx|bitextor/bicleaner-ai-full-en-xx)
            BACKEND="bicleaner"
            MODULE="src.backends.bicleaner"
            MODEL_CANONICAL="en-xx"
            BICLEANER_INST="${BICLEANER_VENV:-${VENV_BASE}/bicleaner_venv}"
            ;;
        es-xx|bitextor/bicleaner-ai-full-es-xx)
            BACKEND="bicleaner"
            MODULE="src.backends.bicleaner"
            MODEL_CANONICAL="es-xx"
            BICLEANER_INST="${BICLEANER_VENV:-${VENV_BASE}/bicleaner_venv}"
            ;;
        de-xx|bitextor/bicleaner-ai-full-de-xx)
            BACKEND="bicleaner"
            MODULE="src.backends.bicleaner"
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
        --prompt-mode)
            PROMPT_MODE="${2:-}"
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
        --max-num-batched-tokens)
            MAX_NUM_BATCHED_TOKENS="${2:-}"
            shift 2
            ;;
        --max-num-seqs)
            MAX_NUM_SEQS="${2:-}"
            shift 2
            ;;
        --response-format)
            RESPONSE_FORMAT="${2:-}"
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
    --execution "$EXECUTION"
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

    run_bicleaner() {
        srun python3 -m "$MODULE" \
            "${COMMON_ARGS[@]}" \
            --model "$MODEL_CANONICAL"
    }

    EXIT_CODE=1
    for attempt in 1 2 3; do
        echo "Attempt $attempt"
        if run_bicleaner; then
            EXIT_CODE=0
            break
        else
            EXIT_CODE=$?
            echo "Attempt $attempt failed with exit code $EXIT_CODE"
        fi
        sleep $((30*attempt))
    done

elif [ "$BACKEND" = "remedy" ]; then

    run_remedy() {
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
    }

    EXIT_CODE=1
    for attempt in 1 2 3; do
        echo "Attempt $attempt"
        if run_remedy; then
            EXIT_CODE=0
            break
        else
            EXIT_CODE=$?
            echo "Attempt $attempt failed with exit code $EXIT_CODE"
        fi
        sleep $((30*attempt))
    done

elif [ "$BACKEND" = "llm" ]; then
    MODEL_REPO="${MODEL_REPO:-$MODEL_CANONICAL}"

    LLM_ARGS=(
        --model "$MODEL_REPO"
        --prompt-mode "$PROMPT_MODE"
        --temperature "$TEMPERATURE"
        --max-tokens "$MAX_TOKENS"
        --max-retries "$MAX_RETRIES"
        --dtype "$VLLM_DTYPE"
        --gpu-memory-utilization "$VLLM_GPU_UTIL"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --max-num-seqs "$MAX_NUM_SEQS"
    )
    # Append optional args only when explicitly set.
    if [ -n "$ENFORCE_EAGER" ]; then
        LLM_ARGS+=(--enforce-eager)
    fi
    if [ -n "$MAX_MODEL_LEN" ]; then
        LLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
    fi
    if [ -n "$RESPONSE_FORMAT" ]; then
        LLM_ARGS+=(--response-format "$RESPONSE_FORMAT")
    fi

    # Build extra singularity --env flags for vLLM engine config.
    #
    # Structured output backend is an ENGINE-LEVEL setting (vLLM ≥0.12).
    # On ROCm/MI250X, xgrammar (default) crashes — use "outlines" instead.
    # Set STRUCTURED_OUTPUTS_BACKEND=outlines when response_format != none.
    EXTRA_VLLM_ENVS=()
    if [ -n "$STRUCTURED_OUTPUTS_BACKEND" ]; then
        EXTRA_VLLM_ENVS+=(--env "VLLM_STRUCTURED_OUTPUTS_BACKEND=${STRUCTURED_OUTPUTS_BACKEND}")
    elif [ -n "$RESPONSE_FORMAT" ] && [ "$RESPONSE_FORMAT" != "none" ]; then
        # Auto-select outlines on ROCm when structured output is requested.
        echo "Auto-setting VLLM_STRUCTURED_OUTPUTS_BACKEND=outlines (xgrammar crashes on ROCm)"
        EXTRA_VLLM_ENVS+=(--env "VLLM_STRUCTURED_OUTPUTS_BACKEND=outlines")
    fi

    run_llm_scoring() {
        singularity run \
            --env "TORCH_HOME=${TORCH_HOME}" \
            --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
            --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
            --env "HF_HOME=${HF_HOME}" \
            --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
            --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
            --env "HF_HUB_OFFLINE=1" \
            --env "TRANSFORMERS_OFFLINE=1" \
            "${EXTRA_VLLM_ENVS[@]}" \
            "$SIF" bash -c "
            source ${VENV_PATH}/bin/activate
            python3 -m $MODULE \
                ${COMMON_ARGS[*]} \
                ${LLM_ARGS[*]}
        "
    }

    EXIT_CODE=1
    for attempt in 1 2 3; do
        echo "Scoring attempt $attempt"
        if run_llm_scoring; then
            EXIT_CODE=0
            break
        else
            EXIT_CODE=$?
            echo "Scoring attempt $attempt failed with exit code $EXIT_CODE"
        fi
        sleep $((30*attempt))
    done

else
    # comet / metricx backends
    run_metric() {
        singularity run \
            --env "HF_HOME=${HF_HOME}" \
            --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
            --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
            --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
            --env "HF_ASSETS_CACHE=${HF_ASSETS_CACHE}" \
            --env "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}" \
            --env "HUGGINGFACE_ASSETS_CACHE=${HUGGINGFACE_ASSETS_CACHE}" \
            --env "HF_HUB_OFFLINE=1" \
            --env "TRANSFORMERS_OFFLINE=1" \
            "$SIF" bash -c "
            source ${VENV_PATH}/bin/activate
            python3 -m $MODULE \
                ${COMMON_ARGS[*]} \
                --model $MODEL_CANONICAL
        "
    }

    EXIT_CODE=1
    for attempt in 1 2 3; do
        echo "Attempt $attempt"
        if run_metric; then
            EXIT_CODE=0
            break
        else
            EXIT_CODE=$?
            echo "Attempt $attempt failed with exit code $EXIT_CODE"
        fi
        sleep $((30*attempt))
    done
fi

echo "=================================="
echo "End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "$MODEL scoring completed successfully"
else
    echo "$MODEL scoring failed with exit code $EXIT_CODE"
fi
echo "=================================="

exit $EXIT_CODE
