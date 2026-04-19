#!/usr/bin/env bash
#
# Inner launcher for one OPUS queue worker. Expected to run under SLURM as
# part of an array; reads MODEL / DB / OUTPUT_BASE / OPUS_ROOT / PLATFORM
# from the environment (exported by submit_array.sh).
#
#SBATCH --job-name=opus_worker
#SBATCH --account=project_462001050
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --output=/projappl/project_462001050/members/ibrahiam/05_quality_estimation/logs/%x-%A_%a.out
#SBATCH --error=/projappl/project_462001050/members/ibrahiam/05_quality_estimation/logs/%x-%A_%a.err

set -euo pipefail

: "${MODEL:?MODEL is required}"
: "${DB:?DB is required}"
: "${OUTPUT_BASE:?OUTPUT_BASE is required}"

PLATFORM="$(printf '%s' "${PLATFORM:-lumi}" | tr '[:upper:]' '[:lower:]')"
OPUS_ROOT="${OPUS_ROOT:-}"
if [ "$PLATFORM" != "lumi" ]; then
    echo "ERROR: OPUS workers only support PLATFORM=lumi (got '$PLATFORM')." >&2
    exit 1
fi

SCRATCH_CACHE="${SCRATCH_CACHE:-/scratch/project_462001050/ibrahiam}"
export HF_HOME="$SCRATCH_CACHE/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HUGGINGFACE_ASSETS_CACHE="$HF_ASSETS_CACHE"
export TORCH_HOME="$SCRATCH_CACHE/.cache/torch"
export TRITON_CACHE_DIR="$SCRATCH_CACHE/.cache/triton"
export XDG_CACHE_HOME="$SCRATCH_CACHE/.cache"
export MASTER_ADDR=127.0.0.1
PORT_JOB_SEED="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}"
PORT_TASK_SEED="${SLURM_ARRAY_TASK_ID:-0}"
export MASTER_PORT=$(( 20000 + ((PORT_JOB_SEED + PORT_TASK_SEED) % 20000) ))

WORKDIR="${WORKDIR:-/projappl/project_462001050/members/ibrahiam/05_quality_estimation}"
VENV_BASE="${VENV_BASE:-/scratch/project_462001050/ibrahiam/envs}"
DEFAULT_OPUS_ROOT="/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2"
OPUS_ROOT="${OPUS_ROOT:-$DEFAULT_OPUS_ROOT}"

# FLORES-aligned runtime knobs.
BATCH_SIZE="${BATCH_SIZE:-8}"
GPUS="${GPUS:-1}"
VLLM_DTYPE="${VLLM_DTYPE:-}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_RETRIES="${MAX_RETRIES:-5}"
PROMPT_MODE="${PROMPT_MODE:-detailed}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
RESPONSE_FORMAT="${RESPONSE_FORMAT:-}"
STRUCTURED_OUTPUTS_BACKEND="${STRUCTURED_OUTPUTS_BACKEND:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
PART_WRITER="${PART_WRITER:-}"
PART_MAX_BYTES="${PART_MAX_BYTES:-}"
PART_MAX_SHARDS="${PART_MAX_SHARDS:-}"

mkdir -p "${WORKDIR}/logs"
cd "$WORKDIR"

echo "=================================="
echo "OPUS worker: model=$MODEL platform=$PLATFORM"
echo "DB=$DB  OUTPUT_BASE=$OUTPUT_BASE  OPUS_ROOT=$OPUS_ROOT"
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Array task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Start time: $(date)"
echo "Master Port: $MASTER_PORT"
echo "Batch size: $BATCH_SIZE  GPUs: $GPUS"
echo "Part writer: ${PART_WRITER:-0}  Part max bytes: ${PART_MAX_BYTES:-<worker-default>}  Part max shards: ${PART_MAX_SHARDS:-<worker-default>}"
echo "=================================="

quote_args() {
    local arg
    for arg in "$@"; do
        printf '%q ' "$arg"
    done
}

resolve_model_config() {
    local m
    m="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"
    QUEUE_MODEL="$MODEL"
    SCORER_MODEL="$MODEL"
    case "$m" in
        wmt22-cometkiwi-da|wmt22-comet|unbabel/wmt22-cometkiwi-da)
            BACKEND="comet"
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
            ;;
        wmt23-cometkiwi-da-xl|wmt23-comet|unbabel/wmt23-cometkiwi-da-xl)
            BACKEND="comet"
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
            ;;
        xcomet|xcomet-xl|unbabel/xcomet-xl)
            BACKEND="comet"
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
            ;;
        metricx24|metricx|metricx-24|metricx-24-hybrid-xl-v2p6|metricx-24-hybrid-xl-v2p6-bfloat16|google/metricx-24-hybrid-xl-v2p6|google/metricx-24-hybrid-xl-v2p6-bfloat16)
            BACKEND="metricx"
            VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
            ;;
        qwen3-14b|qwen/qwen3-14b|qwen3-8b|qwen/qwen3-8b|qwen3-4b-fp8|qwen/qwen3-4b-fp8|qwen3-4b|qwen3-4b-instruct-2507|qwen/qwen3-4b-instruct-2507|qwen3-4b-instruct-2507-fp8|qwen/qwen3-4b-instruct-2507-fp8|qwen3-1.7b|qwen/qwen3-1.7b|qwen3-0.6b|qwen/qwen3-0.6b|m-prometheus-7b|unbabel/m-prometheus-7b|m-prometheus-3b|unbabel/m-prometheus-3b)
            BACKEND="llm"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            ;;
        qwen3-4b-awq|qwen/qwen3-4b-awq)
            BACKEND="llm"
            VENV_PATH="${LLM_VENV:-${VENV_BASE}/vllm_venv}"
            VLLM_DTYPE="float16"
            ;;
        remedy|remedy-9b-22|shaomutan/remedy-9b-22|shaomutan_remedy-9b-22)
            BACKEND="remedy"
            if [ -d "$MODEL" ]; then
                SCORER_MODEL="$MODEL"
            else
                SCORER_MODEL="${REMEDY_MODEL:-/scratch/project_462001050/ibrahiam/envs/images/Models/patched_models/ShaomuTan_ReMedy-9B-22}"
            fi
            ;;
        bicleaner|bicleaner-ai|auto|en-xx|bitextor/bicleaner-ai-full-en-xx|es-xx|bitextor/bicleaner-ai-full-es-xx|de-xx|bitextor/bicleaner-ai-full-de-xx)
            BACKEND="bicleaner"
            ;;
        *)
            echo "ERROR: cannot resolve backend for MODEL=$MODEL" >&2
            exit 1
            ;;
    esac
}

resolve_model_config
echo "Resolved backend: $BACKEND"
echo "Queue model: $QUEUE_MODEL"
echo "Scorer model: $SCORER_MODEL"

VENV_PATH="${VENV_PATH:-}"

if [ "$BACKEND" = "bicleaner" ]; then
    export PYTHONNOUSERSITE=1
    module --force purge
    module load LUMI
    module load partition/G
    module load rocm
    module load lumi-container-wrapper

    export TF_ROCM_FUSION_ENABLE=0
    export SINGULARITY_BIND="/opt/rocm"
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

    BICLEANER_INST="${BICLEANER_VENV:-${VENV_BASE}/bicleaner_venv}"
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
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export HF_ASSETS_CACHE="$HF_HOME/assets"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
    export HUGGINGFACE_ASSETS_CACHE="$HF_ASSETS_CACHE"
    mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_ASSETS_CACHE"

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
    export SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif}"
fi

if [ "$BACKEND" = "llm" ] && [ -z "$MAX_MODEL_LEN" ]; then
    MAX_MODEL_LEN=8192
    echo "LLM max model len: $MAX_MODEL_LEN (default for OPUS LLM workers)"
elif [ "$BACKEND" = "llm" ]; then
    echo "LLM max model len: $MAX_MODEL_LEN (explicit override)"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
fi

compute_walltime_seconds() {
    local job_line token end_s
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        echo ""
        return
    fi
    job_line="$(scontrol show job --oneliner "$SLURM_JOB_ID" 2>/dev/null || true)"
    for token in $job_line; do
        case "$token" in
            EndTime=*)
                end_s="${token#EndTime=}"
                break
                ;;
        esac
    done
    if [ -n "$end_s" ] && [ "$end_s" != "Unknown" ]; then
        local end_epoch now
        end_epoch="$(date -d "$end_s" +%s 2>/dev/null || echo 0)"
        now="$(date +%s)"
        if [ "$end_epoch" -gt "$now" ]; then
            echo $((end_epoch - now))
            return
        fi
    fi
    echo ""  # unknown
}

WALLTIME_SECONDS="$(compute_walltime_seconds)"
if [ -n "$WALLTIME_SECONDS" ]; then
    echo "Remaining walltime seconds: $WALLTIME_SECONDS (resolved from SLURM EndTime)"
else
    echo "Remaining walltime seconds: unknown (SLURM EndTime unavailable or unparsable)"
fi

WORKER_ARGS=(
    --db "$DB"
    --model "$QUEUE_MODEL"
    --backend "$BACKEND"
    --output-base "$OUTPUT_BASE"
    --opus-root "$OPUS_ROOT"
    --batch-size "$BATCH_SIZE"
    --gpus "$GPUS"
)

if [ "$SCORER_MODEL" != "$QUEUE_MODEL" ]; then
    WORKER_ARGS+=(--scorer-model "$SCORER_MODEL")
fi

if [ -n "$WALLTIME_SECONDS" ]; then
    WORKER_ARGS+=(--walltime-seconds "$WALLTIME_SECONDS")
fi
if [ -n "$PART_WRITER" ]; then
    WORKER_ARGS+=(--part-writer)
fi
if [ -n "$PART_MAX_BYTES" ]; then
    WORKER_ARGS+=(--part-max-bytes "$PART_MAX_BYTES")
fi
if [ -n "$PART_MAX_SHARDS" ]; then
    WORKER_ARGS+=(--part-max-shards "$PART_MAX_SHARDS")
fi

# Backend-specific extras
case "$BACKEND" in
    llm)
        WORKER_ARGS+=(
            --prompt-mode "$PROMPT_MODE"
            --temperature "$TEMPERATURE"
            --max-tokens "$MAX_TOKENS"
            --max-retries "$MAX_RETRIES"
            --dtype "${VLLM_DTYPE:-bfloat16}"
            --gpu-memory-utilization "$VLLM_GPU_UTIL"
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
            --max-num-seqs "$MAX_NUM_SEQS"
        )
        if [ -n "$MAX_MODEL_LEN" ]; then
            WORKER_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
        fi
        if [ -n "$RESPONSE_FORMAT" ]; then
            WORKER_ARGS+=(--response-format "$RESPONSE_FORMAT")
        fi
        if [ -n "$ENFORCE_EAGER" ]; then
            WORKER_ARGS+=(--enforce-eager)
        fi
        ;;
    remedy)
        WORKER_ARGS+=(--gpu-memory-utilization "$VLLM_GPU_UTIL")
        WORKER_ARGS+=(--cache-dir "$HF_HOME")
        ;;
esac

WORKER_CMD="$(quote_args python3 -m execution.opus_queue.worker "${WORKER_ARGS[@]}")"
EXTRA_VLLM_ENVS=()
if [ -n "$STRUCTURED_OUTPUTS_BACKEND" ]; then
    EXTRA_VLLM_ENVS+=(--env "VLLM_STRUCTURED_OUTPUTS_BACKEND=${STRUCTURED_OUTPUTS_BACKEND}")
elif [ -n "$RESPONSE_FORMAT" ] && [ "$RESPONSE_FORMAT" != "none" ]; then
    echo "Auto-setting VLLM_STRUCTURED_OUTPUTS_BACKEND=outlines (xgrammar crashes on ROCm)"
    EXTRA_VLLM_ENVS+=(--env "VLLM_STRUCTURED_OUTPUTS_BACKEND=outlines")
fi

set +e
if [ "$BACKEND" = "bicleaner" ]; then
    set -x
    python3 -m execution.opus_queue.worker "${WORKER_ARGS[@]}"
    EXIT_CODE=$?
    set +x
elif [ "$BACKEND" = "remedy" ]; then
    set -x
    singularity exec --rocm -B /scratch -B /pfs -B /projappl "$SIF" env \
        PYTHONPATH="$PYTHONPATH" \
        HF_HOME="$HF_HOME" \
        HF_HUB_CACHE="$HF_HUB_CACHE" \
        HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
        HF_ASSETS_CACHE="$HF_ASSETS_CACHE" \
        HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
        HUGGINGFACE_ASSETS_CACHE="$HUGGINGFACE_ASSETS_CACHE" \
        TRANSFORMERS_OFFLINE=1 \
        HF_HUB_OFFLINE=1 \
        VLLM_TARGET_DEVICE=rocm \
        VLLM_USE_V1=1 \
        VLLM_USE_TRITON_FLASH_ATTN=0 \
        TORCHDYNAMO_DISABLE=1 \
        TORCHINDUCTOR_DISABLE=1 \
        python3 -m execution.opus_queue.worker "${WORKER_ARGS[@]}"
    EXIT_CODE=$?
    set +x
elif [ "$BACKEND" = "llm" ]; then
    set -x
    singularity run \
        --env "TORCH_HOME=${TORCH_HOME}" \
        --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
        --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
        --env "HF_HOME=${HF_HOME}" \
        --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
        --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
        --env "HF_ASSETS_CACHE=${HF_ASSETS_CACHE}" \
        --env "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}" \
        --env "HUGGINGFACE_ASSETS_CACHE=${HUGGINGFACE_ASSETS_CACHE}" \
        --env "HF_HUB_OFFLINE=1" \
        --env "TRANSFORMERS_OFFLINE=1" \
        "${EXTRA_VLLM_ENVS[@]}" \
        "$SIF" bash -lc "
        source ${VENV_PATH}/bin/activate
        ${WORKER_CMD}
    "
    EXIT_CODE=$?
    set +x
else
    set -x
    singularity run "$SIF" bash -lc "
        source ${VENV_PATH}/bin/activate
        ${WORKER_CMD}
    "
    EXIT_CODE=$?
    set +x
fi

echo "=================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================="

exit $EXIT_CODE
