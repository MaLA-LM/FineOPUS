#!/usr/bin/env bash
#
# Per-srun-task script for standard-g OPUS workers. Runs once per GCD
# (8 invocations per node, controlled by SLURM_LOCALID 0..7).
#
# Launched by run_worker_standard_g.sh via:
#   srun --kill-on-bad-exit=1 ./run_worker_standard_g_task.sh
#
# Reads MODEL / MANIFEST_ROOT / BUILD_TAG / TRACE_ROOT / OUTPUT_BASE / OPUS_ROOT /
# LOG_ROOT / runtime knobs from the
# environment (exported by submit_array_standard_g.sh and forwarded by
# the outer launcher).
#
# This script intentionally mirrors scripts/opus/run_worker.sh so the two
# stay easy to compare side-by-side. The standard-g-specific differences are
# per-LOCALID GPU selection plus per-worker port/cache isolation.
#
set -euo pipefail

: "${MODEL:?MODEL is required}"
: "${MANIFEST_ROOT:?MANIFEST_ROOT is required}"
: "${BUILD_TAG:?BUILD_TAG is required}"
: "${TRACE_ROOT:?TRACE_ROOT is required}"
: "${OUTPUT_BASE:?OUTPUT_BASE is required}"

PLATFORM="$(printf '%s' "${PLATFORM:-lumi}" | tr '[:upper:]' '[:lower:]')"
if [ "$PLATFORM" != "lumi" ]; then
    echo "ERROR: OPUS standard-g workers only support PLATFORM=lumi (got '$PLATFORM')." >&2
    exit 1
fi

# ---- per-task identity + GPU selection -----------------------------------
# SLURM_LOCALID is the task's index within the node (0..7 here).
#
# On LUMI standard-g, the documented full-node pattern is to launch 8 tasks
# and set ROCR_VISIBLE_DEVICES=$SLURM_LOCALID inside each task. That is the
# model this script now follows directly.
#
# Ref: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/
# Ref: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/
# Ref: https://rocm.docs.amd.com/en/docs-7.2.1/reference/env-variables.html
LOCAL_ID="${SLURM_LOCALID:-0}"

# Master port: the existing run_worker.sh formula collides for the 8
# workers on a single node (same JOB_ID + same TASK_ID). Adding
# LOCAL_ID*101 (prime, > 7 distinct steps) gives each worker a distinct
# port in the 20000-39999 range with negligible collision probability
# across array tasks.
PORT_JOB_SEED="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}"
PORT_TASK_SEED="${SLURM_ARRAY_TASK_ID:-0}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 20000 + ((PORT_JOB_SEED + PORT_TASK_SEED + LOCAL_ID * 101) % 20000) ))

# ---- shared scratch caches (HF model files: read-only, safe to share) ----
SCRATCH_CACHE="${SCRATCH_CACHE:-/scratch/project_462001050/ibrahiam}"
export HF_HOME="$SCRATCH_CACHE/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HUGGINGFACE_ASSETS_CACHE="$HF_ASSETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$SCRATCH_CACHE/.cache"

# ---- per-worker compile/cache dirs (write-heavy, MUST be unique) ---------
# torch.compile and triton write generated kernels and metadata files.
# Multiple processes writing the same files race and produce corrupt
# caches. Per-worker subdirectories under the job tag avoid this and are
# easy to garbage-collect later.
JOB_TAG="${SLURM_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0}"
export TORCH_HOME="$SCRATCH_CACHE/.cache/torch/${JOB_TAG}/${LOCAL_ID}"
export TRITON_CACHE_DIR="$SCRATCH_CACHE/.cache/triton/${JOB_TAG}/${LOCAL_ID}"
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"

# ---- workdir + venv defaults (mirror run_worker.sh) ----------------------
WORKDIR="${WORKDIR:-/projappl/project_462001050/members/ibrahiam/05_quality_estimation}"
LOG_ROOT="${LOG_ROOT:-/scratch/project_462001050/opus_qe/logs}"
VENV_BASE="${VENV_BASE:-/scratch/project_462001050/ibrahiam/envs}"
DEFAULT_OPUS_ROOT="/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3"
OPUS_ROOT="${OPUS_ROOT:-$DEFAULT_OPUS_ROOT}"

# FLORES-aligned runtime knobs (same defaults as run_worker.sh).
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

mkdir -p "$LOG_ROOT"
cd "$WORKDIR"

echo "----- task GCD=${LOCAL_ID} pid=$$ host=$(hostname) port=${MASTER_PORT} -----"

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
echo "GCD=${LOCAL_ID}: backend=$BACKEND  queue_model=$QUEUE_MODEL  scorer_model=$SCORER_MODEL"

if [ "$BACKEND" = "bicleaner" ]; then
    echo "WARNING: bicleaner is mostly CPU-bound; running 8 instances per node may not improve throughput. Prefer scripts/opus/submit_array.sh (small-g) for this backend." >&2
fi

VENV_PATH="${VENV_PATH:-}"

# Backend-specific environment. The remedy branch overrides HF_HOME to
# point at the offline cache prepared for ReMedy; that override is
# intentional and matches run_worker.sh.
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

    export VLLM_TARGET_DEVICE=rocm
    export VLLM_USE_V1=1
    export VLLM_USE_TRITON_FLASH_ATTN=0
    export TORCHDYNAMO_DISABLE=1
    export TORCHINDUCTOR_DISABLE=1
    export SIF=/scratch/project_462001050/ibrahiam/envs/images/vllm092_rocm.sif
else
    module purge
    module use /appl/local/laifs/modules
    module load lumi-aif-singularity-bindings
    export SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif}"
fi

# Apply the LUMI wrapper pattern after module loads in case any module
# touched GPU visibility env vars.
export MASTER_ADDR=127.0.0.1
export ROCR_VISIBLE_DEVICES="$LOCAL_ID"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

if [ "$BACKEND" = "llm" ] && [ -z "$MAX_MODEL_LEN" ]; then
    MAX_MODEL_LEN=8192
    echo "GCD=${LOCAL_ID}: LLM max model len: $MAX_MODEL_LEN (default for OPUS LLM workers)"
elif [ "$BACKEND" = "llm" ]; then
    echo "GCD=${LOCAL_ID}: LLM max model len: $MAX_MODEL_LEN (explicit override)"
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
    end_s=""
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
    echo ""
}

WALLTIME_SECONDS="$(compute_walltime_seconds)"
if [ -n "$WALLTIME_SECONDS" ]; then
    echo "GCD=${LOCAL_ID}: remaining walltime seconds=$WALLTIME_SECONDS"
else
    echo "GCD=${LOCAL_ID}: remaining walltime seconds unknown"
fi
echo "GCD=${LOCAL_ID}: part_writer=${PART_WRITER:-0} part_max_bytes=${PART_MAX_BYTES:-<worker-default>} part_max_shards=${PART_MAX_SHARDS:-<worker-default>}"

WORKER_ARGS=(
    --mode manifest
    --manifest-root "$MANIFEST_ROOT"
    --build-tag "$BUILD_TAG"
    --trace-root "$TRACE_ROOT"
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
    EXTRA_VLLM_ENVS+=(--env "VLLM_STRUCTURED_OUTPUTS_BACKEND=outlines")
fi

set +e
if [ "$BACKEND" = "bicleaner" ]; then
    python3 -m execution.opus_queue.worker "${WORKER_ARGS[@]}"
    EXIT_CODE=$?
elif [ "$BACKEND" = "remedy" ]; then
    # Keep the standard-g host-side ROCR_VISIBLE_DEVICES wrapper, but make
    # the ReMedy container invocation itself match the known-good small-g
    # launcher as closely as possible. The small-g branch does not add extra
    # device binds or override ROCR/MASTER/TORCH cache envs here.
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
elif [ "$BACKEND" = "llm" ]; then
    # Match the small-g LLM container environment as closely as possible.
    # The only intentional deviation is GPU selection: on standard-g each
    # task still starts from the host-side ROCR wrapper, then switches to the
    # HIP/CUDA visibility vars that vLLM + Ray expect inside the container.
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
        unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
        export HIP_VISIBLE_DEVICES=${LOCAL_ID}
        export CUDA_VISIBLE_DEVICES=${LOCAL_ID}
        ${WORKER_CMD}
    "
    EXIT_CODE=$?
else
    singularity run \
        --env "HF_HOME=${HF_HOME}" \
        --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
        --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
        --env "HF_ASSETS_CACHE=${HF_ASSETS_CACHE}" \
        --env "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}" \
        --env "HUGGINGFACE_ASSETS_CACHE=${HUGGINGFACE_ASSETS_CACHE}" \
        --env "HF_HUB_OFFLINE=1" \
        --env "TRANSFORMERS_OFFLINE=1" \
        "$SIF" bash -lc "
        if [ -f /opt/venv/bin/activate ]; then source /opt/venv/bin/activate; fi
        source ${VENV_PATH}/bin/activate
        unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
        ${WORKER_CMD}
    "
    EXIT_CODE=$?
fi
set -e

echo "----- task GCD=${LOCAL_ID} done exit=$EXIT_CODE -----"
exit $EXIT_CODE
