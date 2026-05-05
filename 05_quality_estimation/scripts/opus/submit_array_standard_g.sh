#!/usr/bin/env bash
#
# Submit one SLURM array of OPUS queue workers, where each array task
# allocates a WHOLE LUMI-G node (4x MI250X = 8 GCDs) and spawns 8 workers
# in parallel via srun. This trades the 200 single-GCD slots of small-g
# for up to 200 nodes x 8 GCDs = 1600 GCDs in flight on standard-g.
#
# This is the high-throughput companion to scripts/opus/submit_array.sh
# (which uses small-g with 1 GCD per array task). Work is pre-assigned
# in a manifest; each node-level array task owns 8 stable worker slots.
#
# Throughput math:
#   --array=0-99%100      => 100 nodes x 8 GCDs =  800 GCDs in flight
#   --array=0-49%50       =>  50 nodes x 8 GCDs =  400 GCDs in flight
#   --array=0-199%200     => 200 nodes x 8 GCDs = 1600 GCDs (saturates the
#                            200-running cap of standard-g; combine with
#                            small-g via submit_array.sh for up to 1800)
#
# Walltime ceiling on standard-g is 48:00:00 (2 days). small-g allows 72h.
#
set -euo pipefail

MODEL=""
ARRAY_SPEC=""
CONCURRENCY=""
TIME_LIMIT="24:00:00"
MANIFEST_ROOT=""
BUILD_TAG=""
TRACE_ROOT="/scratch/project_462001050/opus_qe/shard_trace"
OUTPUT_BASE=""
OPUS_ROOT=""
PARTITION="standard-g"
ACCOUNT="project_462001249"
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
Usage: submit_array_standard_g.sh [args]

Submits a standard-g array. Each array task = 1 whole LUMI-G node = 8 GCDs.
Use scripts/opus/submit_array.sh instead for the per-GCD small-g pattern.

Required:
  --model <key>          Model key (e.g. metricx24, qwen3-4b-instruct-2507).
  --array <a-b>          SLURM array spec (e.g. 0-99). Each index = 1 node.
  --manifest-root <dir>  Root containing <build_tag>/manifest.jsonl.
  --build-tag <tag>      Manifest build tag.
  --output-base <dir>    Shared-storage dir for shard JSONLs / part files.

Optional:
  --trace-root <dir>     Per-worker trace root (default: /scratch/project_462001050/opus_qe/shard_trace).
  --concurrency <int>    SLURM %N cap (joined to --array as a-b%N).
                         Standard-g caps at 200 running jobs.
  --time <HH:MM:SS>      Walltime per array task (default 24:00:00, max
                         48:00:00 on standard-g).
  --opus-root <dir>      Override OPUS root (default: dataset adapter default).
  --account <acct>       SLURM --account (default: project_462001050).
  --partition <part>     SLURM --partition (default: standard-g; this script
                         is intended for standard-g only).
  --batch-size <int>     Per-forward batch size (default in worker: 8).
  --gpus <int>           --gpus passed to the Python worker (default: 1).
                         This is the per-worker GPU count for vLLM tensor
                         parallelism, NOT the SLURM allocation. Each of
                         the 8 workers on a node sees exactly 1 GCD.
  --prompt-mode <mode>   LLM prompt mode: detailed|simple|batch.
  --temperature <float>  LLM sampling temperature.
  --max-tokens <int>     LLM max generation tokens.
  --max-retries <int>    Retries for invalid LLM outputs.
  --dtype <dtype>        LLM dtype (bfloat16, float16, auto).
  --gpu-memory-utilization <float>  vLLM GPU memory fraction.
  --max-num-batched-tokens <int>    vLLM scheduler token budget.
  --max-num-seqs <int>   vLLM scheduler sequence cap.
  --max-model-len <int>  Optional vLLM context cap (default: 8192 for LLM).
  --response-format <fmt>  none|json_object|json_schema.
  --structured-outputs-backend <name>  outlines|xgrammar.
  --enforce-eager        Disable CUDA graphs / torch.compile in vLLM.
  --part-writer          Append multiple shards into worker-owned part files.
  --part-max-bytes <int> Rotate part files before the next shard would exceed this size.
  --part-max-shards <int>  Rotate part files after this many shards per part.
  --db <path>            Deprecated; ignored by manifest workers.
  --extra <str>          Extra args appended verbatim to sbatch.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model)         MODEL="${2:-}"; shift 2 ;;
        --array)         ARRAY_SPEC="${2:-}"; shift 2 ;;
        --concurrency)   CONCURRENCY="${2:-}"; shift 2 ;;
        --time)          TIME_LIMIT="${2:-}"; shift 2 ;;
        --db)            echo "WARNING: --db is deprecated and ignored; use --manifest-root/--build-tag." >&2; shift 2 ;;
        --manifest-root) MANIFEST_ROOT="${2:-}"; shift 2 ;;
        --build-tag)     BUILD_TAG="${2:-}"; shift 2 ;;
        --trace-root)    TRACE_ROOT="${2:-}"; shift 2 ;;
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

if [ -z "$MODEL" ] || [ -z "$ARRAY_SPEC" ] || [ -z "$MANIFEST_ROOT" ] || [ -z "$BUILD_TAG" ] || [ -z "$OUTPUT_BASE" ]; then
    echo "ERROR: --model, --array, --manifest-root, --build-tag, and --output-base are required." >&2
    print_usage >&2
    exit 1
fi

PLATFORM="$(printf '%s' "$PLATFORM" | tr '[:upper:]' '[:lower:]')"
if [ "$PLATFORM" != "lumi" ]; then
    echo "ERROR: OPUS standard-g launcher only supports LUMI (got --platform=$PLATFORM)." >&2
    exit 1
fi

if [ "$PARTITION" != "standard-g" ]; then
    echo "WARNING: this submitter sets node-level #SBATCH directives in run_worker_standard_g.sh that target standard-g. Overriding --partition to '$PARTITION' may not behave as intended." >&2
fi

if [ -n "$CONCURRENCY" ]; then
    ARRAY_ARG="${ARRAY_SPEC}%${CONCURRENCY}"
else
    ARRAY_ARG="$ARRAY_SPEC"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_worker_standard_g.sh"
TASK_SCRIPT="$SCRIPT_DIR/run_worker_standard_g_task.sh"

for path in "$RUNNER" "$TASK_SCRIPT"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: required script not found: $path" >&2
        exit 1
    fi
    if [ ! -r "$path" ]; then
        echo "ERROR: required script is not readable: $path" >&2
        exit 1
    fi
done

MANIFEST_FILE="${MANIFEST_ROOT%/}/${BUILD_TAG}/manifest.jsonl"
if [ ! -r "$MANIFEST_FILE" ]; then
    echo "ERROR: manifest is not readable: $MANIFEST_FILE" >&2
    exit 1
fi

SBATCH_ARGS=(
    --job-name="opus_g_${MODEL}"
    --array="$ARRAY_ARG"
    --time="$TIME_LIMIT"
    --export=ALL,MODEL="$MODEL",MANIFEST_ROOT="$MANIFEST_ROOT",BUILD_TAG="$BUILD_TAG",TRACE_ROOT="$TRACE_ROOT",OUTPUT_BASE="$OUTPUT_BASE",OPUS_ROOT="$OPUS_ROOT",PLATFORM="$PLATFORM",BATCH_SIZE="$BATCH_SIZE",GPUS="$GPUS",PROMPT_MODE="$PROMPT_MODE",TEMPERATURE="$TEMPERATURE",MAX_TOKENS="$MAX_TOKENS",MAX_RETRIES="$MAX_RETRIES",VLLM_DTYPE="$VLLM_DTYPE",VLLM_GPU_UTIL="$VLLM_GPU_UTIL",MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS",MAX_NUM_SEQS="$MAX_NUM_SEQS",MAX_MODEL_LEN="$MAX_MODEL_LEN",RESPONSE_FORMAT="$RESPONSE_FORMAT",STRUCTURED_OUTPUTS_BACKEND="$STRUCTURED_OUTPUTS_BACKEND",ENFORCE_EAGER="$ENFORCE_EAGER",PART_WRITER="$PART_WRITER",PART_MAX_BYTES="$PART_MAX_BYTES",PART_MAX_SHARDS="$PART_MAX_SHARDS",OPUS_STANDARD_G_SCRIPT_DIR="$SCRIPT_DIR"
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

# Throughput hint for the operator.
NODE_COUNT_HINT=""
case "$ARRAY_SPEC" in
    *-*)
        ARR_LO="${ARRAY_SPEC%%-*}"
        ARR_HI_RAW="${ARRAY_SPEC#*-}"
        ARR_HI="${ARR_HI_RAW%%[%,]*}"
        if [[ "$ARR_LO" =~ ^[0-9]+$ ]] && [[ "$ARR_HI" =~ ^[0-9]+$ ]]; then
            NODE_COUNT_HINT=$(( ARR_HI - ARR_LO + 1 ))
        fi
        ;;
esac
if [ -n "$NODE_COUNT_HINT" ]; then
    GCD_HINT=$(( NODE_COUNT_HINT * 8 ))
    if [ -n "$CONCURRENCY" ]; then
        RUNNING_NODES_HINT="$CONCURRENCY"
        if [ "$NODE_COUNT_HINT" -lt "$RUNNING_NODES_HINT" ]; then
            RUNNING_NODES_HINT="$NODE_COUNT_HINT"
        fi
        RUNNING_GCD_HINT=$(( RUNNING_NODES_HINT * 8 ))
        echo "Throughput hint: array spans $NODE_COUNT_HINT nodes ($GCD_HINT GCDs total); at most $RUNNING_NODES_HINT nodes ($RUNNING_GCD_HINT GCDs) in flight at once due to --concurrency=$CONCURRENCY."
    else
        echo "Throughput hint: array spans $NODE_COUNT_HINT nodes ($GCD_HINT GCDs total); SLURM cap is 200 running jobs on standard-g."
    fi
fi

echo "Submitting: sbatch ${SBATCH_ARGS[*]} $RUNNER"
sbatch "${SBATCH_ARGS[@]}" "$RUNNER"
