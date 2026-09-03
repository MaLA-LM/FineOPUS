#!/usr/bin/env bash
#SBATCH --job-name=eval-mt
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --partition=small-g
#SBATCH --time=0-08:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_465002530
#SBATCH --output=logs/eval/%x_%A_%a.out
#SBATCH --error=logs/eval/%x_%A_%a.err

set -euo pipefail

# Slurm executes a private copy of this file from /var/spool/slurmd, so
# BASH_SOURCE[0] does not point back to the repository inside a batch job.
# submit_mt_eval.sh enters evaluation/ before sbatch, making SLURM_SUBMIT_DIR
# the stable source directory. Keep BASH_SOURCE for direct/local execution.
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/mt_eval.py" ]]; then
    SCRIPT_DIR=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

DATA_ROOT="$SCRIPT_DIR/data"
TOKENIZER="$REPO_ROOT/slm_from_scratch/tokenizer/qwen3_5"
TMP_BASE="$SCRIPT_DIR/tmp"
DATASETS="FLORES-200,NTREX-128,BOUQuET_Sentence"
FEW_SHOT=3
LIMIT=""
TASK_MANIFEST=""
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
MODEL_NAME=""
MODEL_DIR=""
CHECKPOINT_NAME=""
CHECKPOINT_DIR=""
HF_MODEL=""
LANGUAGES=""
OUTPUT_DIR=""
KEEP_HF=0
OVERWRITE=0
ENFORCE_EAGER=0
COMET_MODEL="Unbabel/wmt22-comet-da"
COMET_BATCH_SIZE=8
COMET_GPUS=1
NO_COMET=0

usage() {
    cat <<EOF
Usage:
  $0 --task-manifest FILE [--task-id N] [options]
  $0 --model-dir DIR --checkpoint-dir DIR --hf-model NAME \\
     --languages LANG[,LANG...] --output-dir DIR [options]

Options:
  --data-root DIR         Dataset root (default: $DATA_ROOT)
  --tokenizer DIR         Training tokenizer (default: $TOKENIZER)
  --tmp-base DIR          Temporary HF export parent (default: $TMP_BASE)
  --datasets LIST         Comma-separated dataset names
  --few-shot N            Number of demonstrations (default: 3)
  --limit N               Limit scored rows per direction (smoke tests)
  --keep-hf               Keep converted model under OUTPUT_DIR/hf_model
  --overwrite             Re-run completed direction results
  --enforce-eager         Pass enforce_eager=True to vLLM
  --comet-model MODEL     COMET checkpoint (default: $COMET_MODEL)
  --comet-batch-size N    COMET inference batch size (default: 8)
  --comet-gpus N          GPUs used by COMET after vLLM exits (default: 1)
  --no-comet              Disable COMET and only compute BLEU/chrF++
EOF
}

while (( $# )); do
    case "$1" in
        --task-manifest) TASK_MANIFEST="$2"; shift 2 ;;
        --task-id) TASK_ID="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --checkpoint-name) CHECKPOINT_NAME="$2"; shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --hf-model) HF_MODEL="$2"; shift 2 ;;
        --languages) LANGUAGES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --tmp-base) TMP_BASE="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --few-shot) FEW_SHOT="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --keep-hf) KEEP_HF=1; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        --enforce-eager) ENFORCE_EAGER=1; shift ;;
        --comet-model) COMET_MODEL="$2"; shift 2 ;;
        --comet-batch-size) COMET_BATCH_SIZE="$2"; shift 2 ;;
        --comet-gpus) COMET_GPUS="$2"; shift 2 ;;
        --no-comet) NO_COMET=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$TASK_MANIFEST" ]]; then
    [[ -n "$TASK_ID" ]] || { echo "--task-id or SLURM_ARRAY_TASK_ID is required" >&2; exit 2; }
    [[ -f "$TASK_MANIFEST" ]] || { echo "Task manifest not found: $TASK_MANIFEST" >&2; exit 1; }
    task_line=$(sed -n "$((TASK_ID + 2))p" "$TASK_MANIFEST")
    [[ -n "$task_line" ]] || { echo "No task $TASK_ID in $TASK_MANIFEST" >&2; exit 1; }
    IFS=$'\t' read -r manifest_id MODEL_NAME MODEL_DIR CHECKPOINT_NAME CHECKPOINT_DIR HF_MODEL LANGUAGES OUTPUT_DIR <<< "$task_line"
    [[ "$manifest_id" == "$TASK_ID" ]] || {
        echo "Manifest task mismatch: requested $TASK_ID, read $manifest_id" >&2
        exit 1
    }
fi

for value_name in MODEL_DIR CHECKPOINT_DIR HF_MODEL LANGUAGES OUTPUT_DIR; do
    [[ -n "${!value_name}" ]] || { echo "Missing required value: $value_name" >&2; exit 2; }
done
[[ -d "$MODEL_DIR" ]] || { echo "Model directory missing: $MODEL_DIR" >&2; exit 1; }
[[ -d "$CHECKPOINT_DIR" ]] || { echo "Checkpoint directory missing: $CHECKPOINT_DIR" >&2; exit 1; }
[[ -d "$TOKENIZER" ]] || { echo "Tokenizer directory missing: $TOKENIZER" >&2; exit 1; }
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_DIR")}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-$(basename "$CHECKPOINT_DIR")}"

if [[ -f "$OUTPUT_DIR/_SUCCESS" && "$OVERWRITE" == 0 ]]; then
    echo "Already complete, skipping: $OUTPUT_DIR"
    exit 0
fi

mkdir -p "$OUTPUT_DIR" "$TMP_BASE"
JOB_TMP=$(mktemp -d "$TMP_BASE/mt_eval.${SLURM_JOB_ID:-local}.XXXXXX")
HF_OUTPUT="$JOB_TMP/hf_model"
cleanup() {
    if [[ "$KEEP_HF" == 1 && -d "$HF_OUTPUT" ]]; then
        mkdir -p "$OUTPUT_DIR"
        mv -- "$HF_OUTPUT" "$OUTPUT_DIR/hf_model"
    fi
    rm -rf -- "$JOB_TMP"
}
trap cleanup EXIT

echo "Job started: $(date --iso-8601=seconds)"
echo "Model: $MODEL_NAME"
echo "Checkpoint: $CHECKPOINT_NAME"
echo "Languages: $LANGUAGES"
echo "Datasets: $DATASETS"
echo "Output: $OUTPUT_DIR"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.7
source "$SCRIPT_DIR/eval_env/bin/activate"

# The venv includes system packages; ignore an incompatible package installed
# in ~/.local (notably huggingface-hub 1.x shadowing the module's 0.x version).
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_MODULES_CACHE="${JOB_TMP}/hf_modules"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Conversion and vLLM each create a single-process torch.distributed store.
# Array tasks can share a node, so an inherited/default MASTER_PORT causes
# intermittent EADDRINUSE failures. Derive a stable, distinct port per task.
port_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}"
port_task_id="${SLURM_ARRAY_TASK_ID:-0}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + (port_job_id + port_task_id * 997) % 40000))
echo "Distributed rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"

preflight_args=(
    --data-root "$DATA_ROOT"
    --output-dir "$OUTPUT_DIR"
    --languages "$LANGUAGES"
    --datasets "$DATASETS"
    --few-shot "$FEW_SHOT"
    --comet-model "$COMET_MODEL"
    --comet-batch-size "$COMET_BATCH_SIZE"
    --comet-gpus "$COMET_GPUS"
    --preflight-only
    --check-runtime
)
[[ "$NO_COMET" == 0 ]] || preflight_args+=(--no-comet)
python "$SCRIPT_DIR/mt_eval.py" "${preflight_args[@]}"

echo "Converting checkpoint to Hugging Face format: $(date --iso-8601=seconds)"
bash "$SCRIPT_DIR/megatron-to-hf-lumi.sh" \
    "$CHECKPOINT_DIR" "$HF_OUTPUT" "$HF_MODEL" "$TOKENIZER"

eval_args=(
    --model "$HF_OUTPUT"
    --model-name "$MODEL_NAME"
    --checkpoint-name "$CHECKPOINT_NAME"
    --data-root "$DATA_ROOT"
    --output-dir "$OUTPUT_DIR"
    --languages "$LANGUAGES"
    --datasets "$DATASETS"
    --few-shot "$FEW_SHOT"
    --comet-model "$COMET_MODEL"
    --comet-batch-size "$COMET_BATCH_SIZE"
    --comet-gpus "$COMET_GPUS"
)
[[ -z "$LIMIT" ]] || eval_args+=(--limit "$LIMIT")
[[ "$OVERWRITE" == 0 ]] || eval_args+=(--overwrite)
[[ "$ENFORCE_EAGER" == 0 ]] || eval_args+=(--enforce-eager)
[[ "$NO_COMET" == 0 ]] || eval_args+=(--no-comet)

echo "Starting vLLM evaluation: $(date --iso-8601=seconds)"
python "$SCRIPT_DIR/mt_eval.py" "${eval_args[@]}"
echo "Job completed: $(date --iso-8601=seconds)"
