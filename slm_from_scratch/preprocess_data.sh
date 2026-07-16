#!/bin/bash
#SBATCH --job-name=preprocess-data
#SBATCH --output=../logs/preprocess-data/%x_%j.out
#SBATCH --error=../logs/preprocess-data/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --account=project_462001087

set -Eeuo pipefail


on_error() {
    local status=$?
    echo "Error: preprocessing failed with exit code ${status}." >&2
    exit "$status"
}
trap on_error ERR

usage() {
    cat <<'EOF'
Usage: preprocess_data.sh [OPTIONS]

Options override environment variables. For sbatch, you can use --export=INPUT_FILE=...,OUTPUT_PREFIX=... or pass flags after the script name.

  -i, --input FILE            Input JSONL (env: INPUT_FILE)
  -o, --output-prefix PATH    Output prefix (env: OUTPUT_PREFIX)
  -t, --tokenizer MODEL       HuggingFace tokenizer id (default: Qwen/Qwen3.5-9B, env: TOKENIZER_MODEL)
  -w, --workers N            Worker count (default: SLURM_CPUS_PER_TASK or 1)
  -h, --help                 Show this help
EOF
}

CLI_WORKERS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            if [[ $# -lt 2 ]]; then echo "Error: $1 requires a value" >&2; exit 1; fi
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output-prefix)
            if [[ $# -lt 2 ]]; then echo "Error: $1 requires a value" >&2; exit 1; fi
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        -t|--tokenizer)
            if [[ $# -lt 2 ]]; then echo "Error: $1 requires a value" >&2; exit 1; fi
            TOKENIZER_MODEL="$2"
            shift 2
            ;;
        -w|--workers)
            if [[ $# -lt 2 ]]; then echo "Error: $1 requires a value" >&2; exit 1; fi
            CLI_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# Env defaults; CLI was applied above (only specified flags are overwritten)
INPUT_FILE="${INPUT_FILE:-}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3.5-9B}"
if [[ -n "$CLI_WORKERS" ]]; then
    WORKERS="$CLI_WORKERS"
else
    WORKERS=${SLURM_CPUS_PER_TASK:-1}
fi

if [[ -z "$INPUT_FILE" || -z "$OUTPUT_PREFIX" ]]; then
    echo "Error: INPUT_FILE and OUTPUT_PREFIX must be set (use -i/-o or environment variables)." >&2
    usage >&2
    exit 1
fi

if [[ ! -f "$INPUT_FILE" || ! -r "$INPUT_FILE" ]]; then
    echo "Error: input file does not exist or is not readable: $INPUT_FILE" >&2
    exit 1
fi

if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: workers must be a positive integer, got: $WORKERS" >&2
    exit 1
fi

OUTPUT_DIR=$(dirname -- "$OUTPUT_PREFIX")
mkdir -p -- "$OUTPUT_DIR"

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

echo "HF_HOME: ${HF_HOME:-<not set>}"

echo "Configuration:"
echo "  Input file:     $INPUT_FILE"
echo "  Output prefix:  $OUTPUT_PREFIX"
echo "  Tokenizer:      $TOKENIZER_MODEL"
echo "  Workers:        $WORKERS"

RUN_MARKER=$(mktemp "$OUTPUT_DIR/.preprocess-marker.XXXXXX")
trap 'rm -f -- "$RUN_MARKER"' EXIT

srun python ./Megatron-LM/tools/preprocess_data.py \
    --input "$INPUT_FILE" \
    --output-prefix "$OUTPUT_PREFIX" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --workers "$WORKERS" \
    --log-interval 10000 \
    --append-eod

# Megatron's preprocessing work runs in a child process whose failure is not
# propagated by some versions, so verify the expected artifacts explicitly.
OUTPUT_BIN="${OUTPUT_PREFIX}_text_document.bin"
OUTPUT_IDX="${OUTPUT_PREFIX}_text_document.idx"
if [[ ! -s "$OUTPUT_BIN" || ! -s "$OUTPUT_IDX" ||
      "$OUTPUT_BIN" -ot "$RUN_MARKER" || "$OUTPUT_IDX" -ot "$RUN_MARKER" ]]; then
    echo "Error: preprocessing did not create or update non-empty output files:" >&2
    echo "  $OUTPUT_BIN" >&2
    echo "  $OUTPUT_IDX" >&2
    exit 1
fi

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
