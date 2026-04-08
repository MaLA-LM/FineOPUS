#!/bin/bash
# ---------------------------------------------------------------------------
# submit_all.sh
#
# Reads model_to_language_pairs.json, computes the number of chunks per model,
# and submits one SLURM array job per model.
#
# Usage:
#   bash submit_all.sh [--pairs-per-chunk N] [--dry-run]
#
# Options:
#   --pairs-per-chunk N   Number of language pairs per SLURM array task (default: 10)
#   --dry-run             Print sbatch commands without actually submitting
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Configurable paths ----------------------------------------------------
MODEL_PAIRS_JSON="${MODEL_PAIRS_JSON:-$SCRIPT_DIR/model_to_language_pairs.json}"
INPUT_DIR="${INPUT_DIR:-/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_462001069/FineOPUS/intermediate/FineOPUS-Filtered-Stage2-Scored}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOG_DIR="$REPO_ROOT/logs/similarity_scoring"
# ----------------------------------------------------------------------------

PAIRS_PER_CHUNK=10
DRY_RUN=0
FILTER_MODEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pairs-per-chunk)
            PAIRS_PER_CHUNK="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --model)
            FILTER_MODEL="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  FineOPUS Similarity Scoring Job Submission"
echo "======================================================="
echo "  Model pairs JSON : $MODEL_PAIRS_JSON"
echo "  Input dir        : $INPUT_DIR"
echo "  Output dir       : $OUTPUT_DIR"
echo "  Pairs per chunk  : $PAIRS_PER_CHUNK"
echo "  Dry run          : $DRY_RUN"
echo "  Filter model     : ${FILTER_MODEL:-(all)}"
echo "======================================================="

_py_list_models() {
python3 - <<PYEOF
import json, math

with open("$MODEL_PAIRS_JSON") as f:
    data = json.load(f)

filter_model = "$FILTER_MODEL"
for model, pairs in data.items():
    if filter_model and model != filter_model:
        continue
    n_pairs = len(pairs)
    n_chunks = math.ceil(n_pairs / $PAIRS_PER_CHUNK)
    # Support both old (str) and new ([pair, bytes]) formats for size display
    total_gb = sum(e[1] if isinstance(e, list) else 0 for e in pairs) / 1e9
    print(f"{model}|{n_pairs}|{n_chunks}|{total_gb:.1f}")
PYEOF
}

# Preview table
_py_list_models
echo ""

# Submit
while IFS='|' read -r model n_pairs n_chunks total_gb; do
    last_idx=$((n_chunks - 1))
    echo ">>> Model: $model"
    echo "    Pairs: $n_pairs  |  Total: ${total_gb} GB  |  Chunks: $n_chunks  |  Array: 0-${last_idx}"

    SUBMIT_CMD="sbatch \
        --array=0-${last_idx} \
        --export=ALL,MODEL=\"${model}\",TOTAL_CHUNKS=${n_chunks},INPUT_DIR=\"${INPUT_DIR}\",OUTPUT_DIR=\"${OUTPUT_DIR}\",MODEL_PAIRS_JSON=\"${MODEL_PAIRS_JSON}\",BATCH_SIZE=${BATCH_SIZE} \
        ${SCRIPT_DIR}/compute_similarity.sh"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "    [DRY RUN] $SUBMIT_CMD"
    else
        JOB_ID=$(eval $SUBMIT_CMD)
        echo "    Submitted: $JOB_ID"
    fi
    echo ""
done < <(_py_list_models)

echo "Done submitting all jobs."
