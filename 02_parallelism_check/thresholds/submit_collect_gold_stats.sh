#!/bin/bash
# ---------------------------------------------------------------------------
# submit_collect_gold_stats.sh
#
# Reads the best_model_per_lang_pair CSV, determines the set of models that
# actually appear as a best model, and submits one GPU job per model.
# Each job loads FLORES-200 + BOUQuET_Sentence, encodes gold sentences with
# the model, and writes per-(src,tgt) cosine distribution stats.
#
# Usage:
#   bash submit_collect_gold_stats.sh [--dry-run] [--model NAME]
# ---------------------------------------------------------------------------
set -euo pipefail

BEST_MODEL_CSV="../results/best_model_per_lang_pair_by_flores_bouquet_combined_selected_models.csv"
OUTPUT_CSV="./stats/gold_score_stats.csv"
LOG_DIR="../../logs/threshold_gold_stats"
BATCH_SIZE="${BATCH_SIZE:-64}"
DRY_RUN=0
FILTER_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --model)   FILTER_MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR" "./stats"

# Get unique models from best_model CSV (column index 3: src,tgt,model,MRR)
MODELS=$(awk -F',' 'NR>1 && $3!="" {print $3}' "$BEST_MODEL_CSV" | sort -u)

echo "============================================================"
echo "  Submitting collect_gold_stats.py jobs"
echo "============================================================"
echo "  best_model_csv: $BEST_MODEL_CSV"
echo "  output_csv   : $OUTPUT_CSV"
echo "  log_dir      : $LOG_DIR"
echo "  batch_size   : $BATCH_SIZE"
echo "  filter_model : ${FILTER_MODEL:-(all)}"
echo "============================================================"

for MODEL in $MODELS; do
    if [[ -n "$FILTER_MODEL" && "$MODEL" != "$FILTER_MODEL" ]]; then
        continue
    fi

    JOB_NAME="gold_stats-$(echo "$MODEL" | sed 's#.*/##; s#\.#_#g')"
    echo ">>> Model: $MODEL"

    CMD="sbatch \
        --job-name=$JOB_NAME \
        --output=$LOG_DIR/%x_%j.out \
        --error=$LOG_DIR/%x_%j.err \
        --partition=dev-g \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gpus-per-node=1 \
        --mem=64G \
        --time=0-01:00:00 \
        --account=project_462000964 \
        --export=ALL,MODEL=\"${MODEL}\",BEST_MODEL_CSV=\"${BEST_MODEL_CSV}\",OUTPUT_CSV=\"${OUTPUT_CSV}\",BATCH_SIZE=${BATCH_SIZE} \
        ./collect_gold_stats.sh"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "    [DRY RUN] $CMD"
    else
        JOB_ID=$(eval $CMD)
        echo "    Submitted: $JOB_ID"
    fi
done

echo "Done."
