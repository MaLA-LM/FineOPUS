#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ---
DATA_DIR="/scratch/project_462001069/opus_qe/merged"
OUTPUT_FILE="/scratch/project_462001050/FineOPUS/statistics/mala-opus-dedup-2410.csv"
# ---------------------

TASK_ROOT_DIR="${TASK_ROOT_DIR:-$(dirname "$OUTPUT_FILE")/token_count_tasks}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${REPORT_DIR:-$TASK_ROOT_DIR/aggregation_$RUN_ID}"

python "$SCRIPT_DIR/aggregate_count_token.py" \
    --data_dir "$DATA_DIR" \
    --task_root_dir "$TASK_ROOT_DIR" \
    --output_file "$OUTPUT_FILE" \
    --report_dir "$REPORT_DIR"

echo "Aggregation reports written to $REPORT_DIR"
