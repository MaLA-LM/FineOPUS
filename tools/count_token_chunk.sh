#!/bin/bash
set -e # Exit immediately if a command fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional: Clean up old logs
mkdir -p "$SCRIPT_DIR/slurmlog"
rm -f "$SCRIPT_DIR"/slurmlog/cnt_tok*.log

# --- Configuration ---
DATA_DIR="/scratch/project_462001069/opus_qe/merged"
OUTPUT_FILE="/scratch/project_462001050/FineOPUS/statistics/mala-opus-dedup-2410.csv"

# Define how many folders each array job should process.
CHUNK_SIZE=200

# Tune this if tokenizer memory or throughput needs adjustment.
TOKENIZER_BATCH_SIZE=1024
# ---------------------

TASK_ROOT_DIR="${TASK_ROOT_DIR:-$(dirname "$OUTPUT_FILE")/token_count_tasks}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_TASK_DIR="$TASK_ROOT_DIR/run_$RUN_ID"
WORKER_TASK_DIR="$RUN_TASK_DIR/worker_lists"
WORKER_OUTPUT_DIR="$RUN_TASK_DIR/worker_outputs"
TASK_LIST_FILE="$RUN_TASK_DIR/incomplete_folders_opus_2410.txt"
ALL_FOLDERS_FILE="$RUN_TASK_DIR/all_folders.txt"

mkdir -p "$WORKER_TASK_DIR" "$WORKER_OUTPUT_DIR"

# 1. Find all possible folders from DATA_DIR.
echo "Finding all folders in $DATA_DIR..."
find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort > "$ALL_FOLDERS_FILE"

NUM_ALL_FOLDERS=$(wc -l < "$ALL_FOLDERS_FILE")

# 2 & 3. Find processed folders and filter to get incomplete list.
echo "Determining incomplete folders..."
> "$TASK_LIST_FILE"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "No existing output file found. All folders are incomplete."
    cp "$ALL_FOLDERS_FILE" "$TASK_LIST_FILE"
else
    # The output CSV's first column is lang_pair. Language-pair names do not
    # contain commas, so this simple CSV read is enough here.
    awk -F, '
        NR==FNR {
            value = $1
            gsub(/\r/, "", value)
            gsub(/^"|"$/, "", value)
            if (value != "" && value != "lang_pair") {
                completed[value] = 1
            }
            next
        }
        !($0 in completed) {
            print $0
        }
    ' "$OUTPUT_FILE" "$ALL_FOLDERS_FILE" > "$TASK_LIST_FILE"
fi

# 4. Count, split, and submit.
NUM_INCOMPLETE=$(wc -l < "$TASK_LIST_FILE")

if [ "$NUM_INCOMPLETE" -eq 0 ]; then
    echo "All folders are already processed. Nothing to submit."
else
    NUM_JOBS=$(( (NUM_INCOMPLETE + CHUNK_SIZE - 1) / CHUNK_SIZE ))

    echo "Total folders found: $NUM_ALL_FOLDERS"
    echo "Incomplete folders: $NUM_INCOMPLETE"
    echo "Creating $NUM_JOBS worker task files in $WORKER_TASK_DIR"

    for TASK_ID in $(seq 1 "$NUM_JOBS"); do
        START_LINE=$(( (TASK_ID - 1) * CHUNK_SIZE + 1 ))
        END_LINE=$(( TASK_ID * CHUNK_SIZE ))
        sed -n "${START_LINE},${END_LINE}p" "$TASK_LIST_FILE" > "$WORKER_TASK_DIR/worker_${TASK_ID}.txt"
    done

    echo "Submitting $NUM_JOBS array jobs (processing up to $CHUNK_SIZE folders each)..."

    cd "$SCRIPT_DIR"
    sbatch --array=1-$NUM_JOBS \
           --export=ALL,DATA_DIR="$DATA_DIR",TASK_LIST_FILE="$TASK_LIST_FILE",OUTPUT_FILE="$OUTPUT_FILE",CHUNK_SIZE="$CHUNK_SIZE",TOKENIZER_BATCH_SIZE="$TOKENIZER_BATCH_SIZE",WORKER_TASK_DIR="$WORKER_TASK_DIR",WORKER_OUTPUT_DIR="$WORKER_OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
           count_token_chunk.slurm
fi
