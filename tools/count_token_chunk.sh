#!/bin/bash
set -e # Exit immediately if a command fails

rm -f slurmlog/cnt_tok*.log # Use -f to avoid errors if no files exist

# --- Configuration ---
DATA_DIR="/scratch/project_462000675/MaLA-LM/mala-opus-dedup-2410"
OUTPUT_FILE="/scratch/project_462000941/members/shaoxion/FineOPUS/statistics/mala-opus-dedup-2410.csv"
COMPLETE_LIST="completed_folders_opus_2410.txt"

# This file will list the full paths of folders to process
TASK_LIST_FILE="incomplete_folders_opus_2410.txt"

# NEW: Define how many folders each array job should process
CHUNK_SIZE=50
# ---------------------

# 1. Find all possible folders
echo "Finding all folders in $DATA_DIR..."
find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | sort > all_folders.tmp

# 2. Find completed folders
> "$TASK_LIST_FILE" # Clear/create the incomplete list

if [ ! -f "$COMPLETE_LIST" ]; then
    # If completed list doesn't exist, all folders are incomplete
    echo "Completed list not found. Submitting all folders."
    cp all_folders.tmp "$TASK_LIST_FILE"
else
    # Use awk to find incomplete folders
    echo "Checking all folders against $COMPLETE_LIST..."
    
    awk -F, -v all_folders_file="all_folders.tmp" '
        # 1. Read the CSV file first (NR==FNR)
        NR==FNR {
            for (i=1; i<=NF; i++) {
                completed[$i] = 1
            }
            next
        }
        
        # 2. Read the "all_folders.tmp" file second
        {
            if ( ! ($0 in completed) ) {
                print $0
            }
        }
    ' "$COMPLETE_LIST" all_folders.tmp > "$TASK_LIST_FILE"
fi

rm all_folders.tmp # Clean up temporary file

# 3. Count and submit
NUM_INCOMPLETE=$(wc -l < "$TASK_LIST_FILE")

if [ "$NUM_INCOMPLETE" -eq 0 ]; then
    echo "✅ All folders are already processed. Nothing to submit."
    rm "$TASK_LIST_FILE" # Clean up
else
    # NEW: Calculate the number of array jobs needed using ceiling division
    # (NUM_INCOMPLETE + CHUNK_SIZE - 1) / CHUNK_SIZE
    NUM_JOBS=$(( (NUM_INCOMPLETE + CHUNK_SIZE - 1) / CHUNK_SIZE ))

    echo "Found $NUM_INCOMPLETE incomplete folders. Submitting $NUM_JOBS array jobs (processing $CHUNK_SIZE folders each)..."
    
    # MODIFIED: Submit the "worker" script with the correct array size and export CHUNK_SIZE
    sbatch --array=1-$NUM_JOBS \
           --export=ALL,DATA_DIR="$DATA_DIR",COMPLETE_LIST="$COMPLETE_LIST",TASK_LIST_FILE="$TASK_LIST_FILE",OUTPUT_FILE="$OUTPUT_FILE",CHUNK_SIZE="$CHUNK_SIZE" \
           count_token_chunk.slurm
fi