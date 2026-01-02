#!/bin/bash
set -e # Exit immediately if a command fails

# Optional: Clean up old logs
rm -f slurmlog/dedup_*.log 

# --- Configuration ---
DATA_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED"
OUT_DIR="/scratch/project_462000941/FineOPUS/FineOPUS-deduplicated"
STATS_FILE="/scratch/project_462000941/members/shaoxion/FineOPUS/02_deduplication/exact_dedup_stats.csv"

# Temporary Directory for intermediate files during deduplication DUCKDB processing
TMP_DIR="/scratch/project_462000964/FineOPUS/tmp"

# Task List: Temporary file to store folders that still need processing
TASK_LIST_FILE="incomplete_folders_dedup.tmp.txt"

# Job Size: How many folders per single Slurm array task?
#    Deduplication is heavy. Keep this LOW (e.g., 1 to 5) so jobs finish within time limits.
CHUNK_SIZE=10000 
# ---------------------

# 1. Find all possible folders from DATA_DIR
echo "Finding all folders in $DATA_DIR..."

# -printf "%f\n" tells find to print only the folder name (e.g., "en-fr", "de-es")
find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort > all_folders.tmp

NUM_ALL_FOLDERS=$(wc -l < all_folders.tmp)

# 2 & 3. Find processed folders and filter to get incomplete list
echo "Determining incomplete folders..."

# Clear/create the incomplete list file to ensure we start fresh
> "$TASK_LIST_FILE" 

if [ ! -f "$STATS_FILE" ]; then
    # Case A: If no stats file exists, everything is incomplete
    echo "No existing stats file found. All folders are incomplete."
    cp all_folders.tmp "$TASK_LIST_FILE"
else
    # Case B: Exclude folders found in STATS_FILE
    # We assume the 1st column of the CSV is 'dataset_name' (the folder name)
    
    awk -F, '
        NR==FNR {
            # Mark the folder name (1st column) as completed
            # remove quotes if csv has them, though standard awk split handles simple csvs
            completed[$1] = 1
            next
        }
        {
            # Check the lines from all_folders.tmp
            if (!($0 in completed)) {
                print $0
            }
        }
    ' "$STATS_FILE" all_folders.tmp > "$TASK_LIST_FILE"
fi

rm all_folders.tmp # Clean up temporary file

# 4. Count and submit
NUM_INCOMPLETE=$(wc -l < "$TASK_LIST_FILE")

if [ "$NUM_INCOMPLETE" -eq 0 ]; then
    echo "✅ All folders are already processed. Nothing to submit."
    rm "$TASK_LIST_FILE" # Optional: remove empty task file
else
    # Calculate array size using ceiling division
    NUM_JOBS=$(( (NUM_INCOMPLETE + CHUNK_SIZE - 1) / CHUNK_SIZE ))

    echo "Total folders found: $NUM_ALL_FOLDERS"
    echo "Incomplete folders: $NUM_INCOMPLETE"
    echo "Submitting $NUM_JOBS array jobs (processing $CHUNK_SIZE folders each)..."
    
    # Submit the job
    # We pass the paths as environment variables to the Slurm script
    sbatch --array=1-$NUM_JOBS \
           --export=ALL,DATA_DIR="$DATA_DIR",OUT_DIR="$OUT_DIR",TMP_DIR="$TMP_DIR",TASK_LIST_FILE="$TASK_LIST_FILE",STATS_FILE="$STATS_FILE",CHUNK_SIZE="$CHUNK_SIZE" \
           exact_dedup_chunk.slurm
fi