#!/bin/bash
set -e # Exit immediately if a command fails

# Optional: Clean up old logs
rm -f slurmlog/cnt_tok*.log 

# --- Configuration ---
DATA_DIR="/scratch/project_462000941/FineOPUS/fineopus-original"
OUTPUT_FILE="/scratch/project_462000941/members/shaoxion/FineOPUS/statistics/token_counts/fineopus_original.csv"
TASK_LIST_FILE="tmp/incomplete_folders_fineopus_original.txt"

# Define how many folders each array job should process
CHUNK_SIZE=1500
# ---------------------

# 1. Find all possible folders from DATA_DIR
echo "Finding all folders in $DATA_DIR..."

# -printf "%f\n" tells find to print only the file's name (without leading directories)
find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort > all_folders.tmp

NUM_ALL_FOLDERS=$(wc -l < all_folders.tmp)


# 2 & 3. Find processed folders and filter to get incomplete list
echo "Determining incomplete folders..."

# Clear/create the incomplete list file to ensure we start fresh
> "$TASK_LIST_FILE" 

if [ ! -f "$OUTPUT_FILE" ]; then
    # Case A: If no output file exists, everything is incomplete
    echo "No existing output file found. All folders are incomplete."
    cp all_folders.tmp "$TASK_LIST_FILE"
else
    # Case B: Exclude folders found in OUTPUT_FILE
    # We use awk to read the CSV first (NR==FNR), recording the 1st column (path) as completed.
    # Then we read all_folders.tmp, printing only lines NOT in the completed array.
    
    awk -F, '
        NR==FNR {
            # Mark the folder path (1st column) as completed
            completed[$1] = 1
            next
        }
        {
            # Check the lines from all_folders.tmp
            if (!($0 in completed)) {
                print $0
            }
        }
    ' "$OUTPUT_FILE" all_folders.tmp > "$TASK_LIST_FILE"
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
    echo "Incomplted folders: $NUM_INCOMPLETE"
    echo "Submitting $NUM_JOBS array jobs (processing $CHUNK_SIZE folders each)..."
    
    # Removed undefined 'COMPLETE_LIST' variable from exports
    sbatch --array=1-$NUM_JOBS \
           --export=ALL,DATA_DIR="$DATA_DIR",TASK_LIST_FILE="$TASK_LIST_FILE",OUTPUT_FILE="$OUTPUT_FILE",CHUNK_SIZE="$CHUNK_SIZE" \
           count_token_chunk.slurm
fi