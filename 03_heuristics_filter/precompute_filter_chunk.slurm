#!/bin/bash
#
#SBATCH --job-name=precomp       # A descriptive name for your job
#SBATCH --account=project_462001050      # IMPORTANT: Replace with your project ID
#SBATCH --partition=small           # The partition (queue) to submit to.
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks-per-node=1         # Run 1 main task (your python script)
#SBATCH --cpus-per-task=16          # Request 16 CPU cores for this one task
#SBATCH --mem=64G                   # Total memory for all 16 cores (14G/core)
#SBATCH --time=36:00:00              # Maximum execution time (HH:MM:SS)
#SBATCH --output=slurmlog/%x_%A_%a.out.log  # Standard output log file
#SBATCH --error=slurmlog/%x_%A_%a.err.log   # Standard error log file


echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Running in directory $(pwd)"
echo "Job started at $(date)"
echo "SLURM allocated ${SLURM_CPUS_PER_TASK} CPUs to this task."

module load cray-python

# Calculate lines to process from the TASK_LIST_FILE
START_LINE=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE + 1 ))
END_LINE=$(( START_LINE + CHUNK_SIZE - 1 ))

# Read specific lines from the task list file
# sed -n '5,10p' prints lines 5 through 10
FOLDERS_TO_PROCESS=$(sed -n "${START_LINE},${END_LINE}p" "$TASK_LIST_FILE")

echo "--- Job ID: $SLURM_JOB_ID Array ID: $SLURM_ARRAY_TASK_ID ---"
echo "Processing lines $START_LINE to $END_LINE from $TASK_LIST_FILE"

# Loop through the assigned folders
for FOLDER_NAME in $FOLDERS_TO_PROCESS; do
    echo "------------------------------------------------"
    echo "Processing dataset: $FOLDER_NAME"
    
    FULL_INPUT_PATH="${DATA_DIR}/${FOLDER_NAME}"
    echo "Precomputing language pair: $lang_pair"
    ls "$FULL_INPUT_PATH"
    
    # Run the Python script
    python -u precompute_heuristics_filters.py \
        --data_dir "$FULL_INPUT_PATH" \
        --out_dir "$OUT_DIR" \
        --stats_file "$STATS_FILE" \
        --src_col "source_text" \
        --trg_col "target_text"

    echo "Finished processing: $FOLDER_NAME"
done

echo "--- All tasks in this chunk complete ---"
echo "Job finished at $(date)"