#!/bin/bash

#SBATCH --job-name=cnt_tok
#SBATCH --account=project_462000787
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=slurmlog/cnt_tok_%A_%a.out.log
#SBATCH --error=slurmlog/cnt_tok_%A_%a.err.log

# --- Job Execution ---
# Inherited variables from submit.sh:
# DATA_DIR, OUTPUT_FILE, TASK_LIST_FILE, COMPLETE_LIST, CHUNK_SIZE

# 1. Get Task ID and Chunk Size
TASK_ID=$SLURM_ARRAY_TASK_ID
# Set default chunk size to 1 if not exported (a safety fallback)
: "${CHUNK_SIZE:=1}"

# 2. Calculate the line range this task is responsible for
START_LINE=$(( (TASK_ID - 1) * CHUNK_SIZE + 1 ))
END_LINE=$(( TASK_ID * CHUNK_SIZE ))

# 3. Print log info
echo "--------------------------------------------------------"
echo "Starting job $SLURM_ARRAY_JOB_ID, task $TASK_ID"
echo "Running on host: $(hostname)"
echo "Job started at $(date)"
echo "Processing lines: $START_LINE to $END_LINE from $TASK_LIST_FILE"
echo "--------------------------------------------------------"

# 4. Load modules once
module use /appl/local/csc/modulefiles/
module load pytorch

# 5. Extract the block of folders and process them in a loop
sed -n "${START_LINE},${END_LINE}p" "$TASK_LIST_FILE" | while read -r MY_FOLDER; do

    # Safety check: skip if line is empty
    if [ -z "$MY_FOLDER" ]; then
        echo "Task $TASK_ID: Reached empty line (end of list?), skipping."
        continue
    fi

    echo "Task $TASK_ID: Processing folder: $MY_FOLDER"

    # Run your command for this specific folder
    #
    # --- MODIFICATION HERE ---
    # Added --input=none to prevent srun from consuming
    # the stdin of the 'while read' loop.
    #
    srun --input=none python count_token.py --data_dir "$DATA_DIR" \
        --lang_pair $(basename "$MY_FOLDER") \
        --output_file "$OUTPUT_FILE"

    # Log completion for this specific folder
    echo "$MY_FOLDER" >> "$COMPLETE_LIST"

    echo "Task $TASK_ID: Finished and logged $MY_FOLDER"

done

echo "--------------------------------------------------------"
echo "Finished all work for task $TASK_ID at $(date)"
echo "--------------------------------------------------------"