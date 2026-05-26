#!/bin/bash

#SBATCH --job-name=cnt_tok
#SBATCH --account=project_462001050
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=slurmlog/cnt_tok_%A_%a.out.log  # Changed to %A_%a for unique array logs
#SBATCH --error=slurmlog/cnt_tok_%A_%a.err.log     # Changed to %A_%a for unique array logs

# --- Job Execution ---
# The variables DATA_DIR, OUTPUT_FILE, and TASK_LIST_FILE are
# inherited from the 'sbatch --export' command in submit.sh

# 1. Get the task ID (from 1 to N, set by the submit script)
TASK_ID=$SLURM_ARRAY_TASK_ID

# 2. Get this task's folder path from the list
MY_FOLDER=$(sed -n "${TASK_ID}p" "$TASK_LIST_FILE")

# 3. Safety check
if [ -z "$MY_FOLDER" ]; then
    echo "Error: Task ID $TASK_ID could not find a folder in $TASK_LIST_FILE."
    exit 1
fi

# 4. Print log info
echo "--------------------------------------------------------"
echo "Starting job $SLURM_ARRAY_JOB_ID, task $TASK_ID"
echo "Processing folder: $MY_FOLDER"
echo "Running on host: $(hostname)"
echo "Job started at $(date)"
echo "--------------------------------------------------------"

# 5. Run your command
module use /appl/local/csc/modulefiles/
module load pytorch
# for latest DeepSeek
source /scratch/project_462001050/ibrahiam/envs/transformers-latest/bin/activate

export HF_TOKEN

srun python count_token.py --data_dir "$DATA_DIR" \
    --lang_pair "$(basename "$MY_FOLDER")" \
    --output_file "$OUTPUT_FILE"

echo "$MY_FOLDER" >> ${COMPLETE_LIST}

echo "Finished task $TASK_ID at $(date)"