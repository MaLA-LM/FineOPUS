#!/bin/bash

#SBATCH --job-name=cnt_tok
#SBATCH --account=project_462001050
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=slurmlog/cnt_tok_%A_%a.out.log
#SBATCH --error=slurmlog/cnt_tok_%A_%a.err.log

set -e

# --- Job Execution ---
# Inherited variables from count_token_chunk.sh:
# DATA_DIR, OUTPUT_FILE, TASK_LIST_FILE, CHUNK_SIZE, TOKENIZER_BATCH_SIZE,
# WORKER_TASK_DIR, WORKER_OUTPUT_DIR, SCRIPT_DIR

TASK_ID=$SLURM_ARRAY_TASK_ID
: "${CHUNK_SIZE:=400}"
: "${TOKENIZER_BATCH_SIZE:=1024}"

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORKER_TASK_FILE="$WORKER_TASK_DIR/worker_${TASK_ID}.txt"
WORKER_OUTPUT_FILE="$WORKER_OUTPUT_DIR/counts_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.csv"
LOCK_FILE="${OUTPUT_FILE}.lock"

if [ ! -f "$WORKER_TASK_FILE" ]; then
    echo "Worker task file not found: $WORKER_TASK_FILE" >&2
    exit 1
fi

PAIR_COUNT=$(grep -cve '^[[:space:]]*$' "$WORKER_TASK_FILE" || true)

echo "--------------------------------------------------------"
echo "Starting job $SLURM_ARRAY_JOB_ID, task $TASK_ID"
echo "Running on host: $(hostname)"
echo "Job started at $(date)"
echo "Chunk size: $CHUNK_SIZE"
echo "Tokenizer batch size: $TOKENIZER_BATCH_SIZE"
echo "Task list file: $TASK_LIST_FILE"
echo "Worker task file: $WORKER_TASK_FILE"
echo "Worker output file: $WORKER_OUTPUT_FILE"
echo "Language pairs assigned to worker: $PAIR_COUNT"
echo "--------------------------------------------------------"

if [ "$PAIR_COUNT" -eq 0 ]; then
    echo "Worker task file is empty. Nothing to do."
    exit 0
fi

# Load modules once per worker.
module use /appl/local/csc/modulefiles/
module load pytorch

# For latest DeepSeek.
source /scratch/project_462001050/ibrahiam/envs/transformers-latest/bin/activate

export HF_TOKEN

# Run one Python process per SLURM worker. The tokenizer is loaded once inside
# this process and reused for every language pair in WORKER_TASK_FILE.
srun --input=none python "$SCRIPT_DIR/count_token.py" \
    --data_dir "$DATA_DIR" \
    --lang_pairs_file "$WORKER_TASK_FILE" \
    --output_file "$WORKER_OUTPUT_FILE" \
    --processed_file "$OUTPUT_FILE" \
    --tokenizer_batch_size "$TOKENIZER_BATCH_SIZE"

if [ ! -s "$WORKER_OUTPUT_FILE" ]; then
    echo "Worker completed but produced no output rows. Nothing to merge."
    exit 0
fi

if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to safely merge worker CSV files into $OUTPUT_FILE" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

(
    flock -x 200

    if [ ! -s "$OUTPUT_FILE" ]; then
        cat "$WORKER_OUTPUT_FILE" >> "$OUTPUT_FILE"
    else
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
            FNR == 1 {
                next
            }
            {
                value = $1
                gsub(/\r/, "", value)
                gsub(/^"|"$/, "", value)
                if (value != "" && !(value in completed)) {
                    print $0
                    completed[value] = 1
                }
            }
        ' "$OUTPUT_FILE" "$WORKER_OUTPUT_FILE" >> "$OUTPUT_FILE"
    fi
) 200>"$LOCK_FILE"

echo "Merged completed worker CSV into $OUTPUT_FILE"
echo "--------------------------------------------------------"
echo "Finished all work for task $TASK_ID at $(date)"
echo "--------------------------------------------------------"
