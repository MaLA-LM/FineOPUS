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

set -euo pipefail

# --- Job Execution ---
# Inherited variables from count_token_chunk.sh:
# DATA_DIR, MANIFEST_FILE, OUTPUT_FILE, CHUNK_SIZE, TOKENIZER_BATCH_SIZE,
# PARQUET_BATCH_SIZE, WORKER_OUTPUT_DIR, SCRIPT_DIR

TASK_ID=$SLURM_ARRAY_TASK_ID
: "${CHUNK_SIZE:=5}"
: "${TOKENIZER_BATCH_SIZE:=1024}"
: "${PARQUET_BATCH_SIZE:=10000}"

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORKER_OUTPUT_DIR="${WORKER_OUTPUT_DIR:?WORKER_OUTPUT_DIR is required}"
MANIFEST_FILE="${MANIFEST_FILE:?MANIFEST_FILE is required}"
WORKER_OUTPUT_FILE="$WORKER_OUTPUT_DIR/counts_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.csv"

if [ ! -f "$MANIFEST_FILE" ]; then
    echo "Parquet manifest file not found: $MANIFEST_FILE" >&2
    exit 1
fi

mkdir -p "$WORKER_OUTPUT_DIR"

PARQUET_COUNT=$(awk -F'\t' -v task_id="$TASK_ID" '
    NR > 1 && $1 == task_id {
        count++
    }
    END {
        print count + 0
    }
' "$MANIFEST_FILE")

echo "--------------------------------------------------------"
echo "Starting job $SLURM_ARRAY_JOB_ID, task $TASK_ID"
echo "Running on host: $(hostname)"
echo "Job started at $(date)"
echo "Chunk size in parquet files: $CHUNK_SIZE"
echo "Tokenizer batch size: $TOKENIZER_BATCH_SIZE"
echo "Parquet row batch size: $PARQUET_BATCH_SIZE"
echo "Manifest file: $MANIFEST_FILE"
echo "Main direction CSV, not modified by worker: $OUTPUT_FILE"
echo "Worker output file: $WORKER_OUTPUT_FILE"
echo "Parquet files assigned to worker: $PARQUET_COUNT"
echo "Assigned parquet files:"
awk -F'\t' -v task_id="$TASK_ID" '
    NR > 1 && $1 == task_id {
        print "  " $3
    }
' "$MANIFEST_FILE"
echo "--------------------------------------------------------"

if [ "$PARQUET_COUNT" -eq 0 ]; then
    echo "No parquet files assigned to this worker. Nothing to do."
    exit 0
fi

# Load modules once per worker.
module use /appl/local/csc/modulefiles/
module load pytorch

# For latest DeepSeek.
source /scratch/project_462001050/ibrahiam/envs/transformers-latest/bin/activate

export HF_TOKEN

# Run one Python process per SLURM worker. The tokenizer is loaded once inside
# this process and reused for every parquet file assigned to this task.
srun --input=none python "$SCRIPT_DIR/count_token.py" \
    --data_dir "$DATA_DIR" \
    --parquet_manifest_file "$MANIFEST_FILE" \
    --worker_id "$TASK_ID" \
    --output_file "$WORKER_OUTPUT_FILE" \
    --tokenizer_batch_size "$TOKENIZER_BATCH_SIZE" \
    --parquet_batch_size "$PARQUET_BATCH_SIZE"

if [ ! -s "$WORKER_OUTPUT_FILE" ]; then
    echo "Worker completed but produced no output rows."
else
    echo "Worker parquet CSV written to $WORKER_OUTPUT_FILE"
fi

echo "--------------------------------------------------------"
echo "Finished all work for task $TASK_ID at $(date)"
echo "--------------------------------------------------------"
