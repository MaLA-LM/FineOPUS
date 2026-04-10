#!/bin/bash
#SBATCH --job-name=split_parquet
#SBATCH --output=../logs/split_parquet/%x_%A_%a.out
#SBATCH --error=../logs/split_parquet/%x_%A_%a.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --account=project_462001087

# ---------------------------------------------------------------------------
# Edit the variables below, then submit with:
#   sbatch tools/split_parquet.sh                        # single job
#   sbatch --array=0-19 tools/split_parquet.sh           # array job (20 tasks)
# ---------------------------------------------------------------------------

INPUT_DIR=""
OUTPUT_DIR=""
TOTAL_CHUNKS=20        # must match the --array upper bound + 1
MAX_ROWS=10000000
WORKERS=4

# ---------------------------------------------------------------------------

start_time=$(date +%s)
echo "Job started at  : $(date)"
echo "Array task ID   : ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node            : $(hostname)"

CHUNK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "Input dir       : $INPUT_DIR"
echo "Output dir      : ${OUTPUT_DIR:-(in-place)}"
echo "Chunk           : $CHUNK_ID / $TOTAL_CHUNKS"
echo "Max rows/shard  : $MAX_ROWS"
echo "Workers         : $WORKERS"
echo ""

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /scratch/project_462000941/members/zihao/OPUS2410/torch25_env/bin/activate

OUTPUT_ARG=""
if [[ -n "$OUTPUT_DIR" ]]; then
    OUTPUT_ARG="--output-dir $OUTPUT_DIR"
fi

srun python ./split_parquet.py \
    "$INPUT_DIR" \
    $OUTPUT_ARG \
    --max-rows "$MAX_ROWS" \
    --workers  "$WORKERS" \
    --chunk    "$CHUNK_ID" \
    --total-chunks "$TOTAL_CHUNKS"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo ""
echo "Job ended at    : $(date)"
echo "Duration        : $(date -u -d @${duration} +%T)"
