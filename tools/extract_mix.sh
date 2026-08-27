#!/bin/bash
#SBATCH --job-name=fineopus-extract-mix
#SBATCH --output=../logs/fineopus-extract-mix/%x_%A_%a.out
#SBATCH --error=../logs/fineopus-extract-mix/%x_%A_%a.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --account=project_462001087
#
# Extract the token-budgeted multilingual parallel mix.
#
# Submit (one array per dataset, chunked over the target pairs):
#   NUM_CHUNKS=32 ./extract_mix.sh NLLB
#   NUM_CHUNKS=32 ./extract_mix.sh MaLA_Bi
#   NUM_CHUNKS=32 ./extract_mix.sh MaLA_Bi_NLLB
#   NUM_CHUNKS=32 ./extract_mix.sh FineOPUS-Filtered-Stage5
#   NUM_CHUNKS=32 ./extract_mix.sh FineOPUS-Filtered-Stage4
#   QUALITY_PRIORITY=1 NUM_CHUNKS=32 ./extract_mix.sh FineOPUS-Filtered-Stage5
#   QUALITY_PRIORITY=1 NUM_CHUNKS=32 ./extract_mix.sh FineOPUS-Filtered-Stage4
#   NUM_CHUNKS=32 ./extract_mix.sh FineOPUS-Filtered-Stage3
#   NUM_CHUNKS=32 ./extract_mix.sh FineOPUS-Filtered-Stage2
#   NUM_CHUNKS=32 ./extract_mix.sh FineOPUS-Filtered-Stage1
#
# Worker mode is entered automatically when SLURM sets SLURM_ARRAY_TASK_ID.

set -euo pipefail

# Fixed location of the tools directory (BASH_SOURCE points at the SLURM spool
# copy in worker mode, so it cannot be used to locate extract_mix.py).
TOOLS_DIR="${TOOLS_DIR:-/scratch/project_462001427/FineOPUS/tools}"

DATASET="${1:?Usage: ./extract_mix.sh <NLLB|MaLA_Bi|MaLA_Bi_NLLB|FineOPUS-Filtered-Stage5|FineOPUS-Filtered-Stage4|FineOPUS-Filtered-Stage3|FineOPUS-Filtered-Stage2|FineOPUS-Filtered-Stage1>}"
NUM_CHUNKS="${NUM_CHUNKS:-32}"
SEED="${SEED:-42}"
CSV="${CSV:-/scratch/project_462001427/FineOPUS/slm_from_scratch/data/parallel/multilingual_mix/Multilingual-Mix-Parallel.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/project_462001427/FineOPUS/slm_from_scratch/data/parallel/multilingual_mix/_parquet}"
QUALITY_PRIORITY="${QUALITY_PRIORITY:-0}"
QUALITY_STRENGTH="${QUALITY_STRENGTH:-6.0}"

case "$QUALITY_PRIORITY" in
    0) QUALITY_ARGS=() ;;
    1) QUALITY_ARGS=(--quality_priority --quality_strength "$QUALITY_STRENGTH") ;;
    *) echo "QUALITY_PRIORITY must be 0 or 1 (got: $QUALITY_PRIORITY)" >&2; exit 2 ;;
esac

if [ "$QUALITY_PRIORITY" = "1" ]; then
    case "$DATASET" in
        FineOPUS-Filtered-Stage4|FineOPUS-Filtered-Stage5) ;;
        *)
            echo "QUALITY_PRIORITY=1 only supports FineOPUS-Filtered-Stage4/Stage5" >&2
            exit 2
            ;;
    esac
fi

# --------------------------- Worker mode ---------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    module purge
    module use /appl/local/csc/modulefiles/
    module load pytorch
    source /scratch/project_462001050/ibrahiam/envs/transformers-latest/bin/activate

    echo "Worker chunk ${SLURM_ARRAY_TASK_ID}/${NUM_CHUNKS} dataset=${DATASET} $(date)"
    srun --input=none python3 "$TOOLS_DIR/extract_mix.py" \
        --csv "$CSV" \
        --dataset "$DATASET" \
        --output_root "$OUTPUT_ROOT" \
        --seed "$SEED" \
        --chunk "$SLURM_ARRAY_TASK_ID" \
        --total_chunks "$NUM_CHUNKS" \
        "${QUALITY_ARGS[@]}"
    echo "Finished chunk ${SLURM_ARRAY_TASK_ID} at $(date)"
    exit 0
fi

# --------------------------- Submit mode ---------------------------
mkdir -p "$TOOLS_DIR/../logs/fineopus-extract-mix"
LAST=$((NUM_CHUNKS - 1))
echo "Submitting ${NUM_CHUNKS} array jobs for dataset ${DATASET}."
cd "$TOOLS_DIR"
sbatch --array=0-"$LAST" \
       --export=ALL,NUM_CHUNKS="$NUM_CHUNKS",SEED="$SEED",CSV="$CSV",OUTPUT_ROOT="$OUTPUT_ROOT",TOOLS_DIR="$TOOLS_DIR",QUALITY_PRIORITY="$QUALITY_PRIORITY",QUALITY_STRENGTH="$QUALITY_STRENGTH" \
       extract_mix.sh "$DATASET"
