#!/bin/bash
#SBATCH --job-name=benchmarking
#SBATCH --output=../logs/embed_eval/%x_%j.out
#SBATCH --error=../logs/embed_eval/%x_%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --account=project_462000941

start_time=$(date +%s)
echo "Job started at: $(date)"


MODEL="${MODEL}"
DATASET="${DATASET:-Zihao-Li/FLORES-200}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SOURCE_LANG="${SOURCE_LANG:-eng_Latn}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
TARGET_LANGUAGES="${TARGET_LANGUAGES:-}"
NORMALIZE_EMBEDDINGS="${NORMALIZE_EMBEDDINGS:-false}"

# Activate environment
module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../.venv/bin/activate

# Show GPU info
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname
elif command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader
fi

# Build base command
BASE_CMD="srun python ./benchmarking.py \
  --dataset_name \"$DATASET\" \
  --model \"$MODEL\" \
  --split \"$SPLIT\" \
  --source_lang \"$SOURCE_LANG\" \
  --output_dir \"$OUTPUT_DIR\" \
  --batch_size \"$BATCH_SIZE\" \
  --skip_processed"

# Add target languages if specified
if [ -n "$TARGET_LANGUAGES" ]; then
    BASE_CMD="$BASE_CMD --target_languages $TARGET_LANGUAGES"
fi

# Add normalize_embeddings flag if set to true
if [ "$NORMALIZE_EMBEDDINGS" = "true" ]; then
    BASE_CMD="$BASE_CMD --normalize_embeddings"
fi

# Run benchmarking
eval $BASE_CMD


end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job ended at: $(date)"
echo "Duration: $(date -u -d @${duration} +%T)"