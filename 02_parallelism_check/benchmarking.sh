#!/bin/bash
#SBATCH --job-name=embed_eval
#SBATCH --output=../logs/embed_eval/%x_%j.out
#SBATCH --error=../logs/embed_eval/%x_%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --account=project_462000941

start_time=$(date +%s)
echo "Job started at: $(date)"

MODEL="jinaai/jina-embeddings-v3"
DATASET="Zihao-Li/FLORES-200"
SPLIT="test"
BATCH_SIZE=16
SOURCE_LANG="eng_Latn"
OUTPUT_DIR="./results"
# Leave empty for all languages, or specify: TARGET_LANGUAGES="fra_Latn deu_Latn spa_Latn"
TARGET_LANGUAGES=""

# Activate environment
module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
# source ../.venv/bin/activate || source .venv/bin/activate

# Show GPU info
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname
elif command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader
fi

# Run benchmarking
if [ -n "$TARGET_LANGUAGES" ]; then
    srun python ./benchmarking.py \
      --dataset_name "$DATASET" \
      --model "$MODEL" \
      --split "$SPLIT" \
      --source_lang "$SOURCE_LANG" \
      --output_dir "$OUTPUT_DIR" \
      --batch_size "$BATCH_SIZE" \
      --target_languages $TARGET_LANGUAGES \
      --skip_processed
else
    srun python ./benchmarking.py \
      --dataset_name "$DATASET" \
      --model "$MODEL" \
      --split "$SPLIT" \
      --source_lang "$SOURCE_LANG" \
      --output_dir "$OUTPUT_DIR" \
      --batch_size "$BATCH_SIZE" \
      --skip_processed
fi


end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job ended at: $(date)"
echo "Duration: $(date -u -d @${duration} +%T)"