#!/bin/bash

MODELS=(
    "jinaai/jina-embeddings-v3"
    "google/embeddinggemma-300m"
    "Qwen/Qwen3-Embedding-0.6B"
    "intfloat/multilingual-e5-large"
    "Alibaba-NLP/gte-multilingual-base"
)

# Fixed parameters
DATASET="Zihao-Li/FLORES-200"
BATCH_SIZE=16
SOURCE_LANG="eng_Latn"
SPLIT="test"
OUTPUT_DIR="./results"
# Leave empty for all languages, or specify: TARGET_LANGUAGES="fra_Latn deu_Latn spa_Latn"
TARGET_LANGUAGES=""

echo "=========================================="
echo "Submitting benchmarking jobs for ${#MODELS[@]} models"
echo "=========================================="
echo ""

# Loop through each model and submit a job
for MODEL in "${MODELS[@]}"; do
    echo "Preparing job for model: $MODEL"
    
    # Extract model name for job name (replace / with -)
    JOB_NAME=$(echo "$MODEL" | sed 's/\//-/g')
    
    echo "  Submitting job:"
    echo "    MODEL: $MODEL"
    echo "    DATASET: $DATASET"
    echo "    BATCH_SIZE: $BATCH_SIZE"
    echo "    SOURCE_LANG: $SOURCE_LANG"
    echo "    TARGET_LANGUAGES: ${TARGET_LANGUAGES:-all}"
    echo "    JOB_NAME: $JOB_NAME"
    
    # Submit the job
    sbatch --job-name="$JOB_NAME" \
           --export=ALL,MODEL="$MODEL",DATASET="$DATASET",BATCH_SIZE="$BATCH_SIZE",SOURCE_LANG="$SOURCE_LANG",SPLIT="$SPLIT",OUTPUT_DIR="$OUTPUT_DIR",TARGET_LANGUAGES="$TARGET_LANGUAGES" \
           ./benchmarking.sh
    
    echo ""
done
