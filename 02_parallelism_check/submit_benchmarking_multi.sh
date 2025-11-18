#!/bin/bash

# 1024 Dimensionality Models:
# jinaai/jina-embeddings-v3 Qwen/Qwen3-Embedding-0.6B intfloat/multilingual-e5-large
# 768 Dimensionality Models:
# google/embeddinggemma-300m Alibaba-NLP/gte-multilingual-base
MODELS_LIST=(
    "jinaai/jina-embeddings-v3 Qwen/Qwen3-Embedding-0.6B"
    "jinaai/jina-embeddings-v3 intfloat/multilingual-e5-large"
    "intfloat/multilingual-e5-large Qwen/Qwen3-Embedding-0.6B"

    "google/embeddinggemma-300m Alibaba-NLP/gte-multilingual-base"

    "jinaai/jina-embeddings-v3 google/embeddinggemma-300m"
    "jinaai/jina-embeddings-v3 Alibaba-NLP/gte-multilingual-base"
    "intfloat/multilingual-e5-large google/embeddinggemma-300m"
    "intfloat/multilingual-e5-large Alibaba-NLP/gte-multilingual-base"
    "Qwen/Qwen3-Embedding-0.6B google/embeddinggemma-300m"
    "Qwen/Qwen3-Embedding-0.6B Alibaba-NLP/gte-multilingual-base"

    "jinaai/jina-embeddings-v3 Qwen/Qwen3-Embedding-0.6B intfloat/multilingual-e5-large"
)

DATASETS=(
    "Zihao-Li/FLORES-200"
    "Zihao-Li/NTREX-128"
    "Zihao-Li/BOUQuET"
)

# Fixed parameters
BATCH_SIZE=16
SOURCE_LANG="eng_Latn"
SPLIT="test"
OUTPUT_DIR="./results_multi"
# Leave empty for all languages, or specify: TARGET_LANGUAGES="fra_Latn deu_Latn spa_Latn"
TARGET_LANGUAGES=""
# Set to "true" to enable embedding normalization, or "false" to disable
NORMALIZE_EMBEDDINGS="false"

echo "=========================================="
echo "Submitting benchmarking jobs for ${#MODELS_LIST[@]} models"
echo "=========================================="
echo ""

# Loop through each model and submit a job
for DATASET in "${DATASETS[@]}"; do
    for MODELS in "${MODELS_LIST[@]}"; do
        echo "Preparing job for models: $MODELS"
        
        # Extract model names for job name (keep only part after / for each model)
        MODEL_NAMES=$(echo "$MODELS" | sed 's|[^ ]*/||g' | sed 's/ /_/g')
        JOB_NAME="${MODEL_NAMES}-$(echo "$DATASET" | sed 's/.*\///')"
        
        echo "  Submitting job:"
        echo "    MODELS: $MODELS"
        echo "    DATASET: $DATASET"
        echo "    BATCH_SIZE: $BATCH_SIZE"
        echo "    SOURCE_LANG: $SOURCE_LANG"
        echo "    TARGET_LANGUAGES: ${TARGET_LANGUAGES:-all}"
        echo "    NORMALIZE_EMBEDDINGS: $NORMALIZE_EMBEDDINGS"
        echo "    JOB_NAME: $JOB_NAME"
        
        # Submit the job
        sbatch --job-name="$JOB_NAME" \
            --export=ALL,MODELS="$MODELS",DATASET="$DATASET",BATCH_SIZE="$BATCH_SIZE",SOURCE_LANG="$SOURCE_LANG",SPLIT="$SPLIT",OUTPUT_DIR="$OUTPUT_DIR",TARGET_LANGUAGES="$TARGET_LANGUAGES",NORMALIZE_EMBEDDINGS="$NORMALIZE_EMBEDDINGS" \
            ./benchmarking_multi.sh
        
        echo ""
    done
done
