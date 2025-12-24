#!/bin/bash

MODELS=(
    "jinaai/jina-embeddings-v3"
    # "google/embeddinggemma-300m"
    # "Qwen/Qwen3-Embedding-0.6B"
    # "intfloat/multilingual-e5-large"
    # "Alibaba-NLP/gte-multilingual-base"
)

DATASETS=(
    "Zihao-Li/FLORES-200"
    # "Zihao-Li/NTREX-128"
    # "Zihao-Li/BOUQuET"
)

# Fixed parameters
BATCH_SIZE=16
SPLIT="test"
OUTPUT_DIR="./results"
# Path to the language pairs file (each line: source_lang-target_lang, e.g., zho_Hans-eng_Latn)
LANGUAGE_PAIRS_FILE="./language_pairs.txt"
# Set to "true" to enable embedding normalization, or "false" to disable
NORMALIZE_EMBEDDINGS="false"

echo "=========================================="
echo "Submitting benchmarking jobs for ${#MODELS[@]} models"
echo "=========================================="
echo "Language pairs file: $LANGUAGE_PAIRS_FILE"
echo ""

# Loop through each model and submit a job
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "Preparing job for model: $MODEL"
        
        # Extract model name for job name (keep only part after /)
        JOB_NAME="$(echo "$MODEL" | sed 's/.*\///')-$(echo "$DATASET" | sed 's/.*\///')"
        
        echo "  Submitting job:"
        echo "    MODEL: $MODEL"
        echo "    DATASET: $DATASET"
        echo "    BATCH_SIZE: $BATCH_SIZE"
        echo "    LANGUAGE_PAIRS_FILE: $LANGUAGE_PAIRS_FILE"
        echo "    NORMALIZE_EMBEDDINGS: $NORMALIZE_EMBEDDINGS"
        echo "    JOB_NAME: $JOB_NAME"
        
        # Submit the job
        sbatch --job-name="$JOB_NAME" \
            --export=ALL,MODEL="$MODEL",DATASET="$DATASET",BATCH_SIZE="$BATCH_SIZE",SPLIT="$SPLIT",OUTPUT_DIR="$OUTPUT_DIR",LANGUAGE_PAIRS_FILE="$LANGUAGE_PAIRS_FILE",NORMALIZE_EMBEDDINGS="$NORMALIZE_EMBEDDINGS" \
            ./benchmarking.sh
        
        echo ""
    done
done
