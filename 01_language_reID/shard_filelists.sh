#!/bin/bash

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

SOURCE_DIR=""
OUTPUT_DIR=""
NUM_SPLITS=512
USE_RELPATH=true

if [ "$USE_RELPATH" = true ]; then
    python shard_filelists.py \
        --source_dir "$SOURCE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_splits "$NUM_SPLITS" \
        --relpath
else
    python shard_filelists.py \
        --source_dir "$SOURCE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_splits "$NUM_SPLITS"
fi

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"