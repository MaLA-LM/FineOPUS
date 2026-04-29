#!/bin/bash
# SLURM worker script for collect_gold_stats.py
#
# Required env vars (set via submit_collect_gold_stats.sh):
#   MODEL            HF model name
#   BEST_MODEL_CSV   path to best_model_per_lang_pair_*.csv
#   OUTPUT_CSV       path to append gold stats to
#   BATCH_SIZE       encoding batch size
# ---------------------------------------------------------------------------
set -euo pipefail

start_time=$(date +%s)
echo "Job started at   : $(date)"
echo "Model            : $MODEL"
echo "Best-model CSV   : $BEST_MODEL_CSV"
echo "Output CSV       : $OUTPUT_CSV"
echo "Batch size       : $BATCH_SIZE"

# Activate environment following the same convention as compute_similarity.sh
if [[ "$MODEL" == "Alibaba-NLP/gte-multilingual-base" || "$MODEL" == "jinaai/jina-embeddings-v3" ]]; then
    module use /appl/local/csc/modulefiles/
    module load pytorch/2.5
    source /scratch/project_462000941/members/zihao/OPUS2410/torch25_env/bin/activate
else
    module purge
    module load LUMI/25.09
    module load partition/G
    module load rocm/6.4.4
    source /scratch/project_462000941/members/zihao/OPUS2410/.venv/bin/activate
fi

if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname
elif command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader
fi

python3 ./collect_gold_stats.py \
    --model "$MODEL" \
    --best_model_csv "$BEST_MODEL_CSV" \
    --output "$OUTPUT_CSV" \
    --batch_size "$BATCH_SIZE" \
    --skip_existing

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job ended at     : $(date)"
echo "Duration         : $(date -u -d @${duration} +%T)"
