#!/bin/bash
#SBATCH --job-name=similarity_scoring
#SBATCH --output=../logs/similarity_scoring/%x_%A_%a.out
#SBATCH --error=../logs/similarity_scoring/%x_%A_%a.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --account=project_462000964

# ---------------------------------------------------------------------------
# Required environment variables (set via submit_all.sh):
#   MODEL          - HuggingFace model name
#   TOTAL_CHUNKS   - total number of array jobs for this model
#   INPUT_DIR      - root dir of FineOPUS-Filtered-Stage2
#   OUTPUT_DIR     - root dir for scored output parquets
#   MODEL_PAIRS_JSON - path to model_to_language_pairs.json
#   BATCH_SIZE     - encoding batch size (default: 64)
#
# The SLURM array index ($SLURM_ARRAY_TASK_ID) is used as chunk_id.
# ---------------------------------------------------------------------------

start_time=$(date +%s)
echo "Job started at: $(date)"
echo "Array task ID : $SLURM_ARRAY_TASK_ID"
echo "Model         : $MODEL"
echo "Total chunks  : $TOTAL_CHUNKS"

# Defaults
INPUT_DIR="${INPUT_DIR:-/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_462001069/FineOPUS/intermediate/FineOPUS-Filtered-Stage2-Scored}"
MODEL_PAIRS_JSON="${MODEL_PAIRS_JSON:-/scratch/project_462000941/members/zihao/OPUS2410/02_parallelism_check/model_to_language_pairs.json}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Activate environment
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

# Show GPU info
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname
elif command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader
fi

srun python /scratch/project_462000941/members/zihao/OPUS2410/02_parallelism_check/compute_similarity.py \
    --model "$MODEL" \
    --model_pairs_json "$MODEL_PAIRS_JSON" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_id "$SLURM_ARRAY_TASK_ID" \
    --total_chunks "$TOTAL_CHUNKS" \
    --batch_size "$BATCH_SIZE"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job ended at: $(date)"
echo "Duration: $(date -u -d @${duration} +%T)"
