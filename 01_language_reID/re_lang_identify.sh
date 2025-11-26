#!/bin/bash
#SBATCH --job-name=fineopus-reLID-glotlid
#SBATCH --output=../logs/fineopus-reLID-glotlid/%x_%j.out
#SBATCH --error=../logs/fineopus-reLID-glotlid/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --account=project_462000941
#SBATCH --array=0-127

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../.venv/bin/activate

export HF_HOME="/scratch/project_462000941/cache/huggingface"

SOURCE_DIR="/scratch/project_462000941/FineOPUS/fineopus-original"

NUM_PROC=${SLURM_CPUS_PER_TASK:-1}

OUTPUT_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-GlotLID"
MODEL_PATH="/scratch/project_462000941/cache/huggingface/hub/models--cis-lmu--glotlid/snapshots/74cb50b709c9eefe0f790030c6c95c461b4e3b77/model.bin"

# OUTPUT_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-ConLID"
# MODEL_PATH="/scratch/project_462000941/cache/huggingface/hub/models--epfl-nlp--ConLID/snapshots/59e1e21e2301cb87f1c244bff71579a17eafaa42"

FILELIST="./filelists/fineopus-original-filelists-128-shard/filelist_${SLURM_ARRAY_TASK_ID}.txt"

python ./re_lang_identify.py \
  --source_dir "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_proc "$NUM_PROC" \
  --model_path "$MODEL_PATH" \
  --filelist "$FILELIST"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
