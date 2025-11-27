#!/bin/bash
#SBATCH --job-name=fineopus-get-conf-stats-glotlid
#SBATCH --output=../logs/fineopus-get-conf-stats-glotlid/%x_%j.out
#SBATCH --error=../logs/fineopus-get-conf-stats-glotlid/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --account=project_462000941
#SBATCH --array=0-127

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../.venv/bin/activate

export HF_HOME="/scratch/project_462000941/cache/huggingface"

SOURCE_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-GlotLID"
OUTPUT_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-GlotLID-conf-stats"
FILELIST="./filelists/fineopus-original-ReLID-by-GlotLID-filelists-128-shard/filelist_${SLURM_ARRAY_TASK_ID}.txt"

# SOURCE_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-ConLID"
# OUTPUT_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-ConLID-conf-stats"
# FILELIST="./filelists/fineopus-original-ReLID-by-ConLID-filelists-128-shard/filelist_${SLURM_ARRAY_TASK_ID}.txt"

python ./get_conf_stats.py \
  --source_dir "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --filelist "$FILELIST" \
  --job_id "${SLURM_ARRAY_TASK_ID}"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
