#!/bin/bash
#SBATCH --job-name=fineopus-get-conf-stats-glotlid
#SBATCH --output=../logs/fineopus-get-conf-stats-glotlid/%x_%j.out
#SBATCH --error=../logs/fineopus-get-conf-stats-glotlid/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --account=project_462000964
#SBATCH --array=0-127

start_time=$(date +%s)
echo "Job started at: $(date)"

# Create status log directory
STATUS_LOG_DIR="../logs/${SLURM_JOB_NAME}/status"
mkdir -p "$STATUS_LOG_DIR"
STATUS_LOG="$STATUS_LOG_DIR/task_status.log"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../.venv/bin/activate

TEMP_CACHE_DIR="/scratch/project_462000941/cache/huggingface/tmp/hf_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TEMP_CACHE_DIR"
echo "Temporary cache directory: $TEMP_CACHE_DIR"

export HF_HOME="$TEMP_CACHE_DIR"

SOURCE_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-GlotLID"
OUTPUT_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-GlotLID-conf-stats"
FILELIST="./filelists/fineopus-original-ReLID-by-GlotLID-filelists-256-shard/filelist_${SLURM_ARRAY_TASK_ID}.txt"

# SOURCE_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-ConLID"
# OUTPUT_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-ConLID-conf-stats"
# FILELIST="./filelists/fineopus-original-ReLID-by-ConLID-filelists-256-shard/filelist_${SLURM_ARRAY_TASK_ID}.txt"

# Log task start
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTED - Task ${SLURM_ARRAY_TASK_ID} (JobID: ${SLURM_JOB_ID}) - Filelist: $FILELIST" >> "$STATUS_LOG"

python ./get_conf_stats.py \
  --source_dir "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --filelist "$FILELIST" \
  --job_id "${SLURM_ARRAY_TASK_ID}"

# Capture Python script exit code immediately
PYTHON_EXIT_CODE=$?

if [ -d "$TEMP_CACHE_DIR" ]; then
  CACHE_SIZE=$(du -sh "$TEMP_CACHE_DIR" 2>/dev/null | cut -f1)
  echo "Cleaning up temporary cache: $TEMP_CACHE_DIR (Size: $CACHE_SIZE)"
  rm -rf "$TEMP_CACHE_DIR"
  echo "Cache directory deleted"
fi

# Log task completion status
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS - Task ${SLURM_ARRAY_TASK_ID} (JobID: ${SLURM_JOB_ID}) - Filelist: $FILELIST" >> "$STATUS_LOG"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED - Task ${SLURM_ARRAY_TASK_ID} (JobID: ${SLURM_JOB_ID}) - Filelist: $FILELIST" >> "$STATUS_LOG"
  exit 1
fi

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
