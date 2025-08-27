#!/bin/bash
#SBATCH --job-name=mala-opus-conf-stats-conlid
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-06:00:00
#SBATCH --mem=64G
#SBATCH --account=project_462000675
#SBATCH --array=0-127

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

export HF_HOME="/scratch/project_462000964/cache/huggingface"

# SOURCE_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-GlotLID"
# OUTPUT_DIR="./mala-opus-dedup-2410-ReLID-by-GlotLID-conf-stats"
# FILELIST="./mala-opus-dedup-2410-ReLID-by-GlotLID-filelists/filelist_${SLURM_ARRAY_TASK_ID}.txt"

# SOURCE_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-ConLID"
# OUTPUT_DIR="./mala-opus-dedup-2410-ReLID-by-ConLID-conf-stats"
# FILELIST="./mala-opus-dedup-2410-ReLID-by-ConLID-filelists/filelist_${SLURM_ARRAY_TASK_ID}.txt"

python ./get_conf_stats.py \
  --source_dir "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --filelist "$FILELIST" \
  --job_id "${SLURM_ARRAY_TASK_ID}"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
