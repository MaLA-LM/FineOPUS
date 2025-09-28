#!/bin/bash
#SBATCH --job-name=count_parquet_rows_per_pair
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000941

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate


ROOT_DIR="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-V2"
OUTPUT_FILE="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-V2-stats.xlsx"

WORKERS=${SLURM_CPUS_PER_TASK:-$(nproc)}

python ./count_parquet_rows_per_pair.py \
  --root_dir "$ROOT_DIR" \
  --output_file "$OUTPUT_FILE" \
  --workers "$WORKERS"


end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
