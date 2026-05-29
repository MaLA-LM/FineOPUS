#!/bin/bash
#SBATCH --job-name=fineopus-count-parquet-rows-per-pair
#SBATCH --output=../logs/fineopus-count-parquet-rows-per-pair/%x_%j.out
#SBATCH --error=../logs/fineopus-count-parquet-rows-per-pair/%x_%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:30:00
#SBATCH --mem=32G
#SBATCH --account=project_462000964

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../torch25_env/bin/activate


ROOT_DIR=""
OUTPUT_FILE="./xxx.xlsx"

WORKERS=${SLURM_CPUS_PER_TASK:-$(nproc)}


python ./count_parquet_rows_per_pair.py \
  --root_dir "$ROOT_DIR" \
  --output_file "$OUTPUT_FILE" \
  --workers "$WORKERS"


end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
