#!/bin/bash
#SBATCH --job-name=fineopus-move-pairs
#SBATCH --output=../logs/fineopus-move-pairs/%x_%j.out
#SBATCH --error=../logs/fineopus-move-pairs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000964

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../.venv/bin/activate

EXCEL_FILE="./stats/fineopus-original-ReLID-ENSEMBLED-stats.xlsx"
SOURCE_DIR=""
OUTPUT_DIR=""
THRESHOLD=10_000


srun python ./move_pairs.py \
    --excel_file "$EXCEL_FILE" \
    --source_dir "$SOURCE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --threshold "$THRESHOLD" \
    # --dry-run


end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
