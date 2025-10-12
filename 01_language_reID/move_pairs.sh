#!/bin/bash
#SBATCH --job-name=move_pairs
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000941

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

EXCEL_FILE="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-V2-stats.xlsx"
SOURCE_DIR="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-V2"
OUTPUT_DIR="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-V2-DROPPED"
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
