#!/bin/bash
#SBATCH --job-name=hplt2
#SBATCH --account=project_462000675
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=slurmlogs/%x_%j.out.log
#SBATCH --error=slurmlogs/%x_%j.err.log

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load required modules on LUMI
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


BASE_DIR="/scratch/project_462000964/source_data/monolingual/HPLT2.0_cleaned"
language_code="${1:-zho_Hans}"
OUTPUT_DIR="/scratch/project_462000964/FineOPUS/ablation_data/HPLT2.0_cleaned/${LANG_CODE}"
MAX_LINES=20000000

mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Base directory: $BASE_DIR"
echo "  Language code: $LANG_CODE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Max lines per file: $MAX_LINES"
echo "----------------------------------------"

echo "Starting JSONL to Parquet conversion..."
python -u hplt2_lang.py \
    --base-dir "$BASE_DIR" \
    --lang-code "$LANG_CODE" \
    --output-dir "$OUTPUT_DIR" \
    --max-lines "$MAX_LINES"

echo "End Time: $(date)"
echo "Job completed successfully!"