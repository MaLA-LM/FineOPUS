#!/bin/bash
#SBATCH --job-name=fineopus-aggregate-ensembled-relid-for-mix
#SBATCH --output=../logs/fineopus-aggregate-ensembled-relid-for-mix/%x_%j.out
#SBATCH --error=../logs/fineopus-aggregate-ensembled-relid-for-mix/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --account=project_462000964

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../.venv/bin/activate

export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Optional: specify the range of tar files to process
START_IDX=64
END_IDX=127

INPUT_ROOT="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED-TAR"
OUTPUT_ROOT="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED-MIX-${START_IDX}-${END_IDX}"
STAGING_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED-TMP"


python ./aggregate_ensembled_relid_for_mix.py \
    --input_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --max_rows_per_part 1_000_000 \
    --compression zstd \
    --staging_dir "$STAGING_DIR" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX"


end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
