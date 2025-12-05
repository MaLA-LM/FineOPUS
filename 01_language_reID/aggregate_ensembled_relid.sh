#!/bin/bash
#SBATCH --job-name=fineopus-aggregate-ensembled-relid
#SBATCH --output=../logs/fineopus-aggregate-ensembled-relid/%x_%j.out
#SBATCH --error=../logs/fineopus-aggregate-ensembled-relid/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --account=project_462000964

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../.venv/bin/activate


INPUT_ROOT="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED-TAR"
OUTPUT_ROOT="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED-FIRST"
EXTRACT_TMP_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED-TMP"

python ./aggregate_ensembled_relid.py \
  --input_root "$INPUT_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --extract_tmp_dir "$EXTRACT_TMP_DIR"
#   --dry_run


end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
