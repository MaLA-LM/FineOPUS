#!/bin/bash
#SBATCH --job-name=aggregate_ensembled_relid
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=64G
#SBATCH --account=project_462000964

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate


INPUT_ROOT="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-TAR-V2"
OUTPUT_ROOT="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-FIRST-V2"
EXTRACT_TMP_DIR="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-TMP"

python ./aggregate_ensembled_relid.py \
  --input_root "$INPUT_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --extract_tmp_dir "$EXTRACT_TMP_DIR"
#   --dry_run


end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
