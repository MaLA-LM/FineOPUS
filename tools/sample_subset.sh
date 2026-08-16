#!/bin/bash
#SBATCH --job-name=fineopus-sample-subset
#SBATCH --output=../logs/fineopus-sample-subset/%x_%j.out
#SBATCH --error=../logs/fineopus-sample-subset/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-04:00:00
#SBATCH --mem=128G
#SBATCH --account=project_462001087

set -euo pipefail

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

INPUT_DIR="/scratch/project_462001249/MaLA-LM/FineOPUS-ReLID"
OUTPUT_DIR="/scratch/project_462001249/MaLA-LM/FineOPUS-ReLID-sample5k"
SAMPLE_SIZE=5000
SEED=42
# Keep workers=1: large pairs (100M+ rows) are memory-heavy even with streaming.
WORKERS=1

# Optional SLURM array chunking:
#   sbatch --array=0-99 sample_subset.sh
CHUNK=${SLURM_ARRAY_TASK_ID:-}
TOTAL_CHUNKS=${SLURM_ARRAY_TASK_COUNT:-}

EXTRA_ARGS=()
if [[ -n "${CHUNK}" && -n "${TOTAL_CHUNKS}" ]]; then
  EXTRA_ARGS+=(--chunk "${CHUNK}" --total_chunks "${TOTAL_CHUNKS}")
fi

python ./sample_subset.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample_size "${SAMPLE_SIZE}" \
  --seed "${SEED}" \
  --workers "${WORKERS}" \
  "${EXTRA_ARGS[@]}"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
