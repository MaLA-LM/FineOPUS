#!/bin/bash
#SBATCH --job-name=fineopus-merge-pair
#SBATCH --output=../logs/fineopus-merge-pair/%x_%j.out
#SBATCH --error=../logs/fineopus-merge-pair/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --account=project_462000964

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../.venv/bin/activate

export CPU=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$CPU
export MKL_NUM_THREADS=$CPU
export OPENBLAS_NUM_THREADS=$CPU
export NUMEXPR_MAX_THREADS=$CPU
export ARROW_NUM_THREADS=$CPU
export MALLOC_ARENA_MAX=2 


ROOT1="/scratch/project_462000941/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED-MIX-64-127"
ROOT2="/scratch/project_462000964/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED-MIX-128-191"
OUTPUT_ROOT="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED-MIX-64-191"
MAX_ROWS_PER_FILE=100_000_000


srun --cpu-bind=cores python ./merge_pair.py \
    --root1 "$ROOT1" \
    --root2 "$ROOT2" \
    --out-root "$OUTPUT_ROOT" \
    --max-rows-per-file "$MAX_ROWS_PER_FILE"

status=$?

if [ $status -ne 0 ]; then
  echo "Processing failed (exit code $status); skipping removal." >&2

  end_time=$(date +%s)
  echo "Job ended at: $(date)"

  duration=$((end_time - start_time))
  echo "Job duration: $(date -u -d @${duration} +%T)"
  exit $status
fi

echo "Removing $ROOT1 and $ROOT2"
rm -rf "$ROOT1"
rm -rf "$ROOT2"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
