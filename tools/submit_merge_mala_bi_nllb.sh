#!/bin/bash
#SBATCH --job-name=merge_mala_nllb
#SBATCH --account=project_462001087
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/project_462001427/FineOPUS/logs/merge_mala_nllb_%j.out
#SBATCH --error=/scratch/project_462001427/FineOPUS/logs/merge_mala_nllb_%j.err
#SBATCH --signal=B:TERM@120

set -euo pipefail

ROOT=/scratch/project_462001427/FineOPUS
echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date --iso-8601=seconds)"
echo "Canonical text columns: source_text, target_text"
export ARROW_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun "${ROOT}/.venv/bin/python" -u "${ROOT}/tools/merge_mala_bi_nllb.py" \
  --mala-root /scratch/project_462001069/mala-bilingual-translation-corpus \
  --nllb-root /scratch/project_462001069/nllb/nllb-conversion \
  --output-root /scratch/project_462001069/mala-bi-nllb \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --batch-size 250000 \
  --state-log "${ROOT}/logs/merge_mala_bi_nllb_state.jsonl"

echo "Job ${SLURM_JOB_ID} finished at $(date --iso-8601=seconds)"
