#!/usr/bin/env bash
#SBATCH --job-name=aggregate-mt
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --partition=small
#SBATCH --time=0-02:00:00
#SBATCH --account=project_465002530
#SBATCH --output=logs/rescore/%x_%j.out
#SBATCH --error=logs/rescore/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/aggregate_mt_results.py" ]]; then
    SCRIPT_DIR=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.7
source "$SCRIPT_DIR/eval_env/bin/activate"
export PYTHONNOUSERSITE=1

python "$SCRIPT_DIR/aggregate_mt_results.py" "$@"
