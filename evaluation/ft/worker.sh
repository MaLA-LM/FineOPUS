#!/usr/bin/env bash
#SBATCH --job-name=eval-ft-mt
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --partition=small-g
#SBATCH --time=3-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_465002530

set -euo pipefail
PREFLIGHT=0
if [[ "${1:-}" == --preflight ]]; then PREFLIGHT=1; shift; fi
MANIFEST="${1:?Task manifest is required}"
# Explicit source directory survives Slurm's private script copy.
SCRIPT_DIR="${2:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
TASK_ID="${3:-${SLURM_ARRAY_TASK_ID:-}}"
module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.7
source "$SCRIPT_DIR/../eval_env/bin/activate"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/project_462001427/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [[ "$PREFLIGHT" == 1 ]]; then
    exec python "$SCRIPT_DIR/workflow.py" preflight --manifest "$MANIFEST"
fi
mkdir -p "$SCRIPT_DIR/tmp"
JOB_TMP=$(mktemp -d "$SCRIPT_DIR/tmp/eval.${SLURM_JOB_ID:-local}.XXXXXX")
trap 'rm -rf -- "$JOB_TMP"' EXIT
export HF_MODULES_CACHE="$JOB_TMP/hf_modules"
export VLLM_CACHE_ROOT="$JOB_TMP/vllm"
export MASTER_ADDR=127.0.0.1
MASTER_PORT=$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    print(sock.getsockname()[1])
PY
)
export MASTER_PORT
python "$SCRIPT_DIR/workflow.py" worker --manifest "$MANIFEST" --task-id "${TASK_ID:?Model task ID is required}"
