#!/usr/bin/env bash
#SBATCH --job-name=aggregate-ft-mt
#SBATCH --partition=small
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
MANIFEST="${1:?Pass the FT tasks.json manifest}"
SCRIPT_DIR="${2:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PLAN_PYTHON="${FT_PLAN_PYTHON:-/usr/bin/python3.11}"
results_root=$("$PLAN_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["results_root"])' "$MANIFEST")
exec "$PLAN_PYTHON" "$SCRIPT_DIR/../aggregate_mt_results.py" --results-root "$results_root"
