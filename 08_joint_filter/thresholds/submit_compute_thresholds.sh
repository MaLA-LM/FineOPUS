#!/bin/bash
# Submit one job to compute all requested threshold profiles.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/../submit_slurm.py" thresholds "$@"
