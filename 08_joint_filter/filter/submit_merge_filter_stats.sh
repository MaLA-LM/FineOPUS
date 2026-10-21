#!/bin/bash
# Submit one job to validate completed pairs and merge their statistics.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/../submit_slurm.py" merge "$@"
