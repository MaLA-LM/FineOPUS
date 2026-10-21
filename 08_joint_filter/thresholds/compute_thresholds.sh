#!/bin/bash
# SLURM worker; no array/chunk flags: all pairs share one global token budget.
set -euo pipefail
# sbatch runs a spooled copy: the entrypoint must come from the submitter.
JOINT_PYTHON="${1:?Use submit_compute_thresholds.sh to provide the Python executable}"
JOINT_ENTRYPOINT="${2:?Missing absolute Python entrypoint; use submit_compute_thresholds.sh}"
shift 2
if [[ "$JOINT_ENTRYPOINT" != /* || ! -f "$JOINT_ENTRYPOINT" ]]; then
    echo "Invalid Python entrypoint: $JOINT_ENTRYPOINT" >&2
    exit 2
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
exec "$JOINT_PYTHON" -u "$JOINT_ENTRYPOINT" "$@"
