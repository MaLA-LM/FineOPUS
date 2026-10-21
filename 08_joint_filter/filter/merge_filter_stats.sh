#!/bin/bash
# SLURM worker; one job aggregates all completed pair outputs.
set -euo pipefail
# sbatch runs a spooled copy: the entrypoint must come from the submitter.
JOINT_PYTHON="${1:?Use submit_merge_filter_stats.sh to provide the Python executable}"
JOINT_ENTRYPOINT="${2:?Missing absolute Python entrypoint; use submit_merge_filter_stats.sh}"
shift 2
if [[ "$JOINT_ENTRYPOINT" != /* || ! -f "$JOINT_ENTRYPOINT" ]]; then
    echo "Invalid Python entrypoint: $JOINT_ENTRYPOINT" >&2
    exit 2
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
exec "$JOINT_PYTHON" -u "$JOINT_ENTRYPOINT" "$@"
