#!/bin/bash
set -euo pipefail
# sbatch runs a spooled copy: the entrypoint must come from the submitter.
JOINT_PYTHON="${1:?Use submit_apply_thresholds.sh to provide the Python executable}"
JOINT_ENTRYPOINT="${2:?Missing absolute Python entrypoint; use submit_apply_thresholds.sh}"
shift 2
if [[ "$JOINT_ENTRYPOINT" != /* || ! -f "$JOINT_ENTRYPOINT" ]]; then
    echo "Invalid Python entrypoint: $JOINT_ENTRYPOINT" >&2
    exit 2
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec "$JOINT_PYTHON" -u "$JOINT_ENTRYPOINT" "$@" \
    --n_chunks "${SLURM_ARRAY_TASK_COUNT:-1}" --chunk_id "${SLURM_ARRAY_TASK_ID:-0}" --skip_existing
