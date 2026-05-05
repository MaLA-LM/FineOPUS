#!/usr/bin/env bash
#
# Outer launcher: one OPUS queue worker per GCD on a whole LUMI-G node
# (standard-g partition). Each array task allocates one full node
# (4x MI250X = 8 GCDs, ~63 cores, 512 GB) and spawns 8 worker processes
# via srun, one per GCD.
#
# Companion to run_worker.sh, which uses small-g (1 GCD per array task).
# The two scripts coexist; this one is intended for high-throughput
# backfill of large queues.
#
# Reads MODEL / MANIFEST_ROOT / BUILD_TAG / TRACE_ROOT / OUTPUT_BASE / OPUS_ROOT / runtime knobs from the
# environment (exported by submit_array_standard_g.sh).
#
#SBATCH --job-name=opus_worker_g
#SBATCH --account=project_462001249
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --output=/scratch/project_462001050/opus_qe/logs/%x-%A_%a.out
#SBATCH --error=/scratch/project_462001050/opus_qe/logs/%x-%A_%a.err

set -euo pipefail

: "${MODEL:?MODEL is required}"
: "${MANIFEST_ROOT:?MANIFEST_ROOT is required}"
: "${BUILD_TAG:?BUILD_TAG is required}"
: "${TRACE_ROOT:?TRACE_ROOT is required}"
: "${OUTPUT_BASE:?OUTPUT_BASE is required}"

PLATFORM="$(printf '%s' "${PLATFORM:-lumi}" | tr '[:upper:]' '[:lower:]')"
if [ "$PLATFORM" != "lumi" ]; then
    echo "ERROR: standard-g launcher only supports PLATFORM=lumi (got '$PLATFORM')." >&2
    exit 1
fi

WORKDIR="${WORKDIR:-/projappl/project_462001050/members/ibrahiam/05_quality_estimation}"
mkdir -p "${WORKDIR}/logs"
cd "$WORKDIR"

resolve_task_script() {
    local candidate_dir candidate_path
    for candidate_dir in \
        "${OPUS_STANDARD_G_SCRIPT_DIR:-}" \
        "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" \
        "${WORKDIR}/scripts/opus"
    do
        [ -n "$candidate_dir" ] || continue
        candidate_path="${candidate_dir}/run_worker_standard_g_task.sh"
        if [ -r "$candidate_path" ]; then
            printf '%s\n' "$candidate_path"
            return 0
        fi
    done
    return 1
}

# Under sbatch, BASH_SOURCE[0] often points at Slurm's spool copy of this
# wrapper. Resolve the companion task script from the submitter-exported
# script dir first, then fall back to repo-local locations.
TASK_SCRIPT="$(resolve_task_script || true)"
if [ -z "$TASK_SCRIPT" ]; then
    echo "ERROR: per-task script not found or not readable." >&2
    echo "Checked: ${OPUS_STANDARD_G_SCRIPT_DIR:-<unset>}, $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd), ${WORKDIR}/scripts/opus" >&2
    exit 1
fi

echo "=================================="
echo "OPUS standard-g launcher: model=$MODEL"
echo "MANIFEST_ROOT=$MANIFEST_ROOT  BUILD_TAG=$BUILD_TAG  TRACE_ROOT=$TRACE_ROOT"
echo "OUTPUT_BASE=$OUTPUT_BASE"
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Array task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: $(hostname)"
echo "ntasks-per-node=${SLURM_NTASKS_PER_NODE:-?} cpus-per-task=${SLURM_CPUS_PER_TASK:-?} gpus-per-node=${SLURM_GPUS_PER_NODE:-?}"
echo "Start time: $(date)"
echo "=================================="

# srun spawns 8 task-script instances on this node. Each task gets its
# node-local rank in SLURM_LOCALID=0..7, and the task script uses the
# documented LUMI full-node wrapper pattern to select one GCD per worker.
#
# The cpu-bind mask below keeps each task's 7 CPU cores NUMA-affine to its
# assigned GCD.
#
#   LOCALID  GCD   CCD   cores   mask_cpu
#   0        0     6     49-55   0xfe000000000000
#   1        1     7     57-63   0xfe00000000000000
#   2        2     2     17-23   0xfe0000
#   3        3     3     25-31   0xfe000000
#   4        4     0     1-7     0xfe
#   5        5     1     9-15    0xfe00
#   6        6     4     33-39   0xfe00000000
#   7        7     5     41-47   0xfe0000000000
#
# Core 0 of each CCD is left unassigned (LUMI low-noise reservation).
# Ref: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/
# Ref: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/
#
# --kill-on-bad-exit=1 : any shard failure stops the whole node-level job.
#                        Manifest ownership is static, and resume is based
#                        only on successful completion markers; unfinished
#                        shards are retried on explicit resubmission.
#
# Current launch model:
#   * the job reserves one full standard-g node with 8 GCDs
#   * the srun step launches 8 tasks with NUMA-aware CPU binding
#   * each task writes its own stdout/stderr file under a per-array-task log dir
#   * each task selects its physical GCD via ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000"
CPU_BIND="${CPU_BIND},0xfe0000,0xfe000000"
CPU_BIND="${CPU_BIND},0xfe,0xfe00"
CPU_BIND="${CPU_BIND},0xfe00000000,0xfe0000000000"

STEP_LOG_DIR="${WORKDIR}/logs/${SLURM_JOB_NAME:-opus_worker_g}-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$STEP_LOG_DIR"
echo "Task logs: ${STEP_LOG_DIR}"

set +e
srun \
    --cpu-bind="$CPU_BIND" \
    --output="${STEP_LOG_DIR}/gcd-%t.out" \
    --error="${STEP_LOG_DIR}/gcd-%t.err" \
    --kill-on-bad-exit=1 \
    bash "$TASK_SCRIPT"
EXIT_CODE=$?
set -e

echo "=================================="
echo "End time: $(date)"
echo "srun exit code: $EXIT_CODE"
echo "=================================="
exit $EXIT_CODE
