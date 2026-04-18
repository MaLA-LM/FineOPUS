#!/usr/bin/env bash
#
# Reaper loop: reclaims stuck 'running' rows in the OPUS queue DB back to
# 'pending' so another worker can pick them up. Run this in a login-node
# `screen`/`tmux` session, or schedule a one-shot pass from cron every 5 min.
#
set -euo pipefail

DB="${DB:-}"
INTERVAL="${INTERVAL:-300}"
TIMEOUT_MULTIPLIER="${TIMEOUT_MULTIPLIER:-2}"
RESET_FAILED="${RESET_FAILED:-0}"
WORKDIR="${WORKDIR:-/projappl/project_462001050/members/ibrahiam/05_quality_estimation}"
VENV_BASE="${VENV_BASE:-/scratch/project_462001050/ibrahiam/envs}"
VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif}"

print_usage() {
    cat <<'EOF'
Usage: run_reaper.sh [args]

Required:
  --db <path>  Path to shared SQLite queue database.

Optional:
  --interval <sec>             Sweep interval (default 300; use 0 for a one-shot run).
  --timeout-multiplier <f>     cutoff = multiplier * expected_shard_seconds(model) (default 2).
  --reset-failed               Also requeue terminal failed rows after the same cooldown.
  --workdir <dir>              Change into this dir before running.
  --venv-base <dir>            Base directory containing metric_venv.
  --metric-venv <dir>          Full path to the metric virtual environment.
  --sif <path>                 Singularity image to use.
EOF
}

quote_args() {
    local arg
    for arg in "$@"; do
        printf '%q ' "$arg"
    done
}

setup_lumi_container() {
    if [ ! -f "${VENV_PATH}/bin/activate" ]; then
        echo "ERROR: metric venv activation script not found: ${VENV_PATH}/bin/activate" >&2
        exit 1
    fi
    if [ ! -f "$SIF" ]; then
        echo "ERROR: Singularity image not found: $SIF" >&2
        exit 1
    fi

    module purge
    module use /appl/local/laifs/modules
    module load lumi-aif-singularity-bindings
}

while [ $# -gt 0 ]; do
    case "$1" in
        --db)                  DB="${2:-}"; shift 2 ;;
        --interval)            INTERVAL="${2:-}"; shift 2 ;;
        --timeout-multiplier)  TIMEOUT_MULTIPLIER="${2:-}"; shift 2 ;;
        --reset-failed)        RESET_FAILED=1; shift ;;
        --workdir)             WORKDIR="${2:-}"; shift 2 ;;
        --venv-base)           VENV_BASE="${2:-}"; VENV_PATH="${2:-}/metric_venv"; shift 2 ;;
        --metric-venv)         VENV_PATH="${2:-}"; shift 2 ;;
        --sif)                 SIF="${2:-}"; shift 2 ;;
        -h|--help)             print_usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; print_usage >&2; exit 1 ;;
    esac
done

if [ -z "$DB" ]; then
    echo "ERROR: --db is required." >&2
    print_usage >&2
    exit 1
fi

setup_lumi_container
cd "$WORKDIR"

REAPER_CMD="$(quote_args python3 -m execution.opus_queue.tools.reaper \
    --db "$DB" \
    --interval "$INTERVAL" \
    --timeout-multiplier "$TIMEOUT_MULTIPLIER")"
if [ "$RESET_FAILED" = "1" ]; then
    REAPER_CMD+="$(quote_args --reset-failed)"
fi
WORKDIR_Q="$(printf '%q' "$WORKDIR")"
ACTIVATE_Q="$(printf '%q' "${VENV_PATH}/bin/activate")"

echo "Reaper: db=$DB interval=${INTERVAL}s multiplier=${TIMEOUT_MULTIPLIER} reset_failed=${RESET_FAILED}"
echo "Workdir: $WORKDIR"
echo "Metric venv: $VENV_PATH"
echo "SIF: $SIF"

exec singularity run "$SIF" bash -c "
    source ${ACTIVATE_Q}
    cd ${WORKDIR_Q}
    ${REAPER_CMD}
"
