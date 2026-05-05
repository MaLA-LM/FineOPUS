#!/usr/bin/env bash
#
# Post-run merge: concatenate per-shard OPUS outputs into one or more parquet
# files per direction. Run as a regular (non-array) SLURM job after the
# workers drain.
#
#SBATCH --job-name=opus_merge
#SBATCH --account=project_462001249
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.err

set -euo pipefail

DB="${DB:-}"
MANIFEST_ROOT="${MANIFEST_ROOT:-}"
BUILD_TAG="${BUILD_TAG:-}"
TRACE_ROOT="${TRACE_ROOT:-}"
OUTPUT_BASE="${OUTPUT_BASE:-}"
MERGED_BASE="${MERGED_BASE:-}"
MODEL="${MODEL:-}"
DELETE_SHARDS="${DELETE_SHARDS:-0}"
FORCE="${FORCE:-0}"
JOBS="${JOBS:-1}"
WORKDIR="${WORKDIR:-/projappl/project_462001050/members/ibrahiam/05_quality_estimation}"
VENV_BASE="${VENV_BASE:-/scratch/project_462001050/ibrahiam/envs}"
VENV_PATH="${METRIC_VENV:-${VENV_BASE}/metric_venv}"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif}"

print_usage() {
    cat <<'EOF'
Usage: run_merge.sh [args]

Required:
  --output-base <dir>  Base dir of worker JSONLs (<model>/<direction>/*.jsonl).
  --merged-base <dir>  Destination dir for merged parquet files.

Coordination source (choose one):
  --db <path>              Legacy shared SQLite queue DB.
  --manifest-root <dir>    Manifest root containing <build_tag>/manifest.jsonl.
  --build-tag <tag>        Manifest build tag.
  --trace-root <dir>       Trace root containing completion state.jsonl files.

Optional:
  --model <key>        Restrict merge to a single model.
  --jobs <n>           Merge up to n directions in parallel.
  --delete-shards      Delete source shard files after successful merge.
  --force              Re-merge directions whose parquet outputs already exist.
  --workdir <dir>      cd into this dir before running.
  --venv-base <dir>    Base directory containing metric_venv.
  --metric-venv <dir>  Full path to the metric virtual environment.
  --sif <path>         Singularity image to use.
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
        --db)            DB="${2:-}"; shift 2 ;;
        --manifest-root) MANIFEST_ROOT="${2:-}"; shift 2 ;;
        --build-tag)     BUILD_TAG="${2:-}"; shift 2 ;;
        --trace-root)    TRACE_ROOT="${2:-}"; shift 2 ;;
        --output-base)   OUTPUT_BASE="${2:-}"; shift 2 ;;
        --merged-base)   MERGED_BASE="${2:-}"; shift 2 ;;
        --model)         MODEL="${2:-}"; shift 2 ;;
        --jobs)          JOBS="${2:-}"; shift 2 ;;
        --delete-shards) DELETE_SHARDS=1; shift ;;
        --force)         FORCE=1; shift ;;
        --workdir)       WORKDIR="${2:-}"; shift 2 ;;
        --venv-base)     VENV_BASE="${2:-}"; VENV_PATH="${2:-}/metric_venv"; shift 2 ;;
        --metric-venv)   VENV_PATH="${2:-}"; shift 2 ;;
        --sif)           SIF="${2:-}"; shift 2 ;;
        -h|--help)       print_usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; print_usage >&2; exit 1 ;;
    esac
done

if [ -z "$OUTPUT_BASE" ] || [ -z "$MERGED_BASE" ]; then
    echo "ERROR: --output-base and --merged-base are required." >&2
    print_usage >&2
    exit 1
fi

if [ -z "$DB" ] && { [ -z "$MANIFEST_ROOT" ] || [ -z "$BUILD_TAG" ] || [ -z "$TRACE_ROOT" ]; }; then
    echo "ERROR: provide either --db or all of --manifest-root, --build-tag, and --trace-root." >&2
    print_usage >&2
    exit 1
fi

setup_lumi_container
cd "$WORKDIR"
mkdir -p "${WORKDIR}/slurm_logs"

ARGS=(
    --output-base "$OUTPUT_BASE"
    --merged-base "$MERGED_BASE"
    --jobs "$JOBS"
)
if [ -n "$DB" ]; then
    ARGS+=(--db "$DB")
else
    ARGS+=(
        --manifest-root "$MANIFEST_ROOT"
        --build-tag "$BUILD_TAG"
        --trace-root "$TRACE_ROOT"
    )
fi
if [ -n "$MODEL" ]; then
    ARGS+=(--model "$MODEL")
fi
if [ "$DELETE_SHARDS" = "1" ]; then
    ARGS+=(--delete-shards)
fi
if [ "$FORCE" = "1" ]; then
    ARGS+=(--force)
fi

MERGE_CMD="$(quote_args python3 -m execution.opus_queue.tools.merge "${ARGS[@]}")"
WORKDIR_Q="$(printf '%q' "$WORKDIR")"
ACTIVATE_Q="$(printf '%q' "${VENV_PATH}/bin/activate")"

echo "Merging: ${ARGS[*]}"
echo "Workdir: $WORKDIR"
echo "Metric venv: $VENV_PATH"
echo "SIF: $SIF"

exec singularity run "$SIF" bash -c "
    source ${ACTIVATE_Q}
    cd ${WORKDIR_Q}
    ${MERGE_CMD}
"
