#!/usr/bin/env bash
#
# Separate cleanup entrypoint for OPUS merge inputs. This removes source
# JSONL direction directories only when matching merged parquet outputs exist.
#
#SBATCH --job-name=opus_merge_cleanup
#SBATCH --account=project_462001249
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.err

set -euo pipefail

resolve_run_merge_script() {
    local candidate_dir candidate_path
    local candidate_dirs=()

    if [ -n "${OPUS_MERGE_SCRIPT_DIR:-}" ]; then
        candidate_dirs+=("${OPUS_MERGE_SCRIPT_DIR}")
    fi
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        candidate_dirs+=("${SLURM_SUBMIT_DIR}/scripts/opus")
    fi
    candidate_dirs+=("$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")
    candidate_dirs+=("${WORKDIR:-/projappl/project_462001050/members/ibrahiam/05_quality_estimation}/scripts/opus")

    for candidate_dir in "${candidate_dirs[@]}"; do
        [ -n "$candidate_dir" ] || continue
        candidate_path="${candidate_dir}/run_merge.sh"
        if [ -r "$candidate_path" ]; then
            printf '%s\n' "$candidate_path"
            return 0
        fi
    done
    return 1
}

# Under sbatch, BASH_SOURCE[0] can point at Slurm's spool copy of this
# wrapper. Resolve the companion merge script from submit/repo locations.
RUN_MERGE="$(resolve_run_merge_script || true)"
if [ -z "$RUN_MERGE" ]; then
    echo "ERROR: run_merge.sh not found or not readable." >&2
    echo "Checked OPUS_MERGE_SCRIPT_DIR, SLURM_SUBMIT_DIR/scripts/opus, script dir, and WORKDIR/scripts/opus." >&2
    exit 1
fi

exec "$RUN_MERGE" --delete-merged-directions "$@"
