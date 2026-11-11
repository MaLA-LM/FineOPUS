#!/bin/bash
#SBATCH --job-name=move_files
#SBATCH --output=../logs/move_files/%x_%j.out
#SBATCH --error=../logs/move_files/%x_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=16G
#SBATCH --account=project_462001249

# ─── Configuration ────────────────────────────────────────────────────────────
SRC="${SRC:-/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED}"
DST="${DST:-/scratch/project_462001249/MaLA-LM}"
# DST accepts a local directory or [user@]host:/absolute/path (SSH config aliases work).
# Example (copy only; replace the destination path with your Roihu project path):
#   DST='lizihao1@roihu-cpu.csc.fi:/path/to/destination' DELETE_SRC=0 bash tools/move_files.sh
# For Slurm, export SRC/DST/DELETE_SRC before sbatch (the default --export=ALL passes them).
# Remote transfers require rsync on both clusters and non-interactive SSH authentication.
SSH_PORT="${SSH_PORT:-22}"

# Set to 1 to delete source after a successful transfer; 0 to copy and keep source.
# DELETE_SRC=0 still transfers data; it is not a dry run.
DELETE_SRC="${DELETE_SRC:-1}"
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

if [[ "$DELETE_SRC" != 0 && "$DELETE_SRC" != 1 ]]; then
    echo "ERROR: DELETE_SRC must be 0 or 1."
    exit 1
fi
if [[ ! "$SSH_PORT" =~ ^[0-9]{1,5}$ ]] || (( 10#$SSH_PORT < 1 || 10#$SSH_PORT > 65535 )); then
    echo "ERROR: SSH_PORT must be a number between 1 and 65535."
    exit 1
fi

# Keep directory-copy semantics consistent even if SRC has trailing slashes.
while [[ "$SRC" != / && "$SRC" == */ ]]; do SRC="${SRC%/}"; done
SRC_NAME=$(basename -- "$SRC")
if [[ "$SRC_NAME" == / || "$SRC_NAME" == . || "$SRC_NAME" == .. ]]; then
    echo "ERROR: SRC must name a file or directory, not /, . or ..."
    exit 1
fi

REMOTE_HOST=""
DST_PATH="$DST"
if [[ "$DST" =~ ^([^/:]+):(.+)$ ]]; then
    REMOTE_HOST="${BASH_REMATCH[1]}"
    DST_PATH="${BASH_REMATCH[2]}"
    if [[ "$REMOTE_HOST" == -* || "$DST_PATH" != /* ]]; then
        echo "ERROR: Remote DST must use [user@]host:/absolute/path."
        exit 1
    fi
fi
SSH=(ssh -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=30)

# SSH sends one shell command; quote every argument for the remote login shell.
run_remote() {
    local command="" arg
    for arg in "$@"; do
        command+=" '${arg//\'/\'\\\'\'}'"
    done
    "${SSH[@]}" "$REMOTE_HOST" "$command"
}

start_time=$(date +%s)
echo "=== Job started at: $(date) ==="
echo "SRC: $SRC"
echo "DST: $DST"
echo ""

# Validate source
if [ ! -e "$SRC" ]; then
    echo "ERROR: Source does not exist: $SRC"
    exit 1
fi

# Create destination directory if needed
if [[ -n "$REMOTE_HOST" ]]; then
    run_remote mkdir -p -- "$DST_PATH"
else
    mkdir -p -- "$DST_PATH"
fi

echo "=== Starting rsync ==="
# --archive       : preserves permissions, timestamps, symlinks, etc.
# --hard-links    : preserves hard links (important for datasets)
# --partial       : keep partially transferred files on interruption (allows resume)
# --info=progress2: single-line overall progress (better than per-file for many files)
# --no-whole-file : always use delta transfer (safer across filesystems)
RSYNC_OPTIONS=()
if [[ -n "$REMOTE_HOST" ]]; then
    RSYNC_OPTIONS+=(--rsh="ssh -p $SSH_PORT -o BatchMode=yes -o ConnectTimeout=30")
fi
if rsync \
    --archive \
    --hard-links \
    --partial \
    --info=progress2 \
    --no-whole-file \
    --human-readable \
    --stats \
    --protect-args \
    "${RSYNC_OPTIONS[@]}" \
    -- "$SRC" "${DST%/}/"; then
    RSYNC_EXIT=0
else
    RSYNC_EXIT=$?
fi

echo ""
if [ $RSYNC_EXIT -ne 0 ]; then
    echo "ERROR: rsync failed with exit code $RSYNC_EXIT. Source is NOT deleted."
    exit $RSYNC_EXIT
fi

echo "=== rsync completed successfully ==="

# Verify: compare file counts between source and destination
SRC_COUNT=$(find "$SRC" -type f -printf '.' | wc -c)
if [[ -n "$REMOTE_HOST" ]]; then
    DST_COUNT=$(run_remote bash -o pipefail -c \
        'find "$1" -type f -printf "." | wc -c' bash "${DST_PATH%/}/$SRC_NAME")
else
    DST_COUNT=$(find "${DST_PATH%/}/$SRC_NAME" -type f -printf '.' | wc -c)
fi
echo "File count — source: $SRC_COUNT  destination: $DST_COUNT"

if [ "$SRC_COUNT" -ne "$DST_COUNT" ]; then
    echo "ERROR: File counts do not match. Source is NOT deleted."
    exit 1
fi

echo "File counts match."

# Delete source only after verified success
if [ "$DELETE_SRC" -eq 1 ]; then
    echo "=== Deleting source: $SRC ==="
    rm -rf -- "$SRC"
    echo "Source deleted."
else
    echo "DELETE_SRC=0 — source kept at $SRC"
fi

end_time=$(date +%s)
duration=$((end_time - start_time))
echo ""
echo "=== Job ended at: $(date) ==="
echo "=== Total duration: $(date -u -d @${duration} +%T) ==="
