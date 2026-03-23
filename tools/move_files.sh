#!/bin/bash
#SBATCH --job-name=move_files
#SBATCH --output=../logs/move_files/%x_%j.out
#SBATCH --error=../logs/move_files/%x_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=16G
#SBATCH --account=project_462001249

# ─── Configuration ────────────────────────────────────────────────────────────
SRC="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED"
DST="/scratch/project_462001249/MaLA-LM"

# Set to 1 to delete source after a successful transfer; 0 for dry-run / copy only
DELETE_SRC=1
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

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
mkdir -p "$DST"

echo "=== Starting rsync ==="
# --archive       : preserves permissions, timestamps, symlinks, etc.
# --hard-links    : preserves hard links (important for datasets)
# --partial       : keep partially transferred files on interruption (allows resume)
# --info=progress2: single-line overall progress (better than per-file for many files)
# --no-whole-file : always use delta transfer (safer across filesystems)
rsync \
    --archive \
    --hard-links \
    --partial \
    --info=progress2 \
    --no-whole-file \
    --human-readable \
    --stats \
    "$SRC" "$DST/"

RSYNC_EXIT=$?

echo ""
if [ $RSYNC_EXIT -ne 0 ]; then
    echo "ERROR: rsync failed with exit code $RSYNC_EXIT. Source is NOT deleted."
    exit $RSYNC_EXIT
fi

echo "=== rsync completed successfully ==="

# Verify: compare file counts between source and destination
SRC_NAME=$(basename "$SRC")
SRC_COUNT=$(find "$SRC" -type f | wc -l)
DST_COUNT=$(find "$DST/$SRC_NAME" -type f | wc -l)
echo "File count — source: $SRC_COUNT  destination: $DST_COUNT"

if [ "$SRC_COUNT" -ne "$DST_COUNT" ]; then
    echo "ERROR: File counts do not match. Source is NOT deleted."
    exit 1
fi

echo "File counts match."

# Delete source only after verified success
if [ "$DELETE_SRC" -eq 1 ]; then
    echo "=== Deleting source: $SRC ==="
    rm -rf "$SRC"
    echo "Source deleted."
else
    echo "DELETE_SRC=0 — source kept at $SRC"
fi

end_time=$(date +%s)
duration=$((end_time - start_time))
echo ""
echo "=== Job ended at: $(date) ==="
echo "=== Total duration: $(date -u -d @${duration} +%T) ==="
