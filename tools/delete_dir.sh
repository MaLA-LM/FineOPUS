#!/bin/bash
#SBATCH --job-name=delete
#SBATCH --output=../logs/delete/%x_%j.out
#SBATCH --error=../logs/delete/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462001087

start_time=$(date +%s)
echo "Job started at: $(date)"
echo "========================================"

DIRS_TO_DELETE=(
    ""
    ""
)

if [ ${#DIRS_TO_DELETE[@]} -eq 0 ]; then
    echo "ERROR: No directories specified in DIRS_TO_DELETE array!"
    echo "Please edit the script and add directory paths to delete."
    exit 1
fi

echo "Total directories to delete: ${#DIRS_TO_DELETE[@]}"
echo "========================================"

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

delete_dir() {
    local index=$1
    local total=$2
    local dir=$3
    local result_file="$TEMP_DIR/result_$index"
    
    echo "[$index/$total] Starting: $dir"
    
    if [ ! -e "$dir" ]; then
        echo "[$index/$total] ⚠ SKIPPED: $dir (does not exist)"
        echo "skip" > "$result_file"
        return
    fi
    
    local del_start=$(date +%s)
    
    if rm -rf "$dir" 2>&1; then
        local del_end=$(date +%s)
        local del_duration=$((del_end - del_start))
        echo "[$index/$total] ✓ DELETED: $dir (took ${del_duration}s)"
        echo "success" > "$result_file"
    else
        echo "[$index/$total] ✗ FAILED: $dir"
        echo "fail" > "$result_file"
    fi
}

total=${#DIRS_TO_DELETE[@]}
for i in "${!DIRS_TO_DELETE[@]}"; do
    index=$((i + 1))
    delete_dir "$index" "$total" "${DIRS_TO_DELETE[$i]}" &
done

echo "All $total delete jobs launched in parallel, waiting..."
wait
echo ""

success_count=$(grep -l "success" "$TEMP_DIR"/result_* 2>/dev/null | wc -l)
fail_count=$(grep -l "fail" "$TEMP_DIR"/result_* 2>/dev/null | wc -l)
skip_count=$(grep -l "skip" "$TEMP_DIR"/result_* 2>/dev/null | wc -l)

echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "  Successful: $success_count"
echo "  Failed:     $fail_count"
echo "  Skipped:    $skip_count"
echo "  Total:      $total"
echo "========================================"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Total job duration: $(date -u -d @${duration} +%T)"

exit $fail_count
