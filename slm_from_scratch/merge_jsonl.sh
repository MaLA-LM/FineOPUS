#!/bin/bash
#SBATCH --job-name=merge-jsonl
#SBATCH --output=../logs/merge-jsonl/%x_%j.out
#SBATCH --error=../logs/merge-jsonl/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G
#SBATCH --account=project_462001087

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  merge_jsonl.sh [-f] [-p SECONDS] -o OUTPUT_FILE INPUT_DIR [INPUT_DIR ...]

Recursively find and concatenate all *.jsonl files in the input directories.
Files are concatenated in lexicographic path order.

Options:
  -o FILE  Output JSONL file (required)
  -f       Overwrite FILE if it already exists
  -p SEC   Progress update interval in seconds (default: 60)
  -h       Show this help

Example:
  ./merge_jsonl.sh -o combined.jsonl mono_jsonl_dir parallel_jsonl_dir
EOF
}

OUTPUT_FILE=""
OVERWRITE=0
PROGRESS_INTERVAL=60

while getopts ":o:p:fh" option; do
    case "$option" in
        o) OUTPUT_FILE="$OPTARG" ;;
        p) PROGRESS_INTERVAL="$OPTARG" ;;
        f) OVERWRITE=1 ;;
        h)
            usage
            exit 0
            ;;
        :)
            echo "Error: -$OPTARG requires an argument." >&2
            usage >&2
            exit 2
            ;;
        \?)
            echo "Error: unknown option -$OPTARG." >&2
            usage >&2
            exit 2
            ;;
    esac
done
shift $((OPTIND - 1))

if [[ -z "$OUTPUT_FILE" ]]; then
    echo "Error: -o OUTPUT_FILE is required." >&2
    usage >&2
    exit 2
fi
if (( $# == 0 )); then
    echo "Error: at least one input directory is required." >&2
    usage >&2
    exit 2
fi
if [[ "$OUTPUT_FILE" != *.jsonl ]]; then
    echo "Error: output filename must end with .jsonl: $OUTPUT_FILE" >&2
    exit 2
fi
if [[ ! "$PROGRESS_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: progress interval must be a positive integer: $PROGRESS_INTERVAL" >&2
    exit 2
fi

INPUT_DIRS=("$@")
for input_dir in "${INPUT_DIRS[@]}"; do
    if [[ ! -d "$input_dir" ]]; then
        echo "Error: input directory does not exist: $input_dir" >&2
        exit 2
    fi
done

# Canonical paths make excluding an existing output reliable when it is located
# below one of the input directories.
OUTPUT_FILE=$(realpath -m -- "$OUTPUT_FILE")
for i in "${!INPUT_DIRS[@]}"; do
    INPUT_DIRS[$i]=$(realpath -- "${INPUT_DIRS[$i]}")
done

if [[ -e "$OUTPUT_FILE" && "$OVERWRITE" -ne 1 ]]; then
    echo "Error: output already exists: $OUTPUT_FILE (use -f to overwrite)" >&2
    exit 2
fi

OUTPUT_DIR=$(dirname -- "$OUTPUT_FILE")
mkdir -p -- "$OUTPUT_DIR"

read -r TOTAL_FILES TOTAL_BYTES < <(
    find "${INPUT_DIRS[@]}" \
        -type f -name '*.jsonl' ! -path "$OUTPUT_FILE" -printf '%s\n' \
        | awk '{ files += 1; bytes += $1 } END { printf "%d %.0f\n", files, bytes }'
)
if (( TOTAL_FILES == 0 )); then
    echo "Error: no .jsonl files found in the input directories." >&2
    exit 2
fi

TEMP_FILE=$(mktemp --tmpdir="$OUTPUT_DIR" ".$(basename -- "$OUTPUT_FILE").tmp.XXXXXX")
PROGRESS_PID=""
START_TIME=$(date +%s)

stop_progress() {
    if [[ -n "$PROGRESS_PID" ]]; then
        kill "$PROGRESS_PID" 2>/dev/null || true
        wait "$PROGRESS_PID" 2>/dev/null || true
        PROGRESS_PID=""
    fi
}

cleanup() {
    stop_progress
    rm -f -- "$TEMP_FILE"
}
trap cleanup EXIT

show_progress() {
    local copied now elapsed metrics percent copied_gib total_gib speed_mib eta_seconds eta
    copied=$(stat -c '%s' -- "$TEMP_FILE" 2>/dev/null || echo 0)
    now=$(date +%s)
    elapsed=$((now - START_TIME))

    metrics=$(awk -v copied="$copied" -v total="$TOTAL_BYTES" -v elapsed="$elapsed" 'BEGIN {
        percent = total > 0 ? copied * 100 / total : 100
        if (percent > 100) percent = 100
        speed = elapsed > 0 ? copied / elapsed : 0
        eta = speed > 0 && total > copied ? (total - copied) / speed : 0
        printf "%.2f %.2f %.2f %.1f %.0f", percent, copied / 1073741824,
               total / 1073741824, speed / 1048576, eta
    }')
    read -r percent copied_gib total_gib speed_mib eta_seconds <<< "$metrics"

    if (( eta_seconds > 0 )); then
        printf -v eta '%02d:%02d:%02d' \
            $((eta_seconds / 3600)) $(((eta_seconds % 3600) / 60)) $((eta_seconds % 60))
    else
        eta='--:--:--'
    fi

    printf '[%(%F %T)T] Progress: %6s%% | %s / %s GiB | %s MiB/s | ETA %s\n' \
        -1 "$percent" "$copied_gib" "$total_gib" "$speed_mib" "$eta"
}

progress_monitor() {
    while true; do
        sleep "$PROGRESS_INTERVAL"
        show_progress
    done
}

echo "Input files: $TOTAL_FILES"
printf 'Total size : %.2f GiB\n' "$(awk -v bytes="$TOTAL_BYTES" 'BEGIN { print bytes / 1073741824 }')"
show_progress
progress_monitor &
PROGRESS_PID=$!

find "${INPUT_DIRS[@]}" \
    -type f -name '*.jsonl' \
    ! -path "$OUTPUT_FILE" \
    -print0 \
    | sort -z \
    | xargs -0 -r cat -- > "$TEMP_FILE"

show_progress
stop_progress
mv -f -- "$TEMP_FILE" "$OUTPUT_FILE"
trap - EXIT

echo "Output: $OUTPUT_FILE"
ls -lh -- "$OUTPUT_FILE"
