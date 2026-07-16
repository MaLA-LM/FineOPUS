#!/bin/bash
#SBATCH --job-name=merge-bilingual-mix
#SBATCH --output=../logs/merge-bilingual-mix/%x_%A_%a.out
#SBATCH --error=../logs/merge-bilingual-mix/%x_%A_%a.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=2G
#SBATCH --account=project_462001087
#SBATCH --array=0-65%6

set -Eeuo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/data/parallel" ]]; then
    SCRIPT_DIR=$(realpath -- "$SLURM_SUBMIT_DIR")
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/slm_from_scratch/data/parallel" ]]; then
    SCRIPT_DIR=$(realpath -- "$SLURM_SUBMIT_DIR/slm_from_scratch")
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
DATA_ROOT="$SCRIPT_DIR/data"
PARALLEL_ROOT="$DATA_ROOT/parallel/bilingual_mix/_jsonl"
MONOLINGUAL_ROOT="$DATA_ROOT/monolingual/bilingual_mix/_jsonl"
COMBINED_ROOT="$DATA_ROOT/combined/bilingual_mix/_jsonl"

DATASETS=(
    FineOPUS-Filtered-Stage1
    FineOPUS-Filtered-Stage2
    FineOPUS-Filtered-Stage3
    FineOPUS-Filtered-Stage4
    MaLA_Bi
    NLLB
)

LANGUAGES=(
    ara_Arab
    bul_Cyrl
    deu_Latn
    ell_Grek
    fra_Latn
    ita_Latn
    por_Latn
    ron_Latn
    rus_Cyrl
    spa_Latn
    zho_Hans
)

DATASET_COUNT=${#DATASETS[@]}
LANGUAGE_COUNT=${#LANGUAGES[@]}
TASK_COUNT=$((DATASET_COUNT * LANGUAGE_COUNT))
TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-}}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-60}

if [[ -z "$TASK_ID" || ! "$TASK_ID" =~ ^[0-9]+$ || "$TASK_ID" -ge "$TASK_COUNT" ]]; then
    echo "Usage: $0 TASK_ID (0-$((TASK_COUNT - 1)))" >&2
    echo "Under Slurm, TASK_ID is read from SLURM_ARRAY_TASK_ID." >&2
    exit 2
fi
if [[ ! "$PROGRESS_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: PROGRESS_INTERVAL must be a positive integer: $PROGRESS_INTERVAL" >&2
    exit 2
fi

dataset=${DATASETS[$((TASK_ID / LANGUAGE_COUNT))]}
language=${LANGUAGES[$((TASK_ID % LANGUAGE_COUNT))]}
pair="eng_Latn-$language"

parallel_file="$PARALLEL_ROOT/$dataset/$pair/$pair.jsonl"
monolingual_file="$MONOLINGUAL_ROOT/$language.jsonl"
output_dir="$COMBINED_ROOT/$dataset/$pair"
output_file="$output_dir/combined.jsonl"

for input_file in "$parallel_file" "$monolingual_file"; do
    if [[ ! -f "$input_file" || ! -r "$input_file" ]]; then
        echo "Error: input file is missing or unreadable: $input_file" >&2
        exit 1
    fi
    if [[ ! -s "$input_file" ]]; then
        echo "Error: input file is empty: $input_file" >&2
        exit 1
    fi
    if [[ $(tail -c 1 -- "$input_file" | od -An -tuC) != *10* ]]; then
        echo "Error: input file does not end with a newline: $input_file" >&2
        exit 1
    fi
done

parallel_bytes=$(stat -c '%s' -- "$parallel_file")
monolingual_bytes=$(stat -c '%s' -- "$monolingual_file")
expected_bytes=$((parallel_bytes + monolingual_bytes))

mkdir -p -- "$output_dir"

if [[ -e "$output_file" ]]; then
    actual_bytes=$(stat -c '%s' -- "$output_file")
    if [[ "$actual_bytes" -eq "$expected_bytes" ]]; then
        echo "Already complete; skipping: $output_file ($actual_bytes bytes)"
        exit 0
    fi
    echo "Error: existing output has $actual_bytes bytes; expected $expected_bytes: $output_file" >&2
    exit 1
fi

temp_file=$(mktemp --tmpdir="$output_dir" ".combined.jsonl.tmp.${SLURM_JOB_ID:-local}.XXXXXX")
progress_pid=""
start_time=$(date +%s)

stop_progress() {
    if [[ -n "$progress_pid" ]]; then
        kill "$progress_pid" 2>/dev/null || true
        wait "$progress_pid" 2>/dev/null || true
        progress_pid=""
    fi
}

cleanup() {
    stop_progress
    rm -f -- "$temp_file"
}
trap cleanup EXIT

show_progress() {
    local copied now elapsed metrics percent copied_gib total_gib speed_mib eta_seconds eta
    copied=$(stat -c '%s' -- "$temp_file" 2>/dev/null || echo 0)
    now=$(date +%s)
    elapsed=$((now - start_time))

    metrics=$(awk -v copied="$copied" -v total="$expected_bytes" -v elapsed="$elapsed" 'BEGIN {
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

echo "Dataset   : $dataset"
echo "Language  : $language"
echo "Parallel  : $parallel_file ($parallel_bytes bytes)"
echo "Monolingual: $monolingual_file ($monolingual_bytes bytes)"
echo "Output    : $output_file ($expected_bytes expected bytes)"

show_progress
progress_monitor &
progress_pid=$!

cat -- "$parallel_file" "$monolingual_file" > "$temp_file"

show_progress
stop_progress

actual_bytes=$(stat -c '%s' -- "$temp_file")
if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
    echo "Error: temporary output has $actual_bytes bytes; expected $expected_bytes" >&2
    exit 1
fi

mv -- "$temp_file" "$output_file"
trap - EXIT

echo "Complete  : $output_file ($actual_bytes bytes)"
