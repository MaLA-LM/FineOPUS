#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional: Clean up old logs.
mkdir -p "$SCRIPT_DIR/slurmlog"
rm -f "$SCRIPT_DIR"/slurmlog/cnt_tok*.log

# --- Configuration ---
DATA_DIR="/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage4"
OUTPUT_FILE="/scratch/project_462000941/members/zihao/OPUS2410/tools/token_stats/FineOPUS-Filtered-Stage4_Qwen3_5.csv"

# Number of parquet files assigned to each array worker.
CHUNK_SIZE=200

# Tune this if tokenizer memory or throughput needs adjustment.
TOKENIZER_BATCH_SIZE=1024

# Number of parquet rows read into memory at a time.
PARQUET_BATCH_SIZE=10000

# Tokenizer model name or path
TOKENIZER="Qwen/Qwen3.5-9B"
# Suffix used for token count headers in CSV (e.g. n_src_tokens_deepseekv4)
TOKENIZER_NAME="Qwen3_5"
# ---------------------

DIRECTION_HEADER="lang_pair,src_lang,tgt_lang,n_lines,n_src_tokens_space,n_tgt_tokens_space,n_src_tokens_${TOKENIZER_NAME},n_tgt_tokens_${TOKENIZER_NAME}"
PARQUET_HEADER="lang_pair,src_lang,tgt_lang,parquet_file,n_lines,n_src_tokens_space,n_tgt_tokens_space,n_src_tokens_${TOKENIZER_NAME},n_tgt_tokens_${TOKENIZER_NAME}"

TASK_ROOT_DIR="${TASK_ROOT_DIR:-$(dirname "$OUTPUT_FILE")/token_count_tasks}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_TASK_DIR="$TASK_ROOT_DIR/run_$RUN_ID"
WORKER_OUTPUT_DIR="$RUN_TASK_DIR/worker_outputs"

WORKER_CSV_LIST="$RUN_TASK_DIR/all_worker_csvs.txt"
DIRECTION_ROWS_TO_MERGE="$RUN_TASK_DIR/direction_rows_to_merge.csv"
COMPLETED_DIRECTIONS_FILE="$RUN_TASK_DIR/completed_directions.txt"
COMPLETED_PARQUETS_FILE="$RUN_TASK_DIR/completed_parquets.txt"
ALL_FOLDERS_FILE="$RUN_TASK_DIR/all_folders.txt"
INCOMPLETE_DIRECTIONS_FILE="$RUN_TASK_DIR/incomplete_directions.txt"
ALL_PARQUETS_FILE="$RUN_TASK_DIR/all_incomplete_direction_parquets.tsv"
MISSING_PARQUETS_FILE="$RUN_TASK_DIR/missing_parquets.tsv"
DIRECTIONS_WITH_PARQUETS_FILE="$RUN_TASK_DIR/directions_with_parquets.txt"
DIRECTIONS_WITH_MISSING_FILE="$RUN_TASK_DIR/directions_with_missing_parquets.txt"
READY_FOR_AGGREGATION_FILE="$RUN_TASK_DIR/directions_ready_for_aggregation.txt"
NO_PARQUET_FILE="$RUN_TASK_DIR/directions_without_parquets.txt"
MANIFEST_FILE="$RUN_TASK_DIR/parquet_manifest.tsv"

mkdir -p "$RUN_TASK_DIR" "$WORKER_OUTPUT_DIR" "$(dirname "$OUTPUT_FILE")"

if [ "$CHUNK_SIZE" -le 0 ]; then
    echo "CHUNK_SIZE must be positive." >&2
    exit 1
fi

if [ ! -s "$OUTPUT_FILE" ]; then
    printf "%s\n" "$DIRECTION_HEADER" > "$OUTPUT_FILE"
fi

echo "Scanning previous worker CSV files under $TASK_ROOT_DIR..."
if [ -d "$TASK_ROOT_DIR" ]; then
    find "$TASK_ROOT_DIR" -path "*/worker_outputs/*.csv" -type f | sort > "$WORKER_CSV_LIST"
else
    > "$WORKER_CSV_LIST"
fi

mapfile -t WORKER_CSVS < "$WORKER_CSV_LIST"
NUM_WORKER_CSVS=${#WORKER_CSVS[@]}
echo "Worker CSV files found: $NUM_WORKER_CSVS"

echo "Merging completed direction-level worker rows into $OUTPUT_FILE..."
> "$DIRECTION_ROWS_TO_MERGE"
if [ "$NUM_WORKER_CSVS" -gt 0 ]; then
    awk -F, -v direction_header="$DIRECTION_HEADER" '
        function clean(value) {
            gsub(/\r/, "", value)
            gsub(/^"|"$/, "", value)
            return value
        }
        FNR == 1 {
            file_number++
            if (file_number == 1) {
                skip_file = 0
            } else {
                header = $0
                gsub(/\r/, "", header)
                skip_file = (header != direction_header)
                next
            }
        }
        file_number == 1 {
            value = clean($1)
            if (value != "" && value != "lang_pair") {
                completed[value] = 1
            }
            next
        }
        skip_file {
            next
        }
        {
            value = clean($1)
            if (value != "" && value != "lang_pair" && !(value in completed)) {
                print $0
                completed[value] = 1
            }
        }
    ' "$OUTPUT_FILE" "${WORKER_CSVS[@]}" > "$DIRECTION_ROWS_TO_MERGE"
fi

MERGED_DIRECTION_ROWS=$(awk 'END { print NR + 0 }' "$DIRECTION_ROWS_TO_MERGE")
if [ "$MERGED_DIRECTION_ROWS" -gt 0 ]; then
    cat "$DIRECTION_ROWS_TO_MERGE" >> "$OUTPUT_FILE"
fi
echo "Direction-level worker rows added to main CSV: $MERGED_DIRECTION_ROWS"

awk -F, '
    function clean(value) {
        gsub(/\r/, "", value)
        gsub(/^"|"$/, "", value)
        return value
    }
    {
        value = clean($1)
        if (value != "" && value != "lang_pair") {
            print value
        }
    }
' "$OUTPUT_FILE" | sort -u > "$COMPLETED_DIRECTIONS_FILE"

echo "Collecting parquet-level checkpoints from previous worker CSV files..."
> "$COMPLETED_PARQUETS_FILE"
if [ "$NUM_WORKER_CSVS" -gt 0 ]; then
    awk -F, -v parquet_header="$PARQUET_HEADER" '
        function clean(value) {
            gsub(/\r/, "", value)
            gsub(/^"|"$/, "", value)
            return value
        }
        FNR == 1 {
            header = $0
            gsub(/\r/, "", header)
            is_parquet_file = (header == parquet_header)
            next
        }
        !is_parquet_file {
            next
        }
        {
            parquet_file = clean($4)
            if (parquet_file != "" && parquet_file != "parquet_file") {
                print parquet_file
            }
        }
    ' "${WORKER_CSVS[@]}" | sort -u > "$COMPLETED_PARQUETS_FILE"
fi

NUM_COMPLETED_DIRECTIONS=$(awk 'END { print NR + 0 }' "$COMPLETED_DIRECTIONS_FILE")
NUM_COMPLETED_PARQUETS=$(awk 'END { print NR + 0 }' "$COMPLETED_PARQUETS_FILE")
echo "Completed directions known: $NUM_COMPLETED_DIRECTIONS"
echo "Completed parquet files known from worker CSVs: $NUM_COMPLETED_PARQUETS"

echo "Finding all language-pair folders in $DATA_DIR..."
find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort > "$ALL_FOLDERS_FILE"
NUM_ALL_FOLDERS=$(awk 'END { print NR + 0 }' "$ALL_FOLDERS_FILE")

awk -v completed_file="$COMPLETED_DIRECTIONS_FILE" '
    BEGIN {
        while ((getline line < completed_file) > 0) {
            completed[line] = 1
        }
        close(completed_file)
    }
    !($0 in completed) {
        print $0
    }
' "$ALL_FOLDERS_FILE" > "$INCOMPLETE_DIRECTIONS_FILE"

NUM_INCOMPLETE_DIRECTIONS=$(awk 'END { print NR + 0 }' "$INCOMPLETE_DIRECTIONS_FILE")

echo "Building parquet manifest from incomplete directions..."
> "$ALL_PARQUETS_FILE"
> "$NO_PARQUET_FILE"

while IFS= read -r LANG_PAIR; do
    [ -n "$LANG_PAIR" ] || continue

    LANG_PAIR_DIR="$DATA_DIR/$LANG_PAIR"
    if [ ! -d "$LANG_PAIR_DIR" ]; then
        echo "Warning: directory disappeared while building manifest: $LANG_PAIR_DIR" >&2
        continue
    fi

    PARQUET_COUNT=0
    while IFS= read -r PARQUET_NAME; do
        [ -n "$PARQUET_NAME" ] || continue
        printf "%s\t%s/%s\n" "$LANG_PAIR" "$LANG_PAIR" "$PARQUET_NAME" >> "$ALL_PARQUETS_FILE"
        PARQUET_COUNT=$((PARQUET_COUNT + 1))
    done < <(find "$LANG_PAIR_DIR" -maxdepth 1 -type f -name "*.parquet" -printf "%f\n" | sort)

    if [ "$PARQUET_COUNT" -eq 0 ]; then
        printf "%s\n" "$LANG_PAIR" >> "$NO_PARQUET_FILE"
    fi
done < "$INCOMPLETE_DIRECTIONS_FILE"

awk -F'\t' -v completed_file="$COMPLETED_PARQUETS_FILE" '
    BEGIN {
        while ((getline line < completed_file) > 0) {
            completed[line] = 1
        }
        close(completed_file)
    }
    !($2 in completed) {
        print $0
    }
' "$ALL_PARQUETS_FILE" > "$MISSING_PARQUETS_FILE"

cut -f1 "$ALL_PARQUETS_FILE" | sort -u > "$DIRECTIONS_WITH_PARQUETS_FILE"
cut -f1 "$MISSING_PARQUETS_FILE" | sort -u > "$DIRECTIONS_WITH_MISSING_FILE"
awk -v missing_file="$DIRECTIONS_WITH_MISSING_FILE" '
    BEGIN {
        while ((getline line < missing_file) > 0) {
            missing[line] = 1
        }
        close(missing_file)
    }
    !($0 in missing) {
        print $0
    }
' "$DIRECTIONS_WITH_PARQUETS_FILE" > "$READY_FOR_AGGREGATION_FILE"

awk -F'\t' -v chunk_size="$CHUNK_SIZE" '
    BEGIN {
        OFS = "\t"
        print "worker_id", "lang_pair", "parquet_file"
    }
    {
        worker_id = int((NR - 1) / chunk_size) + 1
        print worker_id, $1, $2
    }
' "$MISSING_PARQUETS_FILE" > "$MANIFEST_FILE"

NUM_PARQUET_TASKS=$(awk 'NR > 1 { count++ } END { print count + 0 }' "$MANIFEST_FILE")
NUM_DIRECTIONS_WITH_MISSING=$(awk 'END { print NR + 0 }' "$DIRECTIONS_WITH_MISSING_FILE")
NUM_READY_FOR_AGGREGATION=$(awk 'END { print NR + 0 }' "$READY_FOR_AGGREGATION_FILE")
NUM_WITHOUT_PARQUETS=$(awk 'END { print NR + 0 }' "$NO_PARQUET_FILE")

echo "Total folders found: $NUM_ALL_FOLDERS"
echo "Directions not yet in main/direction-level worker CSVs: $NUM_INCOMPLETE_DIRECTIONS"
echo "Directions with missing parquet work: $NUM_DIRECTIONS_WITH_MISSING"
echo "Directions with parquet rows complete and ready for manual aggregation: $NUM_READY_FOR_AGGREGATION"
echo "Directions without parquet files: $NUM_WITHOUT_PARQUETS"
echo "Missing parquet files to process: $NUM_PARQUET_TASKS"
echo "Manifest file: $MANIFEST_FILE"

if [ "$NUM_PARQUET_TASKS" -eq 0 ]; then
    echo "No parquet files need processing. Nothing to submit."
else
    NUM_JOBS=$(( (NUM_PARQUET_TASKS + CHUNK_SIZE - 1) / CHUNK_SIZE ))

    echo "Submitting $NUM_JOBS array jobs (up to $CHUNK_SIZE parquet files per worker; final worker may have fewer)."

    cd "$SCRIPT_DIR"
    sbatch --array=1-$NUM_JOBS \
           --export=ALL,DATA_DIR="$DATA_DIR",MANIFEST_FILE="$MANIFEST_FILE",OUTPUT_FILE="$OUTPUT_FILE",CHUNK_SIZE="$CHUNK_SIZE",TOKENIZER_BATCH_SIZE="$TOKENIZER_BATCH_SIZE",PARQUET_BATCH_SIZE="$PARQUET_BATCH_SIZE",WORKER_OUTPUT_DIR="$WORKER_OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR",TOKENIZER="$TOKENIZER",TOKENIZER_NAME="$TOKENIZER_NAME" \
           count_token_chunk.slurm
fi
