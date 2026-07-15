#!/bin/bash
#SBATCH --job-name=construct-parallel-data
#SBATCH --output=../logs/construct-parallel-data/%x_%j.out
#SBATCH --error=../logs/construct-parallel-data/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=256G
#SBATCH --account=project_462001087

set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $(basename "$0") --input_folder INPUT --output_folder OUTPUT [options]
  $(basename "$0") INPUT OUTPUT [options]

Options:
  --concat_n_lines N       Number of rows per JSONL block. Use 0 for one block per parquet. Default: 40
  --workers N              Number of Python worker processes per language pair. Default: 8
  --source_text_col NAME   Parquet source text column. Default: source_text
  --target_text_col NAME   Parquet target text column. Default: target_text
  --direction_mode MODE    original or bidirectional_english. Default: bidirectional_english
  -h, --help               Show this help

If INPUT contains language-pair subfolders like eng_Latn-fra_Latn, each one is
converted into OUTPUT/<lang_pair>/. If INPUT itself contains parquet files, it
is converted directly into OUTPUT/.
EOF
}

input_folder=""
output_folder=""
concat_n_lines=40
workers=8
source_text_col="source_text"
target_text_col="target_text"
direction_mode="bidirectional_english"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_folder|--input-folder)
            input_folder="$2"
            shift 2
            ;;
        --output_folder|--output-folder)
            output_folder="$2"
            shift 2
            ;;
        --concat_n_lines|--concat-n-lines)
            concat_n_lines="$2"
            shift 2
            ;;
        --workers)
            workers="$2"
            shift 2
            ;;
        --source_text_col|--source-text-col)
            source_text_col="$2"
            shift 2
            ;;
        --target_text_col|--target-text-col)
            target_text_col="$2"
            shift 2
            ;;
        --direction_mode|--direction-mode)
            direction_mode="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -z "$input_folder" ]]; then
                input_folder="$1"
            elif [[ -z "$output_folder" ]]; then
                output_folder="$1"
            else
                echo "Unexpected argument: $1" >&2
                usage >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$input_folder" || -z "$output_folder" ]]; then
    echo "Both input_folder and output_folder are required." >&2
    usage >&2
    exit 1
fi

if [[ ! -d "$input_folder" ]]; then
    echo "Input folder does not exist: $input_folder" >&2
    exit 1
fi

start_time=$(date +%s)
echo "Job started at: $(date)"

if type module >/dev/null 2>&1; then
    module purge
    module use /appl/local/csc/modulefiles
    module load pytorch
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner=()
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    runner=(srun)
fi

mkdir -p "$output_folder"

run_converter() {
    local pair_input_folder="$1"
    local pair_output_folder="$2"
    local lang_pair

    lang_pair="$(basename "$pair_input_folder")"
    mkdir -p "$pair_output_folder"

    echo "Converting ${pair_input_folder} -> ${pair_output_folder}"
    "${runner[@]}" python ./construct_parallel_data_from_parquet.py \
        --input_folder "$pair_input_folder" \
        --output_folder "$pair_output_folder" \
        --lang_pair "$lang_pair" \
        --source_text_col "$source_text_col" \
        --target_text_col "$target_text_col" \
        --concat_n_lines "$concat_n_lines" \
        --direction_mode "$direction_mode" \
        --workers "$workers"
}

mapfile -t lang_pair_dirs < <(find "$input_folder" -mindepth 1 -maxdepth 1 -type d -name "*-*" | sort)

converted_count=0
if [[ ${#lang_pair_dirs[@]} -gt 0 ]]; then
    for pair_dir in "${lang_pair_dirs[@]}"; do
        if compgen -G "${pair_dir}/*.parquet" >/dev/null; then
            run_converter "$pair_dir" "$output_folder/$(basename "$pair_dir")"
            converted_count=$((converted_count + 1))
        else
            echo "Skipping $(basename "$pair_dir"): no parquet files found"
        fi
    done
else
    if compgen -G "${input_folder}/*.parquet" >/dev/null; then
        run_converter "$input_folder" "$output_folder"
        converted_count=1
    else
        echo "No parquet files or language-pair subfolders found under: $input_folder" >&2
        exit 1
    fi
fi

echo "Converted language-pair folders: $converted_count"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
