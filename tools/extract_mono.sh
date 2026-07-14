#!/bin/bash
#SBATCH --job-name=fineopus-extract-mono
#SBATCH --output=../logs/fineopus-extract-mono/%x_%A_%a.out
#SBATCH --error=../logs/fineopus-extract-mono/%x_%A_%a.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --account=project_462001087
#
# Extract the token-budgeted monolingual multilingual mix as JSONL.
#
# Submit:
#   NUM_CHUNKS=32 ./extract_mono.sh
#
# Optional examples:
#   NUM_CHUNKS=8 LANGS="eng_Latn deu_Latn" ./extract_mono.sh
#   NUM_CHUNKS=32 OVERWRITE=1 ./extract_mono.sh
#
# Worker mode is entered automatically when SLURM sets SLURM_ARRAY_TASK_ID.

set -euo pipefail

# Fixed tools path because BASH_SOURCE points at the SLURM spool copy in worker
# mode.
TOOLS_DIR="${TOOLS_DIR:-/scratch/project_462001427/FineOPUS/tools}"

NUM_CHUNKS="${NUM_CHUNKS:-32}"
SEED="${SEED:-42}"
CSV="${CSV:-/scratch/project_462001427/FineOPUS/slm_from_scratch/data/monolingual/multilingual_mix/monolingual_uniform_450B_quota.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_462001427/FineOPUS/slm_from_scratch/data/monolingual/multilingual_mix/_jsonl}"
FINEWEB_DIR="${FINEWEB_DIR:-/scratch/project_462001069/fineweb-2}"
NEMOTRON_DIR="${NEMOTRON_DIR:-/scratch/project_462001069/nemotron-cc}"
FINEWEB_STATS="${FINEWEB_STATS:-/scratch/project_462001427/FineOPUS/tools/token_stats/FineWeb-2.csv}"
NEMOTRON_STATS="${NEMOTRON_STATS:-/scratch/project_462001427/FineOPUS/tools/token_stats/Nemotron-CC.csv}"
BATCH_SIZE="${BATCH_SIZE:-50000}"
INPUT_TEXT_COLUMN="${INPUT_TEXT_COLUMN:-text}"
OUTPUT_TEXT_KEY="${OUTPUT_TEXT_KEY:-text}"

EXTRA_ARGS=()
if [ -n "${LANGS:-}" ]; then
    # shellcheck disable=SC2206
    LANG_ARRAY=($LANGS)
    EXTRA_ARGS+=(--langs "${LANG_ARRAY[@]}")
fi
if [ "${OVERWRITE:-0}" = "1" ]; then
    EXTRA_ARGS+=(--overwrite)
fi
if [ "${INCLUDE_LANG:-0}" = "1" ]; then
    EXTRA_ARGS+=(--include_lang)
fi

# --------------------------- Worker mode ---------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    module purge
    module use /appl/local/csc/modulefiles/
    module load pytorch
    source /scratch/project_462001050/ibrahiam/envs/transformers-latest/bin/activate

    echo "Worker chunk ${SLURM_ARRAY_TASK_ID}/${NUM_CHUNKS} $(date)"
    srun --input=none python3 "$TOOLS_DIR/extract_mono.py" \
        --csv "$CSV" \
        --output_dir "$OUTPUT_DIR" \
        --fineweb_dir "$FINEWEB_DIR" \
        --nemotron_dir "$NEMOTRON_DIR" \
        --fineweb_stats "$FINEWEB_STATS" \
        --nemotron_stats "$NEMOTRON_STATS" \
        --seed "$SEED" \
        --batch_size "$BATCH_SIZE" \
        --input_text_column "$INPUT_TEXT_COLUMN" \
        --output_text_key "$OUTPUT_TEXT_KEY" \
        --chunk "$SLURM_ARRAY_TASK_ID" \
        --total_chunks "$NUM_CHUNKS" \
        "${EXTRA_ARGS[@]}"
    echo "Finished chunk ${SLURM_ARRAY_TASK_ID} at $(date)"
    exit 0
fi

# --------------------------- Submit mode ---------------------------
mkdir -p "$TOOLS_DIR/../logs/fineopus-extract-mono" "$OUTPUT_DIR"
LAST=$((NUM_CHUNKS - 1))
echo "Submitting ${NUM_CHUNKS} array jobs for monolingual extraction."
cd "$TOOLS_DIR"
sbatch --array=0-"$LAST" \
       --export=ALL,NUM_CHUNKS="$NUM_CHUNKS",SEED="$SEED",CSV="$CSV",OUTPUT_DIR="$OUTPUT_DIR",FINEWEB_DIR="$FINEWEB_DIR",NEMOTRON_DIR="$NEMOTRON_DIR",FINEWEB_STATS="$FINEWEB_STATS",NEMOTRON_STATS="$NEMOTRON_STATS",BATCH_SIZE="$BATCH_SIZE",INPUT_TEXT_COLUMN="$INPUT_TEXT_COLUMN",OUTPUT_TEXT_KEY="$OUTPUT_TEXT_KEY",TOOLS_DIR="$TOOLS_DIR",LANGS="${LANGS:-}",OVERWRITE="${OVERWRITE:-0}",INCLUDE_LANG="${INCLUDE_LANG:-0}" \
       extract_mono.sh
