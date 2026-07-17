#!/bin/bash

# Base directories
BASE_DIR="/scratch/project_462001427/FineOPUS/slm_from_scratch"
JSONL_BASE="${BASE_DIR}/data/combined/bilingual_mix/_jsonl"
BIN_BASE="${BASE_DIR}/data/combined/bilingual_mix/_bin"

# Define languages and datasets
languages=(
    "eng_Latn-ara_Arab"
    "eng_Latn-deu_Latn"
    "eng_Latn-fra_Latn"
    "eng_Latn-por_Latn"
    "eng_Latn-rus_Cyrl"
    "eng_Latn-zho_Hans"
    "eng_Latn-bul_Cyrl"
    "eng_Latn-ell_Grek"
    "eng_Latn-ita_Latn"
    "eng_Latn-ron_Latn"
    "eng_Latn-spa_Latn"
)

datasets=(
    "FineOPUS-Filtered-Stage1"
    "FineOPUS-Filtered-Stage2"
    "FineOPUS-Filtered-Stage3"
    "FineOPUS-Filtered-Stage4"
    "MaLA_Bi"
    "NLLB"
)

# Loop over all datasets and languages to submit jobs
for dataset in "${datasets[@]}"; do
    for lang in "${languages[@]}"; do
        input_file="${JSONL_BASE}/${dataset}/${lang}/combined.jsonl"
        output_prefix="${BIN_BASE}/${dataset}/${lang}/combined"

        # Check if the input file exists before submitting the job
        if [[ -f "$input_file" ]]; then
            # Construct a descriptive job name using dataset and language pair
            job_name="prep-${dataset}-${lang}"
            echo "Submitting job for dataset: ${dataset}, language: ${lang}..."
            sbatch --job-name="$job_name" preprocess_data.sh -i "$input_file" -o "$output_prefix"
        else
            echo "Warning: Input file does not exist, skipping: ${input_file}"
        fi
    done
done
