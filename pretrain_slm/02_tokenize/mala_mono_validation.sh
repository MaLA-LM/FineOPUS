#!/bin/bash
#SBATCH --job-name=tokenize_mala_validation
#SBATCH --output=./slurmlogs/%x_%j.out.log
#SBATCH --error=./slurmlogs/%x_%j.err.log
#SBATCH --partition=debug
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --mem=224G
#SBATCH --account=project_462000675

source ../lumi_config.sh

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999

# Move to the gpt-neox install

data_dir=/scratch/project_462000941/members/shaoxion/FineOPUS/ablation_data
out_dir=/scratch/project_462000964/FineOPUS/ablation_token

mkdir -p "$out_dir/mala/validation"

# Calculate the number of lines in the file
num_lines=$(cat "$data_dir/mala/validation/validation_0001.jsonl" | wc -l)


echo "START: $(date)"

srun --label ../launch.sh tools/datasets/preprocess_data.py     --input "$data_dir/mala/validation/validation_0001.jsonl"     --output-prefix "$out_dir/mala/validation/validation_0001"     --tokenizer-type "SPMTokenizer"     --vocab-file "/scratch/project_462000941/members/shaoxion/FineOPUS/models/gemma3/tokenizer.model"     --num-docs $num_lines     --append-eod     --workers 256

echo "END: $(date)"