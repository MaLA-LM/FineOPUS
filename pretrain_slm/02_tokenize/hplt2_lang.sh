#!/bin/bash

# Configuration
data_dir="/scratch/project_462000964/FineOPUS/ablation_data"
language_code="${1:-zho_Hans}"
echo "Processing language: $language_code"
input_dir="$data_dir/HPLT2.0_cleaned/$language_code"
out_dir="/scratch/project_462000964/FineOPUS/ablation_token"

# Create output directory if it doesn't exist
mkdir -p "$out_dir/HPLT2.0_cleaned/$language_code"

# Check if input directory exists
if [ ! -d "$input_dir" ]; then
    echo "Error: Input directory $input_dir does not exist"
    exit 1
fi

# Find all JSONL files in the input directory
jsonl_files=($(find "$input_dir" -name "*.jsonl" -type f | sort))

if [ ${#jsonl_files[@]} -eq 0 ]; then
    echo "Error: No JSONL files found in $input_dir"
    exit 1
fi

echo "Found ${#jsonl_files[@]} JSONL files to process:"
for file in "${jsonl_files[@]}"; do
    echo "$language_code - $(basename "$file")"
done

# Submit a job for each JSONL file
for jsonl_file in "${jsonl_files[@]}"; do
    # Extract filename without extension for output prefix
    filename=$(basename "$jsonl_file" .jsonl)
    
    echo "Submitting job for: $language_code - $filename"
    
    # Create a temporary SLURM script for this specific file
    temp_script=$(mktemp /tmp/slurm_job_XXXXXX.sh)
    
    cat > "$temp_script" << EOF
#!/bin/bash
#SBATCH --job-name=hplt2_${filename}
#SBATCH --output=./slurmlogs/%x_%j.out.log
#SBATCH --error=./slurmlogs/%x_%j.err.log
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=72:00:00
#SBATCH --mem=224G
#SBATCH --account=project_462000675

source ../lumi_config.sh

export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999

# Input and output paths
input_file="$jsonl_file"
output_prefix="$out_dir/HPLT2.0_cleaned/$language_code/$filename"

# Calculate the number of lines in the file
num_lines=\$(cat "\$input_file" | wc -l)

echo "Processing: \$input_file"
echo "Output prefix: \$output_prefix"
echo "Number of lines: \$num_lines"
echo "START: \$(date)"

srun --label ../launch.sh tools/datasets/preprocess_data.py \\
    --input "\$input_file" \\
    --output-prefix "\$output_prefix" \\
    --tokenizer-type "SPMTokenizer" \\
    --vocab-file "/scratch/project_462000941/members/shaoxion/FineOPUS/models/gemma3/tokenizer.model" \\
    --num-docs \$num_lines \\
    --append-eod \\
    --workers 256

echo "END: \$(date)"
EOF

    # Submit the job and capture job ID
    job_id=$(sbatch "$temp_script" | awk '{print $4}')
    echo "  Submitted job ID: $job_id for file: $filename"
    
    # Clean up temporary script
    rm "$temp_script"
    # Sleep for a short duration to avoid overwhelming the scheduler
    sleep 1
done

echo "All jobs submitted successfully!"
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: ./slurmlogs/"