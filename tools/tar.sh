#!/bin/bash
#SBATCH --job-name=tar
#SBATCH --output=../logs/tar/%x_%j.out
#SBATCH --error=../logs/tar/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462001087

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

INPUT_DIR=""
OUTPUT_FILE=""

echo "Tarring $INPUT_DIR to $OUTPUT_FILE"

tar -cf "$OUTPUT_FILE" "$INPUT_DIR"

if [ $? -ne 0 ]; then
  echo "Tar failed"
  exit 1
fi

echo "Removing $INPUT_DIR"
rm -rf "$INPUT_DIR"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
