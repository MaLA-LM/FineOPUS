#!/bin/bash
#
#SBATCH --job-name=check_conversion       # A descriptive name for your job
#SBATCH --account=project_462000964      # IMPORTANT: Replace with your project ID
#SBATCH --partition=small           # The partition (queue) to submit to.
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --ntasks-per-node=1         # Number of tasks (processes) to run per node
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=64G                    # Memory per node (e.g., 4G, 16G, 64G)
#SBATCH --time=10:30:00             # Maximum execution time (HH:MM:SS)
#SBATCH --output=slurmlog/%x_%j.out.log  # Standard output log file, %j is the job ID
#SBATCH --error=slurmlog/%x_%j.err.log    # Standard error log file

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Running in directory $(pwd)"
echo "Job started at $(date)"

module load cray-python

# This script performs the following tasks:
# 1. check downloaded files for conversion issues
# 2. log any skipped directories and file mismatches
# 3. fix non-standard codes
# 4. remove empty folders
srun python -u check_conversion.py \
        --data_dir /scratch/project_462001050/FineOPUS/opus-conversion \
        --skip_log check_skipdirs.log \
        --mismatch_log check_file_mismatch.log
