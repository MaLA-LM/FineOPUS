#!/bin/bash
#
#SBATCH --job-name=mp_shard       # A descriptive name for your job
#SBATCH --account=project_462000964      # IMPORTANT: Replace with your project ID
#SBATCH --partition=small           # The partition (queue) to submit to. (Good for testing)
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks-per-node=1         # Run 1 main task (your python script)
#SBATCH --cpus-per-task=16          # Request 16 CPU cores for this one task
#SBATCH --mem=512G                   # Total memory for all 16 cores (32G/core)
#SBATCH --time=72:00:00              # Maximum execution time (HH:MM:SS)
#SBATCH --output=slurmlog/%x_%j.out.log  # Standard output log file, %j is the job ID
#SBATCH --error=slurmlog/%x_%j.err.log   # Standard error log file

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Running in directory $(pwd)"
echo "Job started at $(date)"
echo "SLURM allocated ${SLURM_CPUS_PER_TASK} CPUs to this task."

module load cray-python

# Use the environment variable set by Slurm to control your script
# This ensures your script's worker pool matches the allocated CPUs
export WORKERS=${SLURM_CPUS_PER_TASK}

echo "Running python script with ${WORKERS} workers..."

srun python -u sharding.py \
        --data_dir /scratch/project_462001050/FineOPUS/opus-conversion \
        --out_dir /scratch/project_462001050/FineOPUS/fineopus-original \
        --output_file /scratch/project_462000941/members/shaoxion/FineOPUS/statistics/sharding_line_counts.csv \
        --log_file sharding_skipped_dirs.log \
        --pq_mismatch sharding_pq_mismatch.log \
        --num_workers ${WORKERS}

echo "Job finished at $(date)"