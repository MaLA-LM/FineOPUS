#!/bin/bash
#SBATCH --job-name=mala_bilingual
#SBATCH --account=project_462000675
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=slurmlogs/%x_%j.out.log
#SBATCH --error=slurmlogs/%x_%j.err.log

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load required modules on LUMI
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python mala_bilingual.py