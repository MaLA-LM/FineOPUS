#!/bin/bash
#
#SBATCH --job-name=cvt       # A descriptive name for your job
#SBATCH --account=project_462000964      # IMPORTANT: Replace with your project ID
#SBATCH --partition=small           # The partition (queue) to submit to.
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --ntasks-per-node=1         # Number of tasks (processes) to run per node
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=64G                    # Memory per node (e.g., 4G, 16G, 64G)
#SBATCH --time=72:00:00             # Maximum execution time (HH:MM:SS)
#SBATCH --output=slurmlog/cvt_%j.out.log  # Standard output log file, %j is the job ID
#SBATCH --error=slurmlog/cvt_%j.err.log    # Standard error log file

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Running in directory $(pwd)"
echo "Job started at $(date)"

module load cray-python
srun python jsonl2parquet.py --input_dir /scratch/project_462000675/MaLA-LM/mala-opus-dedup-2410
# srun python jsonl2parquet.py --input_dir /scratch/project_462001050/MaLA-LM/mala-bilingual-translation-corpus
# srun python jsonl2parquet.py --input_dir /scratch/project_462000675/MaLA-LM/mala-monolingual-integration
# srun python jsonl2parquet.py --input_dir /scratch/project_462000675/MaLA-LM/mala-monolingual-split
