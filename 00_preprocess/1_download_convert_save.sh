#!/bin/bash
#
#SBATCH --job-name=dl_cvt       # A descriptive name for your job
#SBATCH --account=project_462000964      # IMPORTANT: Replace with your project ID
#SBATCH --partition=small           # The partition (queue) to submit to.
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --ntasks-per-node=1         # Number of tasks (processes) to run per node
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=64G                    # Memory per node (e.g., 4G, 16G, 64G)
#SBATCH --time=72:00:00             # Maximum execution time (HH:MM:SS)
#SBATCH --output=slurmlog/%x_%j.out.log  # Standard output log file, %j is the job ID
#SBATCH --error=slurmlog/%x_%j.err.log    # Standard error log file

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Running in directory $(pwd)"
echo "Job started at $(date)"

module load cray-python


tmp_dir="/scratch/project_462000675/opus_tmp"
mkdir -p ${tmp_dir}
trap 'rm -rf ${tmp_dir}/*' EXIT

srun python -u download_convert_save.py \
        --input_json OPUS_corpus_collection.json \
        --mapping_file_dir /scratch/project_462000941/members/shaoxion/FineOPUS/00_preprocess/OPUS/corpus \
        --out_dir /scratch/project_462000675/opus \
        --tmp_dir $tmp_dir \
        --progress_file dl_cvt_progress.txt \
        --error_log dl_cvt_log.json
