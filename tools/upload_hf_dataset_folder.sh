#!/bin/bash
#SBATCH --job-name=upload_hf_dataset
#SBATCH --output=../logs/upload_hf_dataset/%x_%j.out
#SBATCH --error=../logs/upload_hf_dataset/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462001050

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

# HF_XET_HIGH_PERFORMANCE
# Enabling high performance mode will try to saturate the network bandwidth of this machine and utilize all CPU cores for parallel upload/download activity.
# Consider this analogous to the legacy HF_HUB_ENABLE_HF_TRANSFER=1 environment variable but applied to hf-xet.
export HF_XET_HIGH_PERFORMANCE=1

BASE_PATH="/scratch/project_462000941/FineOPUS/FineOPUS-deduplicated"
REPO_ID="MaLA-LM/FineOPUS-Deduplicated"
PATH_IN_REPO=""
REVISION="main"
BATCH_SIZE=50


srun python ./upload_hf_dataset_folder.py \
    --base_path "$BASE_PATH" \
    --repo_id "$REPO_ID" \
    --path_in_repo "$PATH_IN_REPO" \
    --revision "$REVISION" \
    --batch_size "$BATCH_SIZE"
    # --dry_run

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
