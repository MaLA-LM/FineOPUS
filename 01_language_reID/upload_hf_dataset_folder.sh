#!/bin/bash
#SBATCH --job-name=upload_hf_dataset
#SBATCH --output=../logs/upload_hf_dataset/%x_%j.out
#SBATCH --error=../logs/upload_hf_dataset/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --account=project_462000941

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

export HF_HUB_ENABLE_HF_TRANSFER=1

BASE_PATH="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-V2"
REPO_ID="MaLA-LM/mala-opus-dedup-2410-reLID"
PATH_IN_REPO=""
REVISION="main"
BATCH_SIZE=10


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
