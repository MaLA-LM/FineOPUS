#!/bin/bash
#SBATCH --job-name=delete_hf_folders
#SBATCH --output=../logs/delete_hf_folders/%x_%j.out
#SBATCH --error=../logs/delete_hf_folders/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --account=project_462001050

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

export HF_XET_HIGH_PERFORMANCE=1

# Target dataset repos to clean up (removes same-lang, Zyyy-script, xxx language folders, and _DONE files).
REPO_IDS=(
    "MaLA-LM/FineOPUS-ReLID"
    "MaLA-LM/FineOPUS-Deduplicated"
    "MaLA-LM/FineOPUS-Filtered-Stage1"
    "MaLA-LM/FineOPUS-Filtered-Stage2"
    "MaLA-LM/FineOPUS-Filtered-Stage3"
    "MaLA-LM/FineOPUS-Filtered-Stage4"
    "MaLA-LM/FineOPUS-Filtered-Stage5"
)

REVISION="main"
BATCH_SIZE=100

# Set DRY_RUN=1 to preview only (default). Set DRY_RUN=0 to actually delete.
DRY_RUN="${DRY_RUN:-1}"
DRY_FLAG=""
if [ "$DRY_RUN" = "1" ]; then
    DRY_FLAG="--dry_run"
    echo ">>> DRY RUN mode (no deletions). Set DRY_RUN=0 to delete for real."
fi

for REPO_ID in "${REPO_IDS[@]}"; do
    echo ""
    echo "########################################################"
    echo "# Repo: $REPO_ID"
    echo "########################################################"
    srun python ./delete_hf_dataset_folders.py \
        --repo_id "$REPO_ID" \
        --revision "$REVISION" \
        --batch_size "$BATCH_SIZE" \
        $DRY_FLAG
done

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
