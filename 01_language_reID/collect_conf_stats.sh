#!/bin/bash
#SBATCH --job-name=fineopus-collect-conf-stats-conlid
#SBATCH --output=../logs/fineopus-collect-conf-stats-conlid/%x_%j.out
#SBATCH --error=../logs/fineopus-collect-conf-stats-conlid/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --account=project_462000964


start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../.venv/bin/activate

# INPUT_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-GlotLID-conf-stats"
# TMP_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-GlotLID-conf-stats/tmp_conf_bins"

INPUT_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-ConLID-conf-stats"
TMP_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-ConLID-conf-stats/tmp_conf_bins"

python ./collect_conf_stats.py \
  --input_dir "$INPUT_DIR" \
  --tmp_dir   "$TMP_DIR"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
