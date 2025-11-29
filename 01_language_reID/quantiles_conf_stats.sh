#!/bin/bash
#SBATCH --job-name=fineopus-quantiles-conf-stats-glotlid
#SBATCH --output=../logs/fineopus-quantiles-conf-stats-glotlid/%x_%j.out
#SBATCH --error=../logs/fineopus-quantiles-conf-stats-glotlid/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --account=project_462000964


start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../.venv/bin/activate

OUTPUT_FILE="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-GlotLID-conf-stats/conf_stats_quantiles.json"
TMP_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-GlotLID-conf-stats/tmp_conf_bins"

OUTPUT_FILE="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-ConLID-conf-stats/conf_stats_quantiles.json"
TMP_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-ConLID-conf-stats/tmp_conf_bins"

python ./quantiles_conf_stats.py \
  --collect_stats "$TMP_DIR/_collect_stats.json" \
  --output_file   "$OUTPUT_FILE" \
  --tmp_dir       "$TMP_DIR" \
  --keep-tmp


end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
