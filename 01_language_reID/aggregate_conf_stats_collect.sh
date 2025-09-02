#!/bin/bash
#SBATCH --job-name=mala-opus-conf-stats-glotlid-collect
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-10:00:00
#SBATCH --mem=128G
#SBATCH --account=project_462000941


start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

# INPUT_DIR="./mala-opus-dedup-2410-ReLID-by-GlotLID-conf-stats"
# TMP_DIR="./mala-opus-dedup-2410-ReLID-by-GlotLID-conf-stats/tmp_conf_bins"

# INPUT_DIR="./mala-opus-dedup-2410-ReLID-by-ConLID-conf-stats"
# TMP_DIR="./mala-opus-dedup-2410-ReLID-by-ConLID-conf-stats/tmp_conf_bins"

python ./aggregate_conf_stats_collect.py \
  --input_dir "$INPUT_DIR" \
  --tmp_dir   "$TMP_DIR"

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
