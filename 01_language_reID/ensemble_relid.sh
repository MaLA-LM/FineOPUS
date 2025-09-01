#!/bin/bash
#SBATCH --job-name=mala-opus-ensemble-relid
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --account=project_462000964

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

GLOTLID_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-GlotLID"
CONLID_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-ConLID"
GLOTLID_THR_JSON="./mala-opus-dedup-2410-ReLID-by-GlotLID-conf-stats/aggregated_language_confidence_stats_quantiles.json"
CONLID_THR_JSON="./mala-opus-dedup-2410-ReLID-by-ConLID-conf-stats/aggregated_language_confidence_stats_quantiles.json"
OUT_ROOT="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-ENSEMBLED"


python ./ensemble_relid.py \
  --glotlid_dir "$GLOTLID_DIR" \
  --conlid_dir "$CONLID_DIR" \
  --glotlid_thr_json "$GLOTLID_THR_JSON" \
  --conlid_thr_json "$CONLID_THR_JSON" \
  --out_root "$OUT_ROOT" \
  --part_size 15000000 \
  --compression snappy \
  --strict_check

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
