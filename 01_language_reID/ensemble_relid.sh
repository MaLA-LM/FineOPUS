#!/bin/bash
#SBATCH --job-name=mala-opus-ensemble-relid
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --account=project_462000964
#SBATCH --array=0-63

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

export HF_HOME="/scratch/project_462000964/cache/huggingface"

GLOTLID_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-GlotLID"
CONLID_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-ConLID"
GLOTLID_THR_JSON="./mala-opus-dedup-2410-ReLID-by-GlotLID-conf-stats/aggregated_language_confidence_stats_quantiles.json"
CONLID_THR_JSON="./mala-opus-dedup-2410-ReLID-by-ConLID-conf-stats/aggregated_language_confidence_stats_quantiles.json"
OUT_ROOT="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-ENSEMBLED"
FILELIST="./mala-opus-dedup-2410-ReLID-Relpath-filelists/filelist_${SLURM_ARRAY_TASK_ID}.txt"


python ./ensemble_relid.py \
  --glotlid_dir "$GLOTLID_DIR" \
  --conlid_dir "$CONLID_DIR" \
  --glotlid_thr_json "$GLOTLID_THR_JSON" \
  --conlid_thr_json "$CONLID_THR_JSON" \
  --out_root "$OUT_ROOT" \
  --part_size 15000000 \
  --filelist "$FILELIST" \
  --compression snappy \
  --strict_check

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
