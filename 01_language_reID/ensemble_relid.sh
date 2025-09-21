#!/bin/bash
#SBATCH --job-name=mala-opus-ensemble-relid
#SBATCH --output=../logs/mala-opus-ensemble-relid/%x_%j.out
#SBATCH --error=../logs/mala-opus-ensemble-relid/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --account=project_462000964
#SBATCH --array=0-511

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

export HF_HOME="/scratch/project_462001069/cache/huggingface"

GLOTLID_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-GlotLID"
CONLID_DIR="/scratch/project_462000964/MaLA-LM/mala-opus-dedup-2410-ReLID-by-ConLID"
GLOTLID_THR_JSON="./mala-opus-dedup-2410-ReLID-by-GlotLID-conf-stats/aggregated_language_confidence_stats_quantiles.json"
CONLID_THR_JSON="./mala-opus-dedup-2410-ReLID-by-ConLID-conf-stats/aggregated_language_confidence_stats_quantiles.json"
OUT_ROOT="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-TAR-V2/tmp_${SLURM_ARRAY_TASK_ID}"
FILELIST="./mala-opus-dedup-2410-ReLID-Relpath-filelists/filelist_${SLURM_ARRAY_TASK_ID}.txt"


python ./ensemble_relid.py \
  --glotlid_dir "$GLOTLID_DIR" \
  --conlid_dir "$CONLID_DIR" \
  --glotlid_thr_json "$GLOTLID_THR_JSON" \
  --conlid_thr_json "$CONLID_THR_JSON" \
  --out_root "$OUT_ROOT" \
  --max_rows_per_pair_shard 1000000 \
  --min_rows_per_pair_shard 100000 \
  --filelist "$FILELIST" \
  --compression snappy
  # --strict_check
  
status=$?

if [ $status -ne 0 ]; then
  echo "Processing failed (exit code $status); skipping compression." >&2
  end_time=$(date +%s)
  echo "Job ended at: $(date)"
  duration=$((end_time - start_time))
  echo "Job duration: $(date -u -d @${duration} +%T)"
  exit $status
fi

echo "Starting tar packaging of OUT_ROOT: $OUT_ROOT"

if [ ! -d "$OUT_ROOT" ]; then
  echo "OUT_ROOT directory not found: $OUT_ROOT" >&2
else
  ARCHIVE_BASE="${OUT_ROOT%/}"
  ARCHIVE_DIRNAME=$(basename "$ARCHIVE_BASE")
  ARCHIVE_PARENT=$(dirname "$ARCHIVE_BASE")

  echo "Creating tar archive -> ${ARCHIVE_BASE}.tar"
  if tar -C "$ARCHIVE_PARENT" -cf "$ARCHIVE_PARENT/${ARCHIVE_DIRNAME}.tar" "$ARCHIVE_DIRNAME"; then
    echo "Packaging successful; removing original directory $OUT_ROOT"
    rm -rf "$OUT_ROOT"
  else
    echo "Packaging failed; original directory retained." >&2
  fi
fi

end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
