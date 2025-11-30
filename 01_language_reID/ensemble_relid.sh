#!/bin/bash
#SBATCH --job-name=fineopus-ensemble-relid
#SBATCH --output=../logs/fineopus-ensemble-relid/%x_%j.out
#SBATCH --error=../logs/fineopus-ensemble-relid/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --account=project_462000964
#SBATCH --array=256-383

start_time=$(date +%s)
echo "Job started at: $(date)"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source ../.venv/bin/activate

export HF_HOME="/scratch/project_462000941/cache/huggingface"

GLOTLID_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-GlotLID"
CONLID_DIR="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-ConLID"
GLOTLID_THR_JSON="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-GlotLID-conf-stats/conf_stats_quantiles.json"
CONLID_THR_JSON="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-by-ConLID-conf-stats/conf_stats_quantiles.json"
OUT_ROOT="/scratch/project_462001069/members/zihao/FineOPUS/fineopus-original-ReLID-ENSEMBLED-TAR/tmp_${SLURM_ARRAY_TASK_ID}"
FILELIST="./filelists/fineopus-original-ReLID-relpath-filelists-512-shard/filelist_${SLURM_ARRAY_TASK_ID}.txt"


python ./ensemble_relid.py \
  --glotlid_dir "$GLOTLID_DIR" \
  --conlid_dir "$CONLID_DIR" \
  --glotlid_thr_json "$GLOTLID_THR_JSON" \
  --conlid_thr_json "$CONLID_THR_JSON" \
  --out_root "$OUT_ROOT" \
  --max_rows_per_pair_shard 1_000_000 \
  --min_rows_per_pair_shard 100_000 \
  --filelist "$FILELIST" \
  --compression zstd
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
