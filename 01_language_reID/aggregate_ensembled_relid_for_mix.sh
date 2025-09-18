#!/bin/bash
#SBATCH --job-name=aggregate_ensembled_relid_for_mix
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-06:00:00
#SBATCH --mem=256G
#SBATCH --account=project_462000941

start_time=$(date +%s)
echo "Job started at: $(date)"

module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source /flash/project_462000941/venv/opus2410_env/bin/activate

# tar -cf mala-opus-dedup-2410-ReLID-ENSEMBLED.tar ./mala-opus-dedup-2410-ReLID-ENSEMBLED
# tar -xf mala-opus-dedup-2410-ReLID-ENSEMBLED.tar

export CPU=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$CPU
export MKL_NUM_THREADS=$CPU
export OPENBLAS_NUM_THREADS=$CPU
export NUMEXPR_MAX_THREADS=$CPU
export ARROW_NUM_THREADS=$CPU         # PyArrow/Parquet 读取与 compute 的线程数
export MALLOC_ARENA_MAX=2             # 避免多 arena 导致内存碎片上升

# Optional: specify the range of tar files to process
START_IDX=0
END_IDX=63

INPUT_ROOT="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-TAR"
OUTPUT_ROOT="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-MIX-${START_IDX}-${END_IDX}"
STAGING_DIR="/scratch/project_462000941/members/zihao/OPUS2410/01_language_reID/mala-opus-dedup-2410-ReLID-ENSEMBLED-TMP"


srun --cpu-bind=cores python ./aggregate_ensembled_relid_for_mix.py \
    --input_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --max_rows_per_part 5000 \
    --compression snappy \
    --staging_dir "$STAGING_DIR" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX"


end_time=$(date +%s)
echo "Job ended at: $(date)"

duration=$((end_time - start_time))
echo "Job duration: $(date -u -d @${duration} +%T)"
