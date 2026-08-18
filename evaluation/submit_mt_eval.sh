#!/usr/bin/env bash
# Submit model/checkpoint translation evaluations as a bounded Slurm array.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
MODELS_ROOT="$REPO_ROOT/slm_from_scratch/output"
DATA_ROOT="$SCRIPT_DIR/data"
RESULTS_ROOT="$SCRIPT_DIR/results"
LANGUAGE_MANIFEST="$REPO_ROOT/tools/token_stats/token_count_tasks_Multilingual-Mix-Mono/run_20260714_145253/jsonl_manifest.tsv"
CHECKPOINT="latest"
DATASETS="FLORES-200,NTREX-128,BOUQuET_Sentence"
FEW_SHOT=3
MAX_CONCURRENT=4
LIMIT=""
DRY_RUN=0
FORCE=0
KEEP_HF=0
OVERWRITE=0
ENFORCE_EAGER=0
SKIP_PREFLIGHT=0
MODEL_GLOBS=()

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --checkpoint VALUE       latest (default), all, integer, or iter_NNNNNNN
  --model-glob GLOB        Include model basenames matching GLOB; repeatable
  --datasets LIST          Comma-separated datasets (default: all three)
  --few-shot N             Demonstrations per prompt (default: 3)
  --limit N                Score at most N examples/direction (smoke run)
  --max-concurrent N       Maximum simultaneous array jobs (default: 4)
  --models-root DIR        Trained model root
  --data-root DIR          Prepared dataset root
  --results-root DIR       Evaluation output root
  --language-manifest FILE Multilingual training language manifest
  --force                  Include tasks already carrying _SUCCESS
  --overwrite              Re-run completed directions in included tasks
  --keep-hf                Preserve each converted HF model
  --enforce-eager          Use vLLM eager mode
  --skip-preflight         Submit without local data/runtime checks
  --dry-run                Build and print the manifest, do not submit

Examples:
  $0 --dry-run
  $0 --model-glob '0.4B_Pretrain_eng_Latn-deu*' --limit 20
  $0 --model-glob '0.9B_Pretrain_multilingual_*' --max-concurrent 2
EOF
}

while (( $# )); do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --model-glob) MODEL_GLOBS+=("$2"); shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --few-shot) FEW_SHOT="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --models-root) MODELS_ROOT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --language-manifest) LANGUAGE_MANIFEST="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        --keep-hf) KEEP_HF=1; shift ;;
        --enforce-eager) ENFORCE_EAGER=1; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

mkdir -p "$RESULTS_ROOT/task_manifests" "$SCRIPT_DIR/logs/eval" "$SCRIPT_DIR/tmp"
run_stamp=$(date +%Y%m%d_%H%M%S)
TASK_MANIFEST="$RESULTS_ROOT/task_manifests/mt_eval_${run_stamp}.tsv"

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.7
source "$SCRIPT_DIR/eval_env/bin/activate"
export PYTHONNOUSERSITE=1

manifest_args=(
    --models-root "$MODELS_ROOT"
    --output-root "$RESULTS_ROOT"
    --language-manifest "$LANGUAGE_MANIFEST"
    --checkpoint "$CHECKPOINT"
    --output "$TASK_MANIFEST"
)
for pattern in "${MODEL_GLOBS[@]}"; do manifest_args+=(--model-glob "$pattern"); done
[[ "$FORCE" == 0 ]] || manifest_args+=(--force)
python "$SCRIPT_DIR/build_mt_manifest.py" "${manifest_args[@]}"

task_count=$(( $(wc -l < "$TASK_MANIFEST") - 1 ))
echo "Task manifest: $TASK_MANIFEST"
sed -n '1,6p' "$TASK_MANIFEST"
if (( task_count > 5 )); then echo "... ($task_count tasks total)"; fi

if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run only; no job submitted."
    exit 0
fi

if [[ "$SKIP_PREFLIGHT" == 0 ]]; then
    all_languages=$(python - "$TASK_MANIFEST" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = csv.DictReader(handle, delimiter="\t")
    print(",".join(sorted({lang for row in rows for lang in row["languages"].split(",")})))
PY
)
    python "$SCRIPT_DIR/mt_eval.py" \
        --data-root "$DATA_ROOT" \
        --output-dir "$RESULTS_ROOT/.preflight" \
        --languages "$all_languages" \
        --datasets "$DATASETS" \
        --few-shot "$FEW_SHOT" \
        --preflight-only \
        --check-runtime
fi

worker_args=(
    --task-manifest "$TASK_MANIFEST"
    --data-root "$DATA_ROOT"
    --datasets "$DATASETS"
    --few-shot "$FEW_SHOT"
)
[[ -z "$LIMIT" ]] || worker_args+=(--limit "$LIMIT")
[[ "$KEEP_HF" == 0 ]] || worker_args+=(--keep-hf)
[[ "$OVERWRITE" == 0 ]] || worker_args+=(--overwrite)
[[ "$ENFORCE_EAGER" == 0 ]] || worker_args+=(--enforce-eager)

pushd "$SCRIPT_DIR" >/dev/null
submission=$(sbatch --array="0-$((task_count - 1))%$MAX_CONCURRENT" \
    "$SCRIPT_DIR/convert_and_eval_mt.sh" "${worker_args[@]}")
popd >/dev/null
echo "$submission"
echo "After the array finishes, aggregate with:"
echo "  python $SCRIPT_DIR/aggregate_mt_results.py --results-root $RESULTS_ROOT"
