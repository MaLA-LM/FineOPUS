#!/usr/bin/env bash
# Build an FT-only manifest, validate the runtime, then submit the HF evaluations.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DRY_RUN=0
SKIP_PREFLIGHT=0
PARTITION=""
JOB_TIME=""
ACCOUNT=project_465002530
DEPENDENCY=""
PLAN_ARGS=()
PLAN_PYTHON="${FT_PLAN_PYTHON:-/usr/bin/python3.11}"

usage() {
    cat <<EOF
Usage: bash $0 [options]
  --dry-run                 Write coverage and task manifest without submitting
  --model-glob GLOB         Select FT model names (repeatable; default FT-*)
  --datasets LIST           Default FLORES-200,NTREX-128,BOUQuET_Sentence
  --languages LIST          Optional normalized FT language subset
  --few-shot N              Demonstrations per direction (default 3)
  --limit N                 Smoke test; uses a separate output directory
  --partition NAME          Defaults: dev-g for smoke, small-g for full evaluation
  --job-time TIME           Defaults: 30 minutes for smoke, 3 days for full evaluation
  --account ACCOUNT         Slurm account (default $ACCOUNT)
  --dependency SPEC          Optional sbatch dependency, e.g. afterok:JOB_ID
  --models-root DIR         HF FT model root
  --data-root DIR           Prepared benchmark root
  --results-root DIR        FT output root (must be outside evaluation/results)
  --language-report FILE    WikiANN language CSV
  --language-policy FILE    Reviewed label fixes and dataset aliases
  --comet-model MODEL       Default Unbabel/wmt22-comet-da
  --comet-batch-size N      Default 8
  --no-comet                Compute BLEU and chrF++ only
  --enforce-eager           Disable vLLM graph capture
  --max-model-len N         Default 4096
  --max-new-tokens N        Default 256
  --force                   Include completed models
  --overwrite               Recompute existing directions (implies --force)
  --skip-preflight          Skip submit-time runtime/data checks
EOF
}
while (( $# )); do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --job-time) JOB_TIME="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --overwrite) PLAN_ARGS+=(--overwrite --force); shift ;;
        --force|--no-comet|--enforce-eager) PLAN_ARGS+=("$1"); shift ;;
        --model-glob|--datasets|--languages|--few-shot|--limit|--models-root|--data-root|--results-root|--language-report|--language-policy|--comet-model|--comet-batch-size|--max-model-len|--max-new-tokens)
            PLAN_ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
mkdir -p "$SCRIPT_DIR/tmp" "$SCRIPT_DIR/logs"
pointer=$(mktemp "$SCRIPT_DIR/tmp/manifest.XXXXXX")
trap 'rm -f -- "$pointer"' EXIT
# Planning only requires the system Python; dry runs never load the GPU runtime.
"$PLAN_PYTHON" "$SCRIPT_DIR/workflow.py" plan --manifest-pointer "$pointer" "${PLAN_ARGS[@]}"
manifest=$(cat "$pointer")
task_count=$("$PLAN_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"]))' "$manifest")
if [[ "$task_count" == 0 ]]; then
    echo 'No unfinished model jobs; nothing to submit.'
    exit 0
fi
is_full=$("$PLAN_PYTHON" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["tasks"][0]["settings"]["limit"] is None))' "$manifest")
if [[ "$is_full" == 1 ]]; then
    PARTITION="${PARTITION:-small-g}"
    JOB_TIME="${JOB_TIME:-3-00:00:00}"
else
    PARTITION="${PARTITION:-dev-g}"
    JOB_TIME="${JOB_TIME:-00:30:00}"
fi
if [[ "$PARTITION" == dev-g && "$task_count" != 1 ]]; then
    echo 'dev-g allows one job per user/project. Select one smoke model with --model-glob.' >&2
    exit 2
fi
echo "Plan: $task_count independent model jobs; partition=$PARTITION; time=$JOB_TIME"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "No job submitted (dry-run=$DRY_RUN, unfinished tasks=$task_count)."
    exit 0
fi
if [[ "$SKIP_PREFLIGHT" == 0 ]]; then
    bash "$SCRIPT_DIR/worker.sh" --preflight "$manifest"
fi
sbatch_args=(--parsable --partition="$PARTITION"
    --time="$JOB_TIME" --account="$ACCOUNT" --chdir="$SCRIPT_DIR"
    --output="$SCRIPT_DIR/logs/%x_%j.out" --error="$SCRIPT_DIR/logs/%x_%j.err")
[[ -z "$DEPENDENCY" ]] || sbatch_args+=(--dependency="$DEPENDENCY")
jobs_file="${manifest%.json}.jobs.tsv"
printf 'task_id\tmodel\tjob_id\n' > "$jobs_file"
while IFS=$'\t' read -r task_id model_name; do
    job_id=$(sbatch "${sbatch_args[@]}" --job-name="eval-${model_name}" \
        "$SCRIPT_DIR/worker.sh" "$manifest" "$SCRIPT_DIR" "$task_id")
    printf '%s\t%s\t%s\n' "$task_id" "$model_name" "$job_id" >> "$jobs_file"
    echo "Submitted $job_id: $model_name"
done < <("$PLAN_PYTHON" -c 'import json,sys; [print(str(t["task_id"])+"\t"+t["settings"]["model_name"]) for t in json.load(open(sys.argv[1]))["tasks"]]' "$manifest")
echo "Submitted $task_count independent model jobs. Manifest: $manifest"
echo 'The last successful full-evaluation job aggregates results; no extra job is submitted.'
echo "Manual aggregation after completion: bash $SCRIPT_DIR/aggregate.sh $manifest"
