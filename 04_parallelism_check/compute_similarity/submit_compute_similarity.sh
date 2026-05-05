#!/bin/bash
# ---------------------------------------------------------------------------
# submit_compute_similarity.sh
#
# For each model in model_to_language_pairs.json:
#   1. Enumerate all parquet shards under INPUT_DIR
#   2. Skip shards whose output already contains 'similarity_score'
#   3. Bin-pack remaining shards by file size into SLURM array tasks
#   4. Write a per-model manifest JSON
#   5. Submit one SLURM array job per model
#
# Usage:
#   bash submit_compute_similarity.sh [--tasks N] [--dry-run] [--model NAME]
#
# Options:
#   --tasks N      Max number of SLURM array tasks per model (default: 50)
#   --dry-run      Print sbatch commands without submitting
#   --model NAME   Only process this model (default: all)
# ---------------------------------------------------------------------------
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Configurable paths ----------------------------------------------------
MODEL_PAIRS_JSON="${MODEL_PAIRS_JSON:-$SCRIPT_DIR/model_to_language_pairs.json}"
INPUT_DIR="${INPUT_DIR:-/scratch/project_462001069/FineOPUS/intermediate/FineOPUS-Filtered-Stage2-Split}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_462001069/FineOPUS/intermediate/FineOPUS-Filtered-Stage2-Scored}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOG_DIR="$REPO_ROOT/logs/similarity_scoring"
MANIFEST_DIR="${MANIFEST_DIR:-$SCRIPT_DIR/manifests}"
# ----------------------------------------------------------------------------

N_TASKS=128
DRY_RUN=0
FILTER_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)    N_TASKS="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --model)    FILTER_MODEL="$2"; shift 2 ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR" "$MANIFEST_DIR"

SUBMIT_PLAN="$MANIFEST_DIR/submit_plan.txt"

echo "======================================================="
echo "  FineOPUS Similarity Scoring Job Submission"
echo "======================================================="
echo "  Model pairs JSON : $MODEL_PAIRS_JSON"
echo "  Input dir        : $INPUT_DIR"
echo "  Output dir       : $OUTPUT_DIR"
echo "  Max tasks/model  : $N_TASKS"
echo "  Manifest dir     : $MANIFEST_DIR"
echo "  Dry run          : $DRY_RUN"
echo "  Filter model     : ${FILTER_MODEL:-(all)}"
echo "======================================================="
echo ""
echo "Scanning parquet files and generating manifests..."
echo ""

python3 - <<PYEOF
import json
import heapq
from pathlib import Path

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("[WARN] pyarrow not available; using file-existence check only.")


def is_processed(out_path: Path) -> bool:
    """Return True if the output parquet already has 'similarity_score'."""
    if not out_path.exists():
        return False
    if not HAS_PYARROW:
        return False
    try:
        schema = pq.read_schema(out_path)
        return "similarity_score" in schema.names
    except Exception:
        return False


model_pairs_json = "$MODEL_PAIRS_JSON"
input_dir = Path("$INPUT_DIR")
output_dir = Path("$OUTPUT_DIR")
manifest_dir = Path("$MANIFEST_DIR")
n_tasks_max = $N_TASKS
filter_model = "$FILTER_MODEL"
submit_plan_path = "$SUBMIT_PLAN"

with open(model_pairs_json) as f:
    data = json.load(f)

plan_lines = []

for model, entries in data.items():
    if filter_model and model != filter_model:
        continue

    pairs = [e[0] if isinstance(e, list) else e for e in entries]

    unprocessed = []
    total_parquets = 0
    already_done = 0

    for lang_pair in pairs:
        pair_dir = input_dir / lang_pair
        if not pair_dir.exists():
            continue
        for shard_path in sorted(pair_dir.glob(f"{lang_pair}_shard_*.parquet")):
            total_parquets += 1
            out_path = output_dir / lang_pair / shard_path.name
            if is_processed(out_path):
                already_done += 1
                continue
            unprocessed.append({
                "input_path": str(shard_path),
                "output_path": str(out_path),
                "lang_pair": lang_pair,
                "size": shard_path.stat().st_size,
            })

    n_todo = len(unprocessed)
    total_gb = sum(x["size"] for x in unprocessed) / 1e9

    print(f"Model : {model}")
    print(f"  Parquets  total={total_parquets}  done={already_done}  remaining={n_todo} ({total_gb:.1f} GB)")

    if n_todo == 0:
        print(f"  [SKIP] All parquets already processed.")
        print()
        continue

    # Bin-pack shards into tasks, balancing by file size
    actual_tasks = min(n_tasks_max, n_todo)
    unprocessed.sort(key=lambda x: x["size"], reverse=True)
    heap = [(0, i, []) for i in range(actual_tasks)]
    heapq.heapify(heap)
    for item in unprocessed:
        load, idx, members = heapq.heappop(heap)
        heapq.heappush(heap, (load + item["size"], idx, members + [item]))

    chunks = [[] for _ in range(actual_tasks)]
    for load, idx, members in heap:
        chunks[idx] = members

    # Summarise load balance
    chunk_sizes_gb = [sum(e["size"] for e in c) / 1e9 for c in chunks]
    print(f"  Array tasks: {actual_tasks}  "
          f"load min={min(chunk_sizes_gb):.1f} GB  "
          f"max={max(chunk_sizes_gb):.1f} GB  "
          f"avg={sum(chunk_sizes_gb)/len(chunk_sizes_gb):.1f} GB")

    model_safe = model.replace("/", "__")
    manifest_path = manifest_dir / f"{model_safe}.json"
    with open(manifest_path, "w") as fp:
        json.dump({"model": model, "chunks": chunks}, fp)

    print(f"  Manifest   : {manifest_path}")
    print()
    plan_lines.append(f"{model}|{manifest_path}|{actual_tasks}")

with open(submit_plan_path, "w") as f:
    f.write("\n".join(plan_lines) + ("\n" if plan_lines else ""))
PYEOF

if [[ ! -s "$SUBMIT_PLAN" ]]; then
    echo "No jobs to submit. All parquets already processed."
    exit 0
fi

echo "Submitting jobs..."
echo ""

while IFS='|' read -r model manifest_path n_tasks; do
    [[ -z "$model" ]] && continue
    last_idx=$((n_tasks - 1))
    echo ">>> Model: $model"
    echo "    Array : 0-${last_idx}  |  Manifest: $manifest_path"

    SUBMIT_CMD="sbatch \
        --array=0-${last_idx} \
        --export=ALL,MODEL=\"${model}\",MANIFEST_FILE=\"${manifest_path}\",BATCH_SIZE=${BATCH_SIZE} \
        ${SCRIPT_DIR}/compute_similarity.sh"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "    [DRY RUN] $SUBMIT_CMD"
    else
        JOB_ID=$(eval $SUBMIT_CMD)
        echo "    Submitted: $JOB_ID"
    fi
    echo ""
done < "$SUBMIT_PLAN"

echo "Done submitting all jobs."
