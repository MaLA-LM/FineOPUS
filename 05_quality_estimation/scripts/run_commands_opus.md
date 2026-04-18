# OPUS queue commands

## Shared paths

```bash
export LOOKUP=data/lookups/lookup_OPUS.csv
export OPUS_ROOT=/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2
export DB=/scratch/project_462001050/opus_qe/jobs.db
export OUTPUT_BASE=/scratch/project_462001050/opus_qe/shards
export MERGED_BASE=/scratch/project_462001050/opus_qe/merged
```

## LUMI bootstrap

```bash
# Run once from the repository root after copying / cloning the project to LUMI.
chmod +x envs/*.sh scripts/flores/*.sh scripts/opus/*.sh
```

## Notes

- `scripts/opus/submit_array.sh` is LUMI-only and already defaults to `--account project_462001050 --partition small-g`.
- `LOOKUP` must be a CSV exported from the `lookup_OPUS_split_strategy2` worksheet.
- `--model` must match the exact `winner_model` string written into the queue DB from `data/lookups/lookup_OPUS.csv`. Do not rely on aliases if the DB stores a different string.
- `scripts/opus/submit_array.sh` now exposes the main FLORES-style runtime knobs directly:
  - `--batch-size`
  - `--gpus`
  - `--prompt-mode`
  - `--temperature`
  - `--max-tokens`
  - `--max-retries`
  - `--dtype`
  - `--gpu-memory-utilization`
  - `--max-num-batched-tokens`
  - `--max-num-seqs`
  - `--max-model-len`
  - `--response-format`
  - `--structured-outputs-backend`
  - `--enforce-eager`
- Main knobs to change for queue planning:
  - `--array`: how many worker tasks you submit, for example `0-63` or `0-127`
  - `--concurrency`: Slurm `%N` cap
  - `--time`: walltime per worker
  - `--shard-size-override model:int`: sentences per shard when building or rebuilding the DB
- `execution/opus_queue/planning/shard_planner.py` is the canonical source of truth for OPUS shard-size seeds. Do not duplicate those values elsewhere; use `--shard-size-override model:int` when calibrating one model.

## 1. Create the queue database

```bash
# Dry run: read data/lookups/lookup_OPUS.csv, inspect OPUS directions, compute shard counts,
# but do not create or modify the SQLite DB.
python -m execution.opus_queue.ops.build_queue \
  --lookup data/lookups/lookup_OPUS.csv \
  --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2 \
  --db /scratch/project_462001050/opus_qe/jobs.db \
  --dry-run

# Real build: create / update the SQLite DB with directions and jobs.
module purge  
module use /appl/local/laifs/modules  
module load lumi-aif-singularity-bindings  
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
srun --account=project_462001050 --partition=small --time=01:00:00 --cpus-per-task=16 singularity exec $SIF bash  -c 'source /scratch/project_462001050/ibrahiam/envs/metric_venv/bin/activate && python -u -m execution.opus_queue.ops.build_queue --lookup data/lookups/lookup_OPUS.csv --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2 --db /scratch/project_462001050/opus_qe/jobs.db'

# Rebuild one model after calibration.
# Change --shard-size-override if shards are too short or too long.
python -m execution.opus_queue.ops.build_queue \
  --lookup "$LOOKUP" \
  --opus-root "$OPUS_ROOT" \
  --db "$DB" \
  --reset-pending-for-model metricx24 \
  --shard-size-override metricx24:75000

# If the lookup CSV changed the assigned model for some directions, rebuild those rows.
# Add --force only if you intentionally want to discard existing done rows.
python -m execution.opus_queue.ops.build_queue \
  --lookup "$LOOKUP" \
  --opus-root "$OPUS_ROOT" \
  --db "$DB" \
  --reassign
```

## 2. Start the reaper

```bash
# Default mode: reclaim stale running rows only.
# Keep this running in screen/tmux on a login node.
# Change --interval if you want more or less frequent stale-row sweeps.
# Change --timeout-multiplier if workers are being reaped too aggressively.
bash scripts/opus/run_reaper.sh --db "$DB" --interval 300 --timeout-multiplier 2

# If you see terminal failed shards caused by transient issues
# (scheduler kill, filesystem glitch, model startup crash),
# temporarily allow the reaper to move old failed rows back to pending.
bash scripts/opus/run_reaper.sh --db "$DB" --interval 300 --timeout-multiplier 2 --reset-failed

# Example check before using --reset-failed:
sqlite3 "$DB" <<'EOF'
SELECT model, COUNT(*) AS failed_rows
FROM jobs
WHERE status='failed'
GROUP BY model
ORDER BY failed_rows DESC, model ASC;
EOF
```

## 3. Submit workers for each model

```bash
# Change these per command:
# --array: how many worker slots to create
# --concurrency: scheduler cap (%N)
# --time: wallclock per worker
# --batch-size: per-forward batch size inside the scorer
# --max-tokens / --max-retries / --max-num-batched-tokens / --max-num-seqs:
#   key LLM runtime knobs carried over from FLORES

# COMET family
bash scripts/opus/submit_array.sh --model wmt22-cometkiwi-da --array 0-63 --concurrency 32 --time 24:00:00 --batch-size 8 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array.sh --model wmt23-cometkiwi-da-xl --array 0 --concurrency 1 --time 01:00:00 --batch-size 32 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array.sh --model xcomet-xl --array 0 --concurrency 1 --time 01:00:00 --batch-size 32 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# MetricX
bash scripts/opus/submit_array.sh --model metricx24 --array 0 --concurrency 1 --time 01:00:00 --batch-size 64 --gpus 1 --db /scratch/project_462001050/opus_qe/jobs.db --output-base /scratch/project_462001050/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2

# Qwen family
# LLM runs use the FLORES-known-good recipe (run_commands.md:95):
#   --prompt-mode batch --batch-size 32 --max-tokens 8192 --max-num-batched-tokens 8192
#   --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager
# --enforce-eager is REQUIRED on LUMI/MI250X: vLLM 0.14 + ROCm torch.compile autotuning
# crashes with "'KernelMetadata' object has no attribute 'cluster_dims'" otherwise.
bash scripts/opus/submit_array.sh --model qwen3-14b --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array.sh --model qwen3-8b --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
####
bash scripts/opus/submit_array.sh --model qwen3-4b-instruct-2507 --array 0 --concurrency 1 --time 01:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
####
bash scripts/opus/submit_array.sh --model qwen3-4b-instruct-2507-fp8 --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array.sh --model qwen3-4b-fp8 --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array.sh --model qwen3-4b-awq --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array.sh --model qwen3-1.7b --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array.sh --model qwen3-0.6b --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# Prometheus family (same FLORES-known-good LLM recipe)
bash scripts/opus/submit_array.sh --model m-prometheus-7b --array 0 --concurrency 1 --time 01:00:00 --batch-size 16 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
###
bash scripts/opus/submit_array.sh --model m-prometheus-3b --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# ReMedy
bash scripts/opus/submit_array.sh --model shaomutan_remedy-9b-22 --array 0-2 --concurrency 3 --time 01:00:00 --gpus 1 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# Bicleaner selectors
# If your DB stores auto, use auto. If it stores en-xx / es-xx / de-xx, submit that exact selector.
bash scripts/opus/submit_array.sh --model bicleaner-ai --array 0 --concurrency 1 --time 01:00:00 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array.sh --model en-xx --array 0-31 --concurrency 16 --time 12:00:00 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array.sh --model es-xx --array 0-31 --concurrency 16 --time 12:00:00 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array.sh --model de-xx --array 0-31 --concurrency 16 --time 12:00:00 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
```

## 4. Useful LLM overrides

```bash
# Change prompt style for qwen3-4b-instruct-2507.
bash scripts/opus/submit_array.sh --model qwen3-4b-instruct-2507 --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 8 --prompt-mode simple --max-tokens 256 --max-retries 5 --max-num-batched-tokens 16384 --max-num-seqs 128 --max-model-len 8192 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# Batch prompt mode for LLMs.
bash scripts/opus/submit_array.sh --model qwen3-4b-instruct-2507 --array 0-2 --concurrency 3 --time 01:00:00 --batch-size 32 --prompt-mode batch --max-tokens 256 --max-retries 5 --max-num-batched-tokens 16384 --max-num-seqs 128 --max-model-len 8192 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# Example FLORES-matching LLM settings.
bash scripts/opus/submit_array.sh --model qwen3-4b-instruct-2507 --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
```

## 5. Direct worker runs

```bash
# submit_array.sh now exposes the important FLORES-aligned runtime knobs.
# Direct worker runs are still useful when testing one interactive allocation.

python -m execution.opus_queue.worker \
  --db "$DB" \
  --model qwen3-4b-instruct-2507 \
  --backend llm \
  --output-base "$OUTPUT_BASE" \
  --opus-root "$OPUS_ROOT" \
  --batch-size 16 \
  --prompt-mode detailed \
  --temperature 0.0 \
  --max-tokens 256 \
  --max-retries 5 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 128 \
  --max-model-len 8192
```

## 6. Monitoring

```bash
sqlite3 "$DB" <<'EOF'
SELECT model, status, COUNT(*) FROM jobs GROUP BY model, status;
SELECT event, COUNT(*) FROM run_events GROUP BY event;
SELECT detail FROM run_events WHERE event='done' ORDER BY ts DESC LIMIT 20;
EOF
```

## 7. Merge completed shards

```bash
# Merge one model.
sbatch ./scripts/opus/run_merge.sh --db "$DB" --output-base "$OUTPUT_BASE" --merged-base "$MERGED_BASE" --model metricx24

# Merge everything that is complete.
sbatch ./scripts/opus/run_merge.sh --db "$DB" --output-base "$OUTPUT_BASE" --merged-base "$MERGED_BASE"

# Add --delete-shards if you want to remove shard_*.jsonl after a successful merge.
sbatch ./scripts/opus/run_merge.sh --db "$DB" --output-base "$OUTPUT_BASE" --merged-base "$MERGED_BASE" --model metricx24 --delete-shards
```
