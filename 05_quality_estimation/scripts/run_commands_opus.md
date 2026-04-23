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
  - `--part-writer`
  - `--part-max-bytes`
  - `--part-max-shards`
- `scripts/opus/submit_array_standard_g.sh` exposes the same worker runtime knobs as `submit_array.sh`, including the part-writer flags.
- `scripts/opus/run_merge.sh` reads both legacy `shard_*.jsonl` files and worker-owned `part-*.jsonl` files from `OUTPUT_BASE`.
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
bash scripts/opus/run_reaper.sh --db "$DB" --interval 300 --timeout-multiplier 1 --reset-failed

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
bash scripts/opus/submit_array.sh --model metricx24 --array 0-1 --concurrency 2 --time 01:00:00 --batch-size 64 --gpus 1 --db /scratch/project_462001050/opus_qe/jobs.db --output-base /scratch/project_462001050/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2

# MetricX with worker-owned part files enabled.
bash scripts/opus/submit_array.sh --model metricx24 --array 0-31 --concurrency 16 --time 12:00:00 --batch-size 64 --gpus 1 --part-writer --part-max-bytes 536870912 --part-max-shards 32 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

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
bash scripts/opus/submit_array.sh --model qwen3-4b-instruct-2507 --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --part-writer --part-max-bytes 536870912 --part-max-shards 32 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
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

## 3b. Submit standard-g workers (whole-node, 8 GCDs per array task)

`scripts/opus/submit_array_standard_g.sh` is the high-throughput companion
to `submit_array.sh`. Each array task allocates a whole LUMI-G node
(4x MI250X = 8 GCDs) and spawns 8 worker processes in parallel via `srun`,
each pinned to its own GCD. Use it when you have a large pending queue
and want more than the 200 GCDs that small-g caps at.

Key differences vs `submit_array.sh`:

- Partition is `standard-g` (whole-node billing; 200 running jobs cap;
  walltime max 48h instead of 72h).
- Each `--array` index = 1 node = 8 worker processes = 8 GCDs.
- The two submitters can run side by side against the same DB; SQLite WAL
  plus atomic claims handle concurrent workers safely (verified by
  `execution/opus_queue/tests/test_concurrent_claim.py`).

Throughput math:

| `--array`      | `--concurrency` | Nodes in flight | **GCDs in flight** |
| -------------- | --------------- | --------------- | ------------------ |
| `0-49%50`      | 50              | 50              | 400                |
| `0-99%100`     | 100             | 100             | **800**            |
| `0-199%200`    | 200             | 200             | 1600               |

Combine with `submit_array.sh` (200 GCDs on small-g) for up to ~1800 GCDs
total across both partitions.

Caveats before launching:

- ReMedy / LLM workers must have their model files already populated under
  the shared HF cache on `/scratch`. 8 workers booting at once on a cold
  cache will all try to download in parallel.
- Bicleaner is mostly CPU-bound and benefits little from 8x packing.
  Prefer `submit_array.sh` (small-g) for bicleaner backends. The
  standard-g task script will warn if you submit bicleaner anyway.
- Each worker writes its own per-PID temp files and uses per-LOCALID
  `TRITON_CACHE_DIR` / `TORCH_HOME` under
  `/scratch/.../.cache/{triton,torch}/${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}/${LOCALID}`,
  so caches don't race. These can be garbage-collected after the array
  finishes if disk pressure becomes an issue.

```bash
# COMET family on standard-g.
# 8 array tasks = 8 nodes = 64 GCDs in flight (matches the per-GCD
# small-g pattern of --array 0-63 --concurrency 64 in section 3, but
# with whole-node accounting).
bash scripts/opus/submit_array_standard_g.sh --model wmt23-cometkiwi-da-xl --array 0 --concurrency 1 --time 01:00:00 --batch-size 32 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

bash scripts/opus/submit_array_standard_g.sh --model xcomet-xl --array 0 --concurrency 1 --time 01:00:00 --batch-size 32 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# MetricX on standard-g (smaller jobs are usually fine on small-g; only
# scale up to standard-g if MetricX has a large pending backlog).
bash scripts/opus/submit_array_standard_g.sh --model metricx24 --array 0 --concurrency 1 --time 00:30:00 --batch-size 64 --part-writer --part-max-bytes 536870912 --part-max-shards 32 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# Standard-g with part consolidation enabled.
bash scripts/opus/submit_array_standard_g.sh --model qwen3-4b-instruct-2507 --array 0-99 --concurrency 100 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --part-writer --part-max-bytes 536870912 --part-max-shards 32 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# Qwen family on standard-g (FLORES-known-good LLM recipe; --enforce-eager
# is REQUIRED on LUMI/MI250X to avoid the vLLM 0.14 + ROCm torch.compile
# autotuning crash).
# 100 array tasks = 100 nodes = 800 GCDs in flight = the ~800-GCD target.
bash scripts/opus/submit_array_standard_g.sh --model qwen3-14b --array 0-99 --concurrency 100 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array_standard_g.sh --model qwen3-8b --array 0-99 --concurrency 100 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
###
bash scripts/opus/submit_array_standard_g.sh --model qwen3-4b-instruct-2507 --array 0 --concurrency 1 --time 01:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --part-writer --part-max-bytes 1536870912 --part-max-shards 32 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
###
bash scripts/opus/submit_array_standard_g.sh --model qwen3-4b-instruct-2507-fp8 --array 0-99 --concurrency 100 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array_standard_g.sh --model qwen3-4b-fp8 --array 0-99 --concurrency 100 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array_standard_g.sh --model qwen3-4b-awq --array 0-99 --concurrency 100 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array_standard_g.sh --model qwen3-1.7b --array 0-99 --concurrency 100 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
bash scripts/opus/submit_array_standard_g.sh --model qwen3-0.6b --array 0-99 --concurrency 100 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# Prometheus family (same FLORES-known-good LLM recipe).
bash scripts/opus/submit_array_standard_g.sh --model m-prometheus-7b --array 0-1 --concurrency 2 --time 01:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# ReMedy on standard-g. Confirm /pfs/lustrep3/.../hf is pre-populated;
# 8 workers booting on a cold cache at once will thrash the filesystem.
bash scripts/opus/submit_array_standard_g.sh --model shaomutan_remedy-9b-22 --array 0 --concurrency 1 --time 01:00:00 --part-writer --part-max-bytes 1536870912 --part-max-shards 32 --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"

# Combined throughput example: keep the existing small-g array (200 GCDs)
# AND submit a standard-g array (800 GCDs) for the same model. They share
# the DB; SQLite WAL handles concurrent claims atomically.
# Step 1: existing small-g (already documented in section 3)
bash scripts/opus/submit_array.sh --model qwen3-4b-instruct-2507-fp8 --array 0-127 --concurrency 64 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
# Step 2: add standard-g on top
bash scripts/opus/submit_array_standard_g.sh --model qwen3-4b-instruct-2507-fp8 --array 0-99 --concurrency 100 --time 24:00:00 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db "$DB" --output-base "$OUTPUT_BASE" --opus-root "$OPUS_ROOT"
```

After adding new scripts, refresh permissions on LUMI:

```bash
chmod +x scripts/opus/run_worker_standard_g.sh scripts/opus/run_worker_standard_g_task.sh scripts/opus/submit_array_standard_g.sh
```

If a standard-g job complains about `/var/spool/slurmd/.../run_worker_standard_g_task.sh`,
that is a path-resolution issue inside the batch job, not a missing `chmod` on the repo copy.

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
  --max-model-len 8192 \
  --part-writer \
  --part-max-bytes 536870912 \
  --part-max-shards 32
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

# Add --delete-shards if you want to remove legacy shard files and new part files after a successful merge.
sbatch ./scripts/opus/run_merge.sh --db "$DB" --output-base "$OUTPUT_BASE" --merged-base "$MERGED_BASE" --model metricx24 --delete-shards

# Merge a model after a part-writer run. Merge automatically reads both
# part-*.jsonl and any leftover legacy shard_*.jsonl files in OUTPUT_BASE.
sbatch ./scripts/opus/run_merge.sh --db "$DB" --output-base "$OUTPUT_BASE" --merged-base "$MERGED_BASE" --model qwen3-4b-instruct-2507 --force
```
