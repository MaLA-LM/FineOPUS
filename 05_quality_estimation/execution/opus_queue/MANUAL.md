# OPUS queue execution - operator manual

The `opus_queue` execution strategy scores OPUS sub-direction shards by
claiming work from a shared SQLite queue. Unlike FLORES, the unit of work
is a fixed-size sentence slice inside one direction, and workers pull
shards dynamically from the DB.

Package layout:

```text
execution/opus_queue/
|- db/
|- ops/
|- planning/
|- scoring/
|- tools/
|- worker/
|- build_queue.py
|- merge.py
|- queue_db.py
|- queue_ops.py
|- reaper.py
|- worker.py
`- executor.py

scripts/opus/
|- submit_array.sh
|- run_worker.sh
|- run_reaper.sh
`- run_merge.sh
```

## 1. One-time setup

1. Export the `lookup_OPUS_split_strategy2` worksheet as `lookup_OPUS.csv` and copy it to shared storage readable from login nodes. In this repo, the canonical local path is `data/lookups/lookup_OPUS.csv`.
2. Pick a DB location on shared storage, for example
   `/scratch/project_462001050/opus_qe/jobs.db`.
3. Pre-flight with `--dry-run` to sanity-check the lookup CSV counts:

    ```bash
    python -m execution.opus_queue.build_queue \
        --lookup /path/to/lookup_OPUS.csv \
        --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2 \
        --db /scratch/project_462001050/opus_qe/jobs.db \
        --dry-run
   ```

4. Populate the queue for real:

    ```bash
    python -m execution.opus_queue.build_queue \
        --lookup /path/to/lookup_OPUS.csv \
        --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2 \
        --db /scratch/project_462001050/opus_qe/jobs.db
   ```

Row counts are cached in `FineOPUS_test/.row_counts.json` next to the OPUS
root. Delete that file to force a fresh parquet metadata scan.

## 2. Calibration

Submit a single-task array for one model to sanity-check wallclock:

```bash
bash scripts/opus/submit_array.sh \
    --model metricx24 \
    --array 0-0 \
    --time 1:00:00 \
    --db /scratch/project_462001050/opus_qe/jobs.db \
    --output-base /scratch/project_462001050/opus_qe/shards
```

Inspect `run_events` for `done` rows and shard elapsed times:

```sql
sqlite3 /scratch/project_462001050/opus_qe/jobs.db <<'EOF'
SELECT event, COUNT(*) FROM run_events GROUP BY event;
SELECT detail FROM run_events WHERE event='done' ORDER BY ts DESC LIMIT 20;
EOF
```

If shard runtimes are consistently far from the 20-40 minute target,
adjust the planner seed or use overrides:

```bash
python -m execution.opus_queue.build_queue \
    --lookup /path/to/lookup_OPUS.csv \
    --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2 \
    --db /scratch/project_462001050/opus_qe/jobs.db \
    --reset-pending-for-model metricx24 \
    --shard-size-override metricx24:75000
```

`--reset-pending-for-model` refuses to run if any `done` rows already
exist for that model unless `--force` is supplied. Without `done` rows,
it deletes only non-`done` rows. The same rule applies to `--reassign`.

## 3. Steady state

Submit one array per model with whatever `%N` concurrency the scheduler
allows. `submit_array.sh` now defaults to the LUMI account/partition
(`project_462001050`, `small-g`) and exposes the same core runtime knobs
used in FLORES runs, including `--batch-size`, `--prompt-mode`,
`--max-tokens`, `--max-retries`, `--max-num-batched-tokens`,
`--max-num-seqs`, and `--max-model-len`. For OPUS LLM workers,
`MAX_MODEL_LEN` defaults to `8192` unless you explicitly override it:

```bash
bash scripts/opus/submit_array.sh --model metricx24 --array 0-63 --concurrency 32 --db /scratch/project_462001050/opus_qe/jobs.db --output-base /scratch/project_462001050/opus_qe/shards
bash scripts/opus/submit_array.sh --model qwen3-4b-instruct-2507 --array 0-127 --concurrency 64 --db /scratch/project_462001050/opus_qe/jobs.db --output-base /scratch/project_462001050/opus_qe/shards
bash scripts/opus/submit_array.sh --model wmt23-cometkiwi-da-xl --array 0-63 --concurrency 32 --db /scratch/project_462001050/opus_qe/jobs.db --output-base /scratch/project_462001050/opus_qe/shards

# FLORES-style LLM tuning example
bash scripts/opus/submit_array.sh --model qwen3-4b-instruct-2507 --array 0-127 --concurrency 64 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-retries 5 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 8192 --response-format json_schema --enforce-eager --db /scratch/project_462001050/opus_qe/jobs.db --output-base /scratch/project_462001050/opus_qe/shards
```

Launch the reaper on a login node in `screen` or `tmux`:

```bash
bash scripts/opus/run_reaper.sh --db /scratch/.../jobs.db --interval 300
```

If you want the reaper to give terminal `failed` shards a fresh retry budget
after the same model-specific cooldown, opt in explicitly:

```bash
bash scripts/opus/run_reaper.sh --db /scratch/.../jobs.db --interval 300 --reset-failed
```

Example: after a storage outage or model startup crash leaves a batch of rows
in `failed`, inspect them and then temporarily revive them:

```bash
sqlite3 /scratch/.../jobs.db <<'EOF'
SELECT model, COUNT(*) AS failed_rows
FROM jobs
WHERE status='failed'
GROUP BY model
ORDER BY failed_rows DESC, model ASC;
EOF

bash scripts/opus/run_reaper.sh --db /scratch/.../jobs.db --interval 300 --reset-failed
```

Keep that flag for cases where failures are likely transient (scheduler,
filesystem, model startup). It is off by default so terminal failures stay
visible instead of silently looping forever.

Monitor progress:

```sql
SELECT model, status, COUNT(*) FROM jobs GROUP BY model, status;
```

## 4. Resume after failures

If jobs die, just resubmit the arrays. Workers only claim `pending` rows,
the reaper reclaims stale `running` rows, and stale workers are prevented
from re-finalizing shards they no longer own.

## 5. Completion

When every row for a model is `done`, run the merge job:

```bash
sbatch ./scripts/opus/run_merge.sh \
    --db /scratch/.../jobs.db \
    --output-base /scratch/.../opus_qe/shards \
    --merged-base /scratch/.../opus_qe/merged \
    --model metricx24
```

Each direction gets:

```text
<merged-base>/<model>/<direction_key>.parquet
<merged-base>/<model>/<direction_key>.meta.json
```

The merge metadata records the source DB path and the shard count used to
produce the merged file.

## Output layout summary

```text
<output-base>/
    <model>/
        <direction_key>/
            shard_00000.jsonl
            shard_00001.jsonl
            ...
<merged-base>/
    <model>/
        <direction_key>.parquet
        <direction_key>.meta.json
```

This layout is intentionally flatter than the FLORES hive-style layout.
The `<model>` directory is always the queue/DB model key, and merge
reads `shard_*.jsonl` in numeric shard order and writes them as a
single Parquet file per direction.

## Troubleshooting

- SQLite locking: `queue_db.py` enables WAL, `busy_timeout=30000`, and
  retries `SQLITE_BUSY` with randomized backoff. If locking still flakes
  out on the cluster filesystem, move the DB or replace `queue_db.py`
  with a different backend behind the same API.
- Worker walltime: the launcher computes remaining time with `scontrol`
  and passes `--walltime-seconds` so workers exit cleanly before a hard
  kill.
- Merge ordering: merge sorts by parsed integer `shard_id`, not raw file
  name text.
- Schema bumps: `queue_db.initialize` checks `PRAGMA user_version` and
  refuses incompatible DBs until a migration is added.
