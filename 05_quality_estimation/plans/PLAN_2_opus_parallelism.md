# Plan 2 — OPUS parallelism: sub-direction sharding, SQLite job queue, reaper, merge

## Prerequisite

Plan 1 must be done. This plan assumes `execution/` exists with `execution/base.py::ExecutionStrategy`, a registry in `execution/__init__.py`, and `execution/flores_array/` holding the FLORES strategy. All new code in this plan lives under `execution/opus_queue/` and is registered as a second strategy named `"opus_queue"`. No FLORES code or dataset adapter code is touched.

## Why a different execution model is needed

The FLORES strategy's atomic unit of work is one translation direction. Its assumptions:

- every direction takes a similar, small amount of time (~1012 sentences, bounded),
- a static hash-to-shard mapping distributes work evenly,
- a SLURM array task can finish its bucket within its walltime,
- if a task dies, restarting the array re-does at most one direction.

OPUS breaks all four:

- directions vary from ~4k to ~1.25 billion sentences (see `lookup_OPUS.xlsx`), a ~300,000× spread,
- total workload is ~46.8 billion sentences ≈ 584k GPU-hours across ~9,850 directions and 7 models,
- the largest single direction for metricx24 is ~1.25 billion sentences ≈ 12,488h — no SLURM walltime survives it,
- losing a partial direction on kill loses hours or days of compute.

The fix is to make the atomic unit smaller than a direction: **a shard is a fixed-size slice of sentences within one direction**. Shards are sized so one shard is ~20–40 minutes of wallclock on the assigned model. Workers claim shards from a database-backed queue and commit per-shard outputs, so a kill only loses the in-flight shard.

## Architecture

```
execution/opus_queue/
├── __init__.py
├── executor.py          # OpusQueueExecutor : ExecutionStrategy
├── queue_db.py          # SQLite wrapper: schema, claim_next, mark_done, mark_failed, reset_stale
├── build_queue.py       # CLI: populate jobs table from lookup_OPUS.xlsx + per-direction sentence counts
├── worker.py            # worker loop: claim → run → commit, respects walltime
├── reaper.py            # CLI / daemon: reset stale 'running' rows to 'pending'
├── merge.py             # CLI: concatenate completed shard outputs into one file per direction
├── shard_planner.py     # pure function: (direction, n_sentences, model) → list of (start, end)
└── db/schema.sql        # canonical CREATE TABLE statements

scripts/opus/
├── submit_array.sh      # sbatch wrapper: one array per model
├── run_worker.sh        # inner launcher: env setup + python -m execution.opus_queue.worker
├── run_reaper.sh        # launches the reaper loop
└── run_merge.sh         # post-run merge
```

The database file lives on shared cluster storage at a path the user configures (e.g. `execution/opus_queue/jobs.db`). It is the single source of truth for what is pending, running, done, or failed.

## The SQLite queue

### Schema

The database holds three tables. A coding agent implementing this should write the exact SQL in `execution/opus_queue/db/schema.sql` and load it on first connection.

**Table `directions`** — one row per (model, source language, target language), populated once from `lookup_OPUS.xlsx`.

Columns:

- `direction_key` TEXT — e.g. `eng_Latn-fra_Latn`, matching how `lookup_OPUS.xlsx` names them.
- `model` TEXT — e.g. `metricx24`.
- `src_lang` TEXT — e.g. `eng_Latn`.
- `tgt_lang` TEXT — e.g. `fra_Latn`.
- `n_sentences` INTEGER — total sentences in the direction's OPUS parquet file.
- `shard_size` INTEGER — how many sentences per shard for this model (from the planner).
- `est_hours` REAL — carried over from the spreadsheet for monitoring.
- `created_at` INTEGER — unix timestamp.
- Primary key `(direction_key, model)`.

**Table `jobs`** — one row per shard. This is where workers contend.

Columns:

- `direction_key` TEXT, foreign key to `directions`.
- `model` TEXT, foreign key to `directions`.
- `shard_id` INTEGER — 0-based index inside the direction.
- `start_idx` INTEGER — first sentence index (inclusive).
- `end_idx` INTEGER — last sentence index (exclusive).
- `status` TEXT — one of `pending`, `running`, `done`, `failed`. Default `pending`.
- `worker_id` TEXT NULL — e.g. `${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}.${HOSTNAME}` when running.
- `started_at` INTEGER NULL — unix timestamp when claimed.
- `finished_at` INTEGER NULL — unix timestamp when done/failed.
- `attempts` INTEGER — incremented on each claim. Default 0.
- `last_error` TEXT NULL — short message or traceback header on failure.
- `out_path` TEXT NULL — path on shared storage where the shard's JSONL was written.
- Primary key `(direction_key, model, shard_id)`.
- Index on `(model, status)` so claim queries are fast.
- Index on `(status, started_at)` so the reaper is fast.

**Table `run_events`** — append-only log for observability (optional but recommended).

Columns: `ts`, `worker_id`, `event` (one of `claim`, `done`, `fail`, `reap`, `start`, `exit`), `direction_key`, `model`, `shard_id`, `detail`.

### Required PRAGMAs

When `queue_db.py` opens a connection it must set, in this order:

- `PRAGMA journal_mode=WAL;` — allows concurrent readers and one writer without blocking.
- `PRAGMA synchronous=NORMAL;` — safe with WAL, much faster than FULL.
- `PRAGMA busy_timeout=30000;` — wait up to 30s on contention before erroring.
- `PRAGMA foreign_keys=ON;`.

WAL mode requires the database file to live on a filesystem that supports POSIX locks. Lustre/GPFS on HPC systems support this for a single directory but have footguns around many writers from many nodes — the plan accounts for this by keeping one single writer pattern (see "Claim atomicity" below).

### Claim atomicity

Multiple workers across multiple nodes will try to claim shards at the same moment. The claim must be atomic: exactly one worker gets each shard.

The recipe is a single SQL statement using `UPDATE … RETURNING` (SQLite ≥ 3.35). The statement selects one `pending` row for the given model, flips it to `running`, and returns the row. Because it's a single statement inside an implicit transaction, no two concurrent callers can flip the same row. Pseudocode of the SQL shape (not to be copied literally — the agent writes the final SQL):

> update `jobs` set status='running', worker_id=?, started_at=strftime('%s','now'), attempts=attempts+1 where rowid = (select rowid from jobs where status='pending' and model=? order by (end_idx - start_idx) desc limit 1) returning direction_key, model, shard_id, start_idx, end_idx

Largest-first ordering in the inner SELECT improves packing: big shards start early when the pool is full; small shards fill the tail.

On concurrent `SQLITE_BUSY` the `busy_timeout` PRAGMA handles retries automatically. The worker code must still catch and retry `OperationalError: database is locked` for safety, with a short randomized backoff (say 100–500 ms, up to 5 attempts) before surfacing the error.

## The shard planner

`execution/opus_queue/shard_planner.py` exposes one pure function. Input: `(model_name, n_sentences)`. Output: a list of `(shard_id, start_idx, end_idx)` tuples that partition `[0, n_sentences)` into contiguous non-overlapping ranges, plus the `shard_size` used.

Default `shard_size` per model (seed values; the agent should expose them as constants at the top of the file so they're easy to tune after calibration).

**Unit conversion.** The `rate_per_hour` column in `lookup_OPUS.xlsx` is **FLORES directions per hour**, not sentences per hour. One FLORES direction is ~1000 sentences (devtest split: 1012 sentences), so throughput in sentences per hour is `rate_per_hour × 1000`. The spreadsheet's `est_hours` column follows this convention exactly: `est_hours = n_sentences / (rate_per_hour × 1000)`. A shard sized to ≈30 minutes of wallclock is therefore `0.5 × rate_per_hour × 1000 = rate_per_hour × 500` sentences.

Seed values derived from the actual rates found in `lookup_OPUS.xlsx`:

| Model | `rate_per_hour` (FLORES-dir/h) | sentences / hour | Seed `shard_size` (≈30 min) |
|---|---:|---:|---:|
| metricx24 | 100 | 100,000 | 50,000 |
| wmt23-cometkiwi-da-xl | 60 | 60,000 | 30,000 |
| xcomet-xl | 60 | 60,000 | 30,000 |
| shaomutan_remedy-9b-22 | 33 | 33,000 | 16,500 |
| qwen3-4b-instruct-2507 | 20 | 20,000 | 10,000 |
| m-prometheus-7b | 15 | 15,000 | 7,500 |
| bicleaner-ai | 250 | 250,000 | 125,000 |

These are starting points. The executor's CLI must accept `--shard-size-override <model>:<n>` to let the operator override per model after a calibration run.

**Sanity check the planner math against the spreadsheet.** For any row in `lookup_OPUS.xlsx`, the planner should produce `ceil(n_sentences / shard_size)` shards and each shard should take ≈30 minutes based on the model's rate. Cross-check: the largest metricx24 direction in the sheet is ≈1.25 billion sentences (12,488 est_hours × 100 directions/hour × 1000 sent/direction); at 50,000 sentences per shard that's ~25,000 shards of ~30 min each, matching the 12,488 h total. If this math doesn't line up within a few percent, the `rate_per_hour` interpretation is wrong and the seeds need re-derivation before any SLURM submission.

Edge cases the planner must handle: `n_sentences == 0` → no shards; `n_sentences < shard_size` → one shard covering everything; exact multiple of `shard_size` → no trailing ragged shard.

## `build_queue.py` — one-time population

CLI: `python -m execution.opus_queue.build_queue --lookup /path/to/lookup_OPUS.xlsx --opus-root /scratch/.../FineOPUS-Filtered-Stage2 --db $SCRATCH/opus_qe/jobs.db`.

Behavior:

1. Read `lookup_OPUS.xlsx`, sheet `lookup_OPUS_split_strategy2`. Each row yields `(direction_key, winner_model, est_hours)` where `direction_key` has the form `<src>-<tgt>`. Skip rows whose `winner_model` is empty or explicitly marked as unsupported.
2. For each direction, find the OPUS parquet file on disk under `--opus-root` and count rows. Cache the count so re-runs are fast. Implementation hint: use `pyarrow.parquet.ParquetFile(path).metadata.num_rows` — this does not load the data.
3. Call `shard_planner.plan(model, n)` to get the list of shard ranges.
4. Insert one row into `directions` per (direction, model), and `ceil(n/shard_size)` rows into `jobs` with `status='pending'`.
5. The whole operation must be **idempotent**: use `INSERT OR IGNORE` on primary keys, so re-running the command only inserts missing rows and never clobbers existing status.
6. Print a summary at the end: total directions, total shards, total sentences, per-model breakdown. Write the same summary to `run_events`.

The command must accept `--dry-run` which produces the summary without writing to the DB.

## `worker.py` — the inner loop

Run as `python -m execution.opus_queue.worker --db $SCRATCH/opus_qe/jobs.db --model <model> --output-base $SCRATCH/opus_qe/shards --opus-root ... --walltime-seconds $SLURM_REMAINING`.

On startup:

1. Compute `worker_id = f"{SLURM_JOB_ID}.{SLURM_ARRAY_TASK_ID}.{hostname}.{pid}"`. Emit a `start` event to `run_events`.
2. Load the model once (expensive; never reload per shard). The exact loader is model-specific and is already implemented in `src/*_backend.py` — the worker imports that backend via the existing `models/model_registry.py::resolve_model_spec(model_name)`.
3. Sweep own stale rows: `UPDATE jobs SET status='pending', worker_id=NULL WHERE status='running' AND worker_id=?` with the computed id. This handles the case where this exact array task was requeued after a crash and left rows behind.
4. Enter the claim loop (below).
5. On clean exit, emit an `exit` event.

Claim loop per iteration:

1. Check remaining walltime. If `time_left < 1.5 * expected_shard_seconds`, break and exit cleanly. `expected_shard_seconds` is looked up per model from a table mirroring the shard_size table (the same 20–40 min target).
2. Call `claim_next(model, worker_id)`. If no row is returned the queue for this model is drained → break.
3. Resolve the parquet file for `direction_key`, open a row-group stream, and materialize exactly `[start_idx, end_idx)`.
4. Call the scorer from `src/` for the chosen model on those examples. Use the same `score_entry`-shaped callable the FLORES runner uses today; the dataset adapter is `get_dataset("opus")` so output columns and seen/unseen flags match the existing OPUS frames.
5. Atomic write: the frame is serialized as JSONL to a temp file at `<output_base>/<model>/<direction_key>/shard_<NNNNN>.jsonl.tmp`, the file is `fsync`ed, then renamed to `shard_<NNNNN>.jsonl`. This ordering guarantees no consumer ever sees a half-written shard.
6. Call `mark_done(direction_key, model, shard_id, out_path)`. This updates `status='done'`, `finished_at`, and `out_path` in one statement.
7. Catch any exception around steps 3–5. On exception: call `mark_failed(..., error_summary)` which sets `status='failed'` only if `attempts >= max_attempts` (default 3), otherwise resets to `pending` so it gets retried by another worker. `last_error` is always recorded.

Per-shard outputs live under `<output_base>/<model>/<direction_key>/shard_<NNNNN>.jsonl`. This path scheme is distinct from the FLORES layout (`dataset=/model=/split=/shard=NNN/part-*.jsonl`) and should not collide. It is intentionally flat per-direction so the merge step can glob and concatenate in one pass.

### Scorer reuse

The worker does not re-implement scoring. It imports the same `score_entry` callback the scorer CLIs build today. The simplest pattern: the worker's `--model <name>` dispatches to a small table that, for each model, returns the model-loader + score-entry-factory defined in the corresponding `src/score_*.py`. Plan 1's Step 5 already made scorers dispatch-agnostic; the worker is just a second caller of the same callable.

## `reaper.py` — reclaim stuck rows

Run as `python -m execution.opus_queue.reaper --db ... --interval 300 --timeout-multiplier 2`.

Behavior:

- Every `--interval` seconds, compute `cutoff = now - timeout_multiplier * max(expected_shard_seconds_per_model)` (use the model's value per row, not a global max — see SQL below).
- Execute: `UPDATE jobs SET status='pending', last_error='reaped', worker_id=NULL WHERE status='running' AND started_at < cutoff_for_its_model`.
- Emit one `reap` event per reclaimed row.

The reaper is safe to run multiple times or from multiple hosts — it only widens the pool of pending work; it never removes data. Recommend running one instance in a long-lived login-node `screen`/`tmux` session, or as a scheduled task (`cron` every 5 min).

## `merge.py` — per-direction concatenation

Run as `python -m execution.opus_queue.merge --db ... --output-base ... --merged-base $SCRATCH/opus_qe/merged`.

Behavior:

1. Query directions where all shards have `status='done'`: `SELECT direction_key, model FROM directions d WHERE NOT EXISTS (SELECT 1 FROM jobs j WHERE j.direction_key=d.direction_key AND j.model=d.model AND j.status != 'done')`.
2. For each complete direction, list `shard_<NNNNN>.jsonl` files under its shard dir **sorted by shard_id** and concatenate them into `<merged_base>/<model>/<direction_key>.jsonl`. Because shards were produced from disjoint contiguous ranges, concatenating in shard order recovers the original row alignment exactly.
3. Write a per-direction `<direction_key>.meta.json` with `{n_shards, n_rows, source_db, merged_at}`.
4. Optionally delete the shard files after successful merge (guard with `--delete-shards`, off by default; the DB row `status='done'` is the authoritative signal either way).

The merge step is safe to re-run; each direction is merged independently and completed ones are skipped (check output file existence + row count).

## `executor.py` — the `ExecutionStrategy` wrapper

`OpusQueueExecutor` implements the protocol from `execution/base.py` so OPUS can be driven through the same `--execution` CLI flag the FLORES executor uses.

Its `add_cli_args(parser)` registers:

- `--db` (required): path to the SQLite file.
- `--opus-root`: defaults from the OPUS adapter's default root.
- `--output-base`: where shard JSONLs go.
- `--walltime-seconds`: usually sourced from `SLURM_JOB_END_TIME - now` in the launch script, but override via flag for local runs.
- `--max-attempts`: default 3.
- `--shard-size-override`: repeatable, format `model:int`.
- `--claim-retries`: default 5.

Its `run(args, dataset, model_tag, score_entry)` simply calls `worker.run_loop(...)` with the right arguments. Unlike the FLORES executor, this executor ignores `score_entry` if it is None and instead resolves the scorer internally from the registry — the user can still inject one for testing.

Register it in `execution/__init__.py` under the name `"opus_queue"`.

## SLURM wiring

`scripts/opus/submit_array.sh` is the operator-facing entry point. Usage shape:

> `./submit_array.sh --model metricx24 --array 0-63 --concurrency 32 --time 24:00:00 --db $SCRATCH/opus_qe/jobs.db`

Behavior:

- Submits one `sbatch --array=0-63%32 --time=24:00:00 --gpus=1 scripts/opus/run_worker.sh metricx24 $SCRATCH/opus_qe/jobs.db`.
- `run_worker.sh` sets up the environment (module loads or container) exactly like `scripts/flores/run_slurm.sh` does today, then execs `python -m execution.opus_queue.worker --db "$2" --model "$1" --walltime-seconds $(scontrol show job $SLURM_JOB_ID | extract EndTime)`.
- The array index is irrelevant to correctness — it just means "this is one of N worker slots". Workers do not read `SLURM_ARRAY_TASK_ID` for partitioning; they only use it as part of their `worker_id`.

`scripts/opus/run_reaper.sh` launches the reaper on a login node in a `screen` session. `scripts/opus/run_merge.sh` runs merge as a normal (non-array) batch job after the main run drains.

Each model gets its own array. If you want to run metricx24 and qwen in parallel, submit two arrays. They do not interfere because each worker filters by `--model`.

## Operational runbook

Once implemented, the day-to-day is:

1. **One-time setup.** Copy `lookup_OPUS.xlsx` to shared storage. Run `build_queue.py` to populate `jobs.db`. Run with `--dry-run` first to sanity-check counts against the spreadsheet.
2. **Calibration.** Submit a tiny array (`--array 0-0 --time 1:00:00`) for one model and observe shard wallclock in `run_events`. Tune `shard_size` via `--shard-size-override` and re-run `build_queue.py` **only for that model** if the seeds were badly wrong. `build_queue.py` must support `--reset-pending-for-model <m>` to drop and rebuild jobs for a single model when the shard size is changed mid-run; it never deletes `done` rows without an explicit `--force`.
3. **Steady state.** Submit one array per model at whatever concurrency the cluster allocates. Launch the reaper. Monitor with `SELECT model, status, COUNT(*) FROM jobs GROUP BY model, status;`.
4. **Resume.** If everything dies, just resubmit the arrays. Workers claim only `pending` rows. Nothing needs to be cleaned up by hand — the reaper handles orphaned `running` rows, and atomic writes guarantee no corrupt outputs.
5. **Completion.** When all rows for a model are `done`, run `merge.py` for that model. The FLORES-style downstream `stand_alone_modules/` tools can then consume the merged per-direction JSONLs by pointing them at `<merged_base>/`.

## Edge cases and gotchas a coding agent must handle

- **Parquet row-group streaming.** OPUS parquet files are large; loading the whole file to score a 50-row shard is wasteful. Use `pyarrow.parquet.ParquetFile.iter_batches(batch_size=…)` and skip to `start_idx` with a running counter, stopping at `end_idx`. Do not use `.read()` on the whole file.
- **Node-local scratch vs shared scratch.** The DB and the shard output dir must live on shared storage (visible from every compute node). Model weights can live on node-local cache if the container supports it; that's orthogonal.
- **Lustre/GPFS locking.** SQLite's WAL mode plus Lustre has historically been flaky. If the target cluster is Lustre, pre-test with a throwaway DB and 100 concurrent claimers before running at scale. If it is flaky, the fallback is to replace the SQLite backend with a single Redis instance on a login node; the `queue_db.py` abstraction must be narrow enough that only one file changes.
- **Walltime clock skew.** `scontrol show job` time math is reliable; `date +%s - SLURM_JOB_START_TIME` is not (the env var isn't always set). Have the launcher pass `--walltime-seconds` explicitly.
- **Attempts vs retries.** A failed shard with `attempts < max_attempts` goes back to `pending`, not `failed`. Only when `attempts == max_attempts` does it land in `failed` and require manual inspection. Ensure this is a single atomic UPDATE, not a read-then-write race.
- **Duplicate outputs on retry.** When a shard is retried, its `shard_<NNNNN>.jsonl` from the previous attempt may exist. The atomic rename overwrites cleanly; the DB `out_path` always points to the last successful write. Do not append — always write a full file.
- **Ordering in merge.** Sort by integer `shard_id`, not by filename string. `shard_00010.jsonl` sorts after `shard_00009.jsonl` lexicographically only because of the zero-padding — always re-parse the integer to be safe.
- **`lookup_OPUS.xlsx` mutability.** If the spreadsheet is edited (different model assigned to a direction), `build_queue.py` with default idempotent inserts will not reassign the model. The command must support `--reassign` that, for rows where the `(direction_key, model)` in the sheet differs from the DB, deletes the old model's `jobs` rows (only those still pending) and inserts the new. Warn loudly if any `done` rows would have to be discarded and require an explicit confirmation flag.
- **Schema migrations.** Version the schema (`PRAGMA user_version`). Bump on any change and provide a one-shot migration function in `queue_db.py` so old DBs keep working.

## Deliverables

- `execution/opus_queue/` populated per the structure above, with schema, queue wrapper, shard planner, build command, worker, reaper, merge, and executor.
- `scripts/opus/` with the three shell entry points.
- A `MANUAL.md` inside `execution/opus_queue/` that walks the operator through the runbook above end-to-end.
- A calibration log showing one model run end-to-end on a small subset (≤ 3 directions, ≤ 100 shards) proving: atomic writes work, claim under contention gives unique assignments, reaper reclaims a killed worker's rows, and merge reproduces the original row order.
- Integration test that spins up a tmp SQLite DB, inserts 50 fake jobs, launches 4 in-process "workers" in threads, and asserts no double-processed shards and no missed shards.

## Out of scope

- Changing the OPUS dataset adapter (`dataset/opus_scripts/`, `dataset/adapters/opus.py`). It already works and is what the worker calls into.
- Changing model backends. Workers call `src/*_backend.py` as-is.
- Replacing the FLORES execution strategy or altering its behavior in any way.
- A cross-model scheduler (deciding which model to run next globally). Each model's array drains independently; if coordination is needed later, it goes on top of this foundation.
