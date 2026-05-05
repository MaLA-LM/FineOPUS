# PLAN — Migrate OPUS QE Queue from SQLite to Pre-assigned Manifest


## 1. Problem statement

The OPUS QE pipeline currently coordinates 800+ concurrent GCD workers through a single shared SQLite database (`execution/opus_queue/db/`). Workers fight for the database file lock on every `claim_next` / `mark_done` / `mark_failed` / `log_event` call. Even with `WAL`, `busy_timeout=120s`, and the 30-attempt retry loop in [retry.py:54-64](execution/opus_queue/db/retry.py#L54-L64), workers regularly fail with:

```text
sqlite3.OperationalError: database is locked
```

Root cause: SQLite serialises **all** writers at the file level. Partitioning rows by worker does not help because the lock is on the database file, not on rows. Removing WAL would make this strictly worse (rollback-journal mode blocks readers too).

The fix: stop coordinating through a shared mutable database during the hot path. Pre-assign every pending shard to a specific worker slot at submission time, write the assignment once into a read-only manifest, and have each worker journal its own progress in a private append-only trace file.

## 2. Goals and non-goals

### Goals

1. Remove SQLite from the worker hot path entirely. Workers only do read-only manifest opens and append-only writes to their own files.
2. Pre-assign every pending shard deterministically to a worker slot at submission time.
3. One **global** manifest covering all models, with `model` as a column.
4. Per-worker append-only trace file under `/scratch/project_462001050/opus_qe/shard_trace/` recording every column the `jobs` table records today, including a per-attempt `gpu_seconds_delta` so `gpu_seconds_total` can be reconstructed losslessly.
5. Direction-locality: a direction is processed by one worker end-to-end where possible, spilling to the next worker only at the slot boundary. Goal: ≤2 part files per direction.
6. Reuse the existing scorer code paths and `DirectionPartWriter` unchanged.

### Non-goals (explicit deferrals)

- **No dynamic claim/steal across workers in flight.** Stragglers are handled offline by a `repack` tool between submissions (deferred — design only, no implementation in this plan).
- **No deletion of the existing SQLite schema.** `done` rows stay in `jobs.db` permanently as the historical `gpu_seconds_total` archive. Future analytics will read from both the DB and the per-worker trace files.
- **No FLORES pipeline changes.**
- **No change to backend scorers, prompts, or output JSONL row schema.**

## 3. Worker identity model

### Today

[loop.py:31-36](execution/opus_queue/worker/loop.py#L31-L36) builds:

```python
worker_id = f"{SLURM_JOB_ID}.{SLURM_ARRAY_TASK_ID}.{host}.{pid}"
```

`pid` makes this non-deterministic, so a worker cannot resume "its own" work after a restart.

### New scheme

Two distinct identifiers:

```python
worker_slot_id = f"{model}-a{SLURM_ARRAY_TASK_ID:05d}-l{SLURM_LOCALID}"
worker_run_id  = f"{SLURM_JOB_ID}.{SLURM_ARRAY_TASK_ID}.{SLURM_LOCALID}.{host}.{pid}"
```

- `worker_slot_id` is the **stable identity** used to look up assigned shards in the manifest and to name the trace folder. Restarting an array task on a different node still resolves to the same slot → same shards → resumes from its trace.
- `worker_run_id` is recorded inside trace events for forensics (so we can correlate to SLURM logs).

`model` is part of the slot identifier so one global manifest can host disjoint per-model slot namespaces (`metricx24-a00000-l0` ≠ `qwen3-4b-instruct-2507-a00000-l0`).

### Submitter contract

The submitter must declare **slots per array task** so the manifest builder knows how many slots exist per array index:

| Submitter                      | Partition    | Slots per array task    |
| ------------------------------ | ------------ | ----------------------- |
| `submit_array.sh`              | `small-g`    | 1 (single GCD)          |
| `submit_array_standard_g.sh`   | `standard-g` | 8 (one per LUMI-G GCD)  |

Total slots for a submission with `--array=0-N` is `(N + 1) * slots_per_task`. With `--array=0-79` on standard-g this yields **640 worker slots**.

## 4. Quota computation (per-model)

Inputs:
- `walltime_seconds` — from SLURM assigned walltime (default 40 h = 144 000 s on standard-g for our workloads).
- `EXPECTED_SHARD_SECONDS[model]` — from [shard_planner.py:34-45](execution/opus_queue/planning/shard_planner.py#L34-L45). Models without an entry fall back to `DEFAULT_EXPECTED_SHARD_SECONDS = 1800`.
- `safety_factor` — default `0.85` (covers warmup, vLLM compile, container start, drain time).

Per-worker quota:

```text
quota_shards(model) = floor(walltime_seconds * safety_factor / EXPECTED_SHARD_SECONDS[model])
```

Worked example for `walltime_seconds = 144 000`, `safety = 0.85` → `effective_seconds = 122 400`:

| Model                       | EXPECTED_SHARD_SECONDS | Quota shards/worker |
| --------------------------- | ---------------------: | ------------------: |
| `metricx24`                 |                    600 |                 204 |
| `bicleaner-ai`              |                    300 |                 408 |
| `wmt23-cometkiwi-da-xl`     |                   1500 |                  81 |
| `xcomet-xl`                 |                   1500 |                  81 |
| `shaomutan_remedy-9b-22`    |                   1000 |                 122 |
| `qwen3-4b-instruct-2507`    |                   1200 |                 102 |
| `m-prometheus-7b`           |                   1200 |                 102 |
| `m-prometheus-7b-detailed`  |                   9000 |                  13 |
| anything unlisted           |         1800 (default) |                  68 |

The packer in §6 caps each worker at `quota_shards(model)`. If `total_pending_shards(model) > quota_shards(model) * total_slots(model)`, the packer prints an explicit **shortfall** warning naming the leftover shard count so the operator can size the next submission accordingly.

## 5. File layout

All paths under the project's existing scratch root.

```text
/scratch/project_462001050/opus_qe/
   manifests/
      <build_tag>/                          # e.g. 2026-05-04T12-00-00
         manifest.jsonl                     # one row per (slot → shard) assignment, all models
         manifest.summary.json              # per-model totals, quotas, shortfalls, packing report
         README.txt                         # how to invoke workers against this build_tag
   shard_trace/
      <build_tag>/
         <worker_slot_id>/                  # e.g. metricx24-a00000-l0
            assignment.json                 # frozen copy of this slot's manifest rows
            events.jsonl                    # append-only, one event per state transition
            state.json                      # cached materialised state (rebuildable from events)
```

`build_tag` is generated once by the migration / packing tool. A new `build_tag` is created per submission so re-runs do not collide.

## 6. Manifest schema

### `manifest.jsonl` row

```json
{
  "worker_slot_id": "metricx24-a00000-l0",
  "model": "metricx24",
  "array_task_id": 0,
  "local_id": 0,
  "assignment_seq": 17,
  "direction_key": "eng_Latn-fra_Latn",
  "src_lang": "eng_Latn",
  "tgt_lang": "fra_Latn",
  "shard_id": 4,
  "start_idx": 800000,
  "end_idx": 1000000,
  "shard_size": 200000,
  "expected_seconds": 600
}
```

`assignment_seq` defines the order a worker processes its shards. The packer keeps it contiguous **by direction** so the part-writer accumulates a direction into one file before moving on.

### `manifest.summary.json`

```json
{
  "build_tag": "2026-05-04T12-00-00",
  "created_at": 1714838400,
  "walltime_seconds": 144000,
  "safety_factor": 0.85,
  "models": {
    "metricx24": {
      "slots_declared": 640,
      "quota_shards_per_worker": 204,
      "pending_shards": 110000,
      "assigned_shards": 110000,
      "shortfall_shards": 0,
      "directions_total": 350,
      "directions_split_across_slots": 12,
      "max_load_seconds": 121800,
      "min_load_seconds": 88200,
      "avg_load_seconds": 114375
    }
  }
}
```

`directions_split_across_slots` is the headline metric for "how many part files we'll produce above the floor". Operator-visible.

### Packing algorithm

```text
for model in models_with_pending_shards:
    quota = quota_shards(model)
    slots = pre-allocated slot list for model
    sort directions desc by total shard count
    slot_idx = 0
    seq = 0
    for direction in directions:
        for shard in direction.shards (in shard_id order):
            if slots[slot_idx].used >= quota and slots[slot_idx].used > 0:
                slot_idx += 1
                seq = 0
                if slot_idx >= len(slots):
                    record shortfall, stop assigning this model
            assign shard to slots[slot_idx] with assignment_seq = seq
            seq += 1
            slots[slot_idx].used += 1
```

Properties:
- A direction with ≤ quota shards lands in **exactly one** slot → 1 part file.
- A direction with > quota shards spills only at slot boundaries → 2 part files for the spill direction.
- Greediness is per-direction-contiguous, so direction-locality is the rule.
- Falls back to round-robin if `total_shards / slot_count < small_floor` so we don't overflow slot 0.

## 7. Per-worker trace files

### `events.jsonl` row schema

One row per state change. Covers every column of `jobs` plus per-attempt timing.

```json
{
  "ts": 1714838400,
  "event": "claim" | "done" | "failed" | "skip" | "reset_own_stale",
  "worker_slot_id": "metricx24-a00000-l0",
  "worker_run_id": "12345678.0.0.nid001234.4711",
  "model": "metricx24",
  "direction_key": "eng_Latn-fra_Latn",
  "shard_id": 4,
  "attempt": 1,
  "start_idx": 800000,
  "end_idx": 1000000,
  "started_at": 1714838400,
  "finished_at": null,
  "claim_gpu_count": 1,
  "gpu_seconds_delta": 0.0,
  "out_path": null,
  "last_error": null
}
```

Why **events** rather than a single row per shard:
- Lossless audit trail across retries.
- `gpu_seconds_total` for a shard = `sum(gpu_seconds_delta over events for that shard)`. Equivalent to today's `_GPU_SECONDS_EXPR` ([claims.py:26-33](execution/opus_queue/db/claims.py#L26-L33)) but computed at write time.
- Easy to merge across workers offline.

### Write contract (mirrors today's checkpoint contract in `CLAUDE.md`)

For each shard:
1. `claim` event written → flush → fsync (covers worker crash before scoring starts).
2. Score the shard (existing `_score_shard` flow, untouched).
3. Append payload to part file → flush → fsync.
4. `done` event written with `gpu_seconds_delta = elapsed * claim_gpu_count` → flush → fsync.
5. Update `state.json` cache (best-effort, not fsynced — `events.jsonl` is the source of truth).

On failure: `failed` event with `last_error`. Attempt counter advances. Retry on next loop iteration if `attempt < --max-attempts`.

### Worker startup

1. Read `assignment.json` → list of shards in `assignment_seq` order.
2. Replay `events.jsonl` to compute per-shard latest state (in-memory dict keyed by `shard_id`).
3. Skip shards in `done` state.
4. Re-attempt shards in `running` (i.e. previous run crashed mid-shard) or `failed` (subject to `--max-attempts`).
5. Process remaining `pending` shards in `assignment_seq` order so the part-writer keeps direction-locality.

Idempotency: identical to today. Incomplete event = the shard looks `running` → attempt counter increments → retry happens.

## 8. Worker runtime changes

Concentrated diff. Everything below `commit_shard` (scorers, `DirectionPartWriter`, `shard_io`, `count_detail_rows`, `frame_to_jsonl_bytes`, walltime checks) is **unchanged**.

| File | Change |
| ---- | ------ |
| [worker/loop.py](execution/opus_queue/worker/loop.py) | Replace `claim_next` loop with a manifest-driven iterator over assigned shards. Replace `mark_done` / `mark_failed` / `reset_own_stale` / `log_event` calls with trace-writer equivalents. Keep `_score_shard`, `commit_shard` glue, `DirectionPartWriter`, `walltime` checks. |
| [worker/commit.py](execution/opus_queue/worker/commit.py) | `commit_shard` becomes "write part-file payload, then append `done` event to trace". `out_path` semantics unchanged. |
| [worker/cli.py](execution/opus_queue/worker/cli.py) | Drop `--db`, `--claim-retries`. Add `--manifest-root`, `--trace-root`, `--build-tag`. `worker_slot_id` derived from `SLURM_*` env + `--model`. |
| **New** `execution/opus_queue/trace/writer.py` | Append `claim` / `done` / `failed` events. fsync semantics. |
| **New** `execution/opus_queue/trace/reader.py` | Replay `events.jsonl` → per-shard latest state. |
| **New** `execution/opus_queue/trace/state.py` | In-memory state machine, `state.json` materialisation. |
| **New** `execution/opus_queue/manifest/reader.py` | Stream `manifest.jsonl`, filter by `worker_slot_id`. |
| **New** `execution/opus_queue/planning/assigner.py` | The packing algorithm in §6. Pure function over (pending shards, slot count per model, walltime, safety factor). |
| **New** `execution/opus_queue/tools/migrate_to_manifest/` | One-shot SQLite → manifest migrator (§9). |

`execution/opus_queue/db/` is **not deleted**. It stays for:
- The migration tool (read-only).
- Historical analytics on completed `done` rows.
- The existing `tools/merge` until it's ported to the new layout.

## 9. Migration tool — SQLite → manifest

currently the database has only `pending` and `done`.

`python -m execution.opus_queue.tools.migrate_to_manifest`.


CLI:

```text
--db <path>                     existing jobs.db (read-only)
--manifest-root <dir>           output dir (e.g. /scratch/.../opus_qe/manifests)
--build-tag <str>               optional, default: ISO timestamp
--walltime-seconds <int>        default 144000 (40 h)
--safety-factor <float>         default 0.85
--slots <model:int> ...         repeatable, e.g. --slots metricx24:640 --slots qwen3-4b-instruct-2507:200
--include-status pending,failed,running   default: pending,failed
--dry-run                       print summary without writing manifest
```

Steps:

1. Open the existing `jobs.db` read-only (`mode=ro`).
2. For each `(model, direction_key, shard_id)` row matching `--include-status`:
   - Pull `start_idx`, `end_idx`, `shard_id`, `direction_key`, `model`, `src_lang`, `tgt_lang` from `jobs` ⨝ `directions`.
3. Group by `model`. Validate that `--slots <model>` is provided for every model present.
4. Run the assigner from §6 per model. Concatenate output rows into one `manifest.jsonl`.
5. Write `manifest.summary.json`.
6. **Do not modify the source DB.** `done` rows stay untouched per goal §2.

Output is identical regardless of how many times the tool is run with the same inputs (deterministic ordering by `(model, direction_key, shard_id)`).

## 10. Submitter changes

### `scripts/opus/submit_array_standard_g.sh`

Add:

```text
--manifest-root <dir>     required; the directory containing build_tag subdirs
--build-tag <str>         required; selects which manifest the workers read
--trace-root <dir>        default: /scratch/project_462001050/opus_qe/shard_trace
```

Drop (with one-release deprecation warning):

```text
--db <path>
```

Pre-flight checks before `sbatch`:

1. `manifest-root/<build_tag>/manifest.jsonl` exists and is readable.
2. The manifest contains at least one row with `model == $MODEL`.
3. The number of distinct `array_task_id` values for `model=$MODEL` matches what `--array` implies. Abort with a precise message otherwise:

   ```text
   ERROR: manifest has 100 array_task_ids for model=metricx24,
          --array=0-79 implies 80. Re-pack with --slots metricx24:640
          or submit --array=0-99.
   ```
4. The `WARNING: no queue rows found` probe at [submit_array_standard_g.sh:171-187](scripts/opus/submit_array_standard_g.sh#L171-L187) is replaced by a manifest probe: count assigned shards for `(model, build_tag)` minus shards already marked `done` in trace files for the same `build_tag`. Cheap because the trace tree is per-slot.

### `scripts/opus/run_worker_standard_g_task.sh`

`WORKER_ARGS` gets `--manifest-root`, `--build-tag`, `--trace-root`, drops `--db`. Everything else (vLLM env, CPU bind, ROCR pin per `SLURM_LOCALID`) is unchanged.

### `scripts/opus/submit_array.sh` (small-g)

Same changes; `slots_per_task = 1`.

## 11. Direction → file count guarantee

With the §6 packing algorithm:

- A model whose total pending shards ≤ `quota * slots` and whose biggest direction ≤ `quota` → **every direction lives in exactly one part file** (per part-writer rotation rules in [shard_io.py:182-203](execution/opus_queue/worker/shard_io.py#L182-L203)).
- A direction larger than `quota` shards spills into a second slot and produces a second part file. Reported as `directions_split_across_slots` in `manifest.summary.json`.
- Worst case (model whose every direction > quota): one extra file per direction, capped at `ceil(direction_shards / quota)` files per direction.

This is already strictly better than the current state, where `DirectionPartWriter` files are namespaced by `worker_id_safe` and any direction touched by N workers produces N part files.

## 12. Reconstructing `gpu_seconds_total`

For every shard ever processed under the manifest model:

```text
gpu_seconds_total(direction_key, shard_id, model) =
    sum(event.gpu_seconds_delta
        for event in events if event.event == 'done' or event.event == 'failed'
        and event.shard_id == shard_id
        and event.direction_key == direction_key
        and event.model == model)
```

Per-attempt timing is captured at commit time as `elapsed * claim_gpu_count` (mirrors `_GPU_SECONDS_EXPR` in [claims.py:26-33](execution/opus_queue/db/claims.py#L26-L33)). For shards completed before the migration, query the existing `jobs.db` directly. A future analytics tool will UNION the two sources.

## 13. Phased rollout

Each phase is independently shippable; nothing forces a hard cutover.

### Phase 1 — Migration tool + manifest assigner (offline, no worker changes)

1. Implement `planning/assigner.py` with unit tests on synthetic shard inputs.
2. Implement `tools/migrate_to_manifest/` with `--dry-run` first.
3. Run `--dry-run` against the production `jobs.db` for each model. Eyeball `manifest.summary.json`. Tune `--slots` and `--safety-factor` until shortfalls are zero or acknowledged.
4. Generate the first real `build_tag` manifest. Commit no code that depends on it yet.

### Phase 2 — Worker reads manifest in addition to DB (shadow mode)

1. Implement `manifest/reader.py` and `trace/{writer,reader,state}.py`.
2. Add a `--mode {db,manifest}` flag to `worker/cli.py` (default `db`).
3. In `manifest` mode, the worker reads its assigned shards from the manifest, scores them, writes part files **and** trace events, but also still calls `mark_done` / `mark_failed` against the DB so existing dashboards keep working.
4. Run a tiny `--array=0-1` test on small-g to validate end-to-end. Compare DB state vs. trace state.

### Phase 3 — Cutover (workers stop touching the DB)

1. Drop the DB writes from `manifest` mode. Workers are now lock-free.
2. Remove the `db` mode after one stable run.
3. Update `submit_array*.sh` to require `--manifest-root` / `--build-tag` and to reject `--db`.

### Phase 4 — Cleanup (deferred, separate plan)

1. Port `tools/merge` to read trace files instead of the DB.
2. Build the analytics UNION view across `jobs.db` (historical) + trace files (post-migration).
3. Implement the `repack` straggler tool from §2.

## 14. Open follow-ups (parked, not blocking this plan)

- **Cross-submission straggler redistribution.** Deferred per user. When ready, design a `python -m execution.opus_queue.tools.repack` that scans trace folders, collects unfinished shards, and produces a fresh `build_tag` manifest with new slot counts.
- **Manifest builder for "from scratch"** (no existing DB): later — the migration tool covers our current need.
- **Per-model walltime overrides** (e.g. some models on small-g for 72 h, others on standard-g for 48 h): add `--walltime-seconds <model:int>` to the migration tool when needed.
- **Trace compaction**: if `events.jsonl` ever grows uncomfortably large per worker (it won't with current quotas — ~quota events × ~500 bytes ≈ 100 KB/worker), add a periodic `compact-trace` tool that rewrites events into a snapshot row per shard.

## 15. What stays the same

- Backend scorer code (`src/backends/`, `models/`).
- Output JSONL row schema (`utils/frames.py`, `utils/io.py`).
- `DirectionPartWriter` rotation rules.
- The directory layout under `--output-base`: `<model>/<direction_key>/part-<worker_id_safe>-<seq>.jsonl`.
- The existing `jobs.db` (read-only, retained as historical archive).
- The FLORES pipeline.
- The merge tool (in this phase; deferred port in Phase 4).

## 16. Acceptance criteria

This migration is "done" when:

1. A standard-g `--array=0-79` run for any model completes without a single `database is locked` error.
2. The trace files under `<build_tag>/<worker_slot_id>/` allow lossless reconstruction of every column the `jobs` table records today, including `gpu_seconds_total` (= sum of per-attempt deltas) for every shard processed under the manifest.
3. The number of part files per direction in the output tree is ≤ `directions_split_across_slots[model] + directions_total[model]` from `manifest.summary.json` (i.e. one file per direction in the common case, two for spill directions).
4. A killed-and-resubmitted array task picks up exactly its previously-assigned shards, skips any already in `done` state, and re-attempts shards last seen as `running` or `failed`.
5. `submit_array_standard_g.sh` aborts cleanly with an actionable message when the array spec is inconsistent with the manifest's slot count.
