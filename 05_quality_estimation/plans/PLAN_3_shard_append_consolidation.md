# Plan 3 — Worker-keyed shard append + merge-time filter

## Motivation

Today [shard_io.shard_output_path](execution/opus_queue/worker/shard_io.py) writes one file per shard at `<output_base>/<model>/<direction_key>/shard_<NNNNN>.jsonl`. That is simple and crash-safe (tmp → fsync → atomic rename), but it makes the Lustre scratch filesystem the bottleneck at OPUS scale:

- A high-throughput model like `qwen3-4b-instruct-2507` uses `shard_size = 10_000`. A direction with ≈100M sentences produces ≈10,000 small files (2–5 MB each) in a single directory. The MDT metadata server must track every inode, and merge's `Path.iterdir()` on that directory becomes seconds-to-minutes of blocked I/O during high-contention periods.
- Metricx24 at `shard_size = 200_000` is less pathological (up to ~5,000 files for the largest direction) but still below Lustre's preferred file-size band.
- Raising shard sizes to mask the problem is rejected: a worker that dies mid-shard loses ~30 min of work today; doubling the shard size doubles the worst-case loss. The shard-size knob is the wrong dial for this problem.

Plan 2 ([PLAN_2_opus_parallelism.md](plans/PLAN_2_opus_parallelism.md)) defined the current "one file per shard" contract. This plan replaces that contract with **worker-keyed append** — multiple shards for the same direction land in one part file — while preserving Plan 2's atomic-commit semantics and "DB is ground truth" invariant.

## Design goals

1. **Collapse file count by at least 10×** on the high-shard-count directions without changing `shard_size`.
2. **Preserve "DB is ground truth".** A job is committed iff its `jobs.status='done'` row survives. No file-system-level scan should ever need to infer shard state.
3. **Tolerate every crash class the current design tolerates** — SIGKILL mid-write, stale_noop after fsync, SIGKILL between fsync and `mark_done`, reaper reclamation.
4. **No DB schema migration beyond adding fields to existing tables.** No new tables, no new indices on the hot path.
5. **Workers remain stateless across restarts.** A fresh worker never needs to inspect old part files to decide how to resume.
6. **Backward compatibility during rollout.** Merge must read both the legacy `shard_<NNNNN>.jsonl` layout and the new part-file layout so partially completed runs can be finished.

## Approach: append + row-stamp + DB-filtered merge

Two compounding changes:

**Change A — Output layout.** Instead of one file per shard, each worker maintains at most one open part file per direction it is currently scoring. Multiple shards of the same direction processed by the same worker go into the same file; the file rotates when it exceeds a byte or row budget, and closes when the worker switches directions.

New path scheme:

```
<output_base>/<model>/<direction_key>/part-<worker_id>-<seq>.jsonl
```

Example:
```
shards/qwen3-4b-instruct-2507/eng_Latn-fra_Latn/part-12345.0.node42.8734-0000.jsonl
shards/qwen3-4b-instruct-2507/eng_Latn-fra_Latn/part-12345.0.node42.8734-0001.jsonl
shards/qwen3-4b-instruct-2507/fra_Latn-eng_Latn/part-12345.0.node42.8734-0000.jsonl
```

- `worker_id` is Plan 2's existing `"{SLURM_JOB_ID}.{SLURM_ARRAY_TASK_ID}.{hostname}.{pid}"`. It is unique per worker *process lifetime* — a restart on the same node gets a new PID and therefore a new filename, so two workers can never share a file.
- `seq` is a 4-digit zero-padded integer that the worker increments on rotation. It starts at `0000` when the worker first opens a file for a given direction.
- `worker_id` is filename-sanitized (`/`, `\`, `:` → `_`) by the same helper that already exists in [write_temp_payload](execution/opus_queue/worker/shard_io.py) at line 62. Reuse that function.

**Change B — Row stamping + merge filter.** Every JSONL row emitted by the worker carries two additional fields:

- `shard_id` — integer. Already present in the frame today (added in [loop.py:73](execution/opus_queue/worker/loop.py#L73)); keep it.
- `worker_id` — string. Newly added in the worker before serialization.

At merge time, the authoritative map `(direction_key, shard_id) → winning_worker_id` is read from the `jobs` table (where `status='done'`). Merge then reads every part file under the direction's folder and emits only rows whose `(shard_id, worker_id)` pair matches the DB's winner. Rows from stale_noop'd shards, crashed workers, or reaper-reclaimed attempts are silently filtered out — their `worker_id` never appears in the DB's done-set for that `(direction, shard)`.

The DB remains ground truth: if a row is not in the DB's done-set, its bytes are garbage regardless of whether they are durable on disk.

## Why this is simpler than byte-range tracking

A byte-range scheme would record `(out_path, byte_start, byte_end)` per done job and have merge read only those ranges. That works but requires:

- Two new `INTEGER` columns in `jobs` with a migration.
- Exact offset bookkeeping in the worker (handle.tell() before/after each shard, ftruncate on stale_noop).
- Merge logic that does random-access file reads instead of sequential scans — measurably slower on sequential media, and worse if offsets straddle row boundaries.

Row stamping trades two tiny per-row fields (~20–40 bytes uncompressed) for a dropped schema migration, no offset bookkeeping, and a single sequential pass at merge. The per-row overhead is amortised to near-zero under parquet compression if the format switch in the parallel discussion ever happens; it is bounded even without compression.

## Crash-safety analysis

All four failure modes from the current design must still resolve to "no duplicates, no missing rows, no unreadable files":

| Failure | What lands in the part file | What the DB says | Merge outcome |
|---|---|---|---|
| Happy path | rows tagged `(sid, wid_A)` | `done` with `worker_id=wid_A` | rows included |
| SIGKILL mid-`write()` | partial line, no trailing `\n` | row still `running` / later reset to `pending` | partial line fails `json.loads` → `JSONDecodeError` branch already in [convert.py](execution/opus_queue/tools/merge/convert.py) logs and skips. Shard is re-run by worker `wid_B`; merge picks up `wid_B`'s rows, ignores `wid_A`'s nothing. |
| SIGKILL after fsync, before `mark_done` | full rows tagged `(sid, wid_A)` durably on disk | still `running` under `wid_A`, reaper eventually resets to `pending`, another worker `wid_B` re-scores | DB's winning `worker_id` for `sid` is `wid_B`. Merge keeps `wid_B`'s rows, drops `wid_A`'s. |
| `mark_done` returns `stale_noop` | full rows tagged `(sid, wid_A)` | reaper already reset the row; another worker `wid_B` will re-score | same as above — `wid_A`'s rows filtered out at merge. |
| Reaper reclaim of a still-alive worker | depends on whether the worker's current write finished before the reaper struck | worker gets `stale_noop` on its next `mark_done` | same as above |

No case results in lost rows for a shard that the DB thinks is done. No case results in duplicate rows, because each `(direction, shard)` has exactly one winning `worker_id` in the DB. The existing `JSONDecodeError`-skipping loop in [convert.py:29-36](execution/opus_queue/tools/merge/convert.py#L29-L36) is sufficient to handle torn tails.

## File-level changes

This plan edits a small, bounded set of files. Every other OPUS-pipeline file is untouched.

### 1. `execution/opus_queue/worker/shard_io.py` — new output paths and a part writer

Replace [shard_output_path](execution/opus_queue/worker/shard_io.py#L17-L25) with two helpers:

- `direction_dir(output_base, model, direction_key) -> Path` — returns `<output_base>/<model>/<direction_key>/`.
- `next_part_path(direction_dir, worker_id_safe, seq) -> Path` — returns `direction_dir / f"part-{worker_id_safe}-{seq:04d}.jsonl"`. Pure function; no filesystem side effects.

Add a new class `DirectionPartWriter` that encapsulates the per-direction open handle. Its surface:

- `__init__(output_base, model, worker_id, *, max_bytes, max_shards_per_part)` — stores configuration, no IO.
- `append_shard(frame, direction_key, shard_id) -> Path` — serializes `frame` (already stamped with `shard_id` and `worker_id`) to JSONL, writes to the currently open part file for `direction_key` (opening one if needed, rotating if the post-write size would exceed `max_bytes` *before* writing or if `shards_in_current_part >= max_shards_per_part` *before* writing), flushes, fsyncs, and returns the path written to. The rotation check must be *before* the write so that a single oversize shard is not split across two files — it gets its own file if necessary.
- `close_direction(direction_key)` — flushes and closes the handle for that direction if open.
- `close_all()` — flushes and closes every open handle. Called from the worker's exit path.

Internally the writer holds a dict `{direction_key: (file_handle, current_path, bytes_written, shards_in_file, seq_used)}`. Because the Plan 2 claim loop processes at most one shard at a time from a single direction, the dict's expected size is 1 in steady state but may briefly hold 2 during direction handoff — that's fine and does not require an LRU eviction.

Keep [frame_to_jsonl_bytes](execution/opus_queue/worker/shard_io.py#L42-L55) as-is; it already produces the exact byte format needed. The normalization of numpy scalars, NaN, and Inf does not change.

Drop [write_temp_payload](execution/opus_queue/worker/shard_io.py#L58-L69) and [cleanup_temp_file](execution/opus_queue/worker/shard_io.py#L72-L78) once nothing in the new flow calls them (verify by grep — they are only used from `commit.py`, which this plan rewrites).

Retain [count_detail_rows](execution/opus_queue/worker/shard_io.py#L81-L84).

Helper for worker_id sanitization: extract the `worker_id.replace("/", "_").replace("\\", "_").replace(":", "_")` expression currently inside `write_temp_payload` into a module-level `sanitize_worker_id(worker_id: str) -> str` and reuse it for part filenames.

### 2. `execution/opus_queue/worker/commit.py` — append semantics, no tmp/rename dance

Rewrite [commit_shard](execution/opus_queue/worker/commit.py#L17-L71) to work with a `DirectionPartWriter` instead of atomic rename:

1. Stamp the frame with `worker_id` before serialization. `direction_key` and `shard_id` are already stamped in [loop.py:72-73](execution/opus_queue/worker/loop.py#L72-L73). Keep that in [loop.py](execution/opus_queue/worker/loop.py) — don't move stamping into `commit.py` — and add the `worker_id` stamp right after them.
2. Call `writer.append_shard(frame, direction_key, shard_id)` to get back the `out_path` of the part file the rows landed in.
3. Call `queue_db.mark_done(conn, direction_key, model, shard_id, worker_id, out_path)`. `out_path` now points to the part file rather than a per-shard file — this is the only change the DB sees.
4. If `mark_done` returns `"done"`, emit the `done` event exactly as today.
5. If `mark_done` returns `"stale_noop"`, emit the `stale_noop` event as today. **Do not truncate the part file.** The rows are durable with the stale worker's `worker_id`; merge will filter them out by `(shard_id, worker_id)` mismatch against the DB winner.
6. If an exception bubbles out of `append_shard`, propagate it. The caller ([loop.py](execution/opus_queue/worker/loop.py)) already routes exceptions through `mark_failed`. Before propagating, call `writer.close_direction(direction_key)` so the partially-written file is at least closed — any unflushed tail stays buffered in the Python `BufferedWriter` and is lost with the process, which is exactly what we want.

The function signature changes: it takes `writer: DirectionPartWriter` instead of `out_path: Path`. Update the one call site in [loop.py:202-205](execution/opus_queue/worker/loop.py#L202-L205).

### 3. `execution/opus_queue/worker/loop.py` — hold the writer across shards

Changes inside [run_loop](execution/opus_queue/worker/loop.py#L77):

- Construct `writer = DirectionPartWriter(output_base=output_base, model=queue_model, worker_id=worker_id, max_bytes=..., max_shards_per_part=...)` once, right after computing `worker_id` and before the claim loop.
- Remove the `out_path = shard_output_path(...)` computation at [line 201](execution/opus_queue/worker/loop.py#L201).
- Stamp `frame["worker_id"] = worker_id` right after the existing stamps at [lines 72-73](execution/opus_queue/worker/loop.py#L72-L73).
- Pass `writer` to `commit_shard` instead of `out_path`.
- In the `finally:` of the outer `try`, before `conn.close()`, call `writer.close_all()`. This is the single point where part files are flushed on clean exit.
- Also call `writer.close_all()` inside the exception-handling `except` branches if you add any new top-level ones. Existing per-shard `except Exception` in the inner loop need not touch the writer — individual shard failures do not require closing files; the next shard in the same direction can continue appending to the same part.

New CLI flags on [worker/cli.py](execution/opus_queue/worker/cli.py):

- `--part-max-bytes` (default `536870912`, i.e. 512 MiB).
- `--part-max-shards` (default `32`).

Expose them through `args` and feed them into `DirectionPartWriter(__init__)`. Both are tuning knobs; defaults should be validated against a real run before wider rollout.

### 4. `execution/opus_queue/tools/merge/collect.py` — discover part files, no shard_id in filename

Today [sorted_shard_files](execution/opus_queue/tools/merge/collect.py#L29-L39) matches `^shard_(\d+)\.jsonl$` and returns `list[(shard_id, Path)]`. Replace with:

- `sorted_part_files(shard_dir: Path) -> list[Path]` — returns part files sorted by `(worker_id, seq)` as extracted from filenames. Sorting order matters only for determinism of log output; it does not affect correctness. Accept both the new regex `^part-(.+)-(\d{4})\.jsonl$` and the legacy regex `^shard_(\d+)\.jsonl$` so mid-migration runs merge correctly (see "Migration" below).
- `done_jobs_for_direction(conn, direction_key, model) -> dict[int, str]` — returns `{shard_id: winning_worker_id}`. SQL:
  ```
  SELECT shard_id, worker_id FROM jobs
   WHERE direction_key = ? AND model = ? AND status = 'done'
  ```
  Note `worker_id` is the value stored at `mark_done` time, which is the worker that successfully committed.

Leave [collect_complete_directions](execution/opus_queue/tools/merge/collect.py#L11-L26) unchanged.

### 5. `execution/opus_queue/tools/merge/convert.py` — filter rows at read time

Extend [_iter_shard_record_batches](execution/opus_queue/tools/merge/convert.py#L20-L39) (already exists from the streaming patch) to accept a `winners: dict[int, str]` argument and yield only records whose `(shard_id, worker_id)` tuple is in the map. Skip the row if either field is missing — treat missing as "legacy shard format" and include it only when the filename matches the legacy `shard_<NNNNN>.jsonl` pattern (in legacy mode the shard_id from the filename is authoritative and rows are not filtered by `worker_id`).

Rewrite [merge_direction](execution/opus_queue/tools/merge/convert.py#L42-L102) to:

1. Query `winners = done_jobs_for_direction(conn, direction_key, model)`. If empty, warn and return.
2. List part files via the new `sorted_part_files`.
3. For each file:
   - If it is a new-style `part-*.jsonl`, iterate batches with the `winners` filter applied.
   - If it is a legacy `shard_NNNNN.jsonl`, extract `shard_id` from the filename, confirm `shard_id in winners`, and ingest without per-row filtering (legacy rows don't carry `worker_id`).
4. Stream into the existing `ParquetWriter` as today.
5. Drop the `shard_id` and `worker_id` columns from the parquet schema before writing — they are per-file scaffolding, not per-row output data. Implementation: pop those keys from each record dict before batching. Measure whether it's cheaper to pop per-row in Python or to drop the columns after the `pa.Table.from_pylist` call; either is acceptable.

The returned `(success, n_shards, n_rows)` tuple's meaning changes slightly: `n_shards` is now "count of winning shard_ids in DB" rather than "count of files on disk". Update the one caller ([runner.py](execution/opus_queue/tools/merge/runner.py)) to keep its log wording consistent.

Merge needs access to the DB connection inside `merge_direction`. Today [runner.py:31-40](execution/opus_queue/tools/merge/runner.py#L31-L40) opens a connection and calls `merge_direction` — pass the `conn` in. This is a minor signature change.

### 6. `utils/frames.py` or wherever `OUTPUT_COLUMNS` lives — add `worker_id`

Grep for `OUTPUT_COLUMNS` and the schema of detail/summary rows. Add `worker_id` to the allowed columns so the new stamped field survives `frame_to_jsonl_bytes` without being dropped. Confirm no downstream consumer rejects unknown columns; the `extra="forbid"` style in `models/llm_qe.py` is per-prompt and does not apply to frame schemas.

If `OUTPUT_COLUMNS` is a strict allowlist, widen it; if it's informational, just document the new column.

### 7. `execution/opus_queue/tests/test_merge_roundtrip.py` — update

The existing test writes `shard_00000.jsonl` and `shard_00001.jsonl` and asserts merge concatenates them in shard order. Extend it with:

- A new test `test_merge_filters_orphan_rows`:
  - Seed two jobs; both go to `status='done'` with `worker_id='wid_A'` (simulating two successfully committed shards).
  - Write one new-style part file `part-wid_A-0000.jsonl` containing both shards' rows stamped with `(shard_id, worker_id=wid_A)`.
  - Write a second part file `part-wid_B-0000.jsonl` containing rows stamped with `(shard_id=0, worker_id=wid_B)` — simulating an orphan from a stale_noop replay of shard 0.
  - Run merge. Assert the output contains rows from `wid_A` only; assert the output has exactly `n_rows` equal to the sum of `wid_A`'s contributions.

- A new test `test_merge_reads_legacy_layout`:
  - Seed one done job, write a legacy `shard_00000.jsonl`, assert merge still ingests it.

- Keep the existing order-preservation assertion but rewrite it in terms of `shard_id` ordering of filtered-in rows, not file-name ordering.

### 8. `execution/opus_queue/tests/test_concurrent_claim.py` and `test_worker_state_transitions.py` — audit

Scan both for assumptions about `shard_<NNNNN>.jsonl` existence. If any test asserts a specific file layout, update it. If they only exercise DB state, leave them alone.

## Migration

The runtime change is deploy-time only (new worker binary writes new layout). The migration concern is what happens when a partially-completed run restarts with the new code on top of old-layout shard directories.

Rules:

1. **Workers never rewrite old shard files.** They only ever append to their own new part files. Old `shard_<NNNNN>.jsonl` files sit untouched.
2. **Merge reads both layouts.** Legacy `shard_<NNNNN>.jsonl` files are consumed via the legacy regex in [collect.py](execution/opus_queue/tools/merge/collect.py), without row-level filtering (legacy rows don't carry `worker_id`, and a legacy `shard_NNNNN.jsonl` file is atomically written so its rows are authoritative for that shard_id).
3. **Legacy and new-layout rows for the same `shard_id` can coexist** only if the shard was first completed under the old layout (`shard_<NNNNN>.jsonl` exists) and then re-done under the new layout. That can happen if a row was reset to `pending` after the migration. Resolution: the DB's winning `worker_id` is the one from the new-layout run; merge's `(shard_id, worker_id)` filter drops the legacy row because its file carries no `worker_id` and does not match. Implementation detail: legacy-mode ingestion must check `shard_id in winners AND (no new-style file has already contributed rows for this shard_id)`. Track this with a small `seen_shard_ids_from_new_layout: set[int]` built during the first pass over new-style files, then second pass ingests legacy files for `shard_id`s not in that set.

A one-shot operator script `stand_alone_modules/migrate_shards.py` can optionally rewrite legacy files into the new layout by reading each `shard_<NNNNN>.jsonl`, stamping its rows with the current DB's `worker_id` for that shard, and writing a `part-migrated-NNNN.jsonl`. This is **not required** for correctness; the legacy-reading path in merge suffices. Offer it only if merge-directory walks become prohibitive on mixed-layout directions.

## Lustre-specific notes

- **File count impact.** For a `qwen3-4b-instruct-2507` direction of 100M sentences (10,000 shards), a `--part-max-bytes=512MiB` cap would allow well over 32 small shards per part, so `--part-max-shards=32` becomes the real limiter and file count drops from 10,000 to about 313. For `metricx24` at 200k rows per shard, 512 MiB groups roughly 12-20 shards per part; a 5,000-shard direction drops to about 250-420 files. The shard-count cap still bounds very small-shard models.
- **Byte budget rationale.** 512 MiB is still inside Lustre's sweet spot (64 MiB-1 GiB) while cutting file counts harder than 128 MiB. The tradeoff is larger crash granularity: more rows can sit in a worker's last open part file at SIGKILL. Keep `--part-max-bytes` as a tuning knob and recalibrate after the first real run.
- **Concurrent appends.** Two workers never append to the same file because `worker_id` is unique per process lifetime. No flock, no advisory locks, no POSIX-append shenanigans.
- **`fsync` frequency.** The writer `fsync`s once per shard commit, same cadence as today. It does not `fsync` per row or on rotation; the rename-open-new-part happens inside the same process, and any buffered bytes on a rotated file's `BufferedWriter` are flushed before `close()` by Python's standard library.

## Observability

Add two numeric columns to the `run_events` `detail` field or to a new log channel, whichever is cheaper to query:

- `bytes_written_this_part` at commit time.
- `rotations_this_direction` on the `done` event when a rotation occurred.

A post-run sanity query: `SELECT model, direction_key, COUNT(DISTINCT out_path) AS part_files, COUNT(*) AS shards FROM jobs WHERE status='done' GROUP BY model, direction_key ORDER BY shards DESC LIMIT 20;`. The ratio `shards / part_files` is the compression factor achieved; it should be comfortably above 1 on large directions and near 1 on small ones.

A merge-time sanity check: for each direction, `log.info("direction=%s winners=%d part_files=%d legacy_files=%d rows_kept=%d rows_dropped=%d", ...)`. `rows_dropped` > 0 is expected on any run that had crashes or stale_noops and is the visible evidence that the filter is working.

## Edge cases an implementer must handle

- **Empty directions.** A direction with `n_sentences == 0` has zero jobs and therefore zero part files. `merge_direction` already handles this ([convert.py:47-49](execution/opus_queue/tools/merge/convert.py#L47-L49)) — no change.
- **Worker processes more than one direction per session.** The claim loop ([loop.py](execution/opus_queue/worker/loop.py) line 172 onward) can hand out jobs from different directions in any order. The writer must support this: it holds one handle per direction and rotates each independently. Do not assume directions are processed in contiguous batches.
- **Worker interrupted between `append_shard` and `mark_done`.** The rows are durable, the DB still says `running`. After the reaper resets the row, a different worker re-scores and commits. The first worker's rows are filtered out by merge. No special handling in the worker is required.
- **SIGKILL while `append_shard` is writing a second shard to a part file that already contains a committed shard.** The first shard's rows are already fsynced and committed; the second shard's rows are partial. Merge's `JSONDecodeError` branch skips the partial tail; the first shard's rows are preserved by the `(shard_id, worker_id)` filter. No data loss for the first shard.
- **Worker never processes its last claimed shard because walltime ran out.** Covered by the existing `time_left < 1.5 * exp_seconds` check ([loop.py:131-137](execution/opus_queue/worker/loop.py#L131-L137)); the worker never claims a shard it cannot finish. `writer.close_all()` in the `finally:` still runs.
- **Two directions share a part file.** Cannot happen — `DirectionPartWriter` keys its handle dict by `direction_key`, so a cross-direction append is a programming bug, not a runtime concern. Enforce with an `assert direction_key == self._current_direction` inside the low-level writer if it's ever refactored to a single-handle model.
- **`worker_id` length.** Plan 2's worker_id can exceed filename-friendly limits on pathological hostnames. Lustre allows up to 255-byte filenames. `sanitize_worker_id` should additionally truncate to, say, 180 chars and append a short BLAKE2 hash of the full id to keep names short without introducing collisions. Include a test for this.
- **Reading a row with `shard_id` missing from `winners`.** Expected for orphan rows. Silently drop. Count drops in the merge summary.
- **Reading a row with `shard_id` present in `winners` but `worker_id` not matching.** Same — orphan from a replayed shard. Drop.
- **Reading a row missing both `shard_id` and `worker_id`.** Only valid in the legacy code path; in the new path it's a malformed row. Log a warning, drop.

## Rollout sequence

1. **Land the streaming merge patch** (already done in [convert.py](execution/opus_queue/tools/merge/convert.py)). This is a prerequisite — the new merge behaviour extends the streaming loop.
2. **Ship the writer + worker + merge changes behind a feature flag** `--part-writer` on [worker/cli.py](execution/opus_queue/worker/cli.py), default off. When off, fall back to the current `shard_<NNNNN>.jsonl` path.
3. **Calibrate on one direction** per model. Submit a small array with `--part-writer`, inspect resulting part files, run merge, diff the output parquet against a legacy run.
4. **Flip the default to on.** Leave the flag for rollback.
5. **Remove the legacy path** after one full OPUS pass completes cleanly under the new layout and the legacy regex in merge has been unused for at least one run (verified by the observability counter `legacy_files`).

## Testing strategy

Required tests before enabling the default:

- **Unit test for `DirectionPartWriter`:** append three shards of different sizes to one direction with `max_bytes` set low enough to force one rotation and `max_shards_per_part` set high enough to not trigger. Assert exactly two part files exist, with row counts summing to the input.
- **Unit test for rotation-by-shard-count:** symmetric, but trigger rotation via `max_shards_per_part`.
- **Unit test for direction handoff:** append shard A0 to direction A, shard B0 to direction B, shard A1 to direction A. Assert the two direction directories each contain one part file and row counts match.
- **Merge-level orphan filter:** as described in Testing section above.
- **Legacy ingestion:** as described above.
- **Mixed legacy + new in same direction:** seed one done job for an old-layout shard_00000.jsonl and one done job for a new-layout part-*-0000.jsonl covering shard_id=1. Assert merge output contains both and only both.
- **Crash simulation:** in an in-process test, open a writer, append a full shard, truncate the file to a byte offset 20 bytes short of the trailing `\n` (simulating torn write), close writer. Run merge; assert the torn row is dropped with one `JSONDecodeError` log line and the other rows survive.

Add these to [execution/opus_queue/tests/](execution/opus_queue/tests/) alongside the existing suite; the existing suite's invocation (`python -m execution.opus_queue.tests.test_merge_roundtrip`) can be extended into a shared test-runner module if preferred.

## Deliverables

- [execution/opus_queue/worker/shard_io.py](execution/opus_queue/worker/shard_io.py) with `DirectionPartWriter`, `sanitize_worker_id`, and legacy helpers removed.
- [execution/opus_queue/worker/commit.py](execution/opus_queue/worker/commit.py) rewritten to append via the writer.
- [execution/opus_queue/worker/loop.py](execution/opus_queue/worker/loop.py) holding a writer for the session's lifetime.
- [execution/opus_queue/worker/cli.py](execution/opus_queue/worker/cli.py) with two new knobs plus `--part-writer` feature flag.
- [execution/opus_queue/tools/merge/collect.py](execution/opus_queue/tools/merge/collect.py) with `sorted_part_files` and `done_jobs_for_direction`.
- [execution/opus_queue/tools/merge/convert.py](execution/opus_queue/tools/merge/convert.py) with DB-filter integration.
- [execution/opus_queue/tools/merge/runner.py](execution/opus_queue/tools/merge/runner.py) passing `conn` through.
- Updated and new tests under [execution/opus_queue/tests/](execution/opus_queue/tests/).
- One paragraph added to [execution/opus_queue/MANUAL.md](execution/opus_queue/MANUAL.md) (if it exists; otherwise inline into the top-level README) documenting the new file layout and the `--part-writer` flag.
- A one-shot calibration log from a small array submission proving: rotation triggers correctly, direction handoff works, merge filters orphans, and the on-disk file count is materially lower than a legacy run on the same manifest.

## Out of scope

- **Switching shard serialization to parquet.** Covered by its own discussion; orthogonal to append-consolidation and can ship independently in either order.
- **Bucketed subdirectory layout** (`direction_key/00/part-*.jsonl`). Unnecessary once per-direction file count drops below ~1,000; reconsider only if metrics show residual MDT stalls on the worst directions.
- **Replacing the reaper's reclaim behaviour.** The reaper continues to reset stale `running` rows exactly as today; it does not need to know anything about file layout.
- **Re-keying `worker_id` or changing its format.** The existing `{job}.{task}.{host}.{pid}` composition is sufficient for uniqueness; adding UUIDs or boot-ids is a separate decision.
- **Cross-direction compaction** (merging many part files into fewer after a run completes). If wanted, it's a separate `stand_alone_modules/compact_parts.py` job that can run post-hoc and is not on the hot path.
- **FLORES stage_writer**. [execution/flores_array/stage_writer.py](execution/flores_array/stage_writer.py) already does per-shard part rotation and is not affected by this plan.
