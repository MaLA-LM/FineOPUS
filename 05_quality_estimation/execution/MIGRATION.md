# Execution Migration

Canonical FLORES execution paths now live under `execution/flores_array/`.

Old -> new:

- `python -m dataset.make_manifest` -> `python -m execution.flores_array.make_manifest`
- `dataset.manifest` -> `execution.flores_array.manifest`
- `utils.runner` -> `execution.flores_array.runner`
- `utils.stage_writer` -> `execution.flores_array.stage_writer`
- `utils.hashing.compute_shard_id` -> `execution.flores_array.hashing.compute_shard_id`
- `scripts/run_slurm.sh` -> `scripts/flores/run_slurm.sh`
- `scripts/run_slurm_lumi.sh` -> `scripts/flores/run_slurm_lumi.sh`

The old Python module and top-level script paths have been removed. Update any
local notes, wrappers, or job submission scripts to use the canonical paths
above.

## 2026-04 reorganization

Repository structure was further normalized. Internal module moves:

- `src.common` is now a package: `src/common/{dataset_setup,frames,scoring_stats,tagging}.py`
- Backend implementation and CLI entry points moved to `src/backends/<backend>/`:
  - `python -m src.score_<backend>` → `python -m src.backends.<backend>`
- `models.language_support` is now a package:
  `models/language_support/{base,standard,remedy}.py`
- `dataset.flores200` and `dataset.opus` are the canonical adapter packages
- `execution.flores_array.runner` now only holds `run_scoring`; executor and
  direction helpers moved to `execution/flores_array/{executor,directions}.py`
- `execution.opus_queue` internals are layered under:
  - `execution/opus_queue/db/` — replaces `queue_db.py` / `queue_ops.py`
  - `execution/opus_queue/ops/` — `build_queue` CLI + `lookup_reader`
  - `execution/opus_queue/worker/` — worker CLI, run_loop, shard_io, shard_loader
  - `execution/opus_queue/planning/` — `shard_planner`, `count_cache`
  - `execution/opus_queue/scoring/` — `scorer_factory`
  - `execution/opus_queue/tools/` — `merge` and `reaper` CLIs

CLI entry-point renames:

- `python -m execution.opus_queue.build_queue` → `python -m execution.opus_queue.ops.build_queue`
- `python -m execution.opus_queue.merge` → `python -m execution.opus_queue.tools.merge`
- `python -m execution.opus_queue.reaper` → `python -m execution.opus_queue.tools.reaper`
- `python -m execution.opus_queue.worker` — unchanged (worker package)

All legacy shims were removed on 2026-04-17.
