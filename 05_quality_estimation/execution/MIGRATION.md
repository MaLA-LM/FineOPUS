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

Repository structure was further normalized without changing public CLI entry
points. Internal module moves:

- `src.common` is now a package: `src/common/{dataset_setup,frames,scoring_stats,tagging}.py`
- `src.score_<backend>` remains the public entry point, but implementation now
  lives under `src/backends/<backend>/`
- `models.language_support` is now a package:
  `models/language_support/{base,standard,remedy}.py`
- `dataset.flores200` and `dataset.opus` are the canonical adapter packages
- `execution.flores_array.runner` now only holds `run_scoring`; executor and
  direction helpers moved to `execution/flores_array/{executor,directions}.py`
- `execution.opus_queue` internals are layered under:
  - `execution/opus_queue/db/`
  - `execution/opus_queue/ops/`
  - `execution/opus_queue/worker/`
  - `execution/opus_queue/planning/`
  - `execution/opus_queue/scoring/`
  - `execution/opus_queue/tools/`

Compatibility shims remain at the old import paths for the documented queue
entry points and legacy helper modules.
