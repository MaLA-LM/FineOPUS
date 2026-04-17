# Plan 1 — Isolate the FLORES-200 execution model

## Goal

The current code under `05_quality_estimation/` mixes two concerns:

1. **Dataset I/O** — how to discover directions, load sentence pairs, and build output frames. This layer is already clean: FLORES lives in `dataset/flores200_scripts/` + `dataset/adapters/flores200.py`, OPUS lives in `dataset/opus_scripts/` + `dataset/adapters/opus.py`, and they plug into the shared `DatasetAdapter` in `dataset/mediator.py`.

2. **Parallelism / execution model** — how work is split across SLURM array tasks, how progress is tracked, how outputs are rotated. This layer is **not** dataset-neutral. Everything in `dataset/make_manifest.py`, `dataset/manifest.py`, `utils/hashing.py::compute_shard_id`, `utils/runner.py::resolve_shard_context`, and `utils/stage_writer.py` assumes the FLORES execution pattern: hash each direction into one of N buckets, one SLURM task per bucket, one direction is an atomic unit.

This plan's scope is **layer 2 only**: rename, relocate, and tag everything that currently implements "the FLORES parallelism strategy" so it becomes a self-contained module called `execution/flores_array/`. After this plan is done, a second plan can add `execution/opus_queue/` alongside it without touching any FLORES code.

**Non-goals:** do not change any FLORES behavior; do not move dataset adapters; do not touch any `src/score_*.py` scorer logic; do not rename any column, file path, or output layout that downstream tooling in `stand_alone_modules/` depends on.

## Current architecture (as found)

```
05_quality_estimation/
├── dataset/
│   ├── adapters/{flores200.py, opus.py}       # already separated — do not move
│   ├── flores200_scripts/                     # already separated — do not move
│   ├── opus_scripts/                          # already separated — do not move
│   ├── mediator.py                            # DatasetAdapter registry — keep
│   ├── make_manifest.py                       # ← FLORES-style manifest builder
│   └── manifest.py                            # ← FLORES-style manifest reader
├── utils/
│   ├── runner.py                              # ← contains resolve_shard_context (SLURM array logic)
│   ├── stage_writer.py                        # ← contains ShardStageWriter (shard=NNN layout)
│   └── hashing.py                             # ← contains compute_shard_id
├── src/score_*.py                             # per-backend scorers — keep logic, but entry point changes
├── scripts/{run_slurm.sh, run_slurm_lumi.sh}  # ← SLURM launchers (FLORES-style array)
└── stand_alone_modules/                       # post-processing — mostly dataset-aware, keep
```

The `src/score_*.py` scorers today call `run_scoring(args, dataset, directions, model_tag, score_entry)` in `utils/runner.py`. That function is the single integration point where execution strategy is decided. Plan 1 isolates this integration point so Plan 2 can add a second strategy next to it.

## Target architecture after Plan 1

```
05_quality_estimation/
├── dataset/                                   # unchanged
│   ├── adapters/, flores200_scripts/, opus_scripts/, mediator.py
├── execution/                                 # NEW — holds all parallelism strategies
│   ├── __init__.py                            # exports: get_executor(name)
│   ├── base.py                                # NEW — ExecutionStrategy protocol + shared types
│   └── flores_array/                          # NEW — everything FLORES-specific moves here
│       ├── __init__.py
│       ├── manifest.py                        # moved from dataset/manifest.py
│       ├── make_manifest.py                   # moved from dataset/make_manifest.py
│       ├── shard_context.py                   # moved resolve_shard_context() from utils/runner.py
│       ├── stage_writer.py                    # moved from utils/stage_writer.py
│       ├── hashing.py                         # moved compute_shard_id from utils/hashing.py
│       └── runner.py                          # moved run_scoring() from utils/runner.py
├── utils/                                     # kept but slimmed down
│   ├── args.py, cli.py, frames.py, io.py, logger.py   # unchanged
│   └── hashing.py                             # keeps stable_hash_int only; compute_shard_id moves out
├── src/score_*.py                             # one small change: call execution layer instead of utils.runner
├── scripts/
│   ├── flores/                                # NEW — clear label
│   │   ├── run_slurm.sh                       # moved scripts/run_slurm.sh
│   │   └── run_slurm_lumi.sh                  # moved scripts/run_slurm_lumi.sh
│   └── opus/                                  # left empty; filled by Plan 2
└── stand_alone_modules/                       # unchanged except import-path fixes
```

## Rationale

1. **The `execution/` package becomes the seam.** Any future execution model (OPUS queue, Ray, Dask, local multiprocessing) drops in as a sibling of `flores_array/`. The scorers don't need to know which strategy is in play; they call into a named strategy.

2. **Naming makes the FLORES assumption explicit.** Today a reader sees `dataset/manifest.py` and reasonably assumes "this is the manifest for the dataset" — but it's actually "the manifest for the FLORES execution model, which happens to work on any dataset." Renaming to `execution/flores_array/manifest.py` surfaces this.

3. **Nothing gets deleted.** Every file keeps its content. This plan is 95% rename/relocate, 5% wiring.

4. **Output paths stay byte-identical.** `ShardStageWriter` keeps its `dataset={}/model={}/split={}/shard=NNN/` layout. Downstream tools in `stand_alone_modules/` keep working without changes.

---

## Step-by-step implementation

### Step 1 — Create the `execution/` package scaffold

Create `05_quality_estimation/execution/__init__.py`, `execution/base.py`, and `execution/flores_array/__init__.py`.

`execution/base.py` defines a minimal `ExecutionStrategy` protocol that describes what any strategy must expose. The shape should mirror what `utils/runner.py::run_scoring` currently does: given a `DatasetAdapter`, a model identifier, a `score_entry` callback, and parsed CLI args, run scoring to completion. Concretely the protocol should declare:

- a name (string identifier like `"flores_array"` or `"opus_queue"`)
- an `add_cli_args(parser)` classmethod that registers strategy-specific CLI flags
- a `run(args, dataset, model_tag, score_entry)` method that performs the run

`execution/__init__.py` exposes a small registry: `get_executor(name) -> ExecutionStrategy`. Initially only `"flores_array"` is registered. A `--execution` CLI flag (added in Step 5) will pick between strategies, defaulting to `"flores_array"` so existing invocations behave identically.

### Step 2 — Move FLORES-style manifest code

Move the following files **verbatim** (no logic changes) into `execution/flores_array/`:

| From | To |
|---|---|
| `dataset/make_manifest.py` | `execution/flores_array/make_manifest.py` |
| `dataset/manifest.py` | `execution/flores_array/manifest.py` |

Update the CLI invocation in the moved `make_manifest.py` so its module path becomes `python -m execution.flores_array.make_manifest` (it was `python -m dataset.make_manifest`). Document the old and new commands in the repo `README.md`.

Update every importer. Known callers (search for `from dataset.manifest` and `dataset.make_manifest`):

- `utils/runner.py`
- any script in `stand_alone_modules/` that reads the manifest (grep `read_manifest_entries`, `ManifestEntry`)

Any caller gets its import line rewritten from `from dataset.manifest import ...` to `from execution.flores_array.manifest import ...`.

### Step 3 — Move shard-hashing logic

`utils/hashing.py` currently holds two functions: `stable_hash_int()` (generic, keep in place) and `compute_shard_id()` (FLORES-style). Move **only** `compute_shard_id()` into `execution/flores_array/hashing.py`. Keep `stable_hash_int()` in `utils/hashing.py` since it's a general utility that the OPUS plan will reuse for other purposes.

Update importers: anything that did `from utils.hashing import compute_shard_id` now does `from execution.flores_array.hashing import compute_shard_id`. Grep confirms the callers are `execution/flores_array/manifest.py` (after Step 2) and `utils/runner.py`.

### Step 4 — Move `ShardStageWriter` and runner

Move `utils/stage_writer.py` verbatim to `execution/flores_array/stage_writer.py`.

Split `utils/runner.py` into two parts:

- `execution/flores_array/shard_context.py` — takes the `ShardContext` dataclass and `resolve_shard_context()` function (the pieces that read `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT`).
- `execution/flores_array/runner.py` — takes the `run_scoring()` function. This file now imports `shard_context` and `stage_writer` from its own package, and calls into the manifest reader from its own package. After this move, `utils/runner.py` no longer exists (or is reduced to a deprecation shim for one release that re-exports `run_scoring` with a warning — author's choice).

Wrap `runner.py` in a thin class `FloresArrayExecutor` that implements the `ExecutionStrategy` protocol from Step 1. Its `run()` is the existing `run_scoring` logic, and its `add_cli_args()` registers `--manifest`, `--shard-id`, `--num-shards`, `--max-directions-per-part`, `--target-part-bytes` — i.e., every flag that exists today in `utils/args.py::add_common_scoring_args` that is strategy-specific. Move those flag definitions out of `utils/args.py` and into the executor so generic flags (model name, batch size, root dir, output base, etc.) stay in `utils/args.py`.

Register the executor in `execution/__init__.py` under the name `"flores_array"`.

### Step 5 — Update scorers to go through the execution registry

Every file in `src/score_*.py` currently ends with a `main()` that does:

```
run_scoring(args, dataset, directions, model_tag, lambda entry: score_entry(...))
```

Change the call pattern so scorers no longer import `utils.runner` directly. Instead:

1. `utils/args.py::add_common_scoring_args()` adds a new `--execution` flag, default `"flores_array"`, choices taken from the executor registry.
2. Each scorer's `main()` does `executor = get_executor(args.execution)` and then `executor.run(args, dataset, model_tag, score_entry_callable)`.
3. The executor itself, not the scorer, is responsible for calling `collect_directions(args, dataset)` and looping. (Today that loop is inside `run_scoring`; it stays there, just under a new home.)

This is the only behavioral edit in Plan 1 and it is a pure refactor — default behavior is identical because `--execution` defaults to `"flores_array"`.

### Step 6 — Relocate SLURM launchers

Create `scripts/flores/` and move `scripts/run_slurm.sh` and `scripts/run_slurm_lumi.sh` into it. Add a one-line comment header in each file stating: "This launcher uses the flores_array execution strategy: hash(direction) % num_shards → SLURM array task id." This makes the assumption explicit for anyone reading the scripts in isolation.

Create `scripts/opus/` as an empty directory with a `.gitkeep` so Plan 2 has a clear home.

Search the scripts for hard-coded `python -m dataset.make_manifest` and update to `python -m execution.flores_array.make_manifest`.

### Step 7 — Fix stand-alone modules

`stand_alone_modules/` contains post-processing scripts that read the output directory layout or the manifest. Any script that imports `dataset.manifest` or `utils.stage_writer` needs its import path fixed. The output layout itself is unchanged (`ShardStageWriter` behavior is preserved), so any module that only reads output files on disk needs no change.

Specific modules to audit (grep each for `from dataset.manifest`, `from utils.stage_writer`, `from utils.runner`, `from utils.hashing import compute_shard_id`):

- `stand_alone_modules/check_done/`
- `stand_alone_modules/compact/`
- `stand_alone_modules/create_spreadsheet/`
- `stand_alone_modules/dedup/`
- `stand_alone_modules/lookup_table/`
- `stand_alone_modules/mean_median_ensemble/`
- `stand_alone_modules/normalized_scores/`
- any `patch_*/` directories

### Step 8 — Update documentation

Edit `05_quality_estimation/README.md`:

- Add a new top-level section "Execution strategies" that states: "The pipeline supports pluggable execution strategies under `execution/`. Today only `flores_array` is implemented. OPUS support is planned under `opus_queue` (see Plan 2)."
- Update every command example that mentions `python -m dataset.make_manifest` to the new path.
- Mention the `--execution` CLI flag and its default.
- Document that `scripts/flores/` is the FLORES launcher home and `scripts/opus/` is reserved for OPUS.

### Step 9 — Verification

After all moves, the following must hold. provide the following to the user to check, since you can't run these on LUMI:

1. `python -m execution.flores_array.make_manifest --dataset flores200 --split devtest --num-shards 100 --out /tmp/test.tsv` produces a TSV identical (diffable byte-for-byte) to what `python -m dataset.make_manifest` used to produce for the same arguments.
2. `python -m src.score_comet --dataset flores200 --manifest /tmp/test.tsv --shard-id 0 --num-shards 100 --model wmt22-cometkiwi-da --output-base /tmp/out` writes files to `/tmp/out/dataset=flores200/model=wmt22-cometkiwi-da/split=devtest/shard=000/` in exactly the old layout.
3. `grep -r "from dataset.manifest"` and `grep -r "from utils.runner"` and `grep -r "from utils.stage_writer"` return no results inside `05_quality_estimation/` (or only inside deprecation shims, if any).
4. `grep -r "compute_shard_id"` shows it only imported from `execution.flores_array.hashing`.
5. Unit tests (if present) and at least one end-to-end smoke run on a tiny subset of FLORES (e.g., 2–3 directions, one model, one shard) still complete successfully.
6. No file under `stand_alone_modules/` has a broken import after the moves.

## Deliverables

- performs all nine steps atomically. The diff should be dominated by file renames and import-path edits, with only `src/score_*.py` changes touching logic (and only trivially — switching from direct `run_scoring` call to `get_executor(...).run(...)`).
- Updated `README.md`.
- An end-to-end smoke test log proving step 9.2.
- A short migration note added to the repo (e.g., `execution/MIGRATION.md`) listing the old → new import paths for anyone with in-flight branches.

## Out of scope (explicitly)

- Any changes to `dataset/adapters/`, `dataset/flores200_scripts/`, or `dataset/opus_scripts/`.
- Any changes to the output JSONL schema, directory layout, or checkpoint format.
- Any OPUS execution support — Plan 2 handles that on top of the `execution/` scaffold created here.
- Any changes to model backends (`src/*_backend.py`, `src/score_*.py` scorer bodies) beyond the single-line dispatch change in Step 5.
