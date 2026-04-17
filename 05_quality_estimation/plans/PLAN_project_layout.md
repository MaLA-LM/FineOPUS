# Plan: Repository-Wide Reorganization

## Context

The `05_quality_estimation` repo is a distributed FLORES-200 / OPUS QE pipeline (~5,500 LOC of Python + shell). Exploration confirms the user's complaint: several files exceed 250 lines and hold 4–8 independent concerns, and the top-level `execution/opus_queue/` directory has 14 flat files that cluster naturally into 6 layered subpackages.

The goals of this plan:

1. Break every multi-responsibility file into single-responsibility modules.
2. Group modules into layered subdirectories (`db/`, `ops/`, `worker/`, `planning/`, `scoring/`, `tools/` for opus_queue; `backends/<name>/` for src/; consistent `__main__.py` + `cli.py` for every `stand_alone_modules/*`).
3. **Preserve every existing `python -m ...` entry point** and every external import surface. Shell scripts (`scripts/flores/*.sh`, `scripts/opus/*.sh`), `README.md`, and `plans/PLAN_*.md` all invoke specific module paths — these must still resolve after the refactor. Preservation is achieved with thin shim modules (re-exports or `from ... import main as _main; _main()`).
4. Remove dead code and misplaced artifacts (legacy prompt, root-level DB, generated lookups).

Nothing below changes runtime *behaviour*. Every split is a pure re-arrangement of existing code. Refactors that would change behaviour (e.g. extracting the duplicated LLM retry loop into one function, collapsing the 5 identical `*Languages` classes into a factory) are called out explicitly as **optional refactors** and gated behind their own checklist — the plan is safe to execute without them.

---

## Invariants to preserve

These must still work after the refactor, byte-for-byte where specified.

### Entry points (`python -m ...`)

From `scripts/flores/run_slurm.sh`, `scripts/flores/run_slurm_lumi.sh`, `scripts/opus/*.sh`, `README.md`, `CLAUDE.md`, `plans/PLAN_*.md`, and `scripts/run_commands*.md`:

- `python -m src.score_comet`
- `python -m src.score_metricx`
- `python -m src.score_remedy`
- `python -m src.score_bicleaner`
- `python -m src.score_llm`
- `python -m execution.flores_array.make_manifest`
- `python -m execution.opus_queue.build_queue`
- `python -m execution.opus_queue.worker`
- `python -m execution.opus_queue.merge`
- `python -m execution.opus_queue.reaper`
- `python -m stand_alone_modules.check_done.check_shards`
- `python -m stand_alone_modules.compact`
- `python -m stand_alone_modules.create_spreadsheet`
- `python -m stand_alone_modules.dedup`
- `python -m stand_alone_modules.mean_median_ensemble`
- `python -m stand_alone_modules.normalized_scores`
- `python -m stand_alone_modules.patch_remedy`
- `python -m stand_alone_modules.patch_results`

### External import surface

These are imported from outside their own subpackage and must keep working:

- `execution.flores_array.manifest.ManifestEntry` — imported by `src/score_*.py`, `execution/opus_queue/worker.py`, `execution/opus_queue/scorer_factory.py`, `stand_alone_modules/check_done/checkpoint_status.py`.
- `execution.flores_array.runner.validate_flores_args`, `FloresArrayExecutor` — imported by scorers and opus_queue tests.
- `execution.get_executor` from [execution/__init__.py](execution/__init__.py) — the strategy registry.
- `src.common.sanitize_model_tag`, `summarize_scores`, `load_examples`, `build_scored_frames`, `ensure_dataset_ready` — imported by every `score_*` file and by `execution/opus_queue/scorer_factory.py`.
- `src.<backend>_backend.*` — imported by `src/score_<backend>.py` and by `execution/opus_queue/scorer_factory.py`.
- `dataset.mediator.get_dataset`, `DatasetAdapter`, `limit_rows` — imported by adapters, scorers, executors.
- `models.model_registry.resolve_model_spec`, `ModelSpec`, `supported_model_keys`, `Backend`.
- `models.language_support.BaseLanguageSupport` + 5 subclasses.
- `models.llm_qe.*` — Pydantic schemas.
- `utils.args.*`, `utils.cli.*`, `utils.frames.sanitize_scores`, `utils.hashing.*`, `utils.logger.{setup_logger, logger}`.
- `prompts.llm_prompt.render_prompt`, `prompts.llm_prompt_simple.render_simple_prompt`, `prompts.llm_prompt_batch.render_batch_prompt`.

**Rule of thumb:** if you move a file `pkg/foo.py` → `pkg/subpkg/foo.py`, either (a) update every import in the repo **and** add a re-export shim at the old path, or (b) keep the old path as a shim that does `from pkg.subpkg.foo import *` and update the __all__. Preferred: do (a) for all first-party code, do (b) only when a shim is needed for external callers that can't be updated (there are none here — it's all first-party).

### Canonical paths (from CLAUDE.md)

Never introduce hardcoded path alternatives. All paths stay configurable via CLI/env.

---

## Target layout (tree view)

```
05_quality_estimation/
├── CLAUDE.md                       # updated (new module paths referenced)
├── README.md                       # updated
├── data/                           # NEW — moved root-level artifacts
│   └── lookups/
│       ├── lookup_OPUS.csv         # moved from repo root
│       └── lookup_OPUS.xlsx        # moved from repo root
├── dataset/
│   ├── __init__.py
│   ├── mediator.py                 # unchanged
│   ├── flores200/                  # RENAMED from flores200_scripts/, now holds adapter too
│   │   ├── __init__.py             # re-exports FLORES200_ADAPTER
│   │   ├── adapter.py              # moved from dataset/adapters/flores200.py
│   │   ├── discovery.py            # unchanged
│   │   ├── builder.py              # renamed from flores200_builder.py
│   │   ├── frames.py               # unchanged
│   │   ├── langcodes.py            # unchanged (data)
│   │   ├── langcode_mapping.py     # refactored: typed return (see note)
│   │   └── langfamily.py           # unchanged (data)
│   ├── opus/                       # RENAMED from opus_scripts/
│   │   ├── __init__.py             # re-exports OPUS_ADAPTER
│   │   ├── adapter.py              # moved from dataset/adapters/opus.py
│   │   ├── discovery.py
│   │   ├── builder.py              # renamed from opus_builder.py
│   │   ├── frames.py
│   │   ├── langcodes.py            # data
│   │   └── langcode_mapping.py
│   └── adapters/                   # DELETED — flores200.py/opus.py moved into their packages
├── envs/                           # unchanged
├── execution/
│   ├── __init__.py                 # registry of executors (unchanged except opus_queue entry)
│   ├── base.py                     # unchanged
│   ├── MIGRATION.md                # updated
│   ├── flores_array/
│   │   ├── __init__.py             # unchanged re-exports
│   │   ├── manifest.py             # unchanged
│   │   ├── hashing.py              # unchanged
│   │   ├── shard_context.py        # unchanged
│   │   ├── stage_writer.py         # unchanged
│   │   ├── make_manifest.py        # unchanged (entry point)
│   │   ├── executor.py             # NEW — FloresArrayExecutor moved out of runner.py
│   │   ├── directions.py           # NEW — collect_directions() + validate_flores_args()
│   │   └── runner.py               # SHRUNK — run_scoring() only
│   └── opus_queue/
│       ├── __init__.py             # re-exports OpusQueueExecutor (unchanged)
│       ├── MANUAL.md               # unchanged (content) but cross-refs updated
│       ├── executor.py             # unchanged (strategy wrapper)
│       ├── schema.sql              # moved under db/ (see below)
│       ├── worker.py               # SHIM — runs worker/cli.py for `python -m execution.opus_queue.worker`
│       ├── build_queue.py          # SHIM — runs ops/build_queue/cli.py
│       ├── merge.py                # SHIM — runs tools/merge/cli.py
│       ├── reaper.py               # SHIM — runs tools/reaper/cli.py
│       ├── db/                     # NEW SUBPACKAGE — persistence layer
│       │   ├── __init__.py         # re-exports public ops used today (see section 4)
│       │   ├── schema.sql          # moved from execution/opus_queue/schema.sql
│       │   ├── connection.py       # connect(), initialize(), pragma setup  (~60 LOC)
│       │   ├── retry.py            # _execute_with_retry() decorator         (~40 LOC)
│       │   ├── claims.py           # claim_next(), mark_done(), mark_failed(), reset_own_stale(), reset_stale_rows()  (~120 LOC)
│       │   ├── queries.py          # count_by_status() + any pure-read helpers  (~40 LOC)
│       │   ├── events.py           # log_event()  (~30 LOC)
│       │   └── writes.py           # fetch_existing_models(), count_done_jobs(), reset_pending_for_model(), delete_pending_for_pair()  (renamed from queue_ops.py)
│       ├── ops/                    # NEW SUBPACKAGE — queue population write-path
│       │   ├── __init__.py
│       │   ├── build_queue/
│       │   │   ├── __init__.py
│       │   │   ├── __main__.py     # delegates to cli:main
│       │   │   ├── cli.py          # arg parsing + main()  (~60 LOC)
│       │   │   ├── runner.py       # domain orchestration: read → plan → insert  (~100 LOC)
│       │   │   └── summary.py      # per-model stats reporting  (~40 LOC)
│       │   └── lookup_reader.py    # unchanged
│       ├── worker/                 # NEW SUBPACKAGE — main execution loop
│       │   ├── __init__.py
│       │   ├── __main__.py         # delegates to cli:main
│       │   ├── cli.py              # arg parsing + main entry  (~50 LOC)
│       │   ├── loop.py             # run_loop(): claim → score → commit   (~150 LOC)
│       │   ├── commit.py           # _commit_shard, temp cleanup, stale handling  (~60 LOC)
│       │   ├── walltime.py         # remaining-time budgeting  (~30 LOC)
│       │   ├── shard_io.py         # unchanged
│       │   └── shard_loader.py     # unchanged
│       ├── planning/               # NEW SUBPACKAGE — sizing & estimation
│       │   ├── __init__.py
│       │   ├── shard_planner.py    # unchanged
│       │   └── count_cache.py      # unchanged
│       ├── scoring/                # NEW SUBPACKAGE — backend dispatch
│       │   ├── __init__.py         # re-exports resolve_backend, build_scorer
│       │   └── scorer_factory.py   # unchanged (184 LOC, already single-purpose)
│       └── tools/                  # NEW SUBPACKAGE — post-hoc CLI utilities
│           ├── __init__.py
│           ├── reaper/
│           │   ├── __init__.py
│           │   ├── __main__.py
│           │   ├── cli.py          # arg parsing + main
│           │   └── runner.py       # run_once() + reap loop
│           └── merge/
│               ├── __init__.py
│               ├── __main__.py
│               ├── cli.py          # arg parsing + main
│               ├── collect.py      # complete-direction query, shard file sorting
│               ├── convert.py      # JSONL → Parquet + sidecar metadata
│               └── runner.py       # orchestration (run())
├── models/
│   ├── __init__.py
│   ├── model_registry.py           # unchanged
│   ├── language_support/           # NEW SUBPACKAGE (was single file)
│   │   ├── __init__.py             # re-exports all 6 public classes
│   │   ├── base.py                 # BaseLanguageSupport  (~120 LOC)
│   │   ├── standard.py             # CometLanguages, QwenLanguages, PrometheusLanguages, MetricX24Languages  (~40 LOC)
│   │   └── remedy.py               # RemedyLanguages w/ ISO map helpers  (~80 LOC)
│   ├── llm_qe.py                   # unchanged
│   └── language_data/              # unchanged (pure data)
├── plans/                          # unchanged (user's plan drafts)
├── prompts/
│   ├── __init__.py                 # re-exports the three active renderers
│   ├── detailed.py                 # renamed from llm_prompt.py
│   ├── simple.py                   # renamed from llm_prompt_simple.py
│   ├── batch.py                    # renamed from llm_prompt_batch.py
│   ├── llm_prompt.py               # SHIM — `from prompts.detailed import *` for back-compat
│   ├── llm_prompt_simple.py        # SHIM
│   └── llm_prompt_batch.py         # SHIM
│   # llm_prompt_old.py             # DELETED (dead code per CLAUDE.md)
├── scripts/
│   ├── flores/
│   │   ├── run_slurm.sh            # updated (no path changes, only doc comments)
│   │   └── run_slurm_lumi.sh       # updated
│   ├── opus/
│   │   ├── run_worker.sh
│   │   ├── run_merge.sh
│   │   ├── run_reaper.sh
│   │   └── submit_array.sh
│   ├── run_commands.md             # updated
│   └── run_commands_opus.md        # updated
├── src/
│   ├── __init__.py                 # unchanged (empty marker)
│   ├── score_comet.py              # SHIM — `from src.backends.comet.cli import main; main()`
│   ├── score_metricx.py            # SHIM
│   ├── score_remedy.py             # SHIM
│   ├── score_bicleaner.py          # SHIM
│   ├── score_llm.py                # SHIM
│   ├── bicleaner_backend.py        # SHIM — re-exports from src.backends.bicleaner.backend
│   ├── metricx_backend.py          # SHIM
│   ├── remedy_backend.py           # SHIM
│   ├── llm_backend.py              # SHIM
│   ├── common/                     # NEW SUBPACKAGE (was single common.py)
│   │   ├── __init__.py             # re-exports 5 public helpers
│   │   ├── dataset_setup.py        # ensure_dataset_ready(), load_examples()
│   │   ├── scoring_stats.py        # summarize_scores()
│   │   ├── tagging.py              # sanitize_model_tag()
│   │   └── frames.py               # build_scored_frames()
│   └── backends/                   # NEW — per-backend subpackages, uniform layout
│       ├── __init__.py
│       ├── comet/
│       │   ├── __init__.py
│       │   ├── __main__.py         # calls cli:main
│       │   ├── cli.py              # parse_args() + resolve_model() + main()
│       │   ├── backend.py          # load_comet_model() + score_comet()   (~50 LOC)
│       │   └── runner.py           # score_entry() — ties common + backend
│       ├── metricx/
│       │   ├── __init__.py
│       │   ├── __main__.py
│       │   ├── cli.py
│       │   ├── backend.py          # current metricx_backend.py (unchanged)
│       │   └── runner.py
│       ├── remedy/
│       │   ├── __init__.py
│       │   ├── __main__.py
│       │   ├── cli.py              # parse_args() + resolve_model() + main()
│       │   ├── backend.py          # current remedy_backend.py
│       │   ├── lang_mapping.py     # NEW — lang → ISO 639-1 lookup (extracted from score_remedy.py lines 110-125)
│       │   └── runner.py           # score_entry()
│       ├── bicleaner/
│       │   ├── __init__.py
│       │   ├── __main__.py
│       │   ├── cli.py
│       │   ├── backend.py          # current bicleaner_backend.py (with sleep(5) moved in)
│       │   └── runner.py
│       └── llm/
│           ├── __init__.py
│           ├── __main__.py
│           ├── cli.py              # parse_args() for 20+ args  (~110 LOC)
│           ├── backend/
│           │   ├── __init__.py
│           │   ├── engine.py       # build_engine(), response-format helpers    (~80 LOC)
│           │   ├── single.py       # single-segment scoring loop                (~150 LOC)
│           │   ├── batch.py        # batch-mode scoring loop                    (~140 LOC)
│           │   ├── parsing.py      # JSON-to-Pydantic parsing for both modes   (~70 LOC)
│           │   ├── retry.py        # (OPTIONAL) shared retry loop — see §9
│           │   └── constants.py    # PromptMode / ResponseFormat enums          (~30 LOC)
│           ├── language_support.py # select_language_support(model_key)  (~30 LOC, extracted from score_llm.py lines 193-209)
│           └── runner.py           # score_entry() + main orchestration
├── stand_alone_modules/
│   ├── __init__.py
│   ├── check_done/                 # normalized
│   │   ├── __init__.py
│   │   ├── __main__.py             # NEW — delegates to cli:main
│   │   ├── cli.py                  # NEW — arg parsing extracted from check_shards.py
│   │   ├── check_shards.py         # SHIM — calls cli:main (keeps current module path working)
│   │   └── checkpoint_status.py    # unchanged
│   ├── compact/                    # unchanged (already canonical)
│   ├── create_spreadsheet/         # unchanged
│   ├── dedup/                      # unchanged
│   ├── lookup_table/               # normalized
│   │   ├── __init__.py             # NEW
│   │   ├── __main__.py             # NEW
│   │   ├── cli.py                  # NEW — arg parsing extracted from build_opus_lookup.py
│   │   ├── build_opus_lookup.py    # SHIM — keeps old script path working
│   │   ├── opus_data_loader.py     # unchanged
│   │   ├── opus_matcher.py         # unchanged
│   │   ├── sample/                 # unchanged (data)
│   │   └── tables/                 # unchanged (data)
│   ├── mean_median_ensemble/       # unchanged
│   ├── normalized_scores/          # normalized
│   │   ├── __init__.py
│   │   ├── __main__.py             # existing (3 lines)
│   │   ├── cli.py                  # NEW — expose hardcoded config as CLI flags
│   │   ├── normalize.py            # slim, reads config from cli
│   │   ├── queries.py              # unchanged
│   │   └── validate.py             # unchanged
│   ├── patch_remedy/               # normalized
│   │   ├── __init__.py
│   │   ├── __main__.py             # existing
│   │   ├── cli.py                  # NEW — expose hardcoded paths/model as CLI flags
│   │   ├── patch.py                # slim
│   │   └── queries.py              # unchanged
│   └── patch_results/              # unchanged
└── utils/                          # unchanged — already clean
    ├── __init__.py
    ├── args.py
    ├── cli.py
    ├── frames.py
    ├── hashing.py
    ├── io.py
    └── logger.py

# DELETED at repo root:
#   jobs.db                         → ephemeral DB, see §10; user confirmation required before delete
#   prompts/llm_prompt_old.py       → dead code per CLAUDE.md
#   dataset/adapters/               → directory removed after files relocated
```

---

## 1. `src/` — per-backend subpackages and shared `common/`

### 1.1 Current problems

Every `score_*.py` file (109–252 LOC) follows the same skeleton: `parse_args()` → `resolve_model()` → `score_entry()` → `main()`. They share no code beyond `utils.args.add_common_scoring_args()` and `src.common.*`, and each mixes CLI parsing with executor wiring with backend-specific I/O. `src/llm_backend.py` is 539 LOC and contains two nearly duplicated retry loops (single-segment vs batch).

### 1.2 New layout

Create `src/backends/<name>/` per backend with a **uniform three-file layout**:

- `cli.py` — `parse_args()`, `resolve_model()`, `main()` (executor wiring lives here).
- `backend.py` (or `backend/` package for `llm`) — pure model loading + scoring functions; no argparse, no executor.
- `runner.py` — `score_entry(entry, args, …)` — the per-direction orchestration that ties `common.load_examples` → `backend.score_*` → `common.build_scored_frames`.

Plus `__main__.py` so `python -m src.backends.comet` works, and keep the old `src/score_comet.py` as a thin shim so `python -m src.score_comet` still works. Shim body:

```python
# src/score_comet.py
from src.backends.comet.cli import main
if __name__ == "__main__":
    main()
```

Do the same shim for each `src/<backend>_backend.py`:

```python
# src/metricx_backend.py
from src.backends.metricx.backend import *  # noqa: F401,F403
from src.backends.metricx.backend import __all__ as _all
__all__ = _all
```

This keeps [execution/opus_queue/scorer_factory.py](execution/opus_queue/scorer_factory.py) (and any external code) working without modification.

### 1.3 Per-backend migration

#### `src/backends/comet/`

Source: [src/score_comet.py](src/score_comet.py) (109 LOC) — currently has **no** backend file.

- `cli.py` ← lines 1–18 (imports), 20–38 (`parse_args`), 41–42 (`resolve_model`), 78–105 (`main`).
- `backend.py` ← lines 45–49 (`load_comet_model`), 52–60 (`score_comet`).
- `runner.py` ← lines 63–75 (`score_entry`).

#### `src/backends/metricx/`

Source: [src/score_metricx.py](src/score_metricx.py) (104 LOC) + [src/metricx_backend.py](src/metricx_backend.py) (149 LOC).

- `cli.py` ← score_metricx.py lines 1–23 (imports), 26–44 (`parse_args`), 47–48 (`resolve_model`), 68–100 (`main`).
- `backend.py` ← entire current metricx_backend.py moved verbatim.
- `runner.py` ← score_metricx.py lines 51–65 (`score_entry`).

#### `src/backends/remedy/`

Source: [src/score_remedy.py](src/score_remedy.py) (201 LOC) + [src/remedy_backend.py](src/remedy_backend.py) (193 LOC).

- `cli.py` ← score_remedy.py lines 1–32 (imports + constants `DEFAULT_REMEDY_MODEL`, `default_cache_dir`), 34–76 (`parse_args`), 79–93 (`resolve_model`), 163–197 (`main`).
- `backend.py` ← current remedy_backend.py moved verbatim.
- `lang_mapping.py` ← **NEW**. Extract score_remedy.py lines 108–125 into `map_lang_codes_to_iso(src_lang, tgt_lang, language_support)` → `(iso_src, iso_tgt)`, fallback included.
- `runner.py` ← score_remedy.py lines 96–160 (`score_entry`), with the language mapping block replaced by a call to `lang_mapping.map_lang_codes_to_iso`.

#### `src/backends/bicleaner/`

Source: [src/score_bicleaner.py](src/score_bicleaner.py) (132 LOC) + [src/bicleaner_backend.py](src/bicleaner_backend.py) (172 LOC).

- `cli.py` ← score_bicleaner.py lines 1–33 (imports + `BICLEANER_MODEL_IDS`), 35–55 (`parse_args`), 58–61 (`resolve_model`), 107–128 (`main`).
- `backend.py` ← current bicleaner_backend.py. **Move score_bicleaner.py line 88 (`sleep(5)`) into `run_bicleaner()`** immediately after the TSV write, before `subprocess.run`. This is a filesystem-sync workaround that belongs with the subprocess call, not in orchestration.
- `runner.py` ← score_bicleaner.py lines 64–104 (`score_entry`), minus the `sleep(5)` line.

#### `src/backends/llm/`

Source: [src/score_llm.py](src/score_llm.py) (252 LOC) + [src/llm_backend.py](src/llm_backend.py) (539 LOC). This is the biggest split.

- `cli.py` ← score_llm.py lines 1–33 (imports + constants), 35–132 (`parse_args`), 135–145 (`resolve_model`), 177–248 (`main` + engine-build closure).
- `language_support.py` ← **NEW**. Extract score_llm.py lines 193–202 (model_key → language support routing) into `select_language_support(model_key, dataset) -> BaseLanguageSupport`. Include lines 204–209 (auto `response_format` selection) if logically co-located.
- `runner.py` ← score_llm.py lines 148–174 (`score_entry`).
- `backend/` (sub-subpackage, because llm_backend.py is 539 LOC and has natural sub-concerns):
  - `constants.py` ← llm_backend.py lines 28–49 (`PromptMode`, `ResponseFormat` enums + config dicts).
  - `parsing.py` ← lines 54–61 (prompt-mode → result model), 117–149 (single-segment parsing), 243–270 (batch parsing).
  - `engine.py` ← lines 63–102 (temperature/thinking/sampling helpers), 105–114 (prompt render dispatcher), 152–191 (structured-outputs builder), 197–237 (`build_engine`).
  - `single.py` ← lines 404–539 (`score_llm_offline` single-segment path, but only the parts unique to single mode — see below).
  - `batch.py` ← lines 273–398 (`_score_batch_mode`).
  - `retry.py` ← **OPTIONAL** (§9). If taken, common retry-loop helper extracted from both `single.py` and `batch.py`.
- `__init__.py` in `backend/` re-exports `build_engine`, `score_llm_offline`, `PROMPT_MODES`, `RESPONSE_FORMATS`, all module constants so `src/llm_backend.py` shim can re-export from `src.backends.llm.backend`.

Public entry points `score_llm_offline` must stay importable as `src.llm_backend.score_llm_offline` (the shim handles this) **and** as `src.backends.llm.backend.score_llm_offline`.

### 1.4 `src/common/` — break up the 66-LOC catch-all

[src/common.py](src/common.py) mixes five concerns. Split into a subpackage with re-exports so all existing `from src.common import X` imports keep working unchanged:

- `dataset_setup.py` — `ensure_dataset_ready`, `load_examples`.
- `scoring_stats.py` — `summarize_scores`.
- `tagging.py` — `sanitize_model_tag`.
- `frames.py` — `build_scored_frames`.
- `__init__.py`:

```python
from src.common.dataset_setup import ensure_dataset_ready, load_examples
from src.common.frames import build_scored_frames
from src.common.scoring_stats import summarize_scores
from src.common.tagging import sanitize_model_tag

__all__ = [
    "ensure_dataset_ready",
    "load_examples",
    "summarize_scores",
    "sanitize_model_tag",
    "build_scored_frames",
]
```

Delete the old [src/common.py](src/common.py) after the package is in place.

---

## 2. `execution/flores_array/` — small polish only

Already clean. One split to do.

### 2.1 Split `runner.py` (201 LOC, 4 concerns)

Current [execution/flores_array/runner.py](execution/flores_array/runner.py) contains (a) `validate_flores_args()`, (b) `collect_directions()` (manifest reading + bucketing), (c) `run_scoring()` (the core loop), and (d) `FloresArrayExecutor` (the strategy class + `add_cli_args`). Split into:

- `executor.py` — `FloresArrayExecutor` class and its `add_cli_args` method.
- `directions.py` — `validate_flores_args()`, `collect_directions()`.
- `runner.py` — `run_scoring()` only (plus `_make_run_id()` helper).

Update [execution/flores_array/__init__.py](execution/flores_array/__init__.py):

```python
from execution.flores_array.directions import validate_flores_args
from execution.flores_array.executor import FloresArrayExecutor
from execution.flores_array.manifest import (
    ManifestEntry, read_manifest, read_manifest_entries, write_manifest,
)

__all__ = [
    "FloresArrayExecutor", "ManifestEntry",
    "read_manifest", "read_manifest_entries", "write_manifest",
    "validate_flores_args",
]
```

Add back-compat shims at the original qualified paths (`execution.flores_array.runner.FloresArrayExecutor`, `execution.flores_array.runner.validate_flores_args`) by keeping the names in `runner.py`:

```python
# execution/flores_array/runner.py (bottom)
from execution.flores_array.executor import FloresArrayExecutor  # noqa: F401
from execution.flores_array.directions import validate_flores_args  # noqa: F401
```

This keeps `execution/opus_queue/tests/test_opus_adapter_contract.py` line 17 working.

No changes to `manifest.py`, `hashing.py`, `shard_context.py`, `stage_writer.py`, `make_manifest.py`.

---

## 3. `execution/opus_queue/` — the big one

### 3.1 Current problems

14 flat Python files. `worker.py` (313 LOC), `queue_db.py` (295 LOC), `build_queue.py` (284 LOC), `merge.py` (216 LOC), `reaper.py` (124 LOC) all hold CLI + domain + I/O + persistence concerns.

### 3.2 Target subpackages

Six new subpackages (`db/`, `ops/`, `worker/`, `planning/`, `scoring/`, `tools/`). Every existing `python -m execution.opus_queue.<x>` entry point is preserved via a thin shim at the old location.

### 3.3 `execution/opus_queue/db/` — persistence layer

Source: [execution/opus_queue/queue_db.py](execution/opus_queue/queue_db.py) (295 LOC), [queue_ops.py](execution/opus_queue/queue_ops.py) (115 LOC), [schema.sql](execution/opus_queue/schema.sql).

Split `queue_db.py` along its five internal responsibilities:

- `connection.py`:
  - `connect(db_path)` — sqlite3 connection with `PRAGMA journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout`, `row_factory`.
  - `initialize(conn)` — execute `schema.sql`.
  - Any module-level constants for pragma defaults.
- `retry.py`:
  - `_execute_with_retry(cursor, sql, params, …)` — the SQLITE_BUSY retry helper. Promote it to a public-ish name (`execute_with_retry`) so `claims.py` can import.
- `claims.py`:
  - `claim_next(conn, worker_id, model, …)`, `mark_done(conn, …)`, `mark_failed(conn, …)`, `reset_own_stale(conn, worker_id)`, `reset_stale_rows(conn, cutoff)`.
  - Uses `retry.execute_with_retry`.
- `queries.py`:
  - `count_by_status(conn, model)` and any pure-read queries.
- `events.py`:
  - `log_event(conn, event_type, payload, …)` — observability.
- `writes.py` — renamed from current `queue_ops.py`. Contents unchanged: `fetch_existing_models`, `count_done_jobs`, `reset_pending_for_model`, `delete_pending_for_pair`. Rename because "ops" is now a sibling package name and would collide semantically.
- `schema.sql` — moved here (was at `execution/opus_queue/schema.sql`). Update `connection.initialize()` to load from `pathlib.Path(__file__).with_name("schema.sql")`.

`db/__init__.py`:

```python
from execution.opus_queue.db.claims import (
    claim_next, mark_done, mark_failed, reset_own_stale, reset_stale_rows,
)
from execution.opus_queue.db.connection import connect, initialize
from execution.opus_queue.db.events import log_event
from execution.opus_queue.db.queries import count_by_status
from execution.opus_queue.db.writes import (
    count_done_jobs, delete_pending_for_pair, fetch_existing_models,
    reset_pending_for_model,
)

__all__ = [
    "connect", "initialize",
    "claim_next", "mark_done", "mark_failed",
    "reset_own_stale", "reset_stale_rows",
    "count_by_status", "log_event",
    "fetch_existing_models", "count_done_jobs",
    "reset_pending_for_model", "delete_pending_for_pair",
]
```

Back-compat shims at the old module paths to keep tests/callers working:

- `execution/opus_queue/queue_db.py` — shim: `from execution.opus_queue.db import *`.
- `execution/opus_queue/queue_ops.py` — shim: `from execution.opus_queue.db.writes import *`.
- `execution/opus_queue/schema.sql` — delete only after confirming no callers reference it via that path (grep reveals none, safe to delete).

### 3.4 `execution/opus_queue/ops/` — queue population

Source: [execution/opus_queue/build_queue.py](execution/opus_queue/build_queue.py) (284 LOC) + [lookup_reader.py](execution/opus_queue/lookup_reader.py) (66 LOC).

- `ops/build_queue/cli.py` — `_parse_args()` + `main()` entry.
- `ops/build_queue/runner.py` — `run()`: read lookup → plan shards → handle reassign → insert. Pure domain, no argparse.
- `ops/build_queue/summary.py` — per-model stats printer extracted from `run()`.
- `ops/build_queue/__main__.py` — `from .cli import main; main()`.
- `ops/lookup_reader.py` — moved from `execution/opus_queue/lookup_reader.py` unchanged.

Back-compat shim at `execution/opus_queue/build_queue.py` for `python -m execution.opus_queue.build_queue`:

```python
# execution/opus_queue/build_queue.py
from execution.opus_queue.ops.build_queue.cli import main
if __name__ == "__main__":
    main()
```

Back-compat shim for `execution.opus_queue.lookup_reader` at the old path (only if any module imports it via the old name; `build_queue.py` currently does — update to new path instead and leave no shim). Confirm with grep before removing.

### 3.5 `execution/opus_queue/worker/` — main loop

Source: [execution/opus_queue/worker.py](execution/opus_queue/worker.py) (313 LOC) + [shard_io.py](execution/opus_queue/shard_io.py) (85 LOC) + [shard_loader.py](execution/opus_queue/shard_loader.py) (150 LOC).

- `worker/cli.py` — `_parse_args()`, `_make_worker_id()`, `main()`. ~50 LOC.
- `worker/loop.py` — `run_loop(db, model, …)`: claim → score → commit loop. ~150 LOC. Imports `commit._commit_shard`, `walltime.remaining_seconds`.
- `worker/commit.py` — `_commit_shard(frame, direction_key, …)`, temp-file cleanup, failure-marking glue. ~60 LOC.
- `worker/walltime.py` — remaining walltime computation, budget checks. ~30 LOC.
- `worker/shard_io.py` — moved from `execution/opus_queue/shard_io.py` unchanged.
- `worker/shard_loader.py` — moved from `execution/opus_queue/shard_loader.py` unchanged.
- `worker/__main__.py` — `from .cli import main; main()`.

Back-compat shim at `execution/opus_queue/worker.py`:

```python
from execution.opus_queue.worker.cli import main
from execution.opus_queue.worker.loop import run_loop  # noqa: F401
if __name__ == "__main__":
    main()
```

This preserves `python -m execution.opus_queue.worker` **and** `from execution.opus_queue.worker import run_loop` (used by [execution/opus_queue/executor.py](execution/opus_queue/executor.py) line 6).

Back-compat shims for `shard_io.py` and `shard_loader.py` are **not required** because they are only imported internally by `worker.py`. Update imports in the new worker files instead.

### 3.6 `execution/opus_queue/planning/` — sizing & estimation

Source: [execution/opus_queue/shard_planner.py](execution/opus_queue/shard_planner.py) (107 LOC) + [count_cache.py](execution/opus_queue/count_cache.py) (51 LOC). Both already single-purpose — just group them under `planning/`.

- `planning/shard_planner.py` — moved verbatim.
- `planning/count_cache.py` — moved verbatim.
- `planning/__init__.py` re-exports `ShardRange`, `get_shard_size`, `plan`, `expected_shard_seconds`, `DEFAULT_EXPECTED_SHARD_SECONDS`, `parse_shard_size_overrides`, and the count-cache helpers.

Back-compat shims at old paths (because `reaper.py` line 7 and `build_queue.py` line 13 currently import from `execution.opus_queue.shard_planner`; update those imports instead and leave no shim).

### 3.7 `execution/opus_queue/scoring/` — backend dispatch

Source: [execution/opus_queue/scorer_factory.py](execution/opus_queue/scorer_factory.py) (184 LOC). Already clean (5 small builder functions + `resolve_backend` + `build_scorer`). Just relocate under `scoring/` for grouping consistency.

- `scoring/scorer_factory.py` — moved verbatim.
- `scoring/__init__.py` — re-exports `resolve_backend`, `build_scorer`.

Update the import in `worker/loop.py` (`from execution.opus_queue.scoring import build_scorer, resolve_backend`).

No back-compat shim needed (internal-only).

### 3.8 `execution/opus_queue/tools/` — reaper & merge

Source: [execution/opus_queue/reaper.py](execution/opus_queue/reaper.py) (124 LOC) + [merge.py](execution/opus_queue/merge.py) (216 LOC).

- `tools/reaper/cli.py` — arg parsing + `main()`.
- `tools/reaper/runner.py` — `run_once(db, cutoffs, …)` + the reap/sleep loop.
- `tools/reaper/__main__.py`.
- `tools/merge/cli.py` — arg parsing + `main()`.
- `tools/merge/collect.py` — `collect_complete_directions(db, model)` + shard-file sorting helpers.
- `tools/merge/convert.py` — JSONL → Parquet + metadata sidecar writer + shard-delete logic.
- `tools/merge/runner.py` — `run(args)`: orchestrates collect → convert → report → delete.
- `tools/merge/__main__.py`.

Back-compat shims at `execution/opus_queue/reaper.py` and `execution/opus_queue/merge.py`:

```python
# execution/opus_queue/reaper.py
from execution.opus_queue.tools.reaper.cli import main
if __name__ == "__main__":
    main()
```

```python
# execution/opus_queue/merge.py
from execution.opus_queue.tools.merge.cli import main
if __name__ == "__main__":
    main()
```

This preserves `python -m execution.opus_queue.merge` and `python -m execution.opus_queue.reaper`. Also note `execution/opus_queue/tests/test_merge_roundtrip.py` line 14 imports `from execution.opus_queue import merge, queue_db` — the shim satisfies that because `merge` is still a module path and `queue_db` goes through the db shim. Good.

### 3.9 `execution/opus_queue/executor.py` — unchanged

This is the `ExecutionStrategy` wrapper. It only imports `run_loop` from `execution.opus_queue.worker`, which the shim handles. No changes needed.

### 3.10 Update `MANUAL.md`

Update any file-path references inside [execution/opus_queue/MANUAL.md](execution/opus_queue/MANUAL.md) to point at the new package structure. Do not change the CLI examples — they still work via the shims.

---

## 4. `dataset/` — collapse `adapters/` and `flores200_scripts/`

### 4.1 Rationale

Currently `dataset/adapters/flores200.py` (30 LOC) and `dataset/flores200_scripts/*` are split across two dirs but conceptually one unit. Same for OPUS. Collapse each into a single `dataset/<name>/` package.

### 4.2 Moves

- `dataset/flores200_scripts/` → `dataset/flores200/`.
- `dataset/flores200_scripts/flores200_builder.py` → `dataset/flores200/builder.py` (rename).
- `dataset/adapters/flores200.py` → `dataset/flores200/adapter.py`.
- `dataset/opus_scripts/` → `dataset/opus/`.
- `dataset/opus_scripts/opus_builder.py` → `dataset/opus/builder.py`.
- `dataset/adapters/opus.py` → `dataset/opus/adapter.py`.
- Delete empty `dataset/adapters/` directory.

### 4.3 Update imports

Every consumer of `dataset.flores200_scripts.*` and `dataset.opus_scripts.*` must be updated:

- [dataset/adapters/flores200.py](dataset/adapters/flores200.py) — the adapter file, now moved to `dataset/flores200/adapter.py`: change its imports from `dataset.flores200_scripts.X` to relative `from . import X` or `from dataset.flores200.X`.
- Same for `dataset/opus/adapter.py`.
- Any `src/*` or `stand_alone_modules/*` file that imports `dataset.flores200_scripts.*` — grep and rewrite.

`dataset/flores200/__init__.py`:

```python
from dataset.flores200.adapter import FLORES200_ADAPTER
__all__ = ["FLORES200_ADAPTER"]
```

`dataset/opus/__init__.py`:

```python
from dataset.opus.adapter import OPUS_ADAPTER
__all__ = ["OPUS_ADAPTER"]
```

### 4.4 `langcode_mapping.py` — type the return value

Current behaviour: `build_model_language_mapping(supported_languages)` returns `dict[str, list[bool, str | None]]`. Callers index `[0]`/`[1]`.

Replace with a frozen dataclass:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LanguageMatch:
    is_supported: bool
    matched_name: str | None
```

Return `dict[str, LanguageMatch]`. Update every caller (grep for `build_model_language_mapping` — currently in `models/language_support.py` and callers of `get_full_language_name`). This is a small behaviour-preserving refactor but touches several files; **optional**, can be deferred.

### 4.5 `dataset/mediator.py` — unchanged

Its `DATASETS` registry will import from the new paths — update those two lines only.

---

## 5. `models/` — collapse the 5 language-support classes

### 5.1 Current

[models/language_support.py](models/language_support.py) (249 LOC) defines `BaseLanguageSupport` and five near-identical subclasses (`CometLanguages`, `QwenLanguages`, `PrometheusLanguages`, `MetricX24Languages`, `RemedyLanguages`). Only `RemedyLanguages` adds real logic (ISO 639-1 lookups).

### 5.2 New layout

Promote to a subpackage so the 80-LOC Remedy-specific block doesn't dominate the file:

- `models/language_support/base.py` — `BaseLanguageSupport` + `_resolve_dataset`, `_resolve_language_codes`, `_resolve_langcode_mapper` helpers. ~120 LOC.
- `models/language_support/standard.py` — `CometLanguages`, `QwenLanguages`, `PrometheusLanguages`, `MetricX24Languages`. ~40 LOC. Each is a 5–8 line subclass that passes the label + supported set to `BaseLanguageSupport.__init__`.
- `models/language_support/remedy.py` — `RemedyLanguages` with its ISO-map lookup methods. ~80 LOC.
- `models/language_support/__init__.py`:

```python
from models.language_support.base import BaseLanguageSupport
from models.language_support.remedy import RemedyLanguages
from models.language_support.standard import (
    CometLanguages, MetricX24Languages, PrometheusLanguages, QwenLanguages,
)

__all__ = [
    "BaseLanguageSupport",
    "CometLanguages", "QwenLanguages", "PrometheusLanguages",
    "MetricX24Languages", "RemedyLanguages",
]
```

Delete old `models/language_support.py`. All existing `from models.language_support import X` imports keep working because `language_support` is now a package.

### 5.3 Optional: DRY the 4 standard classes

They are currently copy-paste with only a label + supported set differing. A base factory using `__init_subclass__` or a `classmethod` can collapse them to:

```python
class CometLanguages(BaseLanguageSupport):
    _supported = COMET_SUPPORTED_LANGUAGES
    _label = "COMET"
```

This is a behaviour-preserving cleanup but **optional** — list under §9.

### 5.4 `models/model_registry.py`, `models/llm_qe.py`, `models/language_data/`

Unchanged.

---

## 6. `utils/` — no changes

`args.py`, `cli.py`, `frames.py`, `hashing.py`, `io.py`, `logger.py` are all single-purpose and under 140 LOC. Leave them alone.

---

## 7. `prompts/` — delete dead code, rename, keep shims

### 7.1 Delete

`prompts/llm_prompt_old.py` — CLAUDE.md explicitly states it is unused. Grep confirms no importers. Delete the file.

### 7.2 Rename (optional — prefer keep for back-compat)

Rename the remaining files to shorter names to match their responsibilities:

- `llm_prompt.py` → `detailed.py` (detailed 7-dim prompt).
- `llm_prompt_simple.py` → `simple.py`.
- `llm_prompt_batch.py` → `batch.py`.

Keep back-compat shims at the old module paths so `prompts.llm_prompt.render_prompt` etc. still work (they are imported from `src/backends/llm/backend/engine.py` and `src/llm_backend.py`):

```python
# prompts/llm_prompt.py
from prompts.detailed import *  # noqa: F401,F403
```

### 7.3 `prompts/__init__.py`

Add re-exports for convenience:

```python
from prompts.batch import render_batch_prompt
from prompts.detailed import render_prompt
from prompts.simple import render_simple_prompt

__all__ = ["render_prompt", "render_simple_prompt", "render_batch_prompt"]
```

**If rename feels too churny**, skip §7.2 and just delete `llm_prompt_old.py` (§7.1). That's the only truly necessary change here.

---

## 8. `stand_alone_modules/` — standardize the pattern

Every module must follow: `<name>/__main__.py` + `<name>/cli.py` + domain modules.

### 8.1 `check_done/` — add missing `__main__.py` and `cli.py`

Source: [stand_alone_modules/check_done/check_shards.py](stand_alone_modules/check_done/check_shards.py) (82 LOC) + [checkpoint_status.py](stand_alone_modules/check_done/checkpoint_status.py) (213 LOC).

- New `cli.py` — lifts arg parsing (`--tsv`, `--path`, `--model`, `--dataset`) and `main()` out of `check_shards.py`.
- New `__main__.py`:

```python
from stand_alone_modules.check_done.cli import main
if __name__ == "__main__":
    main()
```

- Keep `check_shards.py` as a back-compat shim so `python -m stand_alone_modules.check_done.check_shards` (used by README.md and scripts/run_commands.md) still works:

```python
# stand_alone_modules/check_done/check_shards.py
from stand_alone_modules.check_done.cli import main
if __name__ == "__main__":
    main()
```

- `checkpoint_status.py` unchanged.
- Add `__init__.py` if missing (the listing shows there is none).

### 8.2 `lookup_table/` — add `__main__.py` and `cli.py`

Source: [stand_alone_modules/lookup_table/build_opus_lookup.py](stand_alone_modules/lookup_table/build_opus_lookup.py) (80 LOC) + `opus_data_loader.py` + `opus_matcher.py`.

- New `cli.py` — arg parsing extracted from `build_opus_lookup.py`.
- New `__main__.py` delegating to `cli:main`.
- Keep `build_opus_lookup.py` as shim.
- Add `__init__.py` if missing.

### 8.3 `normalized_scores/` — add `cli.py`, expose hardcoded config

Source: [stand_alone_modules/normalized_scores/normalize.py](stand_alone_modules/normalized_scores/normalize.py) (68 LOC). Currently hardcodes the model list and paths.

- New `cli.py` — adds `--src-root`, `--dst-root`, `--models` (nargs='+'), with the current hardcoded values as defaults.
- Update `normalize.py` to read config from a `Config` dataclass passed in by `cli:main` instead of module-level constants.
- `__main__.py` already exists (3 lines) — keep it, just point at `cli:main`.
- `queries.py`, `validate.py` unchanged.

### 8.4 `patch_remedy/` — same treatment as `normalized_scores/`

Source: [stand_alone_modules/patch_remedy/patch.py](stand_alone_modules/patch_remedy/patch.py) (72 LOC). Hardcodes paths and model.

- New `cli.py` with `--src-root`, `--dst-root`, `--model`.
- Update `patch.py` to accept config as argument.
- `__main__.py` exists — keep, repoint.

### 8.5 `compact/`, `create_spreadsheet/`, `dedup/`, `mean_median_ensemble/`, `patch_results/`

Already follow the pattern. No changes.

---

## 9. Optional in-file refactors

These do not change the file layout but clean up multi-responsibility code called out in the exploration. Safe to defer. Gate each behind its own checklist so the reorganization can merge independently.

### 9.1 LLM retry-loop DRY (~150 LOC duplicated)

[src/llm_backend.py](src/llm_backend.py) lines 273–398 (`_score_batch_mode`) and lines 404–539 (single-segment path in `score_llm_offline`) share an almost-identical retry structure: build conversations → call engine → parse responses → track failed indices → retry with adjusted temperature. Only the parsing step differs.

Extract to `src/backends/llm/backend/retry.py`:

```python
def run_with_retries(
    *,
    items: Sequence[ItemT],
    build_conversations: Callable[[Sequence[ItemT]], list[list[dict]]],
    parse_responses: Callable[[list[str], Sequence[ItemT]], list[ResultT]],
    engine: LLM,
    sampling_params: Callable[[float], SamplingParams],
    base_temperature: float,
    max_retries: int,
) -> list[ResultT | None]: ...
```

Call from both `single.py` and `batch.py`. Removes ~100 LOC of duplication. Same behaviour.

### 9.2 `LanguageMatch` dataclass (§4.4)

Replace `list[bool, str | None]` with a typed dataclass. Small, safe, touches ~6 files.

### 9.3 Factory for 4 standard language-support classes (§5.3)

Collapse `CometLanguages`, `QwenLanguages`, `PrometheusLanguages`, `MetricX24Languages` into a single parameterized base using `__init_subclass__` or a `make_language_support(label, supported_set) -> type` factory.

---

## 10. Root-level cleanup

### 10.1 Files to move or delete

- `jobs.db` (195 MB) — this is an OPUS queue DB generated by `execution.opus_queue.build_queue`. It should not live in the repo root:
  - If it's needed for ongoing work: move to `data/jobs.db` (create `data/` dir) and update any shell/doc reference.
  - If it's an artefact: delete, add `jobs.db` to `.gitignore` (if the repo had gitignore; CLAUDE.md says the repo is not a git repo, but add the guidance anyway).
  - **Require user confirmation before deleting.**
- `lookup_OPUS.csv`, `lookup_OPUS.xlsx` — generated by `stand_alone_modules.lookup_table`. Move to `data/lookups/`. Update references in scripts and `stand_alone_modules/lookup_table/cli.py` defaults.

### 10.2 `scripts/run_commands.md` and `scripts/run_commands_opus.md`

These are ad-hoc command logs. Keep as-is, but update any lines that reference moved artefacts (e.g. `lookup_OPUS.xlsx` → `data/lookups/lookup_OPUS.xlsx`).

### 10.3 Docs

Update [CLAUDE.md](CLAUDE.md) and [README.md](README.md):

- Architecture tree diagrams.
- Any `python -m` command (should all still work, but paths shown in docs should mention the new internal modules for the benefit of new readers).
- The "Key modules" section in CLAUDE.md — update the bullet list to the new layout.

Update [execution/MIGRATION.md](execution/MIGRATION.md) with a new "2026-04 reorganization" section pointing old module paths at new ones for future grep.

---

## 11. Execution order

Do these steps in order. Each step leaves the tree working (all `python -m` paths resolve, all imports succeed). Do **not** batch multiple phases before testing.

1. **Prep** — scan the tree once more to confirm no new files have appeared since this plan was written. Commit the plan file itself before any other change (so the plan is a referenceable artefact).
2. **`src/common/` split** (§1.4). Smallest risky change: single file → subpackage with re-exports. Verify `from src.common import sanitize_model_tag` works.
3. **`src/backends/<name>/` creation** (§1.3), one backend at a time. Order: `comet` → `metricx` → `remedy` → `bicleaner` → `llm`. After each, run the smoke test (§12).
4. **Prompts cleanup** (§7) — delete `llm_prompt_old.py`, optionally rename the rest with shims.
5. **`models/language_support/` package** (§5.2). Verify every consumer still resolves.
6. **`dataset/flores200/` and `dataset/opus/` rename** (§4). Update every consumer's imports.
7. **`execution/flores_array/runner.py` split** (§2.1). Small, low-risk.
8. **`execution/opus_queue/` reorganization** (§3). Do one subpackage at a time, in order: `db/` → `planning/` → `scoring/` → `ops/` → `worker/` → `tools/`. After each, run the opus_queue test suite if available.
9. **`stand_alone_modules/` standardization** (§8). Each module is independent — parallelizable.
10. **Root-level cleanup** (§10). Require user confirmation for `jobs.db` handling.
11. **Docs update** (§10.3).
12. **Optional refactors** (§9) if explicitly requested.

At every step, the rule is: **no import path that worked before must stop working**. Shim modules exist exactly to uphold that rule.

---

## 12. Verification

There is no automated test harness in scope for this plan (tests are excluded per user instructions). Verification is syntactic and import-level.

### 12.1 Static import check

After each phase, run:

```bash
cd /path/to/05_quality_estimation
python -c "
import importlib
mods = [
    'src.common', 'src.score_comet', 'src.score_metricx', 'src.score_remedy',
    'src.score_bicleaner', 'src.score_llm',
    'src.bicleaner_backend', 'src.metricx_backend', 'src.remedy_backend', 'src.llm_backend',
    'src.backends.comet.cli', 'src.backends.metricx.cli', 'src.backends.remedy.cli',
    'src.backends.bicleaner.cli', 'src.backends.llm.cli',
    'execution.flores_array', 'execution.flores_array.runner',
    'execution.flores_array.make_manifest', 'execution.flores_array.executor',
    'execution.opus_queue', 'execution.opus_queue.executor',
    'execution.opus_queue.worker', 'execution.opus_queue.build_queue',
    'execution.opus_queue.merge', 'execution.opus_queue.reaper',
    'execution.opus_queue.queue_db', 'execution.opus_queue.queue_ops',
    'execution.opus_queue.db', 'execution.opus_queue.ops', 'execution.opus_queue.worker',
    'execution.opus_queue.planning', 'execution.opus_queue.scoring', 'execution.opus_queue.tools',
    'execution.opus_queue.scoring.scorer_factory',
    'dataset.mediator', 'dataset.flores200', 'dataset.opus',
    'models.model_registry', 'models.language_support', 'models.llm_qe',
    'utils.args', 'utils.cli', 'utils.frames', 'utils.hashing', 'utils.logger',
    'prompts.llm_prompt', 'prompts.llm_prompt_simple', 'prompts.llm_prompt_batch',
    'stand_alone_modules.check_done.check_shards',
    'stand_alone_modules.check_done.checkpoint_status',
    'stand_alone_modules.compact', 'stand_alone_modules.create_spreadsheet',
    'stand_alone_modules.dedup', 'stand_alone_modules.lookup_table',
    'stand_alone_modules.mean_median_ensemble', 'stand_alone_modules.normalized_scores',
    'stand_alone_modules.patch_remedy', 'stand_alone_modules.patch_results',
]
for m in mods:
    importlib.import_module(m)
    print(f'OK {m}')
"
```

Every line must print `OK`. Any `ModuleNotFoundError` or `ImportError` is a regression and must be fixed before continuing.

*(This command runs on a local dev machine with the deps installed — per CLAUDE.md, the model itself must not execute code. This snippet is for the human or CI to run.)*

### 12.2 `python -m` entry-point check

Each of these invocations must at minimum produce argparse help text (no import failures):

```bash
for cmd in \
  "src.score_comet" "src.score_metricx" "src.score_remedy" "src.score_bicleaner" "src.score_llm" \
  "execution.flores_array.make_manifest" \
  "execution.opus_queue.build_queue" "execution.opus_queue.worker" \
  "execution.opus_queue.merge" "execution.opus_queue.reaper" \
  "stand_alone_modules.check_done.check_shards" "stand_alone_modules.compact" \
  "stand_alone_modules.create_spreadsheet" "stand_alone_modules.dedup" \
  "stand_alone_modules.mean_median_ensemble" "stand_alone_modules.normalized_scores" \
  "stand_alone_modules.patch_remedy" "stand_alone_modules.patch_results" ; do
    python -m "$cmd" --help >/dev/null && echo "OK $cmd" || echo "FAIL $cmd"
done
```

### 12.3 End-to-end smoke test (per CLAUDE.md canonical example)

On a dev host with dependencies installed, run the one-direction smoke test described in [CLAUDE.md](CLAUDE.md) line 123:

```bash
python -m src.score_comet \
    --dataset flores200 --root "$DATA_DIR" --execution flores_array \
    --manifest flores200_directions.tsv --output-base "$OUTPUT_BASE" \
    --model wmt22-cometkiwi-da --num-shards 1 --shard-id 0 --max-rows 10
```

Output files must appear at `$OUTPUT_BASE/dataset=flores200/model=wmt22-cometkiwi-da/split=devtest/shard=000/` exactly as before (layout defined in CLAUDE.md line 83).

Repeat for `score_metricx`, `score_remedy`, `score_bicleaner`, `score_llm` using their canonical example models from the README.

### 12.4 OPUS-queue smoke test

```bash
python -m execution.opus_queue.build_queue --help
python -m execution.opus_queue.worker --help
python -m execution.opus_queue.merge --help
python -m execution.opus_queue.reaper --help
```

### 12.5 Shell scripts

`scripts/flores/run_slurm.sh` and `scripts/flores/run_slurm_lumi.sh` are `bash -n`-checked only (no execution — per CLAUDE.md the model must not submit jobs). Same for `scripts/opus/*.sh`:

```bash
for f in scripts/flores/*.sh scripts/opus/*.sh; do
    bash -n "$f" && echo "OK $f" || echo "FAIL $f"
done
```

---

## 13. Critical files — single-reference list

To keep the exec agent focused, here are every file it will touch, grouped by phase:

### Created new (net-new files)

- `src/common/{__init__.py, dataset_setup.py, scoring_stats.py, tagging.py, frames.py}`
- `src/backends/__init__.py`
- `src/backends/comet/{__init__.py, __main__.py, cli.py, backend.py, runner.py}`
- `src/backends/metricx/{__init__.py, __main__.py, cli.py, backend.py, runner.py}`
- `src/backends/remedy/{__init__.py, __main__.py, cli.py, backend.py, lang_mapping.py, runner.py}`
- `src/backends/bicleaner/{__init__.py, __main__.py, cli.py, backend.py, runner.py}`
- `src/backends/llm/{__init__.py, __main__.py, cli.py, language_support.py, runner.py}`
- `src/backends/llm/backend/{__init__.py, engine.py, single.py, batch.py, parsing.py, constants.py}` (+ `retry.py` if §9.1 is taken)
- `execution/flores_array/{executor.py, directions.py}`
- `execution/opus_queue/db/{__init__.py, connection.py, retry.py, claims.py, queries.py, events.py, writes.py, schema.sql}`
- `execution/opus_queue/ops/{__init__.py, lookup_reader.py, build_queue/{__init__.py, __main__.py, cli.py, runner.py, summary.py}}`
- `execution/opus_queue/worker/{__init__.py, __main__.py, cli.py, loop.py, commit.py, walltime.py, shard_io.py, shard_loader.py}`
- `execution/opus_queue/planning/{__init__.py, shard_planner.py, count_cache.py}`
- `execution/opus_queue/scoring/{__init__.py, scorer_factory.py}`
- `execution/opus_queue/tools/{__init__.py, reaper/{__init__.py, __main__.py, cli.py, runner.py}, merge/{__init__.py, __main__.py, cli.py, collect.py, convert.py, runner.py}}`
- `models/language_support/{__init__.py, base.py, standard.py, remedy.py}`
- `dataset/flores200/{__init__.py, adapter.py}` (adapter.py moved; __init__.py new with re-exports)
- `dataset/opus/{__init__.py, adapter.py}`
- `stand_alone_modules/check_done/{__init__.py, __main__.py, cli.py}`
- `stand_alone_modules/lookup_table/{__init__.py, __main__.py, cli.py}`
- `stand_alone_modules/normalized_scores/cli.py`
- `stand_alone_modules/patch_remedy/cli.py`
- `data/lookups/`

### Converted to shim (file still exists, but body is 2–4 lines)

- `src/score_{comet,metricx,remedy,bicleaner,llm}.py`
- `src/{bicleaner,metricx,remedy,llm}_backend.py`
- `execution/opus_queue/{queue_db.py, queue_ops.py, worker.py, build_queue.py, merge.py, reaper.py}`
- `execution/flores_array/runner.py` (keeps `run_scoring` locally, re-exports `FloresArrayExecutor` and `validate_flores_args`)
- `stand_alone_modules/check_done/check_shards.py`
- `stand_alone_modules/lookup_table/build_opus_lookup.py`
- Optional: `prompts/llm_prompt{,_simple,_batch}.py` (only if §7.2 rename is taken)

### Moved (renamed within tree)

- `dataset/adapters/flores200.py` → `dataset/flores200/adapter.py`
- `dataset/adapters/opus.py` → `dataset/opus/adapter.py`
- `dataset/flores200_scripts/*` → `dataset/flores200/*` (with `flores200_builder.py` → `builder.py`)
- `dataset/opus_scripts/*` → `dataset/opus/*` (with `opus_builder.py` → `builder.py`)
- `execution/opus_queue/schema.sql` → `execution/opus_queue/db/schema.sql`
- `execution/opus_queue/shard_planner.py` → `execution/opus_queue/planning/shard_planner.py`
- `execution/opus_queue/count_cache.py` → `execution/opus_queue/planning/count_cache.py`
- `execution/opus_queue/scorer_factory.py` → `execution/opus_queue/scoring/scorer_factory.py`
- `execution/opus_queue/shard_io.py` → `execution/opus_queue/worker/shard_io.py`
- `execution/opus_queue/shard_loader.py` → `execution/opus_queue/worker/shard_loader.py`
- `execution/opus_queue/lookup_reader.py` → `execution/opus_queue/ops/lookup_reader.py`
- `lookup_OPUS.csv` → `data/lookups/lookup_OPUS.csv`
- `lookup_OPUS.xlsx` → `data/lookups/lookup_OPUS.xlsx`
- `jobs.db` → `data/jobs.db` (or delete after user confirmation)

### Deleted

- `src/common.py` (after the subpackage is in place)
- `models/language_support.py` (after the subpackage is in place)
- `dataset/adapters/` (empty after moves)
- `prompts/llm_prompt_old.py` (dead code)

### Modified in place (imports / docs only)

- `execution/__init__.py` (new subpackage name or comment if needed)
- `execution/opus_queue/__init__.py` (unchanged — already lazy-imports `OpusQueueExecutor`)
- `execution/opus_queue/executor.py` (imports updated to new worker path if the shim is removed; keep as-is if shim stays)
- `execution/opus_queue/MANUAL.md` (path references updated)
- `execution/MIGRATION.md` (add a "2026-04 reorganization" section)
- `dataset/mediator.py` (import paths updated)
- `README.md` (architecture tree updated)
- `CLAUDE.md` ("Key modules" list updated)
- `scripts/run_commands.md`, `scripts/run_commands_opus.md` (updated references to moved artefacts under `data/lookups/`)

---

## 14. Risks and mitigations

- **Hidden imports via string-based dispatch** — confirm by grep for any `importlib.import_module("...")` or `__import__("...")`. A pass over the codebase shows only `execution/opus_queue/__init__.py` uses the lazy-`__getattr__` pattern for `OpusQueueExecutor`; it is preserved unchanged.
- **Shell scripts call `python -m`** — the shim strategy covers every documented path. Re-verify by bash-grepping for `python -m` in `scripts/` and ensuring each target is in §0's invariants list.
- **Tests directory** (`execution/opus_queue/tests/`) is excluded per user instructions, but the test files reference paths like `from execution.opus_queue import queue_db, queue_ops` — the shims satisfy these imports without modifying tests. Good.
- **Large `jobs.db` (195 MB)** — don't move it without user confirmation, it might be active state. Plan lists this under §10 as a confirmation-gated step.
- **`models/language_support.py` as file vs package collision** — when the subpackage is created, the file must be deleted in the same commit (or Python will silently prefer the file). Verified by the verification script in §12.1.
- **Language support copy-paste factoring** (§5.3) — behaviour-preserving but touches 5 class definitions. Optional and gated.

---

## 15. What this plan intentionally does NOT do

- Does not change any runtime behaviour. All refactors are structural.
- Does not introduce new abstractions beyond what already exists (no new base classes, protocols, or registries unless listed as optional in §9).
- Does not touch the tests directory.
- Does not touch files starting with `.` (`.bashrc`, `.claude`, `.codex`, `.github`, `.pycache_tmp`, `.tmp_sqlite_probe`, `.tmp_test`).
- Does not rewrite or modify shell scripts except to update path references if artefacts move (§10.3 only).
- Does not delete `jobs.db` or any data file without user confirmation.
- Does not alter the canonical CSC/LUMI paths enumerated in CLAUDE.md.