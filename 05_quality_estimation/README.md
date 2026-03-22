# Quality Estimation Pipeline for Machine Translation

A distributed, multi-model quality estimation (QE) pipeline that scores translation output across **all FLORES-200 language pairs** (~42,000 directions) using five different backend types: **COMET**, **MetricX-24**, **ReMedy**, **Bicleaner**, and **LLM-based** (Qwen3, M-Prometheus). Designed to run on HPC clusters using SLURM job arrays for parallelism.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Dataset Module](#dataset-module)
   - [Mediator & Adapter Pattern](#mediator--adapter-pattern)
   - [FLORES-200 Adapter](#flores-200-adapter)
   - [Language Code Mapping](#language-code-mapping)
   - [Discovery & Data Loading](#discovery--data-loading)
   - [Manifest System](#manifest-system)
3. [Models Module](#models-module)
   - [Model Registry](#model-registry)
   - [Registered Models](#registered-models)
   - [Language Support System](#language-support-system)
   - [LLM QE Data Models](#llm-qe-data-models)
4. [Scoring Backends (src/)](#scoring-backends-src)
   - [Common Utilities](#common-utilities)
   - [COMET Scorer](#comet-scorer)
   - [MetricX-24 Scorer](#metricx-24-scorer)
   - [ReMedy Scorer](#remedy-scorer)
   - [Bicleaner Scorer](#bicleaner-scorer)
   - [LLM Scorer](#llm-scorer)
5. [Prompts Module](#prompts-module)
   - [Single-Segment Prompt](#single-segment-prompt)
   - [Batch Prompt](#batch-prompt)
6. [Utils Module](#utils-module)
   - [CLI Arguments](#cli-arguments)
   - [Runner (Pipeline Orchestrator)](#runner-pipeline-orchestrator)
   - [Stage Writer & Checkpointing](#stage-writer--checkpointing)
   - [Output Format](#output-format)
   - [Hashing & Sharding](#hashing--sharding)
   - [Logging](#logging)
7. [How to Run](#how-to-run)
   - [Step 1: Create a Manifest](#step-1-create-a-manifest)
   - [Step 2: Submit SLURM Jobs](#step-2-submit-slurm-jobs)
   - [Step 3: Manual / Local Runs](#step-3-manual--local-runs)
   - [SLURM Defaults (Mahti vs LUMI)](#slurm-defaults-mahti-vs-lumi)
   - [Environment Variables](#environment-variables)
   - [ Stand alone modules/features](#stand-alone-modulesfeatures)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          CLI Entry Points                                │
│   python -m src.score_comet / score_metricx / score_remedy / ...         |
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │    utils/runner.py       │
                    │  (Pipeline Orchestrator) │
                    │  - resolve shard context │
                    │  - collect directions    │
                    │  - loop & checkpoint     │
                    └──┬──────────────┬────────┘
                       │              │
          ┌────────────▼──┐    ┌──────▼───────────────┐
          │  dataset/     │    │  utils/stage_writer  │
          │  (FLORES-200) │    │  (JSONL output +     │
          │  load pairs   │    │   checkpointing)     │
          └───────────────┘    └──────────────────────┘
                       │
        ┌──────────────┼───────────────────────────┐
        │              │              │            │
   ┌────▼────┐  ┌──────▼─────┐ ┌─────▼────┐ ┌────▼─────┐
   │  COMET  │  │  MetricX   │ │  ReMedy  │ │   LLM    │
   │ Backend │  │  Backend   │ │ Backend  │ │ Backend  │
   │(PyTorch)│  │(MT5 model) │ │(CLI+vLLM)│ │(vLLM+    │
   │         │  │            │ │          │ │ httpx)   │
   └─────────┘  └────────────┘ └──────────┘ └──────────┘
```

Each scorer:
1. Parses CLI args and resolves the model via the **model registry**
2. Reads the **manifest** to get all language-pair directions
3. Uses **shard context** (from SLURM array or CLI) to pick its subset
4. For each direction: loads data, scores, writes JSONL with checkpointing
5. **Checkpointing** allows safe restarts — already-scored directions are skipped

---

## Dataset Module

### Mediator & Adapter Pattern

File: `dataset/mediator.py`

The dataset system uses an **adapter pattern** so different datasets can be plugged in without changing the scoring code. Currently only FLORES-200 is implemented, but the architecture supports adding more.

**`DatasetAdapter`** is a frozen dataclass that defines the interface every dataset must provide:

| Field | Type | Purpose |
|-------|------|---------|
| `id` | `str` | Unique identifier (e.g., `"flores200"`) |
| `default_root` | `str \| Path` | Default filesystem path to the dataset |
| `split_values` | `tuple[str, ...]` | Valid split names (e.g., `("dev", "devtest")`) |
| `load_parallel` | `Callable` | Loads source + target sentence pairs |
| `limit_rows` | `Callable` | Truncates examples to a max count |
| `discover_directions` | `Callable` | Finds all src→tgt language pairs on disk |
| `expected_detail_rows` | `Callable` | Computes expected row counts per direction |
| `language_codes` | `dict[str, str]` | Maps language codes → display names |
| `langcode_to_name` | `Callable` | Maps language codes → model-supported language names |

**Registration**: The `DATASETS` dict maps dataset IDs to adapter instances. Lookup is via `get_dataset(dataset_id)` which normalizes the input (lowercase, strip) and returns the adapter. If not found, it raises `SystemExit` listing available datasets.

```python
# Usage
from dataset import get_dataset
ds = get_dataset("flores200")  # returns FLORES200_ADAPTER
```

### FLORES-200 Adapter

File: `dataset/adapters/flores200.py`

Wires together all FLORES-200 sub-modules into a single `DatasetAdapter` instance:

```python
FLORES200_ADAPTER = DatasetAdapter(
    id="flores200",
    default_root="/scratch/project_462001050/downstream_benchmarks/flores200",
    split_values=("dev", "devtest"),
    load_parallel=load_flores200_parallel,
    limit_rows=limit_rows,
    discover_directions=discover_directions,
    expected_detail_rows=expected_detail_rows,
    language_codes=flores200_langcodes,       # 200+ language code → name mapping
    langcode_to_name=build_model_language_mapping,
)
```

### Language Code Mapping

File: `dataset/flores200_scripts/langcode_mapping.py`

It matches FLORES-200 display names to model-supported language names using a **four-stage fuzzy matching pipeline**.

**Problem**: Each model defines its own set of supported language names (e.g., COMET uses `"Oriya"`, FLORES uses `"Odia"` for the same language). This module bridges that gap.

#### Stage 1: Exact Match
Case-insensitive string comparison between the FLORES display name and the model's language set.

```
"English" vs model set {"english", "french", ...} → match "english" ✓
```

#### Stage 2: Alias Match
Checks a hardcoded `SPECIAL_ALIASES` dictionary for known name discrepancies:

```python
SPECIAL_ALIASES = {
    "Odia":             ["Oriya"],               # FLORES says "Odia", COMET says "Oriya"
    "Northern Kurdish":  ["Kurdish (Kurmanji)"],  # FLORES vs COMET naming
    "Yue Chinese":      ["Cantonese"],            # FLORES vs Qwen naming
    "Eastern Panjabi":  ["Punjabi"],              # FLORES vs common name
}
```

For example, when checking COMET support for `"Odia"`:
1. Stage 1 fails (COMET doesn't have "Odia")
2. Stage 2 tries alias `"Oriya"` → COMET has it → match ✓

#### Stage 3: Left-Trim Match
Since flores200 language names are usually very specific and model languages are broad, we progressively drops leftmost words to handle qualifier prefixes:

```
"South Azerbaijani" → try "Azerbaijani" → match if found ✓
"Modern Standard Arabic" → try "Standard Arabic" → try "Arabic" → match ✓
```

#### Stage 4: Strip Qualifiers
Removes parenthesized content, then retries exact + left-trim:

```
"Chinese (Simplified)" → "Chinese" → try exact → try left-trim
"Kashmiri (Arabic script)" → "Kashmiri" → try exact → match ✓
```

**Main export**: `build_model_language_mapping(model_languages: set[str]) -> dict[str, list]`

Returns a dict mapping every FLORES language code to `[is_supported: bool, matched_name: str | None]`:

```python
mapping = build_model_language_mapping({"English", "French"})
# {"eng_Latn": [True, "English"], "fra_Latn": [True, "French"], "deu_Latn": [False, None], ...}
```
### Language codes

1) `dataset/flores200_scripts/langcodes.py` — Maps all 200+ FLORES language codes to display names, are copied from the official flores200 repository
2) `dataset/flores200_scripts/langfamily.py` created using `glottolog` library, maps codes to language families

**Note**: `Basque`, `chinese (simplified)` and `chinese (Traditional)` have no language families, thus were added manually in the `langfamily.py` dict.
`chinese` as `Sino-Tibetan`, `Basque` as `Basque`.

### Discovery & Data Loading

#### Discovery (`dataset/flores200_scripts/discovery.py`)

Discovers all available translation directions on disk by scanning the FLORES-200 directory structure.

**File naming convention**: `{langcode}.{split}` (e.g., `eng_Latn.dev`, `spa_Latn.devtest`)

**Algorithm** for `discover_directions(root, split)`:
1. Resolve splits: if `split=None` or `"all"`, use both `["dev", "devtest"]`
2. For each split, glob `*.<split>` files to find all language codes
3. Generate all permutations of `(src, tgt)` where `src != tgt`
4. Return list of `(src_lang, tgt_lang, split, src_path)` tuples

**Hardcoded expected row counts**:
```python
EXPECTED_ROWS = {"dev": 997, "devtest": 1012}
```

The `expected_detail_rows()` function reconciles these hardcoded expectations with actual file line counts. If the actual count differs, the actual count takes precedence.

#### Data Loading (`dataset/flores200_scripts/flores200_builder.py`)

**`load_flores200_parallel(src_lang, tgt_lang, *, split, root)`**:
1. Validates split is `"dev"` or `"devtest"`
2. Reads lines from `{root}/{split}/{src_lang}.{split}` and `{root}/{split}/{tgt_lang}.{split}`
3. Verifies both files have the same number of lines
4. Returns list of `{"src": "...", "tgt": "..."}` dicts

**`limit_rows(examples, max_rows)`**: Truncates to `min(max_rows, len(examples))`. Returns all if `max_rows` is `None`.


### Manifest System

The manifest is a TSV file that lists all language-pair directions to score, with deterministic shard assignments. The whole point of the manifest file is to run multiple array tasks of one model on the same dataset with no additional headache of concurrency handling.

#### Creating a Manifest (`dataset/make_manifest.py`)

CLI tool invoked as `python -m dataset.make_manifest`.

**Algorithm**:
1. Discover all directions from the dataset on disk
2. Sort by `(split, src_lang, tgt_lang)` for reproducibility
3. For each direction, compute a deterministic `shard_id` via BLAKE2b hashing
4. Write TSV with columns: `src_lang`, `tgt_lang`, `split`, `shard_id`

**Example output** (`flores200_directions.tsv`):
```
src_lang    tgt_lang    split    shard_id
ace_Arab    ace_Latn    dev      3
ace_Arab    acm_Arab    dev      7
...
```

#### Reading a Manifest (`dataset/manifest.py`)

**`ManifestEntry`** dataclass: `(src_lang, tgt_lang, split, shard_id)`

- `read_manifest_entries()` — Parses TSV, validates columns, returns `ManifestEntry` objects
- `write_manifest()` — Writes TSV using `compute_shard_id()` for shard assignment

---

## Models Module

### Model Registry

File: `models/model_registry.py`

Central registry of all supported QE models. Each model is a **`ModelSpec`** frozen dataclass:

| Field | Type | Purpose |
|-------|------|---------|
| `key` | `str` | Primary identifier |
| `backend` | `Literal["comet","metricx","llm","bicleaner","remedy"]` | Backend type |
| `model_id` | `str` | HuggingFace ID or custom identifier |
| `aliases` | `tuple[str, ...]` | Alternative names |
| `tokenizer_id` | `str \| None` | Separate tokenizer (MetricX only) |
| `max_length` | `int \| None` | Max sequence length |
| `score_adjuster` | `Callable \| None` | Post-processing function for scores |

**Resolution algorithm** (`resolve_model_spec(name, backend)`):
1. Normalize input (lowercase, strip whitespace)
2. Look up in a pre-built dict that indexes by: key, model_id, and all aliases
3. Return `(ModelSpec, resolved_key)` or raise `ValueError` with supported list

### Registered Models

#### COMET Backend (3 models)
| Key | Model ID | Aliases |
|-----|----------|---------|
| `wmt22-cometkiwi-da` | `Unbabel/wmt22-cometkiwi-da` | `wmt22-comet` |
| `wmt23-cometkiwi-da-xl` | `Unbabel/wmt23-cometkiwi-da-xl` | `wmt23-comet` |
| `xcomet-xl` | `Unbabel/XCOMET-XL` | `xcomet` |

#### MetricX Backend (1 model)
| Key | Model ID | Tokenizer | Max Length | Score Adjuster |
|-----|----------|-----------|------------|----------------|
| `metricx24` | `google/metricx-24-hybrid-xl-v2p6` | `google/mt5-xl` | 1536 | `metricx_adjust` |

**MetricX score adjustment**: MetricX-24 outputs scores on a 0–25 scale (lower = better). The `metricx_adjust` function converts to 0–1 (higher = better):
```python
def metricx_adjust(score: float) -> float:
    return 1.0 - (score / 25.0)
```

#### LLM Backend (5 models)
| Key | Model ID |
|-----|----------|
| `qwen3-14b` | `Qwen/Qwen3-14B` |
| `qwen3-8b` | `Qwen/Qwen3-8B` |
| `qwen3-4b-instruct-2507` | `Qwen/Qwen3-4B-Instruct-2507` (alias: `qwen3-4b`) |
| `m-prometheus-7b` | `Unbabel/M-Prometheus-7B` |
| `m-prometheus-3b` | `Unbabel/M-Prometheus-3B` |

#### Bicleaner Backend (4 models)
| Key | Model ID | Notes |
|-----|----------|-------|
| `auto` | `auto` | Auto-selects based on language pair |
| `en-xx` | `bitextor/bicleaner-ai-full-en-xx` | English source |
| `es-xx` | `bitextor/bicleaner-ai-full-es-xx` | Spanish source |
| `de-xx` | `bitextor/bicleaner-ai-full-de-xx` | German source |

#### ReMedy Backend (1 model)
| Key | Model ID |
|-----|----------|
| `remedy` | `ShaomuTan/ReMedy-9B-22` |

### Language Support System

File: `models/language_support.py`

Each model has a language support class that tracks which of the 200+ FLORES languages the model was trained on or supports.

**Base class `_BaseLanguageSupport`**:
- Receives a set of language name strings the model supports
- Uses `build_model_language_mapping()` from the dataset module to map FLORES codes to model language names
- Provides `is_code_supported(code)`, `support_status(code)`, `get_full_language_name(code)`

**Concrete classes and their coverage**:

| Class | Model | Languages Supported |
|-------|-------|-------------------|
| `CometLanguages` | COMET family | 102 languages |
| `QwenLanguages` | Qwen3 family | 121 languages (most broad) |
| `MetricX24Languages` | MetricX-24 | 103 languages |
| `PrometheusLanguages` | M-Prometheus | 32 languages  |
| `RemedyLanguages` | ReMedy | 36 languages  |

**Note**: `ReMedy` uses `Gemma-2` as its backbone. I did not find an official list of supported languages, but the model was evaluated on about 140 language pairs. ReMedy uses ISO 3166-1 alpha-2 language codes in its interface, so I filtered the FLORES-200 language codes by keeping the languages that had corresponding alpha-2 codes and treated those as supported by ReMedy. I also patched the ReMedy framework to include these languages, because the original implementation only covered about 38 languages. This was necessary because the language codes are inserted directly into the quality estimation prompt within their framework.

**Special handling for ReMedy**: The `RemedyLanguages` class includes `_iso639_1_from_language()` which maps language names to ISO 639-1 codes using a dedicated `REMEDY_ISO_MAP` dictionary (147 entries). This is needed because the remedy-score CLI expects ISO codes, not full names.

### LLM QE Data Models

File: `models/llm_qe.py`

Pydantic models for structured LLM scoring output (all use `extra="forbid"` to reject unexpected fields):

**`DimScores`** — 7 quality dimensions, each scored 0–10 (StrictInt):
- `accuracy_completeness` — Meaning preservation, no additions/omissions
- `terminology_consistency` — Technical term correctness
- `fluency_coherence` — Natural, grammatical flow
- `style_tone_audience` — Appropriate tone and style
- `locale_formatting` — Numbers, dates, punctuation localization
- `technical_integrity` — Entities, units, code, markup preserved
- `cultural_appropriateness` — Cultural sensitivity

**`QEResult`** — Single-segment result: `dims_0to10` + `overall_0to100` (0–100)

**`QEBatchItem`** — Single item in a batch: `id` + `dims_0to10` + `overall_0to100`

**`QEBatchResult`** — Batch wrapper: `results: list[QEBatchItem]`

---

## Scoring Backends (src/)

### Common Utilities

File: `src/common.py`

- **`ensure_dataset_ready(args, dataset)`** — Sets `args.root` to dataset's default root if not specified
- **`summarize_scores(scores)`** — Computes mean and median. **NaN values are replaced with 0** as a penalty (treats LLM failures as worst-case). Returns `(mean, median)` or `(None, None)` if empty.
- **`sanitize_model_tag(name)`** — Converts model name to a filesystem-safe tag (lowercase, replace non-alphanumeric with hyphens)
- **`load_examples(entry, args, dataset)`** — Loads parallel corpus for a direction and applies `max_rows` limit

### COMET Scorer

File: `src/score_comet.py`

Uses the official COMET library (Unbabel).

**Scoring algorithm**:
1. Load model via `download_model()` + `load_from_checkpoint()`
2. For each direction:
   - Transform examples to COMET format: `{"src": source_text, "mt": target_text}` (reference-free, no `"ref"` key)
   - Call `model.predict(samples, batch_size=8, gpus=1)`
   - Extract `prediction["scores"]` → list of floats (0–1 scale)

**Default args**: `--model wmt22-cometkiwi-da`, `--batch-size 8`, `--gpus 1`

### MetricX-24 Scorer

File: `src/score_metricx.py` + `src/metricx_backend.py`

Uses `MT5ForRegression`.

**Scoring algorithm**:
1. Load tokenizer (`google/mt5-xl`, legacy=False, use_fast=False) and model (`google/metricx-24-hybrid-xl-v2p6`)
2. For each example, construct input tokens:
   ```
   "source: {source}" + " candidate: {candidate}" + " reference: "
   ```
   Note: Even though the reference field is empty, `Metricx-24` HF card specify that it must be included.
3. Truncate to `max_length=1536`, pad batch, forward pass with `torch.no_grad()`
4. Apply score adjustment: `1.0 - (score / 25.0)` to convert from 0–25 (lower=better) to 0–1 (higher=better)

**Default args**: `--model metricx24`, `--batch-size 8`, `--gpus 1`

### ReMedy Scorer

File: `src/score_remedy.py` + `src/remedy_backend.py`

Uses the `remedy-score` CLI tool, which internally uses vLLM.

**Scoring algorithm**:
1. Resolve model (supports both registered names and local directory paths), we are currently using a local patched one (with 140 supported languages).
2. For each direction:
   - Convert FLORES language names to ISO 639-1 codes via `_iso639_1_from_language()`
   - **Fallback**: If a language isn't recognized, defaults to `"en"` with a warning
   - Write parallel files to a temp directory under `{output_base}/.tmp_remedy` (not `/tmp`, which is too small on HPC)
   - File naming: `{src_lang}.src` and `{src_lang}-{tgt_lang}.hyp`
   - Run `remedy-score --model {id} --src_file ... --mt_file ... --no_ref --calibrate --num_gpus {n} --gpu_memory_utilization {pct}`
   - Read calibration scores from output file: `{model_dir}/{src}-{tgt}_calibration_scores.txt`

**Port conflict avoidance**: Each SLURM array task gets a unique `MASTER_PORT` computed from the job ID, task ID, and iteration counter:
```
port = 20000 + ((job_seed + task_seed + iteration) % 20000)
```

**Default args**: `--model remedy`, `--gpus 1`, `--gpu-memory-utilization 0.9`, `--cache-dir` from `HF_HUB_CACHE` env

### Bicleaner Scorer

File: `src/score_bicleaner.py` + `src/bicleaner_backend.py`

Uses the `bicleaner-ai-classify` CLI tool.

**Scoring algorithm**:
1. Resolve model key. The `"auto"` key triggers language-based selection:
   - If source language is Spanish → use `es-xx` model
   - If source language is German → use `de-xx` model
   - Otherwise → use `en-xx` model (default)
   - If source doesn't match, fall back to target language
2. Convert FLORES codes to ISO 639-1 using a mapping dict
3. Write examples to TSV (tab-separated, one pair per line)
4. Wait 5 seconds (filesystem sync on HPC)
5. Run `bicleaner-ai-classify --scol 1 --tcol 2 -s {src} -t {tgt} --disable_hardrules input.tsv output.tsv model_id`
6. Read scores from the last column of the output TSV

**Default args**: `--model auto`

### LLM Scorer

File: `src/score_llm.py` + `src/llm_backend.py`

Uses a **vLLM server** for inference and **async HTTP** (httpx) for concurrent requests.

**Architecture**:
```
┌──────────────┐     HTTP/JSON      ┌──────────────┐
│  score_llm   │ ──────────────────→│  vLLM Server │
│  (httpx      │    concurrency=16  │  (GPU)       │
│   async)     │ ←──────────────────│              │
└──────────────┘                    └──────────────┘
```

**Scoring algorithm**:
1. Split examples into **micro-batches** (default 32 segments per batch)
2. Render a batch prompt for each micro-batch (all segments in one prompt)
3. Dispatch all batches as concurrent async HTTP requests (bounded by semaphore)
4. Each request:
   - POST to `{api_base}/chat/completions` with JSON schema response format
   - Chat template kwargs disable thinking tokens (`enable_thinking=False`)
   - Strip any `<think>...</think>` blocks from response (Qwen3 safety net)
   - Parse JSON response into `QEBatchResult` pydantic model
   - Validate: correct count, sequential IDs, all scores in range
   - Convert `overall_0to100` to 0–1 scale: `score / 100.0`
5. **Retry logic**: Up to `max_retries + 3` attempts per batch
   - Transient HTTP errors (timeouts, connection): exponential backoff `min(2^attempt, 60)` seconds
   - JSON/validation errors: immediate retry
6. **Failure handling**: Failed batches produce NaN scores (penalized as 0 in summary)

**Default args**: `--model Qwen/Qwen3-14B`, `--batch-size 8`, `--temperature 0.0`, `--max-tokens 4096`, `--max-retries 5`, `--micro-batch-size 32`, `--concurrency 16`, `--api-base http://127.0.0.1:8000/v1`

---

## Prompts Module

### Single-Segment Prompt

File: `prompts/llm_prompt.py`

Used by the legacy single-segment scoring path. Each source/target pair gets its own prompt:

```
Source language:{source_lang} , Source text:
```{source_seg}```

Target language: {target_lang} , Machine Translation text:
```{target_seg}```

Task: Reference-free MT quality scoring for this single segment.
Score each dimension as an integer 0..10 (higher=better), then overall 0..100.

Dimensions:
1) accuracy_completeness (meaning preserved, no add/omit)
2) terminology_consistency
3) fluency_coherence
4) style_tone_audience
5) locale_formatting
6) technical_integrity (entities/units/code/markup preserved)
7) cultural_appropriateness

Output: ONLY valid JSON, exactly this shape:
{
  "dims_0to10": {
    "accuracy_completeness": 0-10,
    "terminology_consistency": 0-10,
    ...
  },
  "overall_0to100": 0-100
}
```

### Batch Prompt

File: `prompts/llm_prompt_batch.py`

The **production prompt** used by `score_llm_batched()`. Evaluates multiple segments in a single LLM call for efficiency.

**Structure**:
1. System instruction: "You are a professional translation quality evaluator."
2. Item block: Each segment is wrapped with `--- Item {id} ---` header containing source/target language and text
3. Task description: Same 7 dimensions as single-segment
4. JSON schema: Expects `{"results": [{"id": N, "dims_0to10": {...}, "overall_0to100": N}, ...]}`
5. Instruction to return exactly `batch_size` items ordered by ID

**`render_batch_prompt(examples, src_lang, tgt_lang, *, start_id=0)`**:
- Takes a list of `Example` dicts with `"src"` and `"tgt"` keys
- Assigns sequential IDs starting from `start_id`
- Returns the complete prompt string

There is also an older prompt version in `prompts/llm_prompt_old.py` with more verbose instructions (not used in production).

---

## Utils Module

### CLI Arguments

File: `utils/args.py`

`add_common_scoring_args(parser)` adds these arguments to all scorers:

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--dataset` | str | `"flores200"` | No | Dataset identifier |
| `--root` | str | None | No | Dataset root directory (falls back to adapter default) |
| `--output-base` | str | — | **Yes** | Base output directory |
| `--batch-size` | int | varies | No | Batch size for prediction |
| `--gpus` | int | varies | No | Number of GPUs |
| `--manifest` | str | — | **Yes** | TSV manifest file path |
| `--shard-id` | int | None | No | Worker shard ID (auto-detected from SLURM) |
| `--num-shards` | int | None | No | Total shard count (auto-detected from SLURM) |
| `--max-directions-per-part` | int | 25 | No | Rotate output file after N directions |
| `--target-part-bytes` | int | 67108864 (64 MiB) | No | Target output file size before rotation |
| `--max-rows` | int | None | No | Cap rows per direction (debugging) |

Validation is in `utils/cli.py` — checks all required args are present and numeric constraints are met.

### Runner (Pipeline Orchestrator)

File: `utils/runner.py`

The `run_scoring()` function is the **core pipeline loop** used by every scorer.

**Algorithm**:

1. **Resolve shard context**: Determine this worker's `shard_id` and `num_shards`
   - First checks CLI args (`--shard-id`, `--num-shards`)
   - Falls back to SLURM env vars (`SLURM_ARRAY_TASK_ID`, `SLURM_ARRAY_TASK_COUNT`)
   - Validates: `0 <= shard_id < num_shards`

2. **Generate run ID**: `{UTC timestamp}-pid{PID}` (e.g., `20240315T143022Z-pid12345`)

3. **Main loop** over manifest directions:
   ```
   for each ManifestEntry in directions:
       compute direction_key = "src_lang->tgt_lang"
       compute entry_shard_id (from manifest or hash)

       if entry_shard_id != my_shard_id:
           skip (not my work)

       if direction_key in writer.committed_direction_keys:
           skip (already checkpointed)

       frame = score_entry(entry)  # scorer-specific callback
       writer.add_direction(frame, direction_key)
   ```

4. **Cleanup**: Close all writers in `finally` block

The `collect_directions()` function reads the manifest and validates splits against the dataset's allowed values.

### Stage Writer & Checkpointing

File: `utils/stage_writer.py`

**`ShardStageWriter`** manages output files with automatic rotation and crash-safe checkpointing.

**Output directory structure**:
```
{output_base}/
  dataset={dataset}/
    model={model_tag}/
      split={split}/
        shard={shard_id:03d}/
          part-{run_id}-000000.jsonl
          part-{run_id}-000001.jsonl
          checkpoint.jsonl
```

**File rotation triggers** (whichever comes first):
- Part file exceeds `target_part_bytes` (default 64 MiB)
- Part file contains `max_directions_per_part` directions (default 25)

**Checkpointing mechanism**:
1. After writing all rows (summary + details) for a direction to the part file:
   - Flush the file buffer
   - `fsync()` to disk (ensures durability)
   - Append the summary record to `checkpoint.jsonl`
   - Flush + fsync the checkpoint file
   - Add direction_key to in-memory `committed_direction_keys` set
2. On restart, `_load_checkpoint()` reads `checkpoint.jsonl` and rebuilds the set of completed directions

This means: if the process crashes between writing data and checkpointing, the data will be re-scored on restart (safe, idempotent).

**Value normalization** (`_normalize_value()`):
- Extracts numpy scalars via `.item()`
- Converts NaN/Inf to None
- Ensures JSON-serializable output

### Output Format

File: `utils/io.py` + `utils/frames.py`

Each direction produces a pandas DataFrame with two row types, serialized as **JSONL** (one JSON object per line):

```json
{"row_type":"summary","model_name":"Unbabel/XCOMET-XL","dataset":"flores200","split":"dev","src_lang":"ace_Arab","tgt_lang":"ace_Latn","src_lang_seen":false,"tgt_lang_seen":false,"mean":0.25574539594750184,"median":0.22343234717845917,"score":null,"src_txt":null,"tgt_txt":null,"direction_key":"ace_Arab->ace_Latn","shard_id":0}
{"row_type":"detail","model_name":"Qwen/Qwen3-14B","dataset":"flores200","split":"dev","src_lang":"ace_Arab","tgt_lang":"ace_Latn","src_lang_seen":null,"tgt_lang_seen":null,"mean":null,"median":null,"score":0.4,"src_txt":"txt","tgt_txt":"txt", "direction_key":"ace_Arab->ace_Latn","shard_id":0}
```

There are two types of rows, `summary`(mean, median and seen/unseen) and `detail`(all others).


**Column order** (OUTPUT_COLUMNS):
`row_type`, `model_name`, `dataset`, `split`, `src_lang`, `tgt_lang`, `src_lang_seen`, `tgt_lang_seen`, `mean`, `median`, `score`, `src_txt`, `tgt_txt`

The runner also adds `direction_key` and `shard_id` columns before writing.

### Hashing & Sharding

File: `utils/hashing.py`

- **`stable_hash_int(text)`** — BLAKE2b hash (16-byte digest) → unsigned big-endian integer. Deterministic and stable across runs.
- **`direction_key(src, tgt)`** — Returns `"{src}->{tgt}"` string
- **`compute_shard_id(key, num_shards)`** — Returns `stable_hash_int(key) % num_shards`

This ensures each language pair direction is always assigned to the same shard, regardless of worker ordering.

### Logging

File: `utils/logger.py`

Centralized logging with stdout/stderr splitting and optional file rotation.

**Logger name**: `"quality_estimation"`

**Output routing**:
- DEBUG/INFO/WARNING → stdout
- ERROR → stderr
- Optionally: all levels to a rotating log file

**Environment variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_FILE` | None | Path to main log file |
| `LOG_ERR_FILE` | None | Path to error-only log file |
| `LOG_MAX_BYTES` | 10 MB | Max file size before rotation |
| `LOG_BACKUP_COUNT` | 5 | Number of old log files to keep |

**Format**: `%(asctime)s | %(levelname)s | %(module)s | %(message)s`

Noisy loggers (`urllib3`, `asyncio`) are suppressed to WARNING level.

---

## How to Run

### Step 1: Create a Manifest

Generate a manifest TSV that lists all language-pair directions with shard assignments:

```bash
# Single shard (no parallelism)
python -m dataset.make_manifest \
    --dataset flores200 \
    --split all \
    --num-shards 1 \
    --out flores200_directions.tsv

# Multi-shard examples for different models:
python -m dataset.make_manifest --dataset flores200 --split all --num-shards 15  --out flores200_directions_bicleaner.tsv

python -m dataset.make_manifest --dataset flores200 --split all --num-shards 35  --out flores200_directions_metricx.tsv

python -m dataset.make_manifest --dataset flores200 --split all --num-shards 57  --out flores200_directions_xcomet.tsv

python -m dataset.make_manifest --dataset flores200 --split all --num-shards 70  --out flores200_directions_comet23.tsv
```

### Step 2: Submit SLURM Jobs

#### Mahti (NVIDIA A100)

```bash
# Bicleaner (15 shards)
sbatch --array=0-14 \
    --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 \
    scripts/run_slurm.sh --manifest flores200_directions_bicleaner.tsv --model bicleaner

# MetricX-24 (35 shards)
sbatch --array=0-34 \
    --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 \
    scripts/run_slurm.sh --manifest flores200_directions_metricx.tsv --model metricx24

# XCOMET (57 shards)
sbatch --array=0-56 \
    --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 \
    scripts/run_slurm.sh --manifest flores200_directions_xcomet.tsv --model xcomet

# COMET-23 (70 shards)
sbatch --array=0-69 \
    --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 \
    scripts/run_slurm.sh --manifest flores200_directions_comet23.tsv --model comet23

# ReMedy (104 shards)
sbatch --array=0-103 \
    --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 \
    scripts/run_slurm.sh --manifest flores200_directions_remedy.tsv --model remedy
```

#### LUMI (AMD MI250X)

```bash
# M-Prometheus-7B (180 shards)
sbatch --array=0-179 \
    --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 \
    scripts/run_slurm_lumi.sh --manifest flores200_directions_prometheus.tsv --model m-prometheus-7b

# Qwen3-4B (190 shards)
sbatch --array=0-189 \
    --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 \
    scripts/run_slurm_lumi.sh --manifest flores200_directions_qwen4b.tsv --model qwen3-4b
```

#### Rerunning Failed Shards

Specify only the failed shard IDs in the `--array` flag:

```bash
sbatch --array=81,128 \
    --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 \
    scripts/run_slurm_lumi.sh --manifest flores200_directions_qwen4b.tsv --model qwen3-4b --num-shards 190
```

### Step 3: Manual / Local Runs

Run a single shard directly (useful for debugging):

```bash
python -m src.score_comet \
    --dataset flores200 \
    --root /scratch/project_2008161/downstream_benchmarks/flores200 \
    --manifest flores200_directions.tsv \
    --output-base /scratch/project_2008161/QE_flores200_scores \
    --model xcomet \
    --num-shards 8 \
    --shard-id 0
```

### SLURM Defaults (Mahti vs LUMI)

| Parameter | Mahti (`run_slurm.sh`) | LUMI (`run_slurm_lumi.sh`) |
|-----------|------------------------|---------------------------|
| **Account** | `project_2008161` | `project_462001050` |
| **Partition** | `gpusmall` | `small-g` |
| **GPU** | 1x NVIDIA A100 | 1x AMD MI250X |
| **CPUs per task** | 7 | 7 |
| **Memory** | 60 GB | 60 GB |
| **Time limit** | 24 hours | 72 hours |
| **Default model** | `wmt22-cometkiwi-da` | `wmt22-cometkiwi-da` |
| **Batch size** | 8 | 8 |
| **vLLM dtype** | `bfloat16` | `bfloat16` |
| **vLLM GPU utilization** | 0.90 | 0.90 |
| **vLLM max-num-batched-tokens** | 4096 | 8192 |
| **LLM concurrency** | 16 | 32 |
| **LLM micro-batch size** | 32 | 16 |
| **Temperature** | 0.0 | 0.0 |
| **Max tokens** | 8192 | 8192 |
| **Max retries** | 5 | 5 |
| **Execution method** | Direct `python` / `venv` | Singularity containers |
| **Module loads** | `pytorch` | `LUMI`, `partition/G`, `rocm` |
| **vLLM port range** | 40000–59999 | 40000–59999 |
| **MASTER_PORT range** | (not set) | 20000–39999 |
| **Retry attempts** | 3 (backoff: 30s, 60s, 120s) | 3 (backoff: 30s, 60s, 120s) |

### Environment Variables

These can be set before `sbatch` or exported via `--export`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace authentication token |
| `ROOT` / `DATA_DIR` | Platform-specific | Path to FLORES-200 data |
| `OUTPUT_BASE` / `OUTPUT_DIR` | Platform-specific | Base output directory |
| `BATCH_SIZE` | 8 | Model batch size |
| `GPUS` | 1 | Number of GPUs |
| `MAX_DIRECTIONS_PER_PART` | 25 | Directions per output part file |
| `TARGET_PART_BYTES` | 67108864 (64 MiB) | Target part file size |
| `MODEL` | `wmt22-cometkiwi-da` | Model key or ID |
| `VLLM_HOST` | `127.0.0.1` | vLLM server host |
| `VLLM_PORT` | computed | vLLM server port |
| `VLLM_DTYPE` | `bfloat16` | vLLM data type |
| `VLLM_GPU_UTIL` | 0.90 | vLLM GPU memory utilization |
| `TEMPERATURE` | 0.0 | LLM sampling temperature |
| `MAX_TOKENS` | 8192 | LLM max output tokens |
| `MAX_RETRIES` | 5 | LLM retry count |
| `CONCURRENCY` | 16 (Mahti) / 32 (LUMI) | Max concurrent LLM requests |
| `MICRO_BATCH_SIZE` | 32 (Mahti) / 16 (LUMI) | Segments per LLM request |

# Stand alone modules/features

### check done, model name as directory name
```bash
python -m stand_alone_modules.check_done.check_shards --tsv flores200_directions_bicleaner.tsv

python -m stand_alone_modules.check_done.check_shards --tsv flores200_directions_qwen4b.tsv --model qwen3-4b-instruct-2507 --path /scratch/project_462001050/QE_flores200_scores/dataset=flores200

python -m stand_alone_modules.check_done.check_shards --tsv flores200_directions_qwen14b.tsv --model qwen3-14b --path /scratch/project_462001050/QE_flores200_scores/dataset=flores200

python -m stand_alone_modules.check_done.check_shards --tsv flores200_directions_remedy.tsv --model shaomutan_remedy-9b-22 --path /scratch/project_462001050/QE_flores200_scores/dataset=flores200
```


### check deduplication and remove if they exist (scan/apply)
```bash
python -m stand_alone_modules.dedup scan --dataset-path /scratch/project_462001050/QE_flores200_scores/dataset=flores200 --output .

python -m stand_alone_modules.dedup apply --plan /scratch/.../dataset=flores200/dedup_plan.json
```

### patch results seen/unseen + null scores
```bash
# Step 1: patch (creates *-patched.jsonl files alongside originals)
python -m stand_alone_modules.patch_results patch --model-path /scratch/project_462001050/QE_flores200_scores/dataset=flores200/model=m-prometheus-7b

# Step 2: replace (swaps patched files into original names, deletes old)
python -m stand_alone_modules.patch_results replace --model-path /scratch/project_462001050/QE_flores200_scores/dataset=flores200/model=m-prometheus-7b
```

### compcation module: compact jsonl files to parquet files
```bash
singularity run $SIF python -m stand_alone_modules.compact
  --output-base /scratch/project_462001050/QE_flores200_scores
  --dataset flores200
  --name raw_scores
  --target-part-bytes 671088640
  --workers 9
```