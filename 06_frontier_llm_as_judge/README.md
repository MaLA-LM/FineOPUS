# 06 — Frontier LLM as judge

Score every row of a sharded parallel corpus with a frontier LLM (Azure-hosted
DeepSeek-V4-Flash by default) and append an `llm_judge_score` column
(the model's `overall_0to100` field, normalized to a 0.0–1.0 float).

The pipeline targets **FineOPUS-Filtered-Stage4** by default; paths are
overridable via CLI flags.

## Input layout

```
<dataset_dir>/
  abk_Cyrl-eng_Latn/
    abk_Cyrl-eng_Latn_shard_0.parquet
    abk_Cyrl-eng_Latn_shard_1.parquet
  abk_Cyrl-fra_Latn/
    ...
```

Each parquet shard must contain at least the columns `source_text` and
`target_text`. All other columns are preserved verbatim in the output.

## Output layout

Mirrors the input. Each output shard has the same schema as the input plus
one column:

* `llm_judge_score` — `float64` in `[0.0, 1.0]`, the LLM's `overall_0to100`
  score divided by 100 (and clipped). `null` if the LLM call (or JSON
  parsing) failed for that row after retries.

A `_DONE` sentinel is written into each per-pair output directory, and a
per-task stats CSV records one row per processed language pair (see below).

### Large-shard checkpointing

Shards with more than `--checkpoint_every_rows` rows (default: 1,000,000) are
scored in segments. Each segment is written to a part file under a hidden
directory:

```
<out_dir>/<lang_pair>/.<shard_stem>.parts/<shard_stem>.part.00000.parquet
```

When all parts for a shard finish, they are merged into the final
`<shard_stem>.parquet`. Existing part files are reused on resume after an
interrupt; re-run with `--skip_existing` to pick up where you left off.

## Resource-class filtering

Language pairs can be selected by **directional resource-class combo**
(`src_class-tgt_class`), using the Joshi et al. 6-class taxonomy
(`Joshi-et-al-6-classes.json`).

The precomputed mapping lives in `fineopus_pair_class_combinations.json`:

```json
{ "0-0": ["awa_Deva-ory_Orya", "mag_Deva-ory_Orya"], "0-1": [ ... ], ... }
```

Pass the combos you want to score:

```bash
python3 llm_judge.py \
    --dataset_dir ... \
    --out_dir ... \
    --class_combos "0-0,0-1" \
    --pair_combos_json ./fineopus_pair_class_combinations.json
```

When `--class_combos` is empty (default), every pair directory under
`--dataset_dir` is scored. Same-language pairs (`src == tgt`) are skipped
unless `--include_same_lang` is set.

### Balancing SLURM tasks by row count

`split_pair_combos.py` splits one class combo into N balanced parts (by total
row count from `../tools/parquet_rows_per_pair_stats/fineopus-filtered-stage4-row-counts.xlsx`),
skipping pairs already recorded in `stats/llm_judge_stats.csv`:

```bash
python3 split_pair_combos.py --class-combo 3-1 --parts 14
# writes fineopus_pair_class_combinations_3-1_part1.json, _part2.json, ...
```

Point `--pair_combos_json` at a part file when submitting a subset of pairs
per array task.

## Credentials

API keys live in `.env` next to `llm_judge.py` as `AZURE_API_KEY_1` …
`AZURE_API_KEY_11` (plus optional legacy `AZURE_API_KEY`). The script also
accepts `AZURE_OPENAI_API_KEY` if referenced via `--api_key_env`.

`azure_api_key_registry.sh` maps each key env var to its Azure endpoint,
deployment, and per-key TPM/RPM limits. `submit_llm_judge.sh` sources this
registry automatically from `--api-key-env`, so you usually only pass the key
name:

```bash
bash submit_llm_judge.sh --api-key-env AZURE_API_KEY_5 --tasks 4 --dry-run
```

Override endpoint, deployment, or global rate limits explicitly with
`--endpoint` / `--deployment` / `--tpm-total` / `--rpm-total` when needed.

Per-key TPM/RPM ceilings in the registry (e.g. 250k/250 vs 1M/1k) are used
as defaults; the submitter divides the chosen total budget evenly across
array tasks.

## Quick start (single machine)

```bash
module use /appl/local/csc/modulefiles/ && module load pytorch/2.5

python3 llm_judge.py \
    --dataset_dir /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage4 \
    --out_dir     /scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage4-LLMScored \
    --endpoint    https://fineopus-step6.services.ai.azure.com/openai/v1/ \
    --batch_size 10 \
    --concurrency 32 \
    --tpm_limit 225000 \
    --rpm_limit 225 \
    --skip_existing
```

Use `--max_rows N` to cap total scored rows across all pairs (test mode).
Use `--class_combos` to restrict which pairs are processed.

## SLURM array

`submit_llm_judge.sh` splits language pairs across N tasks (round-robin by
pair index) and divides the global TPM/RPM budget evenly across them:

```bash
bash submit_llm_judge.sh \
    --tasks 4 \
    --api-key-env AZURE_API_KEY_2 \
    --dataset-dir /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage4 \
    --out-dir     /scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage4-LLMScored \
    --class-combos "3-1" \
    --batch-size 10 --concurrency 32
```

When `--tpm-total` / `--rpm-total` are omitted, values come from the registry
for the chosen `--api-key-env`. Each array task runs `run_llm_judge.sh`, which
passes `--n_chunks` / `--chunk_id` from `SLURM_ARRAY_TASK_COUNT` /
`SLURM_ARRAY_TASK_ID`.

Logs land in `../logs/fineopus-llm-judge/`. Stats for multi-task runs are
written per chunk as `stats/llm_judge_stats.csv.chunk0000.csv`, etc.

### Resume / skip logic

With `--skip_existing` (enabled by default in the SLURM worker):

* Skip a language pair if its `_DONE` sentinel exists or it is already in the
  stats CSV.
* Skip individual output shards whose parquet file already exists.
* Reuse existing checkpoint part files when resuming a large shard.

## Rate limiting

A sliding-60s token bucket tracks both tokens (prompt + completion) and
request counts. Token counts are first reserved using an estimate
(`len(prompt)/3 + 60 + 20 * batch_size`); after each response the bucket is
reconciled with the real `usage.prompt_tokens + usage.completion_tokens`
from the server.

A **token-cost self-calibrator** (`TokenCalibration`) learns a multiplicative
correction from the first few responses (`--calibration_warmup`, default 5)
so later estimates track actual usage more closely.

When running multiple SLURM tasks the submitter divides the global TPM/RPM
budget across tasks so combined traffic stays under the per-key ceiling.

## Prompt

Each request scores `batch_size` segments at once. The prompt:

* States source/target language (passes through the FLORES-200 codes from
  the directory name, e.g. `abk_Cyrl`, `eng_Latn`).
* Asks for a single `overall_0to100` score per segment, while listing the
  quality aspects the model should weigh together.
* Demands a strict JSON shape; we strip code fences and tolerate stray
  prose before/after the JSON object.

Source and target text are truncated to `--max_chars_per_field` (default
2000) before sending.

If a batch fails after `--max_retries` attempts, the corresponding rows get
`null` for `llm_judge_score` and the shard continues.

Content-filter blocks (Azure "Responsible AI", HTTP 400 with
`finish_reason=content_filter`) are deterministic, so they are **not** retried.
Instead, when a multi-row batch is blocked, each row is re-scored individually
so only the truly offending segment(s) fail and the rest still get a score.

## Post-run analysis

`analyze_qe_llm_scores.py` streams `qe_score` and `llm_judge_score` from
scored parquets and prints distribution summaries (and optional plots):

```bash
module load pytorch/2.5

python3 analyze_qe_llm_scores.py \
    --scored_dir /scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage4-LLMScored \
    --sample_rows 500000 \
    --plot_dir stats/score_dist_plots
```

## Stats CSV

Default path: `stats/llm_judge_stats.csv` (or `.chunkNNNN.csv` per array
task). Columns:

| Column | Description |
|--------|-------------|
| `lang_pair` | e.g. `awa_Deva-ory_Orya` |
| `source_lang`, `target_lang` | Parsed from directory name |
| `n_shards_in`, `n_shards_out` | Input / output shard counts |
| `rows_total`, `rows_scored`, `rows_failed` | Row counts |
| `mean_score` | Mean of non-null `llm_judge_score` values |
| `elapsed_sec` | Wall time for the pair |

## Files

| File | Role |
|------|------|
| `llm_judge.py` | Main async scorer |
| `run_llm_judge.sh` | SLURM worker (loads `pytorch/2.5`, invokes the script) |
| `submit_llm_judge.sh` | SLURM submitter / budget splitter |
| `azure_api_key_registry.sh` | Key → endpoint / deployment / TPM / RPM mapping |
| `split_pair_combos.py` | Split one class combo into balanced SLURM parts |
| `analyze_qe_llm_scores.py` | Post-hoc QE vs LLM score analysis |
| `fineopus_pair_class_combinations.json` | Full pair → class-combo mapping |
| `fineopus_pair_class_combinations.csv` | Same mapping in CSV form |
| `Joshi-et-al-6-classes.json` | Language → resource class (0–5) |
| `.env` | `AZURE_API_KEY_1=...`, etc. (not committed) |
| `stats/llm_judge_stats.csv[.chunkNNNN.csv]` | Per-pair run stats |

Python deps (`pyarrow`, `openai`, `python-dotenv`, etc.) are provided by the
`pytorch/2.5` module on CSC.
