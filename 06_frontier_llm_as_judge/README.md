# 06 — Frontier LLM as judge

Score every row of a sharded parallel corpus with a frontier LLM (Azure-hosted
DeepSeek-V4-Flash by default) and append an `llm_judge_score` column
(the model's `overall_0to100` field, normalized to a 0.0–1.0 float).

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
per-task stats CSV records `rows_total / rows_scored / rows_failed /
mean_score / elapsed_sec` for every processed language pair.

## Credentials

`AZURE_API_KEY` must live in `.env` next to `llm_judge.py`:

```
AZURE_API_KEY=<your-azure-key>
```

The script also accepts `AZURE_OPENAI_API_KEY`.

## Quick start (single machine)

```bash
module use /appl/local/csc/modulefiles/ && module load pytorch/2.5

python3 llm_judge.py \
    --dataset_dir /scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage3 \
    --out_dir     /scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage3-LLMScored \
    --batch_size 10 \
    --concurrency 32 \
    --tpm_limit 900000 \
    --rpm_limit 900 \
    --skip_existing
```

## SLURM array

`submit_llm_judge.sh` splits the language pairs across N tasks and divides
the global TPM/RPM budget evenly across them:

```bash
bash submit_llm_judge.sh \
    --tasks 4 \
    --dataset-dir /scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage3 \
    --out-dir     /scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage3-LLMScored \
    --batch-size 10 --concurrency 32 \
    --tpm-total 900000 --rpm-total 900
```

Logs land in `../../logs/llm_judge/`.

## Rate limiting

A sliding-60s token bucket tracks both tokens (prompt + completion) and
request counts. Token counts are first added by an estimate
(`len(prompt)/3 + 60 + 20 * batch_size`); after each response we reconcile
the bucket with the real `usage.prompt_tokens + usage.completion_tokens`
value reported by the server.

The defaults (`tpm_limit=900_000`, `rpm_limit=900`) leave ~10% headroom
under the documented 1M TPM / 1K RPM ceiling. When running multiple SLURM
tasks the submitter divides those budgets across tasks so the combined
traffic stays under the global ceiling.

## Prompt

Each request scores `batch_size` segments at once. The prompt:

* States source/target language (passes through the FLORES-200 codes from
  the directory name, e.g. `abk_Cyrl`, `eng_Latn`).
* Asks for a single `overall_0to100` score per segment, while listing the
  quality aspects the model should weigh together.
* Demands a strict JSON shape; we strip code fences and tolerate stray
  prose before/after the JSON object.

If a batch fails after `--max_retries` attempts, the corresponding rows get
`null` for `llm_judge_score` and the shard continues.

Content-filter blocks (Azure "Responsible AI", HTTP 400 with
`finish_reason=content_filter`) are deterministic, so they are **not** retried.
Instead, when a multi-row batch is blocked, each row is re-scored individually
so only the truly offending segment(s) fail and the rest still get a score.

## Files

* `llm_judge.py` — main async scorer
* `run_llm_judge.sh` — SLURM worker that loads `pytorch/2.5` and invokes the script
* `submit_llm_judge.sh` — SLURM submitter / budget splitter
* `requirements.txt` — Python deps (already present in `pytorch/2.5`)
* `.env` — `AZURE_API_KEY=...`
* `stats/llm_judge_stats.csv[.chunkNNNN.csv]` — per-pair stats
