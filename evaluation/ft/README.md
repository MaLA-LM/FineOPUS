# Translation evaluation for fine-tuned models

By default, evaluate all eight models: FineOPUS Stages 1–6, MaLA, and NLLB.
Use the existing `../mt_eval.py` **3-shot English ↔ target language** prompts,
data splits, and metrics: BLEU (flores200 tokenizer), chrF++, and COMET
(`Unbabel/wmt22-comet-da`). As in the original workflow, the first three NTREX
test pairs serve as demonstrations and are excluded from scoring.

## Language coverage

The language list comes from a [pinned version of the GlotEval WikiANN CSV](https://github.com/MaLA-LM/GlotEval/blob/74183ec1f0c8b18cdb36af7057580142dc3061e0/benchmark_data_loader/data_langid_alignment/alignment_reports/wikiann_langs_script_report.csv).
This directory includes the original snapshot. `language_policy.json` records
label corrections and aliases for each dataset.

The CSV contains 176 rows: six have unresolved language codes; `en` and `simple`
serve as English anchors; and `be` and `be-x-old` normalize to the same code.
This leaves **167 distinct non-English language/script combinations**.

| Dataset | Exact matches | Covered through mapping | Total covered / 167 | Coverage | Unique benchmark codes | Directions per model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLORES-200 | 103 | 17 | 120 | 71.86% | 119 | 238 |
| NTREX-128 | 88 | 3 | 91 | 54.49% | 91 | 182 |
| BOUQuET_Sentence | 94 | 19 | 113 | 67.66% | 112 | 224 |

Evaluate each benchmark code once. For example, both `als_Latn` and `sqi_Latn`
map to FLORES `als_Latn`: coverage counts both source codes, while evaluation
creates only one pair of translation directions.
Across all three datasets, there are **644 directions per model and 5,152
directions across eight models**.

Full details are available in [reports/coverage.tsv](reports/coverage.tsv) and
[reports/coverage.json](reports/coverage.json). The TSV preserves every source
row with one of these statuses: `exact`, `mapped`, `unavailable`, `unresolved`,
or `english_anchor`.

Coverage includes the explicit macrolanguage-to-variety mappings in the policy
file, such as `nor→nob`, `kur→kmr`, and `que→quy`. These mappings evaluate the
benchmark variety; they do not imply coverage of every variety in a
macrolanguage. Exact dataset codes take precedence. BOUQuET Levantine and
Egyptian Arabic are not counted as Standard Arabic coverage.

Entries marked with `?` in the source retain a `source_uncertain` flag in the
report. Unless explicitly corrected in the policy file, only the trailing `?`
is removed; the reported language and script are retained. For example, `zh`
remains `zho_Hant`, without inferring additional Simplified Chinese coverage.
`ja` is corrected to `jpn_Jpan`, and `simple` to English. Classical Chinese,
Min Nan, and Cantonese use `lzh_Hant`, `nan_Latn`, and `yue_Hant`, respectively,
to avoid counting them as modern Chinese.
The labels `bat-smg`, `bh`, `eml`, `fiu-vro`, `map-bms`, and `sh` remain
unresolved rather than receiving guessed codes.

Regenerate coverage reports without a GPU environment:

```bash
/usr/bin/python3.11 evaluation/ft/workflow.py coverage
```

## Submission and results

Run from the FineOPUS repository root:

```bash
# Generate coverage reports and a task manifest without submitting jobs
bash evaluation/ft/submit.sh --dry-run

# Smoke test one model and language across all three datasets in one dev-g job, including COMET
bash evaluation/ft/submit.sh \
  --model-glob 'FT-FineOPUS-Filtered-Stage1-sample5k' \
  --languages deu_Latn --limit 3 --partition dev-g --job-time 00:30:00

# Evaluate all models: eight independent small-g jobs, one per model
bash evaluation/ft/submit.sh

# Optionally wait for a successful smoke job before starting the eight jobs
bash evaluation/ft/submit.sh --dependency afterok:SMOKE_JOB_ID
```

Full evaluation submits **eight independent jobs, one per model**, without
sharding or a job array. Each job loads its model once, evaluates the covered
languages for each of the three datasets, and then computes COMET scores.
Each job requests one GPU, eight CPUs, 64 GB of memory, and up to three days
(adjustable with `--job-time`). Jobs use the existing `evaluation/eval_env`
environment and Slurm account `project_465002530`.
Repeat `--model-glob` to select a subset of models.

Smoke tests using `--limit` default to `dev-g` with a 30-minute time limit;
full evaluation defaults to `small-g`. Override the partition with `--partition`.
According to the [LUMI partition documentation](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/partitions/),
`dev-g` permits only one queued or running job per user and per project.
Select one model for a smoke test; the script rejects attempts to submit
multiple models to `dev-g` at once.

Full results are stored in `evaluation/ft/results/3shot/<config_fingerprint>/`;
smoke results in `evaluation/ft/results/smoke_3shot_limit3/<config_fingerprint>/`;
and logs in `evaluation/ft/logs/`.
The configuration fingerprint includes evaluation parameters, the language
policy, and model file signatures to keep results from different configurations
in separate aggregate directories. FT weights are loaded without conversion
or modification. Existing from_scratch scripts and `evaluation/results/`
remain unchanged. The script rejects output paths that overlap the original
results directory or model directories.

Each manifest directory contains `coverage.tsv/json`, `tasks.json`, and,
after submission, `tasks.jobs.tsv`. Each model's `final/` directory contains
`task.json`, per-direction results for each dataset, `summary.tsv/json`, and
success markers. The manifest records a separate language list for each dataset
to preserve its intended evaluation scope.
Rerunning with the same parameters skips successful models and reuses completed
direction results for unfinished models. Use `--overwrite` to recompute results.
Workers reject an old plan if model files have changed since planning.

The last successful model job generates `all_results.tsv` and `all_results.json`
within its own process; **no additional aggregation job is submitted**.
Aggregation waits for all model tasks in the current submission to complete
and uses a file lock to prevent duplicate concurrent aggregation.
To aggregate manually:

```bash
bash evaluation/ft/aggregate.sh /path/to/task_manifests/RUN/tasks.json
```

Aggregation reads only the configuration directory associated with the manifest.
Smoke results carry a smoke status and are excluded from full evaluation
aggregates. Avoid using the entire `evaluation/ft/results/` directory as an
aggregation root, since it may contain multiple configurations.

## Excel export

Export a workbook using the tables, ranking rules, and styling from
`evaluation/export_mt_results_excel.py`:

```bash
uv run --no-project evaluation/ft/export_results_excel.py \
  evaluation/ft/results/3shot/<config_fingerprint>/all_results.json
```

The output is `ft_mt_results.xlsx` beside the source JSON. Use `--output FILE.xlsx`
to choose another destination. The workbook contains the same 12 worksheets as
the shared exporter, including raw data, coverage, language and dataset summaries,
model rankings, and separate English-to-X and X-to-English averages.
Final HF checkpoints retain their `final` labels; no training iteration is inferred.

## Validation

```bash
/usr/bin/python3.11 -m unittest discover -s evaluation/ft -p 'test_*.py' -v
bash -n evaluation/ft/submit.sh evaluation/ft/worker.sh evaluation/ft/aggregate.sh
```
