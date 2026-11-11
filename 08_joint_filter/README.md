# Stage6 Joint Filtering

Reuse LID, heuristics, similarity, QE, and optional LLM scores from Stage5. The workflow follows
`06_frontier_llm_as_judge`: **collect statistics and generate multiple threshold profiles first, then apply a selected profile separately**.
No scoring models are called, no new models are trained, and no training experiments are submitted.

## Quick Start

Enter this directory from the repository root. The submission scripts below can be called from any directory;
relative paths are resolved against the calling directory.
Python defaults to the repository's `.venv/bin/python` when available. Use `--python /absolute/path/python`
to specify an interpreter available on the compute nodes. See `requirements.txt` for dependencies.
All four phases are submitted to SLURM through `submit_*.sh`; only submission commands run on the login node.
Set the SLURM account, input data, token table, tokenizer, and output directory for your environment.
Pass `--account` and the relevant path options explicitly rather than relying on environment-specific script defaults.
The storage directory and SLURM compute account are configured independently.
On submission, stdout contains only the job ID, while the full sbatch command is written to stderr,
so the job ID can be captured directly to set dependencies.
The submitter explicitly passes the absolute path of the Python entry point to the worker, avoiding changes
in script location caused by the Slurm spool directory.
Submit through `submit_*.sh`; the worker's two positional arguments are the Python interpreter and the Python entry point, in that order.

```bash
cd 08_joint_filter

# Replace these placeholders with values for your environment.
ACCOUNT="your-slurm-account"
INPUT_DIR="/path/to/stage5"
OUTPUT_DIR="/path/to/stage6"
TOKEN_STATS_CSV="/path/to/token-stats.csv"
TOKENIZER_JSON="/path/to/tokenizer.json"
STATS_DIR="$PWD/thresholds/stats/stage5_v1"

# 1. Submit statistics collection only: fit transformations on a sample, then scan all data to save per-language-pair score histograms.
COLLECT_JOB=$(bash thresholds/submit_collect_data_stats.sh \
  --tasks 64 \
  --account "$ACCOUNT" \
  --stats-dir "$STATS_DIR" \
  --scored-dir "$INPUT_DIR" \
  --token-stats-csv "$TOKEN_STATS_CSV" \
  --sample-rows 50000 \
  --token-mode estimated)
```

The token table must contain per-language-pair row counts matching the Stage5 input.
The examples below reuse the variables defined above; choose a tokenizer compatible with your token statistics.
Before the first run, append `--dry-run` to the submission command to inspect the full sbatch command.

Submit threshold computation with an `afterok` dependency to wait for the entire statistics array to succeed.
This step reads only the small statistics files and does not scan the raw corpus:

```bash
THRESHOLDS_JOB=$(bash thresholds/submit_compute_thresholds.sh \
  --account "$ACCOUNT" \
  --dependency "afterok:${COLLECT_JOB%%;*}" \
  --stats-dir "$STATS_DIR" \
  --profiles 'relaxed=1.2e12,balanced=1e12,strict=8e11')
```

This produces:

```text
thresholds/stats/stage5_v1/
  run.json
  pairs/<lang_pair>/
    calibration.json
    histogram.npz
    summary.json
  profiles/
    thresholds_relaxed.csv
    thresholds_relaxed.json
    thresholds_balanced.csv
    thresholds_balanced.json
    thresholds_strict.csv
    thresholds_strict.json
```

Each CSV and its matching JSON form a profile; the JSON records the rules, statistics snapshot, and CSV validation metadata.
Additional targets can be added:

```bash
bash thresholds/submit_compute_thresholds.sh \
  --account "$ACCOUNT" \
  --stats-dir "$STATS_DIR" \
  --profiles 'medium=1.1e12,aggressive=9e11'
```

After selecting a profile, submit filtering separately:

```bash
FILTER_JOB=$(bash filter/submit_apply_thresholds.sh \
  --tasks 64 \
  --account "$ACCOUNT" \
  --dependency "afterok:${THRESHOLDS_JOB%%;*}" \
  --thresholds-csv "$STATS_DIR/profiles/thresholds_balanced.csv" \
  --out-dir "$OUTPUT_DIR" \
  --scored-dir "$INPUT_DIR" \
  --tokenizer-json "$TOKENIZER_JSON" \
  --count-tokens)
```

`--count-tokens` counts retained samples exactly during output using the tokenizer supplied via
`--tokenizer-json`, without special tokens.
Omit this option for faster processing with estimated token counts.
Different threshold profiles must write to different output directories; completed data is not overwritten.

Submit aggregation with a dependency to wait for all filtering tasks to succeed:

```bash
bash filter/submit_merge_filter_stats.sh \
  --account "$ACCOUNT" \
  --dependency "afterok:${FILTER_JOB%%;*}" \
  --out-dir "$OUTPUT_DIR"
```

The output root contains `filter_stats.csv`, `filter_stats.json`, `filter_stats_corpora.csv`, and
`filter_stats_languages.csv`. The language table reports the total tokens in bilingual pairs associated with
each language, including both sides. Summing its language rows does not give the total corpus size.
To inspect progress during filtering, submit aggregation independently with `--allow-partial`;
the results explicitly list missing language pairs.
Omit `--dependency` when upstream tasks have already completed. For independent submissions, an actual job ID
can also be used, for example `--dependency "afterok:<job-id>"`.

## Token Budget Semantics

In the default `estimated` mode, **row counts in score bins are exact over the full input, but token counts within bins are estimated**.
Source and target character counts are accumulated separately, then scaled using the existing true source-side
and target-side token totals for each language pair.
The input total therefore matches the existing CSV, but retained token counts may deviate as filtering changes
the token-to-character ratio.
`1T` is an estimated target; the first output is not guaranteed to fall within ±1%. `--count-tokens` reports the actual result.

For a more accurate budget, use the following during statistics collection. Use this new statistics directory
for subsequent threshold computation and filtering:

```bash
bash thresholds/submit_collect_data_stats.sh \
  --tasks 64 \
  --account "$ACCOUNT" \
  --scored-dir "$INPUT_DIR" \
  --stats-dir "$PWD/thresholds/stats/stage5_exact_v1" \
  --token-mode exact \
  --tokenizer-json "$TOKENIZER_JSON"
```

`exact` mode does not require a token CSV and performs CPU tokenization on **all input text**, at a substantially higher cost.
Filtering automatically reuses the same tokenizer and reports exact retained token counts.
Both modes use stable hashing to select a specified fraction within the boundary bin, so the budget is an
expected value rather than a knapsack solution accurate to a single token.
Small language pairs or very long rows can introduce discretization error. To change the budget, generate
thresholds under a new profile name and write to a new output directory.
Samples removed by hard rules or lacking all semantic evidence are never restored to reach the target.

## Joint Score

Configuration is centralized in `config.json`. Configuration changes require a new statistics directory;
changing only the token target does not require another scan.

1. Detect copies after NFC and whitespace normalization; discard rows with an empty source or target.
2. Cross-script copies: discard identical texts with at least 20 letters when they clearly conflict with the labeled script.
3. Wrong language: discard text with at least 40 letters when both LID systems identify the same other language
   with confidence ≥0.90 and letters from the labeled script account for <20% of all letters.
   Unknown scripts are not used as evidence for removal.
4. Exclude rows where similarity and QE are both missing or non-finite, recording them separately as `missing_semantics`.
   An available LLM score does not bypass this rule.
5. On the remaining sampled data, fit similarity quantiles P per language pair, QE quantiles Q grouped by QE model,
   and LLM quantiles J. Sampling is uniform without replacement over global row indices for the entire language pair,
   covering all shards and row groups without prefix bias.
6. Compute LID consistency L and five groups of heuristic risks—length imbalance, copying, repetition, numbers,
   and structure—and combine the risks into H.
7. Compute a weighted geometric mean with P/Q/H/L weights of 0.35/0.35/0.20/0.10.
   Multiply the P weight by 0.5 when MRR <0.8, or by 0.75 when no benchmark is available.
   Multiply the Q weight by 0.5 when either side is unsupported by the QE model, or by 0.75 when support status is unknown.
8. When an LLM score is available, its weight is 0.20 and the normalized base score components have a combined weight of 0.80.
   Multiply the total score by 0.5 when at least two valid score groups are <0.20;
   multiply it by another 0.25 when copy risk ≥0.90 and L<0.40.

Corrected character ratios and word lengths use floating-point arithmetic, and scripts without spaces are
identified from the target language in the directory name. The old truncated ratios and word-length fields
affected by the `trg_lang` column-name issue are not read. Input columns are preserved as-is;
corrected features are used only during scoring.
Existing exact source and target maximum word lengths are reused. Missing ancillary heuristics are not treated
as evidence of a bad sample; constant calibration features are disabled.
Number risk is 0 when there are no numbers; entities shorter than 12 letters receive no soft copy penalty.
Punctuation risk is scaled by text length.
The meaning of the user-defined regular expression in `filter_regex_src` is unknown, so it is not used for automatic removal or score penalties.
Language aliases can be configured. Hans/Hant/Hani are treated as the same script for comparison, while Cantonese is not merged into Mandarin.

Quantiles use at most 257 nodes, with tied values assigned their midrank. Scores serve only for ranking and are
not quality probabilities; language pairs do not compete directly on raw total scores.
QE models absent from the sample fall back to the language pair's QE distribution;
when no calibration values are available at all, a neutral quantile of 0.5 is used.

## Generating Multiple Threshold Profiles

For each language pair, joint scores are divided into 10,000 bins, with separate statistics for two coverage
groups: with and without LLM scores. Initial budget weights are:

| Stage5 tokens per language pair | Retention weight |
| --- | ---: |
| <0.1B | 90% |
| 0.1B–1B | 80% |
| 1B–10B | 61% |
| ≥10B | 60% |

A shared coefficient `c` is solved across all language pairs, allocating
`min(eligible tokens, c × original tokens × retention weight)` to reach the profile's total target.
If too few eligible tokens are available, all eligible data is retained and the shortfall is reported in the profile JSON.
Within each language pair, groups with and without LLM scores use the same token retention fraction,
so missing LLM coverage does not itself create a ranking penalty.

Each language pair has two CSV rows, with `llm_present=0/1`; `T` is a human-readable boundary score.
**Filtering uses both `boundary_bin` and `boundary_fraction`**:
all rows above the boundary bin are retained, rows below it are removed, and rows within it are selected
using a stable hash of the input filename, row index, and seed.
This handles discrete LLM scores and large numbers of tied samples, independently of batch size or the number of SLURM tasks.
The CSV therefore cannot be passed directly to the old `06/.../filter`; use the joint filtering scripts in this directory.

## Execution and Resume Constraints

- All four submitters support `--tasks/--account/--partition/--cpus/--mem/--time/--dependency/--python/--dry-run`.
  Other arguments are forwarded to the Python worker; submitters accept both `--scored-dir` and `--scored_dir`.
- Statistics collection and filtering default to 64 array tasks, each with 4 CPUs, 32G memory, and a 2-day time limit.
  Language pairs are assigned deterministically; each large language pair is handled by a single task.
  Threshold computation uses one task with 1 CPU, 16G memory, and a 2-hour limit; aggregation uses one task with
  1 CPU, 8G memory, and a 2-hour limit.
  Threshold computation and aggregation do not create arrays and reject `--tasks` values greater than 1,
  preventing concurrent writes to the same result.
- Statistics collection reads the row groups touched by sampling, then performs one full histogram scan.
  Peak memory depends on one row group, sample size, and batch size. Filtering independently streams the raw data once.
- Use `--pairs eng_Latn-fra_Latn,abk_Cyrl-eng_Latn` to generate separate pilot statistics for selected language pairs.
  Use a separate statistics directory and a token target appropriate for this subset when computing thresholds.
  Omit `--pairs` for the full production statistics run.
- Any shard read failure fails the task; the shard is neither skipped nor marked complete.
  Missing thresholds, incomplete statistics tasks, and token CSV row-count mismatches are errors.
- Statistics record input filenames, sizes, mtimes, row counts, and schemas; filtering verifies that the snapshot
  has not changed. This is not a full hash of the input contents.
- Output is written to a temporary directory per language pair, validated, then published atomically with `_DONE.json`.
  Collection and filtering submission scripts default to `--skip_existing`; only completed items with matching
  configuration and file checks are skipped. Failed language pairs are restarted from scratch.
  Forcefully terminating processes may leave hidden temporary directories, which are not treated as completed data on resume.
- Reusing an output directory from another profile, nesting input and output directories, or reusing a statistics
  directory with a different configuration is rejected.
- Rules are not adjusted automatically; training and other stages are not launched or rerun, and Stage5 is not deleted.

## Pre-submission Checks

```bash
bash thresholds/submit_collect_data_stats.sh \
  --account "$ACCOUNT" \
  --scored-dir "$INPUT_DIR" \
  --token-stats-csv "$TOKEN_STATS_CSV" \
  --stats-dir "$STATS_DIR" --dry-run
bash thresholds/submit_compute_thresholds.sh \
  --account "$ACCOUNT" \
  --stats-dir "$STATS_DIR" --dry-run
bash filter/submit_apply_thresholds.sh \
  --account "$ACCOUNT" \
  --scored-dir "$INPUT_DIR" \
  --thresholds-csv "$STATS_DIR/profiles/thresholds_balanced.csv" \
  --out-dir "$OUTPUT_DIR" --dry-run
bash filter/submit_merge_filter_stats.sh \
  --account "$ACCOUNT" \
  --out-dir "$OUTPUT_DIR" --dry-run
```

`--dry-run` only prints submission commands; it neither runs Python data-processing scripts nor submits jobs to SLURM.
