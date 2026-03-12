### ReLID pipeline for FineOPUS

This directory contains the end‑to‑end **re‑labelled language ID (ReLID)** pipeline used for the FineOPUS corpus. The goal is to re‑estimate source/target language labels for all sentence pairs using multiple LID models, derive per‑language confidence thresholds, and then produce final per‑language‑pair Parquet shards for downstream training and analysis.

The scripts are written for a SLURM cluster (CSC environment) but the logical flow is independent of the specific scheduler.

---

### High‑level processing flow

1. **Prepare filelists**
   - Use `shard_filelists.py` (via `shard_filelists.sh`) to scan an input Parquet tree and split it into balanced text filelists:
     - Input: original FineOPUS Parquet under one root directory.
     - Output: `filelists/*/filelist_*.txt` where each file contains a subset of Parquet paths (absolute or relative).
   - These filelists are later consumed by SLURM array jobs so that work is evenly distributed.

2. **Run language identification with multiple models**
   - `re_lang_identify.sh` launches a SLURM array over the filelists and calls `re_lang_identify.py`.
   - `re_lang_identify.py`:
     - Loads each input Parquet file (columns must include at least `source_text`, `target_text`, `conv_src_lang`, `conv_tgt_lang`).
     - Runs LID with either **GlotLID** (`fastText` model) or **ConLID** (`ConLID.from_pretrained`), depending on `--model_path`.
     - Adds predicted language IDs and confidences:
       - `source_predlang_id`, `source_predlang_conf`
       - `target_predlang_id`, `target_predlang_conf`
     - Writes the result to a mirrored Parquet tree under:
       - `fineopus-original-ReLID-by-GlotLID` or
       - `fineopus-original-ReLID-by-ConLID`.
   - Re‑runs are idempotent: if an output file already exists with the same row count, it is skipped.

3. **Collect per‑language confidence statistics**
   - First stage: per‑shard stats
     - `get_conf_stats.sh` runs a SLURM array over ReLID outputs and calls `get_conf_stats.py`.
     - `get_conf_stats.py` reads Parquet files, extracts the four prediction/confidence columns, and accumulates **lists of confidences per language**.
     - Each array job writes `conf_stats_*.json.gz` under a model‑specific `*-conf-stats` directory.
   - Second stage: global aggregation
     - `collect_conf_stats.sh` calls `collect_conf_stats.py`:
       - Reads all `conf_stats_*.json.gz`.
       - For each language, appends confidences to a float16 `.bin` file in a temporary directory and accumulates counts, sums, sums of squares, and min/max.
       - Writes a summary JSON `_collect_stats.json` describing global stats and the temp directory.
   - Third stage: quantile & threshold computation
     - `quantiles_conf_stats.sh` calls `quantiles_conf_stats.py` with `_collect_stats.json`.
     - `quantiles_conf_stats.py`:
       - Loads per‑language confidence bins from the `.bin` files.
       - Computes mean, median, standard deviation and other statistics.
       - Derives a **per‑language confidence threshold** `thr` (clipped to \[0.3, 0.9\]) typically using `median − std`.
       - Writes `conf_stats_quantiles.json`, which is later used to decide whether a model prediction is “confident enough” for each language.

4. **Ensemble GlotLID and ConLID predictions**
   - `ensemble_relid.sh` runs a SLURM array and calls `ensemble_relid.py`.
   - `ensemble_relid.py`:
     - For each relative file path (from `filelists/fineopus-original-ReLID-relpath-filelists-512-shard`):
       - Loads the corresponding GlotLID and ConLID Parquet files.
       - Looks up per‑language thresholds from:
         - `fineopus-original-ReLID-by-GlotLID-conf-stats/conf_stats_quantiles.json`
         - `fineopus-original-ReLID-by-ConLID-conf-stats/conf_stats_quantiles.json`
     - For each row:
       - Computes final source and target languages by:
         - Starting from the original `conv_src_lang` / `conv_tgt_lang` (from the file path),
         - Accepting a model prediction only if its confidence exceeds the language‑specific threshold,
         - Returning the original language unless **both** models agree on the same language and are confident.
       - Adds `src_lang` and `tgt_lang` plus bookkeeping fields (original langs, both models’ predictions and confidences).
       - Routes the row into a language‑pair bucket `"{src_lang}-{tgt_lang}"`.
     - Buffers are periodically flushed into pair‑specific Parquet shards under a temporary output root, with configurable `max_rows_per_pair_shard` and `min_rows_per_pair_shard`.
     - Very small buckets are redirected to a `_mixed` pool.
   - After processing each shard, `ensemble_relid.sh` **tars** the temporary output directory and removes the original tree, producing `tmp_*.tar` under `fineopus-original-ReLID-ENSEMBLED-TAR`.

5. **Aggregate ensembled outputs into final per‑pair directories**
   - `aggregate_ensembled_relid.sh` calls `aggregate_ensembled_relid.py`:
     - Iterates over all `tmp_*.tar` archives in `fineopus-original-ReLID-ENSEMBLED-TAR`.
     - Extracts per‑pair subdirectories (excluding `_mixed`).
     - Moves and renames Parquet files into a final per‑pair structure under `fineopus-original-ReLID-ENSEMBLED-FIRST`, numbering them as `{pair}_part_000.parquet`, `{pair}_part_001.parquet`, etc.
     - If a pair already has existing parts, indexing continues from the current maximum.
   - `aggregate_ensembled_relid_for_mix.sh` + `aggregate_ensembled_relid_for_mix.py`:
     - Specifically process `_mixed` Parquet shards inside `tmp_*.tar`:
       - Extracts only `_mixed` directories.
       - Reads all `_mixed` Parquet files, splits them again by `(src_lang, tgt_lang)` at the row level, and appends rows to the corresponding pair under `fineopus-original-ReLID-ENSEMBLED-MIX-*`.
       - Ensures each part file contains at most a configurable number of rows (`max_rows_per_part`) and continues numbering from any existing parts.

6. **Post‑processing utilities**
   - `count_parquet_rows_per_pair.py` + `count_parquet_rows_per_pair.sh`:
     - Count total rows per language pair under a given root, writing an aggregated CSV/JSON for later analysis.
   - `move_pairs.py` + `move_pairs.sh`:
     - Read a per‑pair stats Excel file (e.g. with `language_pair` and `total_rows` columns).
     - Move all pair directories with `total_rows < threshold` from a **source** root to a **destination** root.
     - Supports a `--dry-run` mode to inspect planned moves.
   - `merge_pair.py` + `merge_pair.sh`:
     - Given two different roots containing per‑pair directories, merge them into a new `out-root`:
       - For each pair (union of all pairs in both roots), build a PyArrow dataset over all Parquet files.
       - Rewrite the data into chunked, size‑capped Parquet parts in the output root, aligning schemas when necessary.
       - After a successful run, the original roots can be deleted by the wrapper script.

---

### Typical end‑to‑end usage (conceptual)

1. **Prepare sharded filelists** from the original FineOPUS Parquet tree:
   - Configure `SOURCE_DIR` and `OUTPUT_DIR` in `shard_filelists.sh`.
   - Run `shard_filelists.sh` to create `filelist_*.txt` shards.

2. **Run LID with GlotLID and ConLID**:
   - Adjust paths in `re_lang_identify.sh` (source corpus root, output root, model path, filelists).
   - Submit the SLURM job:
     - First with the GlotLID model,
     - Then with the ConLID model (comment/uncomment the relevant `OUTPUT_DIR` / `MODEL_PATH` sections).

3. **Compute confidence stats and thresholds** for each model:
   - Update paths in `get_conf_stats.sh`, `collect_conf_stats.sh`, `quantiles_conf_stats.sh` for each model output.
   - Run:
     - `get_conf_stats.sh` (SLURM array),
     - `collect_conf_stats.sh`,
     - `quantiles_conf_stats.sh`.
   - This yields `conf_stats_quantiles.json` for GlotLID and ConLID.

4. **Ensemble predictions into language‑pair Parquet shards**:
   - Ensure `GLOTLID_DIR`, `CONLID_DIR`, and the two `*_THR_JSON` paths in `ensemble_relid.sh` are correct.
   - Provide `filelists` over the ReLID results (e.g. `fineopus-original-ReLID-relpath-filelists-512-shard`).
   - Submit `ensemble_relid.sh` as a SLURM array; it will write pair‑wise Parquet under temporary directories and tar them.

5. **Assemble and clean up final outputs**:
   - Run `aggregate_ensembled_relid.sh` to assemble all per‑pair Parquet parts into a canonical tree (excluding `_mixed`).
   - Optionally run `aggregate_ensembled_relid_for_mix.sh` to re‑distribute `_mixed` shards back into per‑pair outputs.
   - Use `count_parquet_rows_per_pair.sh` + `move_pairs.sh` / `merge_pair.sh` for statistics, balancing, and combining different versions of the corpus.

This README intentionally focuses on the **overall flow**. For precise CLI options, refer to the docstrings and argument parsers inside each `*.py` script.

