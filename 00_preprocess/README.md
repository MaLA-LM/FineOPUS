## 00_preprocess Overview

This stage prepares OPUS parallel corpora for later FineOPUS experiments. It queries available corpora, downloads the raw zip bundles, converts parallel bitext into Parquet shards with standardized language codes, validates the results, and reshards them for efficient downstream consumption.

The workflow is broken into four numbered steps, each backed by a Slurm script for HPC environments and a Python module for direct execution.

## Prerequisites

- Python 3.9+ with `requests`, `pyarrow`, `langcodes`, `GlotScript` (`pip install GlotScript`), and `langcodes[data]` for script detection.
- Adequate disk space for temporary zips (`temp_processing/`) and Parquet outputs.
- Optional: Slurm cluster with `cray-python` module to run the provided `*.sh` job scripts unchanged.

## Directory Contents

| File | Role |
| ---- | ---- |
| `0_collect_corpora_list.sh` | Calls `get_corpora_list.py`, and clones the OPUS repo for language mappings. |
| `get_corpora_list.py` | Hits the OPUS REST API (`https://opus.nlpl.eu/opusapi/`) and saves the response as `OPUS_API_collection.json`. |
| `1_download_convert_save.sh` | Slurm wrapper that calls `download_convert_save.py` on the compute cluster. |
| `download_convert_save.py` | Downloads each corpus zip, extracts aligned text, maps languages to ISO-639/script codes, and writes sharded Parquet files plus progress/error logs. |
| `2_check_conversion.sh` | Slurm wrapper running `check_conversion.py` to validate folder names and scripts. |
| `check_conversion.py` | Normalizes language codes, fixes directory names, logs mismatches, and prunes empty folders using GlotScript-based script detection. |
| `3_sharding.sh` | Slurm wrapper running `sharding.py` with a large memory, multi-core allocation. |
| `sharding.py` | Re-shards the validated Parquet data into consistent 100M-row shards, enforces schema checks, and records per-pair counts in CSV. |
| `OPUS_API_collection.json` | Cached corpus metadata retrieved from OPUS. |

## End-to-End Workflow

1. **Collect corpus metadata**
   - Run `./0_collect_corpora_list.sh`.  
   - Outputs: `OPUS_API_collection.json` plus a local clone of `Helsinki-NLP/OPUS` so language mappings under `OPUS/corpus/<corpus>/<version>/language_mappings.tsv` are available.

2. **Download, convert, and save Parquet shards**
   - Script entry point:  
     ```
     python download_convert_save.py \
       --input_json OPUS_API_collection.json \
       --mapping_file_dir /path/to/OPUS/corpus \
       --out_dir /path/to/opus-conversion \
       --tmp_dir /path/to/temp_processing \
       --progress_file processed_urls.txt \
       --error_log errors.json
     ```
   - What happens:
     - Streams each corpus zip via HTTP (`requests`), extracts it into `tmp_dir`, and searches for `.<src>`/`.<tgt>` files.
     - Determines ISO-639-3 + script codes using provided mappings; falls back to `langcodes` conversion if no mapping exists or logs failures to `error_log`.
     - Builds rows containing `source_text`, `target_text`, original metadata (corpus, version, url, original langs), and converted codes, then writes Snappy-compressed Parquet shards under `out_dir/<src>-<tgt>/`.
     - Tracks completed URLs in `processed_urls.txt` so the job can be resumed safely.
     - Cleans temporary extraction folders and the downloaded zip regardless of success.

3. **Validate conversions and fix language codes**
   - Script entry point:  
     ```
     python check_conversion.py \
       --data_dir /path/to/opus-conversion \
       --skip_log check_skipdirs.log \
       --mismatch_log check_file_mismatch.log
     ```
   - Responsibilities:
     - Walks each `<lang>-<lang>` folder, normalizes codes to the `xxx_Ssss` pattern, and uses GlotScript to infer scripts when only ISO-639-3 codes are present.
     - Moves Parquet files into newly named folders when normalization changes a code, resolving filename conflicts as needed.
     - Logs skipped directories (invalid names) and mismatch operations to the respective log files.
     - Performs a bottom-up scan at the end to delete empty directories.

4. **Shard for downstream training**
   - Script entry point:  
     ```
     python sharding.py \
       --data_dir /path/to/opus-conversion \
       --out_dir /path/to/fineopus-original \
       --output_file /path/to/statistics/sharding_line_counts.csv \
       --log_file sharding_skipped_dirs.log \
       --pq_mismatch sharding_pq_mismatch.log \
       --num_workers $(nproc)
     ```
   - Key behavior:
     - Validates that each input directory matches `xxx_Ssss-xxx_Ssss`.
     - Splits Parquet batches into shards capped at 100,000,000 lines, writing them under `out_dir/<pair>/`.
     - Detects schema mismatches via `pyarrow` and logs them without aborting the rest of the pipeline.
     - Writes a cumulative CSV summarizing per-pair line counts.

The numbered Slurm scripts (`0_*.sh`–`3_*.sh`) mirror these steps and can be submitted sequentially on the CSC Mahti/LUMI environment (update account IDs, paths, and modules as needed). Each script loads `cray-python`, allocates resources, and runs the Python module via `srun`.

## Generated Artifacts

- `OPUS_API_collection.json`: Full OPUS corpus metadata used as input for downloads.
- `<out_dir>/<src>-<tgt>/*.parquet`: Sharded, Snappy-compressed bilingual datasets with metadata columns.
- `processed_urls.txt`: Progress file that lets `download_convert_save.py` resume after interruptions.
- `dl_cvt_log_fix.json` (or custom `--error_log`): Records language-code failures and other per-corpus issues.
- `check_skipdirs.log`, `check_file_mismatch.log`: Diagnostics from `check_conversion.py`.
- `sharding_line_counts.csv`: Aggregated row counts written by `sharding.py`.
- `sharding_skipped_dirs.log`, `sharding_pq_mismatch.log`: Sharding-specific warnings.

## Tips & Troubleshooting

- **Resuming downloads**: Delete the corresponding line from `processed_urls.txt` if a corpus needs to be reprocessed; otherwise the script will skip it.
- **Mapping files missing**: The OPUS repo clone must stay in sync with the corpora available through the API. If a `language_mappings.tsv` is absent, the script falls back to heuristic conversions—review `dl_cvt_log_fix.json` for those cases.
- **GlotScript dependency**: `check_conversion.py` imports `GlotScript.sp`. Install from PyPI (`pip install GlotScript`) and verify the model data is available; otherwise language normalization will fall back to the `und_Zyyy` placeholder code.
- **Large temporary footprint**: Use a fast local scratch SSD/NVMe for `--tmp_dir` to avoid I/O bottlenecks and enable the cleanup trap in the Slurm scripts to reclaim space even if jobs fail.
- **Parallel sharding**: `sharding.py` relies on Python’s `multiprocessing.Pool`. Set `--num_workers` to match available CPUs (Slurm script sets it via `$SLURM_CPUS_PER_TASK`).

With these steps completed, the `fineopus-original` directory will contain standardized, large sharded Parquet datasets ready for subsequent filtering and training stages.

