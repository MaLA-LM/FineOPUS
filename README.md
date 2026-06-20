# FineOPUS: Multilingual Parallel Corpora Preparation & Evaluation Pipeline

Welcome to the **FineOPUS** repository. This project provides a robust, end-to-end, high-performance computing (HPC) pipeline designed to download, clean, normalize, filter, and evaluate multilingual parallel corpora derived from the OPUS collection. 

The pipeline is organized into numbered stages (**00–07**) that lead you from raw, unverified downloads to a clean, high-quality translation corpus suitable for training large language models (LLMs) or machine translation (MT) systems. It is optimized to run on SLURM-managed clusters (such as CSC Mahti and LUMI) via job arrays, but also supports local execution for debugging.

---

## 📂 Repository Directory Layout

*   [00_preprocess](./00_preprocess) — Metadata retrieval, zip streaming, ISO language mapping, validation, and initial 100M-line sharding.
*   [01_language_reID](./01_language_reID) — Multi-model (GlotLID + ConLID) language re-identification, confidence quantiles, and ensembling.
*   [02_deduplication](./02_deduplication) — SQL-based exact bitext deduplication using DuckDB disk-based memory spilling.
*   [03_heuristics_filter](./03_heuristics_filter) — Two-stage hybrid filtering (univariate adaptive heuristics & unsupervised multivariate Isolation Forest).
*   [04_parallelism_check](./04_parallelism_check) — Embedding-based semantic similarity scoring, model selection (MRR), and thresholding.
*   [05_quality_estimation](./05_quality_estimation) — Distributed scoring across 5 backends: COMET, MetricX-24, ReMedy, Bicleaner, and offline vLLM (Qwen3, M-Prometheus).
*   [06_frontier_llm_as_judge](./06_frontier_llm_as_judge) — High-quality evaluation using Azure-hosted DeepSeek-V4-Flash with token bucket rate limiters.
*   [07_human_eval](./07_human_eval) — Static HTML-based interface for broad quality evaluation (languages, fluency, translation correspondence).

---

## 🛠️ Detailed Stage Breakdown

### [00_preprocess](./00_preprocess)
Queries, downloads, normalizes, and reshards raw OPUS zip bundles.
*   **Key Scripts:**
    *   [get_corpora_list.py](./00_preprocess/get_corpora_list.py): Retrieves metadata from the OPUS REST API.
    *   [download_convert_save.py](./00_preprocess/download_convert_save.py): Streams zips, extracts parallel texts, maps to standardized ISO 639-3 + script codes, and writes Snappy-compressed Parquet.
    *   [check_conversion.py](./00_preprocess/check_conversion.py): Uses `GlotScript` to validate and normalize language codes.
    *   [sharding.py](./00_preprocess/sharding.py): Consolidates data into balanced 100M-row shards.
*   **Prerequisites:** Python 3.9+, `pyarrow`, `requests`, `GlotScript`, `langcodes[data]`.

### [01_language_reID](./01_language_reID)
Re-estimates source and target language labels to handle mislabeled translation text.
*   **Flow:**
    1.  [shard_filelists.py](./01_language_reID/shard_filelists.py): Partitions Parquet files into balanced file lists.
    2.  [re_lang_identify.py](./01_language_reID/re_lang_identify.py): Predicts language labels using **GlotLID** and **ConLID**.
    3.  [get_conf_stats.py](./01_language_reID/get_conf_stats.py) & [collect_conf_stats.py](./01_language_reID/collect_conf_stats.py): Accumulate prediction confidences per language into binary float16 files.
    4.  [quantiles_conf_stats.py](./01_language_reID/quantiles_conf_stats.py): Derives a per-language threshold (`median - std` clipped to `[0.3, 0.9]`).
    5.  [ensemble_relid.py](./01_language_reID/ensemble_relid.py): Ensembles predictions. Replaces the original label *only* when both models confidently agree on the same new language. routes small volume pairs into a `_mixed` bucket.
    6.  [aggregate_ensembled_relid.py](./01_language_reID/aggregate_ensembled_relid.py): Restructures the ensembled `.tar` shards into canonical pair folders.

### [02_deduplication](./02_deduplication)
Executes exact duplicate removal across sentence pairs.
*   **Key Scripts:**
    *   [exact_dedup.py](./02_deduplication/exact_dedup.py): Uses **DuckDB** SQL commands (`SELECT DISTINCT`) to remove duplicates. Integrates memory limit configs (`--memory_limit`) and disk-based temp directories to prevent Out-Of-Memory (OOM) failures on massive datasets.
    *   [dedup_exact_all.sh](./02_deduplication/dedup_exact_all.sh): Runs SLURM job arrays dynamically based on incomplete folders logged in `exact_dedup_stats.csv`.
*   **Dependencies:** `duckdb`, Python 3.8+.

### [03_heuristics_filter](./03_heuristics_filter)
Implements a hybrid out-of-core filtering pipeline.
*   **Architecture:**
    *   **Stage 1 (Deterministic):** [filter_feature_eda.py](./03_heuristics_filter/filter_feature_eda.py) calculates empirical percentiles per language pair and generates adaptive threshold boundaries. [filter_stage1.py](./03_heuristics_filter/filter_stage1.py) drops extreme outliers (character ratios, HTML tags, severe length mismatches).
    *   **Stage 2 (Unsupervised):** [filter_stage2.py](./03_heuristics_filter/filter_stage2.py) trains an `IsolationForest` model on a subset of the data (capped at 400M rows) to detect and isolate multidimensional anomalies.
*   **Technical Details:** Streams data in batches using PyArrow (`iter_batches`) to maintain a bounded memory footprint.

### [04_parallelism_check](./04_parallelism_check)
Embedding-based filter to drop semantic mismatches.
*   **Flow:**
    1.  **Benchmarking:** Evaluates embedding models (Harrier, Multilingual E5, GTE, Jina, Qwen) on gold FLORES-200 and BOUQuET test sets.
    2.  **Scoring:** Scores every parallel sentence pair via cosine similarity using the best-performing model for that language pair.
    3.  **Thresholding:** Computes custom thresholds $T$ based on the gold test distributions and benchmark Mean Reciprocal Rank (MRR), clipped to `[0.3, 0.95]`.
    4.  **Filtering:** Streams Parquet shards and drops rows with scores below $T$.

### [05_quality_estimation](./05_quality_estimation)
A modular, distributed scoring pipeline supporting both **FLORES-200** and **OPUS** datasets.
*   **Supported Backends:**
    *   **COMET:** Translation evaluation.
    *   **MetricX-24:** Hybrid regression model (outputs adjusted from `0-25` to `0-1`).
    *   **ReMedy:** Gemma-2 based offline model calibrated using isolated master ports. Patched to support 140+ language configurations.
    *   **Bicleaner:** Standard Bitextor classifier.
    *   **LLM Scorer:** Offline vLLM execution of Qwen3/M-Prometheus models using structured schemas.
*   **Key Infrastructure:**
    *   **Manifest System:** Deterministic shard assignment using BLAKE2b hashing.
    *   **Stage Writer:** Safe checkpointing (`checkpoint.jsonl`) and transaction tracking.

### [06_frontier_llm_as_judge](./06_frontier_llm_as_judge)
Scores sentence pairs with a frontier model (Azure-hosted DeepSeek-V4-Flash).
*   **Features:**
    *   **Rate Limiting:** Implements sliding-60s token buckets for TPM/RPM budgets.
    *   **RAI Fallback:** When Azure Responsible AI content filters block a multi-row batch, the script automatically falls back to individual scoring, saving failed segments as `null` and continuing without losing progress.

### [07_human_eval](./07_human_eval)
Lightweight, serverless human evaluation interface that generates static HTML pages to audit translation quality.
*   **Evaluation Criteria:** Focuses on broad quality verification: 1) language code correctness, 2) sentence fluency/naturalness, and 3) translation equivalence.
*   **Flow:** 
    1.  [0_create_sample_data.py](./07_human_eval/0_create_sample_data.py): Streams data from Hugging Face and pseudo-samples it into shuffled JSONL files in `data/samples/`.
    2.  [1_create_annotation_html.py](./07_human_eval/1_create_annotation_html.py): Reads the sample data and compiles an interactive index page and individual language pair annotation pages under `data/annotation_html`.
    3.  [create_annotation.sh](./07_human_eval/create_annotation.sh): Helper script orchestrating the sample creation and HTML generation.
*   **Storage:** Since it runs serverless in the browser, annotation progress is saved in local storage and can be exported as JSONL files.

---

## 🚀 Getting Started

### 1. Environment Setup
Create a virtual environment and install dependencies. Note that individual folders (such as `02_deduplication`, `03_heuristics_filter`, `05_quality_estimation`, etc.) may require specialized dependencies (like `duckdb`, `comet-mt`, `vllm`):

```bash
# General setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. SLURM Clusters (CSC / LUMI)
Job array submissions are supported by shell wrappers in each directory. Before running, configure your SLURM accounts, partitions, and HF API Tokens in the corresponding environment wrappers.

```bash
# Example: Submitting Stage 1 Heuristic Filtering
cd 03_heuristics_filter
sbatch --array=1-16 3_filter_stage1.sh
```

---

## 🤝 Contributing & Maintenance

When modifying this repository:
1. Preserve all existing comments and docstrings.
2. If folders are relocated or stages are renumbered, update:
    * The global directory paths in the root `.gitignore`.
    * This global README.md file.
    * References in scripts across all stage directories.
