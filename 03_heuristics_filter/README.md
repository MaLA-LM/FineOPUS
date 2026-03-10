# Heuristics Filtering Pipeline

This step implements a **two‑stage hybrid architecture**:

* **Stage 1 – Data‑Driven Deterministic Filtering** (adaptive, language‑pair‑specific thresholds)
* **Stage 2 – Unsupervised Multivariate Anomaly Detection** (IsolationForest)

## 🏗️ System Architecture

The pipeline consists of three sequential components:

0. **Pre-compute Filter Features:** Pre-computes a set of heuristic filter features
1. **Exploratory Data Analysis (EDA):** Streams raw Parquet shards, computes global and per-language-pair summary statistics, and generates adaptive filtering thresholds.
2. **Stage 1 (Deterministic Filter):** Applies the computed adaptive thresholds and boolean flags to deterministically drop extreme univariate outliers (e.g., severe length mismatches, heavy repetition).
3. **Stage 2 (Unsupervised Filter):** Trains an `IsolationForest` on a representative sample of each language pair to learn the "normal" multivariate manifold, then streams the data to isolate and drop complex, multi-dimensional anomalies.

---

## 🚀 Key Technical Features

* **Out‑of‑Core Batch Streaming:** Data is never fully loaded into Pandas. `pyarrow.parquet.ParquetFile.iter_batches()` is used to stream datasets of any size with a bounded memory footprint.
* **Fractional Downsampling:** EDA and Stage‑2 training subsample large language pairs to a maximum of **400 million rows** (~200 GB RAM peak) to guarantee statistical validity without OOM.
* **Adaptive Thresholds:** EDA now computes multiple empirical percentiles (0.1%, 1%, 5%, median, 95%, 99%, 99.9%) and derives **smart lower/upper bounds** by comparing the local tail with the language‑pair‑specific 4‑σ theoretical range. Features are categorized as two‑sided, lower‑bound only, or upper‑bound only, and logical floors (e.g. ≥1 for lengths, ≥0 for match scores) are enforced.
* **Slurm Stride Arrays:** A small stride value interleaves thousands of lang‑pairs across a job array.

---

## 📂 Repository Structure

```text
├── 0_precompute_all.sh             # wrapper around `precompute_heuristics_filters.py`
├── precompute_heuristics_filters.py # compute all heuristic features and save to Parquet
├── filter_feature_eda.py           # compute per‑LP stats & smart thresholds
├── filter_stage1.py                # Stage 1 deterministic filtering by lang‑pair
├── filter_stage2.py                # Stage 2 IsolationForest inference by lang‑pair
├── 2_run_eda.sh                    # Slurm array for EDA on precomputed filter features
├── 3_filter_stage1.sh              # Slurm array for separate Stage 1
├── 4_filter_stage2.sh              # Slurm array for separate Stage 2
├── lang_pairs.txt                  # generated list of language-pair folders
└── README.md
```

---

## ⚙️ Execution Pipeline

### Step 0: Pre-compute Filter Features

The first stage converts raw text pairs into a set of heuristic features. The helper script runs over every Parquet shard, computes statistics (lengths, ratios, punctuation, numerals, repetition, HTML/regex flags, etc.), and writes the results back alongside the original Parquet files.

```bash
sbatch 0_precompute_all.sh
```


### Step 1: Exploratory Data Analysis (EDA)

Generate a list of all language pairs present in your data root. This text file is used by the Slurm stride arrays to map tasks to specific languages.

```bash
ls -1d /path/to/data/* | xargs -n 1 basename > lang_pairs.txt

```

Run the EDA script to dynamically generate the thresholds. This script calculates multiple empirical percentiles (0.1 % through 99.9 %) per language pair and then computes a **smart lower/upper bound** by comparing the local tail against the pair‑specific 4‑σ theoretical interval. The resulting `filtering_thresholds.csv` is automatically pruned of any completely-empty columns and can be inspected or manually tweaked if desired.

```bash
python filter_feature_eda.py \
    --data_root /path/to/raw_data \
    --out_dir /path/to/eda_outputs

```

**Outputs Generated:**

* `summary_by_langpair.csv` – raw statistics for every (langpair, feature)
* `bool_rates_by_langpair.csv` and `bool_rates_global.csv` – boolean flag prevalences
* `filtering_thresholds.csv` – adaptive thresholds, ready for Stage 1
* `summary_global.csv` (optional) – aggregated feature moments across all pairs

### Step 2: Stage 1 Deterministic Filtering

Submit the Stage 1 Slurm array. This job reads the `filtering_thresholds.csv` and streams the Parquet files, keeping only rows that fall within the acceptable bounds.

```bash
sbatch --array=1-16 3_filter_stage1.sh

```

* **Checkpointing:** Progress is logged to `stage1_tracking.csv`. If the job times out, simply resubmit the identical `sbatch` command.

### Step 3: Stage 2 Unsupervised Filtering

Submit the Stage 2 Slurm array. This runs on the output directory of Stage 1. It fits an Isolation Forest model (conservatively dropping `contamination=0.0001` or 0.01% of data) to catch multivariate garbage that bypassed the heuristic thresholds.

```bash
sbatch --array=1-16 4_filter_stage2.sh
```

* **Imputation Strategy:** Missing feature values are filled with the feature median computed during the training sample. This behaviour is identical in both the standalone `filter_stage2.py` and the hybrid pipeline.

---

## 📊 Monitoring Progress

Both Stage 1 and Stage 2 utilize a centralized tracking CSV. You can instantly monitor the progress of your Slurm array without digging through log files using this terminal one-liner:

```bash
echo "Completed: $(($(wc -l < logs/stage1_tracking.csv) - 1)) / $(wc -l < lang_pairs.txt)"

```

---

## ⚠️ Memory & Performance Notes

* **EDA & Model Training:** The upper bound for in‑memory samples is controlled by `MAX_SAMPLE_SIZE = 400_000_000`. 
* **Inference / Filtering:** Stage 1 and Stage 2 scripts stream Parquet in batches (default 50‑100 k rows).
* **Slurm CPU Binding:** Remember to set `ARROW_NUM_THREADS=$SLURM_CPUS_PER_TASK` (or export via module) to avoid thread oversubscription and achieve maximum throughput.
