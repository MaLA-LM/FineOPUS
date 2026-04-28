# 02_parallelism_check

Embedding-based parallelism quality filter for FineOPUS-Filtered-Stage2.

The pipeline selects the best embedding model per language pair, scores every
sentence pair with that model, derives per-pair similarity thresholds, and
filters out low-quality translations.

```
benchmarking  →  compute_similarity  →  thresholds  →  filter
```

---

## Pipeline overview

### 1. `benchmarking/` — Model selection

Evaluate a pool of multilingual embedding models on held-out parallel corpora
to find the best model for each language pair.

**Benchmarks used**

| Benchmark | Granularity |
|-----------|------------|
| FLORES-200 | sentence |
| BOUQuET | sentence + paragraph |

**Metric:** Mean Reciprocal Rank (MRR) on a bitext retrieval task — each source
sentence is used as a query and the correct target is ranked against a pool of
negatives.

**Models evaluated**

| Model | Size |
|-------|------|
| `microsoft/harrier-oss-v1-0.6b` | 0.6 B |
| `microsoft/harrier-oss-v1-270m` | 270 M |
| `intfloat/multilingual-e5-large` | 560 M |
| `intfloat/multilingual-e5-small` | 118 M |
| `Alibaba-NLP/gte-multilingual-base` | 305 M |
| `jinaai/jina-embeddings-v3` | 570 M |
| `jinaai/jina-embeddings-v5-text-{nano,small}` | 70/305 M |
| `codefuse-ai/F2LLM-v2-{160M,330M,0.6B}` | 160–600 M |
| `google/embedding-gemma-300m` | 300 M |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6 B |

**Key scripts**

| Script | Purpose |
|--------|---------|
| `benchmarking.py` | Single-model evaluation over all language pairs |
| `analyze_models.py` | Aggregate results → best model per pair CSVs |
| `submit_benchmarking.sh` | SLURM submission wrapper |
| `language_pairs.txt` | List of `src_Script-tgt_Script` pairs to evaluate |


---

### 2. `compute_similarity/` — Scoring

Encode every `(source_text, target_text)` row in the FineOPUS-Filtered-Stage2
parquet shards with the best model for that language pair and write a
`similarity_score` column (cosine similarity) to the output shards.

**Key scripts**

| Script | Purpose |
|--------|---------|
| `compute_similarity.py` | Per-shard encoding + scoring (one SLURM task) |
| `precompute_sizes.py` | Pre-scan shard sizes for load balancing |
| `submit_compute_similarity.sh` | Enumerate unprocessed shards, bin-pack by size into array tasks, write manifests, submit SLURM array jobs |
| `model_to_language_pairs.json` | Maps each model to the language pairs it should score |

---

### 3. `thresholds/` — Threshold computation

Derive a per-language-pair similarity threshold `T` that separates
good translations from noise.  The threshold formula anchors on the gold
distribution (when available) and the benchmark MRR score.

**Inputs**

| Input | Produced by |
|-------|-------------|
| `stats/data_score_stats.csv` | `collect_data_stats.py` (percentile stats over all scored rows) |
| `stats/gold_score_stats.csv` | `collect_gold_stats.py` (percentile stats over gold-parallel test sets) |
| `../results/best_model_per_lang_pair_by_flores_bouquet_combined_selected_models.csv` | Step 1 |

**Threshold formula (simplified)**

```
T_gold = max(gold_p01, gold_mean − k·gold_std)
T_data = data_p10

if   MRR ≥ high_mrr:  T_raw = max(T_gold, T_data)   # high confidence
elif MRR ≥ low_mrr:   T_raw = max(T_gold − 0.02, T_data)
elif no benchmark:     T_raw = data_p01               # fallback
else (low MRR):        T_raw = data_p01               # distrust model

T = clip(T_raw, t_floor, min(gold_p25, t_cap, data_p50, keep_cap))
```

Confidence levels (`high` / `medium` / `low_mrr` / `no_benchmark`) are recorded
in the output CSV.

**Key scripts**

| Script | Purpose |
|--------|---------|
| `compute_thresholds.py` | Full threshold derivation + diagnostic plots |
| `collect_data_stats.py` | Compute score percentiles over scored shards |
| `collect_gold_stats.py` | Compute score percentiles over gold test sets |

**Outputs** (`stats/`)

- `thresholds.csv` — versioned threshold tables; columns include `T`,
  `confidence`, `has_gold`, `kept_fraction_est`, and all intermediate values
- `plots/` — per-model and per-pair diagnostic figures

---

### 4. `filter/` — Applying thresholds

Stream every scored shard and keep only rows where
`similarity_score ≥ T` for their language pair.

**Key scripts**

| Script | Purpose |
|--------|---------|
| `apply_thresholds.py` | Per-chunk filtering + per-task stats CSV |
| `merge_filter_stats.py` | Merge chunk stats into a summary table |
| `submit_apply_thresholds.sh` | SLURM array submission |


---