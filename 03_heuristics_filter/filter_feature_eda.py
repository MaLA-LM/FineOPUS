#!/usr/bin/env python3
"""
Memory-Efficient EDA for FineOPUS precomputed filter features.

This script:
- Streams files in batches to maintain a strict, low memory footprint.
- Dynamically subsamples massive language pairs to a maximum threshold (e.g., 3M rows).
- Checks `summary_by_langpair.csv` to skip already processed languages.
- Appends per-language stats, bools, and correlations to disk immediately.
"""

import argparse
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ----------------------------
# Configuration & Setup
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUMERIC_COLS = [
    "src_char_len", "trg_char_len", "src_word_len", "trg_word_len",
    "src_max_word_len", "trg_max_word_len", "src_avg_word_len", "trg_avg_word_len",
    "char_len_ratio", "word_len_ratio", "score_term_punct", "score_numerals",
    "score_lcs_ratio", "score_levenshtein", "score_repeat_src", "score_repeat_trg",
]

BOOL_COLS = ["filter_html_src", "filter_html_trg"]

PERCENTILES = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
STAT_RENAME_MAP = {
    '0.1%': 'p001', '1%': 'p01', '5%': 'p05', '50%': 'median', 
    '95%': 'p95', '99%': 'p99', '99.9%': 'p999'
}

ABSOLUTE_SAFEGUARDS = {
    'char_len_ratio': {'lower': 0.25, 'upper': 4.0},   # A translation is rarely 4x longer
    'word_len_ratio': {'lower': 0.25, 'upper': 4.0},
    'score_lcs_ratio': {'lower': 0.01},                # Only drop if LCS is virtually 0%
    'score_levenshtein': {'lower': 0.01},              
    'src_char_len': {'lower': 1, 'upper': 2500},
    'trg_char_len': {'lower': 1, 'upper': 2500},
    'src_word_len': {'lower': 1, 'upper': 500},
    'trg_word_len': {'lower': 1, 'upper': 500},
    'score_numerals': {'lower': 0.1},                  # Only drop if numeral match is terrible
    'score_term_punct': {'lower': 0.0},                # Punctuation differs widely; rarely drop
    'score_repeat_src': {'upper': 150},                # Allow up to 150 chars of natural repetition
    'score_repeat_trg': {'upper': 150},
    'src_max_word_len': {'upper': 150},                # E.g., long URLs
    'trg_max_word_len': {'upper': 150},
    'src_avg_word_len': {'upper': 50},
    'trg_avg_word_len': {'upper': 50},
}

MAX_SAMPLE_SIZE = 400_000_000  # Cap in-memory rows to 400 Million (~130GB raw, peaks at ~200GB during pandas operations)

# ----------------------------
# Core Functions
# ----------------------------

def generate_summary_stats(df: pd.DataFrame, numeric_cols: list, exact_total_rows: int) -> pd.DataFrame:
    available_cols = [c for c in numeric_cols if c in df.columns]
    if not available_cols:
        return pd.DataFrame()

    stats_df = df[available_cols].describe(percentiles=PERCENTILES).T
    stats_df.rename(columns=STAT_RENAME_MAP, inplace=True)
    
    # Estimate missing count relative to the true total row count
    missing_ratio = df[available_cols].isna().mean()
    stats_df['missing'] = (missing_ratio * exact_total_rows).astype(int)
    stats_df['count'] = exact_total_rows  # Reflect the true dataset size in the logs
    
    stats_df.index.name = 'feature'
    cols_order = ["count", "missing", "min", "p001", "p01", "p05", "median", 
                  "p95", "p99", "p999", "max", "mean", "std"]
    return stats_df[cols_order].reset_index()

def aggregate_global_stats(by_lp_df: pd.DataFrame) -> pd.DataFrame:
    global_rows = []
    
    for feature, group in by_lp_df.groupby('feature'):
        valid = group[group['count'] > 0]
        if valid.empty: continue
            
        N = valid['count'].values
        means = valid['mean'].values
        stds = valid['std'].values
        
        total_count = N.sum()
        total_missing = valid['missing'].sum()
        global_min = valid['min'].min()
        global_max = valid['max'].max()
        global_mean = np.average(means, weights=N)
        
        if total_count > 1:
            v_intra = np.sum((N - 1) * (stds ** 2))
            v_inter = np.sum(N * ((means - global_mean) ** 2))
            global_var = (v_intra + v_inter) / (total_count - 1)
            global_std = np.sqrt(global_var)
        else:
            global_std = np.nan
            
        global_rows.append({
            'feature': feature, 'count': total_count, 'missing': total_missing,
            'min': global_min, 'p001': np.nan, 'p01': np.nan, 'p05': np.nan,
            'median': np.nan, 'p95': np.nan, 'p99': np.nan, 'p999': np.nan,
            'max': global_max, 'mean': global_mean, 'std': global_std
        })
        
    return pd.DataFrame(global_rows)

def compute_smart_thresholds(by_lp_df: pd.DataFrame) -> pd.DataFrame:
    thresholds = []
    
    two_sided_features = [
        'char_len_ratio', 'word_len_ratio', 
        'score_lcs_ratio', 'score_levenshtein',
        'src_char_len', 'trg_char_len', 
        'src_word_len', 'trg_word_len'
    ]
    
    lower_bound_features = [
        'score_numerals', 'score_term_punct'
    ]
    
    upper_bound_features = [
        'score_repeat_src', 'score_repeat_trg',
        'src_max_word_len', 'trg_max_word_len',
        'src_avg_word_len', 'trg_avg_word_len'
    ]

    for (lp, feat), row in by_lp_df.set_index(['langpair', 'feature']).iterrows():
        lower, upper = np.nan, np.nan
        
        # 1. Calculate Local Statistical Bounds
        p001, p999 = row['p001'], row['p999']
        l_mean, l_std = row['mean'], row['std']
        
        sigma_lower = l_mean - 4 * l_std if pd.notna(l_std) else p001
        sigma_upper = l_mean + 4 * l_std if pd.notna(l_std) else p999
        
        stat_lower = min(p001, sigma_lower)
        stat_upper = max(p999, sigma_upper)

        # 2. Apply Absolute Linguistic Safeguards
        safe_l = ABSOLUTE_SAFEGUARDS.get(feat, {}).get('lower', 0.0)
        safe_u = ABSOLUTE_SAFEGUARDS.get(feat, {}).get('upper', float('inf'))

        if feat in two_sided_features:
            # We use `min` for lower bounds to take the *widest* (most forgiving) limit
            lower = min(stat_lower, safe_l)
            # We use `max` for upper bounds to take the *highest* limit
            upper = max(stat_upper, safe_u)
            
        elif feat in lower_bound_features:
            lower = min(stat_lower, safe_l)
            
        elif feat in upper_bound_features:
            upper = max(stat_upper, safe_u)

        if pd.notna(lower) or pd.notna(upper):
            thresholds.append({
                'langpair': lp, 
                'feature': feat, 
                'threshold_lower': lower, 
                'threshold_upper': upper
            })
            
    # 3. Pivot and Format
    thresh_df = pd.DataFrame(thresholds)
    
    if not thresh_df.empty:
        thresh_df = thresh_df.pivot(
            index='langpair', 
            columns='feature', 
            values=['threshold_lower', 'threshold_upper']
        )
        
        thresh_df.columns = [
            f"{col[1]}_{col[0].replace('threshold_', '')}" 
            for col in thresh_df.columns
        ]
        
        thresh_df.reset_index(inplace=True)
        thresh_df.dropna(axis=1, how='all', inplace=True)
        
    return thresh_df

# ----------------------------
# Main Execution
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Memory-Efficient EDA for FineOPUS.")
    parser.add_argument("--data_root", type=Path, required=True, help="Root directory containing language-pair subfolders.")
    parser.add_argument("--out_dir", type=Path, default=Path("eda_outputs"), help="Output directory for EDA results.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    by_lp_path = args.out_dir / "summary_by_langpair.csv"
    bool_by_lp_path = args.out_dir / "bool_rates_by_langpair.csv"
    corr_path = args.out_dir / "spearman_corr_all.csv"

    if not args.data_root.exists():
        raise RuntimeError(f"Data root does not exist: {args.data_root}")

    langpairs = sorted([d for d in args.data_root.iterdir() if d.is_dir()])
    if not langpairs:
        raise RuntimeError(f"No language pair folders found under: {args.data_root}")

    # --- Checkpoint / Resume Logic ---
    processed_lps = set()
    if by_lp_path.exists():
        try:
            existing_df = pd.read_csv(by_lp_path, usecols=['langpair'])
            processed_lps = set(existing_df['langpair'].unique())
            logging.info(f"Found {len(processed_lps)} already processed language pairs. Resuming...")
        except Exception: pass

    logging.info(f"Discovered {len(langpairs)} total language pairs. Processing sequentially...")
    cols_to_load = NUMERIC_COLS + BOOL_COLS
    
    # --- Sequentially Process and Append ---
    for lp_dir in langpairs:
        lp_name = lp_dir.name
        if lp_name in processed_lps:
            continue
            
        parquet_files = sorted(lp_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        # 1. Fast Metadata Scan for Total Rows
        total_rows = 0
        try:
            for p_file in parquet_files:
                total_rows += pq.ParquetFile(p_file).metadata.num_rows
        except Exception as e:
            logging.error(f"Failed to read metadata for {lp_name}: {e}")
            continue

        if total_rows == 0:
            continue

        # Calculate dynamic sampling fraction to prevent memory overflow
        sample_frac = min(1.0, MAX_SAMPLE_SIZE / total_rows)
        logging.info(f"Loading {lp_name}... Total Rows: {total_rows}. Sampling Fraction: {sample_frac:.4f}")

        # 2. Stream and Downsample Data
        sampled_chunks = []
        try:
            for p_file in parquet_files:
                pf = pq.ParquetFile(p_file)
                # Stream in chunks of 100,000 rows
                for batch in pf.iter_batches(batch_size=100_000, columns=[c for c in cols_to_load if c in pf.schema.names]):
                    batch_df = batch.to_pandas()
                    if sample_frac < 1.0:
                        sampled_chunks.append(batch_df.sample(frac=sample_frac, random_state=42))
                    else:
                        sampled_chunks.append(batch_df)
        except Exception as e:
            logging.warning(f"Could not load data chunks for {lp_name}: {e}")
            continue
            
        if not sampled_chunks:
            continue

        df = pd.concat(sampled_chunks, ignore_index=True)

        # 3. Compute & Append Per-LP Stats
        lp_stats = generate_summary_stats(df, NUMERIC_COLS, exact_total_rows=total_rows)
        if not lp_stats.empty:
            lp_stats.insert(0, 'langpair', lp_name)
            lp_stats.to_csv(by_lp_path, mode='a', header=not by_lp_path.exists(), index=False)
            
        # 4. Compute & Append Boolean Rates
        available_bools = [c for c in BOOL_COLS if c in df.columns]
        if available_bools:
            bool_rates = df[available_bools].mean().reset_index()
            bool_rates.columns = ['feature', 'true_rate']
            bool_rates.insert(0, 'langpair', lp_name)
            bool_rates.to_csv(bool_by_lp_path, mode='a', header=not bool_by_lp_path.exists(), index=False)

        # 5. Compute & Append Correlations
        numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
        if numeric_present:
            corr = df[numeric_present].corr(method="spearman").reset_index()
            corr.rename(columns={'index': 'feature_1'}, inplace=True)
            corr_melted = corr.melt(id_vars='feature_1', var_name='feature_2', value_name='spearman_corr')
            corr_melted.insert(0, 'langpair', lp_name)
            corr_melted.to_csv(corr_path, mode='a', header=not corr_path.exists(), index=False)

        del df, sampled_chunks # Explicitly free RAM immediately

    # --- Global Aggregations ---
    if not by_lp_path.exists():
        raise RuntimeError("No stats were generated. Check your data.")

    logging.info("All pairs processed. Loading full stats for global aggregation...")
    full_by_lp_df = pd.read_csv(by_lp_path)

    # global_df = aggregate_global_stats(full_by_lp_df)
    # global_df.to_csv(args.out_dir / "summary_global.csv", index=False)
    
    thresholds_df = compute_smart_thresholds(full_by_lp_df)
    thresholds_df.to_csv(args.out_dir / "filtering_thresholds.csv", index=False)

    if bool_by_lp_path.exists():
        full_bool_df = pd.read_csv(bool_by_lp_path)
        bool_agg = pd.merge(full_bool_df, full_by_lp_df[full_by_lp_df['feature'] == NUMERIC_COLS[0]][['langpair', 'count']], on='langpair')
        global_bools = [{'feature': f, 'true_rate': np.average(g['true_rate'], weights=g['count'])} for f, g in bool_agg.groupby('feature')]
        pd.DataFrame(global_bools).to_csv(args.out_dir / "bool_rates_global.csv", index=False)

    logging.info("EDA finished successfully.")

if __name__ == "__main__":
    main()