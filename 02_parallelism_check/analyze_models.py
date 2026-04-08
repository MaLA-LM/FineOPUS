#!/usr/bin/env python3

import argparse
import pandas as pd
import glob
import os

def load_dataset(results_dir, prefix):
    csv_files = glob.glob(os.path.join(results_dir, f"{prefix}_*.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("best_model")
                 and not os.path.basename(f).startswith("model_summary")]
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['source_lang'] != df['target_lang']]
    return df


def parse_time_to_seconds(time_str):
    """Convert time string 'HH:MM:SS' to total seconds."""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return float('inf')  # Invalid format -> slowest
    except (ValueError, AttributeError):
        return float('inf')  # Invalid format -> slowest


# Model processing time mapping (time to process all language pairs)
# Format: "HH:MM:SS" - lower is faster
MODEL_TIME_SECONDS = {
    'microsoft/harrier-oss-v1-0.6b': 61285,      # 17:01:25
    'microsoft/harrier-oss-v1-270m': 40225,      # 11:10:25
    'intfloat/multilingual-e5-large': 34124,     # 09:28:44
    'intfloat/multilingual-e5-small': 16325,   # 04:32:05
    'Alibaba-NLP/gte-multilingual-base': 18914,  # 05:15:14
    'jinaai/jina-embeddings-v3': 87045,          # 24:10:45
    'codefuse-ai/F2LLM-v2-0.6B': 60461,          # 16:47:41
    'jinaai/jina-embeddings-v5-text-small': 130901,  # 36:21:41
    'codefuse-ai/F2LLM-v2-330M': 35646,          # 09:54:06
    'jinaai/jina-embeddings-v5-text-nano': 55902,  # 15:31:42
    'codefuse-ai/F2LLM-v2-160M': 21806,          # 06:03:26
    'codefuse-ai/F2LLM-v2-80M': 19634,           # 05:27:14
    'google/embeddinggemma-300m': 62679,           # 17:24:39
    'Qwen/Qwen3-Embedding-0.6B': 83035,           # 23:03:55
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude-models', nargs='+', default=[],
                        help='Model names (or substrings) to exclude. '
                             'A model is excluded if any given substring appears in its name.')
    args = parser.parse_args()

    results_dir = "/scratch/project_462000941/members/zihao/OPUS2410/02_parallelism_check/results"

    flores_df = load_dataset(results_dir, "FLORES-200")
    bouquet_df = load_dataset(results_dir, "BOUQuET_Sentence")

    if args.exclude_models:
        def should_exclude(model_name):
            return any(pat in model_name for pat in args.exclude_models)
        excluded_flores = flores_df['model'].apply(should_exclude)
        excluded_bouquet = bouquet_df['model'].apply(should_exclude)
        print(f"Excluding models matching: {args.exclude_models}")
        print(f"  Removed {excluded_flores.sum()} rows from FLORES-200")
        print(f"  Removed {excluded_bouquet.sum()} rows from BOUQuET_Sentence")
        flores_df = flores_df[~excluded_flores]
        bouquet_df = bouquet_df[~excluded_bouquet]
        print(f"  Remaining models: {sorted(flores_df['model'].unique())}")
        print()

    print(f"FLORES-200:       {len(flores_df)} rows")
    print(f"BOUQuET_Sentence: {len(bouquet_df)} rows")

    flores_df['dataset'] = 'FLORES-200'
    bouquet_df['dataset'] = 'BOUQuET_Sentence'

    merge_keys = ['model', 'source_lang', 'target_lang']
    both = pd.merge(flores_df[merge_keys + ['MRR', 'avg_rank']],
                    bouquet_df[merge_keys + ['MRR', 'avg_rank']],
                    on=merge_keys, suffixes=('_flores', '_bouquet'), how='outer')

    both['MRR'] = both[['MRR_flores', 'MRR_bouquet']].mean(axis=1)
    both['avg_rank'] = both[['avg_rank_flores', 'avg_rank_bouquet']].mean(axis=1)

    n_both = both['MRR_flores'].notna() & both['MRR_bouquet'].notna()
    n_flores_only = both['MRR_flores'].notna() & both['MRR_bouquet'].isna()
    n_bouquet_only = both['MRR_flores'].isna() & both['MRR_bouquet'].notna()
    print(f"\nLanguage-pair × model combinations:")
    print(f"  In both datasets (averaged): {n_both.sum()}")
    print(f"  FLORES-200 only:             {n_flores_only.sum()}")
    print(f"  BOUQuET_Sentence only:       {n_bouquet_only.sum()}")
    print(f"  Total:                       {len(both)}")

    combined_df = both[merge_keys + ['MRR', 'avg_rank']].copy()
    combined_df['lang_pair'] = combined_df['source_lang'] + ' -> ' + combined_df['target_lang']
    
    print("=" * 80)
    print("1. Average MRR for each model (sorted by MRR in descending order)")
    print("=" * 80)
    
    model_avg_mrr = combined_df.groupby('model')['MRR'].mean().sort_values(ascending=False)
    for model, mrr in model_avg_mrr.items():
        print(f"  {model:50s}: {mrr:.6f}")
    
    print(f"\nBest model (by average MRR): {model_avg_mrr.index[0]}")
    print(f"Best average MRR: {model_avg_mrr.iloc[0]:.6f}")
    
    print("\n")
    print("=" * 80)
    print("2. Average rank for each model (sorted by avg_rank in ascending order, lower is better)")
    print("=" * 80)
    
    model_avg_rank = combined_df.groupby('model')['avg_rank'].mean().sort_values(ascending=True)
    for model, rank in model_avg_rank.items():
        print(f"  {model:50s}: {rank:.4f}")
    
    print(f"\nBest model (by average avg_rank): {model_avg_rank.index[0]}")
    print(f"Best average rank: {model_avg_rank.iloc[0]:.4f}")
    
    print("\n")
    print("=" * 80)
    print("3. Best model for each language pair (by MRR)")
    print("=" * 80)
    
    # Add a column to indicate if model is ensemble (0 for non-ensemble, 1 for ensemble)
    # When MRR is the same, prefer non-ensemble models
    combined_df['is_ensemble'] = combined_df['model'].str.startswith('ensemble_').astype(int)

    # Add processing time column (lower is faster)
    combined_df['processing_time'] = combined_df['model'].map(MODEL_TIME_SECONDS).fillna(float('inf'))

    # Sort by: lang_pair, MRR descending, is_ensemble ascending (non-ensemble first),
    # then processing_time ascending (faster model first when tied on MRR)
    combined_df_sorted = combined_df.sort_values(
        ['lang_pair', 'MRR', 'is_ensemble', 'processing_time'],
        ascending=[True, False, True, True]
    )
    # Take the first (best) row for each language pair
    best_per_pair = combined_df_sorted.groupby('lang_pair').first().reset_index()
    best_per_pair = best_per_pair[['source_lang', 'target_lang', 'model', 'MRR']].sort_values('MRR', ascending=False)
    
    model_wins = best_per_pair['model'].value_counts()
    print("\nModel wins statistics (how many language pairs it is the best on):")
    for model, wins in model_wins.items():
        total_pairs = len(best_per_pair)
        pct = wins / total_pairs * 100
        print(f"  {model:50s}: {wins:5d} / {total_pairs} ({pct:.1f}%)")

    output_file = os.path.join(results_dir, "best_model_per_lang_pair_by_flores_bouquet_combined_selected_models.csv")
    best_per_pair.to_csv(output_file, index=False)
    print(f"\nBest model for each language pair has been saved to: {output_file}")
    
    print("\n")
    print("=" * 80)
    print("4. Top 20 language pairs with highest MRR and their best models")
    print("=" * 80)
    top_20 = best_per_pair.head(20)
    for _, row in top_20.iterrows():
        print(f"  {row['source_lang']:12s} -> {row['target_lang']:12s}: {row['model']:45s} MRR={row['MRR']:.6f}")
    
    print("\n")
    print("=" * 80)
    print("5. Bottom 20 language pairs with lowest MRR and their best models (these are the hardest for all models to handle)")
    print("=" * 80)
    bottom_20 = best_per_pair.tail(20)
    for _, row in bottom_20.iterrows():
        print(f"  {row['source_lang']:12s} -> {row['target_lang']:12s}: {row['model']:45s} MRR={row['MRR']:.6f}")

    print("\n")
    print("=" * 80)
    print("6. Oracle average MRR comparison (with vs without ensemble models)")
    print("=" * 80)
    
    # With ensemble: best model for each language pair (already computed above)
    # best_per_pair already has the best model per pair (preferring non-ensemble when tied)
    # But here we want the absolute best including ensemble
    best_with_ensemble = combined_df.loc[combined_df.groupby('lang_pair')['MRR'].idxmax()]
    oracle_mrr_with_ensemble = best_with_ensemble['MRR'].mean()
    
    # Without ensemble: only consider non-ensemble models
    non_ensemble_df = combined_df[combined_df['is_ensemble'] == 0]
    best_without_ensemble = non_ensemble_df.loc[non_ensemble_df.groupby('lang_pair')['MRR'].idxmax()]
    oracle_mrr_without_ensemble = best_without_ensemble['MRR'].mean()
    
    print(f"\n  Oracle avg MRR (with ensemble models):    {oracle_mrr_with_ensemble:.6f}")
    print(f"  Oracle avg MRR (without ensemble models): {oracle_mrr_without_ensemble:.6f}")
    print(f"  Improvement from ensemble:                {oracle_mrr_with_ensemble - oracle_mrr_without_ensemble:.6f} ({(oracle_mrr_with_ensemble - oracle_mrr_without_ensemble) / oracle_mrr_without_ensemble * 100:.2f}%)")

    summary_file = os.path.join(results_dir, "model_summary_by_flores_bouquet_combined_selected_models.csv")
    summary_df = pd.DataFrame({
        'model': model_avg_mrr.index,
        'avg_MRR': model_avg_mrr.values,
        'avg_rank': [model_avg_rank[m] for m in model_avg_mrr.index],
        'num_best_pairs': [model_wins.get(m, 0) for m in model_avg_mrr.index],
        'time_seconds': [MODEL_TIME_SECONDS.get(m, float('inf')) for m in model_avg_mrr.index]
    })
    summary_df.to_csv(summary_file, index=False)
    print(f"\nModel summary has been saved to: {summary_file}")

if __name__ == "__main__":
    main()

