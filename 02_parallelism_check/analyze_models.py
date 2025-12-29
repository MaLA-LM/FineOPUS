#!/usr/bin/env python3

import pandas as pd
import glob
import os

def main():
    results_dir = "/scratch/project_462000941/members/zihao/OPUS2410/02_parallelism_check/results"
    csv_files = glob.glob(os.path.join(results_dir, "FLORES-200_*.csv"))
    
    print(f"Found {len(csv_files)} model result files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    print()
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    combined_df['lang_pair'] = combined_df['source_lang'] + ' -> ' + combined_df['target_lang']
    
    combined_df = combined_df[combined_df['source_lang'] != combined_df['target_lang']]
    
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
    
    # Sort by MRR descending, then by is_ensemble ascending (non-ensemble first)
    combined_df_sorted = combined_df.sort_values(['lang_pair', 'MRR', 'is_ensemble'], 
                                                  ascending=[True, False, True])
    # Take the first (best) row for each language pair
    best_per_pair = combined_df_sorted.groupby('lang_pair').first().reset_index()
    best_per_pair = best_per_pair[['source_lang', 'target_lang', 'model', 'MRR']].sort_values('MRR', ascending=False)
    
    model_wins = best_per_pair['model'].value_counts()
    print("\nModel wins statistics (how many language pairs it is the best on):")
    for model, wins in model_wins.items():
        total_pairs = len(best_per_pair)
        pct = wins / total_pairs * 100
        print(f"  {model:50s}: {wins:5d} / {total_pairs} ({pct:.1f}%)")

    output_file = os.path.join(results_dir, "best_model_per_lang_pair.csv")
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

    summary_file = os.path.join(results_dir, "model_summary.csv")
    summary_df = pd.DataFrame({
        'model': model_avg_mrr.index,
        'avg_MRR': model_avg_mrr.values,
        'avg_rank': [model_avg_rank[m] for m in model_avg_mrr.index],
        'num_best_pairs': [model_wins.get(m, 0) for m in model_avg_mrr.index]
    })
    summary_df.to_csv(summary_file, index=False)
    print(f"\nModel summary has been saved to: {summary_file}")

if __name__ == "__main__":
    main()

