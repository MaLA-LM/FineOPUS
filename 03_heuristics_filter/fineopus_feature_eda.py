#!/usr/bin/env python3

"""

Proper EDA for FineOPUS precomputed filter features.



This script:

- samples multiple language-pairs

- reads one shard per language-pair

- computes:

  (1) global summary stats

  (2) per-language-pair summary stats

  (3) boolean flag rates (global + per-language-pair)

  (4) correlation matrix (Spearman)

- produces plots:

  (1) histograms for key numeric features

  (2) boxplots across language pairs for key features

  (3) correlation heatmap



Usage (Mahti):

  module load python-data



  python fineopus_feature_eda_proper.py \

    --data_root /projappl/project_2008161/FineOPUS/deduplicated_filter_precompute \

    --out_dir eda_outputs \

    --num_langpairs 10 \

    --rows_per_langpair 20000 \

    --plot

"""



import os

import glob

import argparse

import numpy as np

import pandas as pd

import pyarrow.parquet as pq

import matplotlib.pyplot as plt





# ----------------------------

# Features to analyze

# ----------------------------

NUMERIC_COLS = [

    # LID confidence

    "src_predlang_conf_glotlid", "tgt_predlang_conf_glotlid",

    "src_predlang_conf_conlid", "tgt_predlang_conf_conlid",



    # Length features

    "src_char_len", "trg_char_len",

    "src_word_len", "trg_word_len",

    "src_max_word_len", "trg_max_word_len",

    "src_avg_word_len", "trg_avg_word_len",

    "char_len_ratio", "word_len_ratio",



    # Heuristic scores

    "score_term_punct",

    "score_numerals",

    "score_lcs_ratio",

    "score_levenshtein",

    "score_repeat_src",

    "score_repeat_trg",

]



BOOL_COLS = [

    "filter_html_src",

    "filter_html_trg",

    "filter_regex_src",

]



PLOT_HIST_COLS = [

    "src_predlang_conf_glotlid",

    "tgt_predlang_conf_glotlid",

    "char_len_ratio",

    "word_len_ratio",

    "score_numerals",

    "score_term_punct",

    "score_lcs_ratio",

    "score_levenshtein",

    "score_repeat_src",

    "score_repeat_trg",

]



PLOT_BOXPLOT_COLS = [

    "src_predlang_conf_glotlid",

    "char_len_ratio",

    "word_len_ratio",

    "score_numerals",

    "score_levenshtein",

    "score_repeat_src",

]





def list_langpairs(data_root: str):

    return sorted([

        d for d in os.listdir(data_root)

        if os.path.isdir(os.path.join(data_root, d))

    ])





def pick_first_parquet(langpair_dir: str):

    files = sorted(glob.glob(os.path.join(langpair_dir, "*.parquet")))

    return files[0] if files else None





def read_sample(file_path: str, cols, nrows: int, seed: int):

    pf = pq.ParquetFile(file_path)

    available = set(pf.schema.names)

    use_cols = [c for c in cols if c in available]



    table = pf.read(columns=use_cols)

    df = table.to_pandas()



    if len(df) > nrows:

        df = df.sample(n=nrows, random_state=seed)



    return df





def summary_stats(series: pd.Series):

    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    valid = np.sum(~np.isnan(x))

    if valid == 0:

        return {

            "count": 0, "missing": int(np.sum(np.isnan(x))),

            "min": np.nan, "p01": np.nan, "p05": np.nan,

            "median": np.nan, "p95": np.nan, "p99": np.nan,

            "max": np.nan, "mean": np.nan, "std": np.nan

        }



    return {

        "count": int(valid),

        "missing": int(np.sum(np.isnan(x))),

        "min": float(np.nanmin(x)),

        "p01": float(np.nanpercentile(x, 1)),

        "p05": float(np.nanpercentile(x, 5)),

        "median": float(np.nanpercentile(x, 50)),

        "p95": float(np.nanpercentile(x, 95)),

        "p99": float(np.nanpercentile(x, 99)),

        "max": float(np.nanmax(x)),

        "mean": float(np.nanmean(x)),

        "std": float(np.nanstd(x)),

    }





def main():

    parser = argparse.ArgumentParser(description="Proper EDA for FineOPUS filter features.")

    parser.add_argument("--data_root", required=True, help="Root directory containing language-pair subfolders.")

    parser.add_argument("--out_dir", default="eda_outputs", help="Output directory for EDA results.")

    parser.add_argument("--num_langpairs", type=int, default=10, help="How many language pairs to analyze.")

    parser.add_argument("--rows_per_langpair", type=int, default=20000, help="Rows sampled per language pair.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")

    parser.add_argument("--plot", action="store_true", help="Generate plots.")

    args = parser.parse_args()



    os.makedirs(args.out_dir, exist_ok=True)

    plot_dir = os.path.join(args.out_dir, "plots")

    os.makedirs(plot_dir, exist_ok=True)



    langpairs = list_langpairs(args.data_root)

    if not langpairs:

        raise RuntimeError(f"No language pair folders found under: {args.data_root}")



    selected = langpairs[:args.num_langpairs]

    print("Selected language pairs:", selected)



    all_rows = []

    for lp in selected:

        lp_dir = os.path.join(args.data_root, lp)

        parquet_file = pick_first_parquet(lp_dir)

        if parquet_file is None:

            print(f"[WARN] No parquet found for {lp}")

            continue



        print(f"Loading {lp}: {os.path.basename(parquet_file)}")

        df = read_sample(parquet_file, NUMERIC_COLS + BOOL_COLS, args.rows_per_langpair, args.seed)

        df["langpair"] = lp

        all_rows.append(df)



    if not all_rows:

        raise RuntimeError("No data loaded. Check data_root and folder structure.")



    data = pd.concat(all_rows, ignore_index=True)

    print("Total sampled rows:", len(data))



    # ----------------------------

    # Global summary

    # ----------------------------

    global_summary = []

    for col in NUMERIC_COLS:

        if col not in data.columns:

            continue

        stats = summary_stats(data[col])

        stats["feature"] = col

        global_summary.append(stats)



    global_df = pd.DataFrame(global_summary)[

        ["feature", "count", "missing", "min", "p01", "p05", "median", "p95", "p99", "max", "mean", "std"]

    ].sort_values("feature")



    global_path = os.path.join(args.out_dir, "summary_global.csv")

    global_df.to_csv(global_path, index=False)

    print("Saved:", global_path)



    # ----------------------------

    # Per-language-pair summary

    # ----------------------------

    rows = []

    for lp, group in data.groupby("langpair"):

        for col in NUMERIC_COLS:

            if col not in group.columns:

                continue

            stats = summary_stats(group[col])

            stats["langpair"] = lp

            stats["feature"] = col

            rows.append(stats)



    by_lp_df = pd.DataFrame(rows)[

        ["langpair", "feature", "count", "missing", "min", "p01", "p05", "median", "p95", "p99", "max", "mean", "std"]

    ].sort_values(["feature", "langpair"])



    by_lp_path = os.path.join(args.out_dir, "summary_by_langpair.csv")

    by_lp_df.to_csv(by_lp_path, index=False)

    print("Saved:", by_lp_path)



    # ----------------------------

    # Boolean rates (global + per-langpair)

    # ----------------------------

    bool_global = []

    for col in BOOL_COLS:

        if col not in data.columns:

            continue

        rate = float(pd.Series(data[col]).astype("float").mean())

        bool_global.append({"feature": col, "true_rate": rate})



    bool_global_df = pd.DataFrame(bool_global)

    bool_global_path = os.path.join(args.out_dir, "bool_rates_global.csv")

    bool_global_df.to_csv(bool_global_path, index=False)

    print("Saved:", bool_global_path)



    bool_rows = []

    for lp, group in data.groupby("langpair"):

        for col in BOOL_COLS:

            if col not in group.columns:

                continue

            rate = float(pd.Series(group[col]).astype("float").mean())

            bool_rows.append({"langpair": lp, "feature": col, "true_rate": rate})



    bool_by_lp_df = pd.DataFrame(bool_rows).sort_values(["feature", "langpair"])

    bool_by_lp_path = os.path.join(args.out_dir, "bool_rates_by_langpair.csv")

    bool_by_lp_df.to_csv(bool_by_lp_path, index=False)

    print("Saved:", bool_by_lp_path)



    # ----------------------------

    # Correlation matrix

    # ----------------------------

    numeric_present = [c for c in NUMERIC_COLS if c in data.columns]

    corr = data[numeric_present].corr(method="spearman")

    corr_path = os.path.join(args.out_dir, "spearman_corr.csv")

    corr.to_csv(corr_path)

    print("Saved:", corr_path)



    # ----------------------------

    # Plots

    # ----------------------------

    if args.plot:

        # histograms

        for col in PLOT_HIST_COLS:

            if col not in data.columns:

                continue

            x = pd.to_numeric(data[col], errors="coerce").dropna()

            if len(x) == 0:

                continue

            plt.figure()

            plt.hist(x, bins=80)

            plt.title(f"Histogram: {col}")

            plt.xlabel(col)

            plt.ylabel("count")

            out = os.path.join(plot_dir, f"hist_{col}.png")

            plt.savefig(out, dpi=200)

            plt.close()



        # correlation heatmap

        plt.figure(figsize=(12, 10))

        plt.imshow(corr.values, aspect="auto")

        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)

        plt.yticks(range(len(corr.index)), corr.index, fontsize=6)

        plt.title("Spearman correlation heatmap")

        plt.colorbar()

        out = os.path.join(plot_dir, "spearman_corr_heatmap.png")

        plt.tight_layout()

        plt.savefig(out, dpi=250)

        plt.close()



        # boxplots per language pair

        # keep at most 20 langpairs in boxplot for readability

        lp_order = selected[:min(len(selected), 20)]

        for col in PLOT_BOXPLOT_COLS:

            if col not in data.columns:

                continue

            sub = data[data["langpair"].isin(lp_order)][["langpair", col]].copy()

            sub[col] = pd.to_numeric(sub[col], errors="coerce")

            sub = sub.dropna()

            if len(sub) == 0:

                continue



            plt.figure(figsize=(12, 5))

            sub.boxplot(column=col, by="langpair", rot=90)

            plt.title(f"Boxplot: {col} (by langpair)")

            plt.suptitle("")

            plt.xlabel("langpair")

            plt.ylabel(col)

            out = os.path.join(plot_dir, f"box_{col}_by_langpair.png")

            plt.tight_layout()

            plt.savefig(out, dpi=250)

            plt.close()



    print("EDA finished successfully.")

    print(f"Results saved under: {args.out_dir}")





if __name__ == "__main__":

    main()


