#!/usr/bin/env python3

"""

Hybrid FineOPUS filtering pipeline



Stage 1 (deterministic): remove extreme/noisy pairs using precomputed features.

Stage 2 (optional, unsupervised): IsolationForest on numeric features; remove a tiny fraction.



Input structure:

  data_root/

    langpairA/*.parquet

    langpairB/*.parquet

    ...



Outputs:

  out_root/

    langpairA/*.parquet

    langpairB/*.parquet

    filtering_log.csv



Designed for large datasets: processes Parquet in batches.

"""



import os

import glob

import argparse

import numpy as np

import pandas as pd

import pyarrow as pa

import pyarrow.parquet as pq

import pyarrow.compute as pc



from sklearn.ensemble import IsolationForest





# -----------------------------

# Helpers for Arrow types

# -----------------------------

def _combine(arr):

    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr



def _as_bool(table, col):

    arr = _combine(table[col])

    return pc.cast(arr, pa.bool_())



def _as_float(table, col):

    arr = _combine(table[col])

    return pc.cast(arr, pa.float64())





# -----------------------------

# Stage 1: Deterministic rules

# -----------------------------

def stage1_keep_mask(table: pa.Table, args) -> pa.Array:

    """

    Return a boolean mask array (length = table.num_rows): True means KEEP.

    Robust to ChunkedArray columns.

    """

    cols = set(table.schema.names)

    n = table.num_rows



    # Start with all True (BooleanArray, not scalar/expression)

    keep = pa.array([True] * n, type=pa.bool_())



    # HTML flags

    if args.drop_html:

        if "filter_html_src" in cols:

            keep = pc.and_(keep, pc.invert(_as_bool(table, "filter_html_src")))

        if "filter_html_trg" in cols:

            keep = pc.and_(keep, pc.invert(_as_bool(table, "filter_html_trg")))



    # Regex flag

    if args.drop_regex and "filter_regex_src" in cols:

        keep = pc.and_(keep, pc.invert(_as_bool(table, "filter_regex_src")))



    # Repetition thresholds

    if "score_repeat_src" in cols:

        keep = pc.and_(keep, pc.less_equal(_as_float(table, "score_repeat_src"), pa.scalar(float(args.max_repeat))))

    if "score_repeat_trg" in cols:

        keep = pc.and_(keep, pc.less_equal(_as_float(table, "score_repeat_trg"), pa.scalar(float(args.max_repeat))))



    # Length ratio thresholds

    if "char_len_ratio" in cols:

        keep = pc.and_(keep, pc.less_equal(_as_float(table, "char_len_ratio"), pa.scalar(float(args.max_char_ratio))))

    if "word_len_ratio" in cols:

        keep = pc.and_(keep, pc.less_equal(_as_float(table, "word_len_ratio"), pa.scalar(float(args.max_word_ratio))))



    # LID confidence thresholds

    if args.min_lid_conf is not None:

        thr = pa.scalar(float(args.min_lid_conf))

        if "src_predlang_conf_glotlid" in cols:

            keep = pc.and_(keep, pc.greater_equal(_as_float(table, "src_predlang_conf_glotlid"), thr))

        if "tgt_predlang_conf_glotlid" in cols:

            keep = pc.and_(keep, pc.greater_equal(_as_float(table, "tgt_predlang_conf_glotlid"), thr))



    return keep





def stage1_counts(table: pa.Table, keep_mask: pa.Array):

    n = table.num_rows

    kept = int(pc.sum(pc.cast(keep_mask, pa.int64())).as_py())

    return n, kept, n - kept





# ------------------------------------

# Stage 2: Unsupervised (conservative)

# ------------------------------------

DEFAULT_STAGE2_FEATURES = [

    "src_char_len", "trg_char_len",

    "src_word_len", "trg_word_len",

    "char_len_ratio", "word_len_ratio",

    "score_term_punct", "score_numerals",

    "score_lcs_ratio", "score_levenshtein",

    "score_repeat_src", "score_repeat_trg",

    "src_predlang_conf_glotlid", "tgt_predlang_conf_glotlid",

]



def stage2_isoforest_keep(df: pd.DataFrame, feature_cols, contamination: float):

    """

    Fit IsolationForest; return keep_mask (bool np array) and scores.

    Keep predicted normal points; remove ~contamination fraction.

    """

    X = df[feature_cols].copy()

    for c in feature_cols:

        X[c] = pd.to_numeric(X[c], errors="coerce")



    # conservative missing handling: median imputation

    X = X.fillna(X.median(numeric_only=True))



    model = IsolationForest(

        n_estimators=200,

        contamination=contamination,

        random_state=42,

        n_jobs=-1,

    )

    model.fit(X)



    pred = model.predict(X)                 # 1 normal, -1 anomaly

    scores = model.decision_function(X)     # higher = more normal

    keep = (pred == 1)

    return keep, scores





# -----------------------------

# Process one langpair folder

# -----------------------------

def process_langpair(langpair_dir: str, out_dir: str, args):

    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(langpair_dir, "*.parquet")))

    if not files:

        return []



    logs = []



    for fp in files:

        fname = os.path.basename(fp)

        out_fp = os.path.join(out_dir, fname)



        pf = pq.ParquetFile(fp)

        writer = None



        total_in = 0

        total_s1_kept = 0

        total_s2_kept = 0



        for batch in pf.iter_batches(batch_size=args.batch_size):

            table = pa.Table.from_batches([batch])



            # Stage 1

            keep1 = stage1_keep_mask(table, args)

            n, kept1, _ = stage1_counts(table, keep1)



            total_in += n

            total_s1_kept += kept1



            table1 = table.filter(keep1)

            if table1.num_rows == 0:

                continue



            # Stage 2 (optional)

            if args.stage2:

                present = [c for c in args.stage2_features if c in table1.schema.names]

                if len(present) >= 4 and table1.num_rows >= 20:

                    df_feat = table1.select(present).to_pandas()

                    keep2_np, _scores = stage2_isoforest_keep(df_feat, present, args.contamination)

                    table2 = table1.filter(pa.array(keep2_np))

                else:

                    table2 = table1

            else:

                table2 = table1



            total_s2_kept += table2.num_rows



            # Write out

            if writer is None:

                writer = pq.ParquetWriter(out_fp, table2.schema, compression="SNAPPY")

            writer.write_table(table2)



        if writer:

            writer.close()



        logs.append({

            "file": fname,

            "input_rows": total_in,

            "stage1_kept": total_s1_kept,

            "stage2_kept": total_s2_kept,

            "stage1_removed": total_in - total_s1_kept,

            "stage2_removed": total_s1_kept - total_s2_kept,

        })



    return logs





def main():

    ap = argparse.ArgumentParser("Hybrid filtering: deterministic + optional unsupervised")

    ap.add_argument("--data_root", required=True, help="Root folder containing langpair subfolders")

    ap.add_argument("--out_root", required=True, help="Output root folder")

    ap.add_argument("--langpairs", default=None, help="Comma-separated langpairs; if omitted, process ALL")

    ap.add_argument("--batch_size", type=int, default=50000)



    # Stage 1 thresholds

    ap.add_argument("--drop_html", action="store_true")

    ap.add_argument("--drop_regex", action="store_true")

    ap.add_argument("--max_repeat", type=float, default=2.0)

    ap.add_argument("--max_char_ratio", type=float, default=10.0)

    ap.add_argument("--max_word_ratio", type=float, default=5.0)

    ap.add_argument("--min_lid_conf", type=float, default=None)



    # Stage 2

    ap.add_argument("--stage2", action="store_true")

    ap.add_argument("--contamination", type=float, default=0.005, help="Fraction removed by stage2 (e.g., 0.005=0.5%)")

    ap.add_argument("--stage2_features", default=",".join(DEFAULT_STAGE2_FEATURES))



    args = ap.parse_args()

    args.stage2_features = [x.strip() for x in args.stage2_features.split(",") if x.strip()]



    # choose langpairs

    if args.langpairs:

        selected = [x.strip() for x in args.langpairs.split(",") if x.strip()]

    else:

        selected = sorted([

            d for d in os.listdir(args.data_root)

            if os.path.isdir(os.path.join(args.data_root, d))

        ])



    os.makedirs(args.out_root, exist_ok=True)



    all_logs = []

    for lp in selected:

        lp_dir = os.path.join(args.data_root, lp)

        if not os.path.isdir(lp_dir):

            continue

        out_dir = os.path.join(args.out_root, lp)

        print(f"[INFO] Processing {lp}")

        logs = process_langpair(lp_dir, out_dir, args)

        for r in logs:

            r["langpair"] = lp

        all_logs.extend(logs)



    if all_logs:

        df = pd.DataFrame(all_logs)

        log_path = os.path.join(args.out_root, "filtering_log.csv")

        df.to_csv(log_path, index=False)

        print("[INFO] Saved log:", log_path)



    print("[DONE]")





if __name__ == "__main__":

    main()


