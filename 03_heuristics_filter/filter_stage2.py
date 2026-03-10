#!/usr/bin/env python3
"""
Unsupervised FineOPUS filtering pipeline (Single Lang-Pair HPC Array Version).

This script:
1. Receives a single language pair via argument.
2. Checks a central CSV log to see if it has already been processed (Resume feature).
3. Pre-trains an IsolationForest on a random sample of the language pair.
4. Streams the Parquet files in batches, applying the model.
5. Logs errors to a dedicated error file.
6. Appends the final processing stats to the central CSV log upon completion.
"""

import os
import glob
import argparse
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest

DEFAULT_FEATURES = [
    "src_char_len", "trg_char_len", "src_word_len", "trg_word_len",
    "char_len_ratio", "word_len_ratio", "score_term_punct", "score_numerals",
    "score_lcs_ratio", "score_levenshtein", "score_repeat_src", "score_repeat_trg",
    "src_predlang_conf_glotlid", "tgt_predlang_conf_glotlid",
]

# ----------------------------
# Logging Setup
# ----------------------------
def setup_logging(error_log_path: str):
    """Configures dual-output logging: INFO to console, ERROR to dedicated file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler for Errors
    if error_log_path:
        os.makedirs(os.path.dirname(error_log_path) or '.', exist_ok=True)
        fh = logging.FileHandler(error_log_path, mode='a')
        fh.setLevel(logging.ERROR) # Only log ERROR and CRITICAL to this file
        fh.setFormatter(formatter)
        logger.addHandler(fh)

# ------------------------------------
# Unsupervised ML (Isolation Forest)
# ------------------------------------
def train_isolation_forest(langpair_dir: str, features: list, contamination: float, sample_size: int = 200000):
    files = glob.glob(os.path.join(langpair_dir, "*.parquet"))
    if not files:
        return None, None

    sample_dfs = []
    rows_collected = 0
    
    for fp in files:
        try:
            schema = pq.ParquetFile(fp).schema.names
            cols_to_read = [c for c in features if c in schema]
            if not cols_to_read:
                continue
                
            df = pq.read_table(fp, columns=cols_to_read).to_pandas()
            if not df.empty:
                sample_dfs.append(df)
                rows_collected += len(df)
            if rows_collected >= sample_size:
                break
        except Exception as e:
            logging.error(f"Error sampling {fp} for training: {e}")
            continue
            
    if not sample_dfs:
        return None, None

    train_data = pd.concat(sample_dfs, ignore_index=True)
    if len(train_data) > sample_size:
        train_data = train_data.sample(n=sample_size, random_state=42)

    for col in features:
        if col not in train_data.columns:
            train_data[col] = np.nan

    X = train_data[features].apply(pd.to_numeric, errors='coerce')
    feature_medians = X.median(numeric_only=True).fillna(0)
    X = X.fillna(feature_medians)

    logging.info(f"Training IsolationForest on {len(X)} sampled rows (contamination={contamination})...")
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X)

    return model, feature_medians

# -----------------------------
# Main Processing Pipeline
# -----------------------------
def process_single_langpair(langpair_dir: str, out_dir: str, args) -> tuple:
    """Processes the files and returns (original_rows, kept_rows)."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(langpair_dir, "*.parquet")))
    
    total_in_all = 0
    total_kept_all = 0

    if not files:
        logging.error(f"No parquet files found in {langpair_dir}.")
        return total_in_all, total_kept_all

    # 1. Pre-train Model
    model, feature_medians = train_isolation_forest(langpair_dir, args.features, args.contamination)
    if model is None:
        logging.error(f"Could not train model for {args.lang_pair}. No valid training data extracted.")
        return total_in_all, total_kept_all

    # 2. Process Files in Batches
    for fp in files:
        fname = os.path.basename(fp)
        out_fp = os.path.join(out_dir, fname)
        
        # Internal file-level checkpoint
        if os.path.exists(out_fp):
            try:
                # Quickly read the row count of the existing output file to keep stats accurate
                total_in_all += pq.ParquetFile(fp).metadata.num_rows
                total_kept_all += pq.ParquetFile(out_fp).metadata.num_rows
            except Exception:
                pass
            continue 

        try:
            pf = pq.ParquetFile(fp)
        except Exception as e:
            logging.error(f"Failed to read input file {fp}: {e}")
            continue

        writer = None
        file_in, file_kept = 0, 0

        for batch in pf.iter_batches(batch_size=args.batch_size):
            table = pa.Table.from_batches([batch])
            file_in += table.num_rows

            if table.num_rows == 0:
                continue

            try:
                df_batch = table.to_pandas()
                for col in args.features:
                    if col not in df_batch.columns:
                        df_batch[col] = np.nan
                        
                X_batch = df_batch[args.features].apply(pd.to_numeric, errors='coerce')
                X_batch = X_batch.fillna(feature_medians)
                
                preds = model.predict(X_batch)
                keep_mask = pa.array((preds == 1))
                
                filtered_table = table.filter(keep_mask)
                file_kept += filtered_table.num_rows

                if writer is None:
                    writer = pq.ParquetWriter(out_fp, filtered_table.schema, compression="SNAPPY")
                writer.write_table(filtered_table)
                
            except Exception as e:
                logging.error(f"Error processing batch in {fp}: {e}")
                continue

        if writer:
            writer.close()

        total_in_all += file_in
        total_kept_all += file_kept
        logging.info(f"[{args.lang_pair}] Filtered {fname}: In={file_in}, Kept={file_kept}")

    return total_in_all, total_kept_all

def main():
    ap = argparse.ArgumentParser("Unsupervised Parquet Filtering (Single Lang-Pair)")
    ap.add_argument("--lang_pair", required=True, help="Specific language pair to process (e.g., en-fi)")
    ap.add_argument("--data_root", required=True, help="Input root directory")
    ap.add_argument("--out_root", required=True, help="Output root directory")
    ap.add_argument("--log_csv", required=True, help="Path to central tracking CSV file")
    ap.add_argument("--error_log", required=True, help="Path to central error log text file")
    
    ap.add_argument("--batch_size", type=int, default=50000)
    ap.add_argument("--contamination", type=float, default=0.005)
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES))

    args = ap.parse_args()
    args.features = [x.strip() for x in args.features.split(",") if x.strip()]

    # Initialize logging
    setup_logging(args.error_log)
    logging.info(f"--- Booting job for {args.lang_pair} ---")

    # ----------------------------
    # Checkpoint / Resume Check
    # ----------------------------
    if os.path.exists(args.log_csv):
        try:
            completed_df = pd.read_csv(args.log_csv)
            if args.lang_pair in completed_df['langpair'].values:
                logging.info(f"Language pair '{args.lang_pair}' is already marked complete in {args.log_csv}. Exiting gracefully.")
                return
        except Exception as e:
            logging.error(f"Could not read tracking CSV {args.log_csv}: {e}. Proceeding with processing.")

    # ----------------------------
    # Execution
    # ----------------------------
    lp_dir = os.path.join(args.data_root, args.lang_pair)
    out_dir = os.path.join(args.out_root, args.lang_pair)
    
    if not os.path.isdir(lp_dir):
        logging.error(f"Input directory does not exist: {lp_dir}")
        return

    original_rows, kept_rows = process_single_langpair(lp_dir, out_dir, args)

    # ----------------------------
    # Update Tracking CSV
    # ----------------------------
    # We append a single row to the central CSV to mark completion
    try:
        new_row = pd.DataFrame([{
            'langpair': args.lang_pair,
            'original_rows': original_rows,
            'kept_rows': kept_rows
        }])
        
        # mode='a' appends to the file safely. If it doesn't exist, header=True writes the columns.
        file_exists = os.path.isfile(args.log_csv)
        new_row.to_csv(args.log_csv, mode='a', header=not file_exists, index=False)
        logging.info(f"[{args.lang_pair}] Completed and recorded. Original: {original_rows}, Kept: {kept_rows}")
        
    except Exception as e:
        logging.error(f"Failed to write completion status to {args.log_csv} for {args.lang_pair}: {e}")

if __name__ == "__main__":
    main()