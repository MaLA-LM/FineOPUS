#!/usr/bin/env python3
"""
FineOPUS Parquet Filter - Stage 1: Heuristic Filtering (Single Lang-Pair)

This script:
- Relies on PyArrow's native C++ multithreading for fast I/O.
- Receives a single language pair via argument.
- Checks a central CSV log to see if it has already been processed (Resume feature).
- Loads per-language numeric thresholds.
- Dynamically detects boolean flags in the Parquet files and drops `True` instances.
- Filters out rows that violate the dynamic numeric thresholds.
- Appends the final processing stats to the central CSV log upon completion.
"""

import argparse
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------
# Logging Setup
# ----------------------------
def setup_logging(error_log_path: str):
    """Configures dual-output logging: INFO to console, ERROR to dedicated file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if error_log_path:
        os.makedirs(os.path.dirname(error_log_path) or '.', exist_ok=True)
        fh = logging.FileHandler(error_log_path, mode='a')
        fh.setLevel(logging.ERROR)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

# ----------------------------
# Core Logic
# ----------------------------
def load_filter_rules(thresholds_path: Path):
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")

    thresh_df = pd.read_csv(thresholds_path)
    thresholds_dict = thresh_df.set_index('langpair').to_dict(orient='index')

    return thresholds_dict

def filter_dataframe(df: pd.DataFrame, lp_thresholds: dict) -> tuple:
    initial_len = len(df)
    
    # 1. Discover Boolean Filters for HTML
    # We look for columns explicitly named with 'filter_html'
    bool_cols = [
        col for col in df.columns 
        if col.startswith('filter_html')
    ]
    
    # Apply Boolean Filters
    for b_col in bool_cols:
        # Drop rows where the boolean flag is True (filling NaNs with False to be safe)
        df = df[df[b_col].fillna(False) == False]
            
    # 2. Apply Numeric Thresholds
    for rule, bound in lp_thresholds.items():
        if pd.isna(bound):
            continue
            
        # Safely slice the last 6 characters to extract the base feature name
        if rule.endswith('_lower'):
            base_feature = rule[:-6]
            is_lower = True
            is_upper = False
        elif rule.endswith('_upper'):
            base_feature = rule[:-6]
            is_lower = False
            is_upper = True
        else:
            continue

        if base_feature in df.columns:
            # We keep rows that meet the threshold OR are missing the score entirely
            if is_lower:
                df = df[df[base_feature].isna() | (df[base_feature] >= bound)]
            if is_upper:
                df = df[df[base_feature].isna() | (df[base_feature] <= bound)]

    return df, initial_len

# ----------------------------
# Main Execution
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Filter Parquet files using EDA thresholds (Single Lang-Pair).")
    parser.add_argument("--lang_pair", required=True, help="Specific language pair to process (e.g., en-fi)")
    parser.add_argument("--data_root", type=Path, required=True, help="Root directory of raw Parquet files.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for cleaned Parquet files.")
    parser.add_argument("--thresholds_file", type=Path, required=True, help="Path to filtering_thresholds.csv")
    parser.add_argument("--log_csv", required=True, help="Path to central tracking CSV file")
    parser.add_argument("--error_log", required=True, help="Path to central error log text file")
    args = parser.parse_args()

    setup_logging(args.error_log)
    logging.info(f"--- Booting Stage 1 job for {args.lang_pair} ---")
    
    # Verify PyArrow is utilizing the CPU cores allocated by Slurm
    logging.info(f"PyArrow C++ Thread Pool Size: {pa.cpu_count()} threads")
    
    # 1. Checkpoint / Resume Check
    if os.path.exists(args.log_csv):
        try:
            completed_df = pd.read_csv(args.log_csv)
            if args.lang_pair in completed_df['langpair'].values:
                logging.info(f"Language pair '{args.lang_pair}' is already marked complete in {args.log_csv}. Exiting gracefully.")
                return
        except Exception as e:
            logging.error(f"Could not read tracking CSV {args.log_csv}: {e}. Proceeding.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Rules
    try:
        thresholds_dict = load_filter_rules(args.thresholds_file)
    except Exception as e:
        logging.error(f"Failed to load filter rules: {e}")
        return

    if args.lang_pair not in thresholds_dict:
        logging.error(f"No thresholds found for {args.lang_pair}. Exiting.")
        return

    lp_thresholds = thresholds_dict[args.lang_pair]

    # 3. Discover Data
    lp_dir = args.data_root / args.lang_pair
    if not lp_dir.is_dir():
        logging.error(f"Input directory does not exist: {lp_dir}")
        return

    out_lp_dir = args.out_dir / args.lang_pair
    out_lp_dir.mkdir(exist_ok=True)
    parquet_files = sorted(lp_dir.glob("*.parquet"))

    if not parquet_files:
        logging.error(f"No parquet files found in {lp_dir}.")
        return

    total_files_processed = 0
    total_rows_kept = 0
    total_rows_dropped = 0
    total_original = 0

    # 4. Process File-by-File
    for p_file in parquet_files:
        out_file = out_lp_dir / p_file.name
        
        # Internal file-level checkpoint
        if out_file.exists():
            try:
                # Quickly read metadata to keep stats accurate
                orig_rows = pq.ParquetFile(p_file).metadata.num_rows
                kept_rows = pq.ParquetFile(out_file).metadata.num_rows
                
                total_original += orig_rows
                total_rows_kept += kept_rows
                total_rows_dropped += (orig_rows - kept_rows)
                
                total_files_processed += 1
            except Exception:
                pass
            continue

        try:
            # Read using PyArrow engine for speed
            df = pd.read_parquet(p_file, engine='pyarrow')
        except Exception as e:
            logging.error(f"Failed to read {p_file}: {e}")
            continue

        if df.empty:
            df.to_parquet(out_file, engine='pyarrow', index=False)
            total_files_processed += 1
            continue

        # Apply filters
        clean_df, initial_len = filter_dataframe(df, lp_thresholds)
        final_len = len(clean_df)
        dropped = initial_len - final_len

        # Save the cleaned file
        clean_df.to_parquet(out_file, engine='pyarrow', index=False)

        total_files_processed += 1
        total_original += initial_len
        total_rows_kept += final_len
        total_rows_dropped += dropped
        
        logging.debug(f"[{args.lang_pair}] Processed {p_file.name}: Kept {final_len} / {initial_len} (Dropped {dropped})")

    logging.info(f"[{args.lang_pair}] Done. Files: {total_files_processed} | Original: {total_original} | Kept: {total_rows_kept} | Dropped: {total_rows_dropped}")

    # Calculate retention rate safely
    retention_rate = (total_rows_kept / total_original) if total_original > 0 else 0.0
    math_is_valid = (total_original - total_rows_kept) == total_rows_dropped
    
    # 5. Update Tracking CSV
    try:
        new_row = pd.DataFrame([{
            'langpair': args.lang_pair,
            'original_rows': total_original,
            'kept_rows': total_rows_kept,
            'dropped_rows': total_rows_dropped,
            'retention_rate': retention_rate,
            'math_check_passed': math_is_valid
        }])
        
        file_exists = os.path.isfile(args.log_csv)
        new_row.to_csv(args.log_csv, mode='a', header=not file_exists, index=False)
        logging.info(f"[{args.lang_pair}] Successfully recorded completion in {args.log_csv}.")
        
    except Exception as e:
        logging.error(f"Failed to write completion status to {args.log_csv} for {args.lang_pair}: {e}")

if __name__ == "__main__":
    main()