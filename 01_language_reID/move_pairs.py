#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import shutil
import sys
import pandas as pd
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    ap = argparse.ArgumentParser(
        description="Move language-pair folders whose total_rows < threshold based on an Excel stats file."
    )
    ap.add_argument("--excel_file", required=True,
                    help="Path to Excel file (e.g., mala-opus-dedup-2410-ReLID-ENSEMBLED-V2-stats.xlsx)")
    ap.add_argument("--source_dir", required=True,
                    help="Source root folder containing language-pair subfolders")
    ap.add_argument("--output_dir", required=True,
                    help="Destination root folder (will be created if missing)")
    ap.add_argument("--threshold", type=int, default=10000,
                    help="Move pairs with total_rows < threshold (default: 10000)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be moved, but do not actually move")
    args = ap.parse_args()

    logging.info("Arguments:")
    logging.info(f"  Excel File: {args.excel_file}")
    logging.info(f"  Source Directory: {args.source_dir}")
    logging.info(f"  Output Directory: {args.output_dir}")
    logging.info(f"  Threshold: {args.threshold}")
    logging.info(f"  Dry Run: {args.dry_run}")

    src = Path(args.source_dir).resolve()
    dst = Path(args.output_dir).resolve()
    if not src.exists():
        logging.error(f"[ERROR] Source not found: {src}")
        sys.exit(1)
    dst.mkdir(parents=True, exist_ok=True)

    # Read Excel
    try:
        df = pd.read_excel(args.excel_file)
    except Exception as e:
        logging.error(f"[ERROR] Failed to read Excel {args.excel_file}: {e}")
        sys.exit(1)

    cols = {c.strip().lower(): c for c in df.columns}
    lp_col = cols["language_pair"]
    tr_col = cols["total_rows"]

    df = df[[lp_col, tr_col]].dropna()
    df_low = df[df[tr_col] < args.threshold].copy()

    to_move = set(df_low[lp_col].astype(str).str.strip())

    # Actual subdirectories in the source directory
    candidates = [p for p in src.iterdir() if p.is_dir()]

    moved_rows = []
    skipped_rows = []

    # Execute (or dry-run) moving
    for p in sorted(candidates):
        name = p.name
        if name in to_move:
            src_path = p
            dst_path = dst / name
            action = "DRY-MOVE" if args.dry_run else "MOVE"
            # Ensure the parent directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            # If the target already exists, append a suffix to avoid overwriting
            final_dst = dst_path
            suffix_i = 1
            while final_dst.exists():
                final_dst = dst / f"{name}__dup{suffix_i}"
                suffix_i += 1

            if args.dry_run:
                logging.info(f"[{action}] {src_path}  ->  {final_dst}")
            else:
                try:
                    # Same mount point will go atomic rename, different disk will automatically roll back to copy+remove
                    shutil.move(str(src_path), str(final_dst))
                    logging.info(f"[{action}] {src_path}  ->  {final_dst}")
                except Exception as e:
                    logging.error(f"[ERROR] Failed to move {src_path} -> {final_dst}: {e}")
                    continue

            moved_rows.append((name, str(src_path), str(final_dst)))
        else:
            skipped_rows.append(name)

    logging.info("\n=== Summary ===")
    logging.info(f"Threshold: total_rows < {args.threshold}")
    logging.info(f"Pairs to move per Excel: {len(to_move)}")
    logging.info(f"Moved (or planned): {len(moved_rows)}")
    logging.info(f"Skipped (not in Excel list or >= threshold): {len(skipped_rows)}")

if __name__ == "__main__":
    main()
