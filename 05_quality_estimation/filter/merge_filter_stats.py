#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the per-task filter_stats.csv.chunk*.csv files produced by
apply_thresholds.py into a single filter_stats.csv, and print a quick
global summary (totals, median kept fraction, pairs with biggest cuts).
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats_output",
        default=str(Path(__file__).resolve().parent / "stats/filter_stats.csv"),
        help="Target CSV. Chunk files at {stats_output}.chunk*.csv are merged "
             "into this path.",
    )
    parser.add_argument(
        "--remove_chunks", action="store_true",
        help="Delete the per-task chunk CSVs after a successful merge.",
    )
    parser.add_argument(
        "--top_n", type=int, default=20,
        help="Print the top-N pairs with lowest kept_fraction.",
    )
    args = parser.parse_args()

    out = Path(args.stats_output)
    chunks = sorted(out.parent.glob(out.stem + ".chunk*.csv"))
    if not chunks:
        logger.error(f"No chunk files found matching {out.stem}.chunk*.csv "
                     f"under {out.parent}")
        sys.exit(1)

    logger.info(f"Merging {len(chunks)} chunk files into {out}")
    frames = []
    for c in chunks:
        try:
            frames.append(pd.read_csv(c))
        except Exception as e:
            logger.warning(f"  skipping {c.name}: {e}")

    df = pd.concat(frames, ignore_index=True)
    # In case a pair was written twice (e.g. resume without skip_existing),
    # keep the last occurrence.
    df = df.drop_duplicates(subset=["lang_pair"], keep="last")
    df = df.sort_values(["source_lang", "target_lang"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df):,} rows -> {out}")

    # ---- global summary ----
    rb = df["rows_before"].astype(int).sum()
    ra = df["rows_after"].astype(int).sum()
    kept = (ra / rb) if rb > 0 else 0.0
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Pairs processed      : {len(df):,}")
    logger.info(f"Rows before -> after : {rb:,} -> {ra:,}  ({kept:.2%} kept)")
    logger.info(f"Median kept fraction : {df['kept_fraction'].astype(float).median():.2%}")
    logger.info(f"Min    kept fraction : {df['kept_fraction'].astype(float).min():.2%}")
    logger.info(f"Max    kept fraction : {df['kept_fraction'].astype(float).max():.2%}")
    logger.info(f"Pairs keeping < 50%  : {(df['kept_fraction'].astype(float) < 0.5).sum():,}")
    logger.info("")

    logger.info(f"Top {args.top_n} pairs with smallest kept_fraction:")
    worst = df.sort_values("kept_fraction").head(args.top_n)
    for _, r in worst.iterrows():
        logger.info(
            f"  {r['lang_pair']:<25}  T={float(r['T']):.3f}  "
            f"{int(r['rows_before']):>10,} -> {int(r['rows_after']):>10,}  "
            f"({float(r['kept_fraction']):.1%})  conf={r['confidence']}"
        )

    if args.remove_chunks:
        for c in chunks:
            c.unlink()
        logger.info(f"Removed {len(chunks)} chunk files.")


if __name__ == "__main__":
    main()
