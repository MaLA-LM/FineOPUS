#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply per-language-pair thresholds (from compute_thresholds.py) to the scored
FineOPUS-Filtered-Stage2-Scored parquet shards.

For each language pair:
  - Look up T from thresholds.csv
  - Stream every input shard, filter rows where similarity_score >= T
  - Write the survivors into output shards of up to --max_rows rows each,
    named '{src}-{tgt}_shard_{K}.parquet' inside '{out_dir}/{src}-{tgt}/'
  - Record rows-before / rows-after in a per-task stats CSV

Designed to be run as a SLURM array job: each task processes
round-robin chunks[chunk_id] of language pairs.

Usage (single machine):
  python apply_thresholds.py \
      --scored_dir /scratch/.../FineOPUS-Filtered-Stage2-Scored \
      --thresholds_csv ../thresholds/stats/thresholds.csv \
      --out_dir /scratch/.../FineOPUS-Filtered-Stage3-Thresholded \
      --stats_output stats/filter_stats.csv \
      --max_rows 10000000

Usage (SLURM array, one task per chunk):
  python apply_thresholds.py \
      ... --n_chunks $SLURM_ARRAY_TASK_COUNT --chunk_id $SLURM_ARRAY_TASK_ID
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


STATS_COLUMNS = [
    "lang_pair", "source_lang", "target_lang",
    "T", "confidence",
    "n_shards_in", "n_shards_out",
    "rows_before", "rows_after",
    "kept_fraction",
]

SCORE_COL = "qe_score"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def enumerate_pairs_from_scored_dir(
    scored_dir: Path, include_same_lang: bool
) -> List[Tuple[str, str]]:
    """List every '{src}-{tgt}' subdirectory under scored_dir that contains
    at least one parquet shard."""
    if not scored_dir.exists():
        raise FileNotFoundError(f"scored_dir does not exist: {scored_dir}")
    pairs: List[Tuple[str, str]] = []
    for child in sorted(scored_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if "-" not in name:
            continue
        src, tgt = name.split("-", 1)
        if not include_same_lang and src == tgt:
            continue
        if not any(child.glob(f"{name}.part-*.parquet")):
            continue
        pairs.append((src, tgt))
    return pairs


def list_shards(pair_dir: Path, lang_pair: str) -> List[Path]:
    return sorted(pair_dir.glob(f"{lang_pair}.part-*.parquet"))


def load_thresholds(path: Path) -> Dict[Tuple[str, str], Dict[str, object]]:
    """Return {(src, tgt): {'T': float, 'confidence': str}} from thresholds.csv."""
    df = pd.read_csv(path)
    required = {"source_lang", "target_lang", "T"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if "confidence" not in df.columns:
        df["confidence"] = ""
    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    for _, r in df.iterrows():
        try:
            T = float(r["T"])
        except (TypeError, ValueError):
            continue
        if pd.isna(T):
            continue
        out[(str(r["source_lang"]), str(r["target_lang"]))] = {
            "T": T,
            "confidence": str(r.get("confidence", "")),
        }
    return out


# ---------------------------------------------------------------------------
# Rotating writer
# ---------------------------------------------------------------------------

class RotatingParquetWriter:
    """Write a stream of pyarrow Tables into {out_dir}/{base}_shard_{K}.parquet
    files, rolling over whenever the current file would exceed max_rows."""

    def __init__(
        self,
        out_dir: Path,
        base_name: str,
        max_rows: int,
        schema: pa.Schema,
        compression: str = "zstd",
    ):
        self.out_dir = out_dir
        self.base_name = base_name
        self.max_rows = max_rows
        self.schema = schema
        self.compression = compression
        self.shard_idx = 0
        self.writer: Optional[pq.ParquetWriter] = None
        self.rows_in_current = 0
        self.total_rows = 0

    def _open_new(self) -> None:
        path = self.out_dir / f"{self.base_name}_shard_{self.shard_idx}.parquet"
        self.writer = pq.ParquetWriter(path, self.schema, compression=self.compression)
        self.rows_in_current = 0

    def write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        # Ensure schemas match exactly to avoid spurious writer errors when
        # input shards have slightly different type metadata.
        if table.schema != self.schema:
            table = table.cast(self.schema)
        offset = 0
        remaining = table.num_rows
        while remaining > 0:
            if self.writer is None:
                self._open_new()
            space = self.max_rows - self.rows_in_current
            take = min(space, remaining)
            self.writer.write_table(table.slice(offset, take))
            self.rows_in_current += take
            self.total_rows += take
            offset += take
            remaining -= take
            if self.rows_in_current >= self.max_rows:
                self.writer.close()
                self.writer = None
                self.shard_idx += 1

    def close(self) -> int:
        """Close the current writer (if any) and return the number of shards
        actually produced."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            n = self.shard_idx + 1
        else:
            n = self.shard_idx  # we already rolled past the last full shard
        # If we never wrote a single row, no shards exist.
        if self.total_rows == 0:
            n = 0
        return n


# ---------------------------------------------------------------------------
# Per-pair filter
# ---------------------------------------------------------------------------

def filter_pair(
    src: str,
    tgt: str,
    scored_dir: Path,
    out_root: Path,
    T: float,
    max_rows: int,
    batch_size: int,
    compression: str,
) -> Tuple[int, int, int, int]:
    """Filter a single language pair. Returns
    (rows_before, rows_after, n_shards_in, n_shards_out)."""
    lang_pair = f"{src}-{tgt}"
    in_dir = scored_dir / lang_pair
    in_shards = list_shards(in_dir, lang_pair)
    if not in_shards:
        logger.warning(f"  {lang_pair}: no input shards under {in_dir}")
        return 0, 0, 0, 0

    out_dir = out_root / lang_pair
    # Clean any stale output from a previous incomplete run
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_before = 0
    rows_after = 0
    writer: Optional[RotatingParquetWriter] = None

    for shard in in_shards:
        try:
            pf = pq.ParquetFile(shard)
        except Exception as e:
            logger.warning(f"  {lang_pair}: failed to open {shard.name}: {e}")
            continue
        rows_before += pf.metadata.num_rows

        if writer is None:
            writer = RotatingParquetWriter(
                out_dir, lang_pair, max_rows=max_rows,
                schema=pf.schema_arrow, compression=compression,
            )

        for batch in pf.iter_batches(batch_size=batch_size):
            if batch.num_rows == 0:
                continue
            table = pa.Table.from_batches([batch])
            if SCORE_COL not in table.column_names:
                logger.warning(
                    f"  {lang_pair}: '{SCORE_COL}' missing in {shard.name}; "
                    "passing all rows through."
                )
                writer.write(table)
                rows_after += table.num_rows
                continue
            mask = pc.greater_equal(table[SCORE_COL], pa.scalar(T, type=table[SCORE_COL].type))
            mask = pc.fill_null(mask, False)
            filtered = table.filter(mask)
            rows_after += filtered.num_rows
            writer.write(filtered)

    n_shards_out = writer.close() if writer is not None else 0

    # Sentinel for skip_existing on reruns
    (out_dir / "_DONE").write_text(
        f"T={T}\nrows_before={rows_before}\nrows_after={rows_after}\n"
    )
    return rows_before, rows_after, len(in_shards), n_shards_out


# ---------------------------------------------------------------------------
# Stats CSV I/O
# ---------------------------------------------------------------------------

def read_existing_pairs(stats_csv: Path) -> set:
    if not stats_csv.exists():
        return set()
    try:
        df = pd.read_csv(stats_csv)
        return set(df["lang_pair"].astype(str).tolist())
    except Exception:
        return set()


def append_stats_row(stats_csv: Path, row: dict) -> None:
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = stats_csv.exists()
    with open(stats_csv, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=STATS_COLUMNS)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in STATS_COLUMNS})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scored_dir",
        default="/scratch/project_462001069/FineOPUS/intermediate/FineOPUS-Filtered-Stage2-Scored",
    )
    parser.add_argument(
        "--thresholds_csv",
        default=str(Path(__file__).resolve().parent.parent / "thresholds/stats/thresholds.csv"),
    )
    parser.add_argument(
        "--out_dir",
        default="/scratch/project_462001069/FineOPUS/intermediate/FineOPUS-Filtered-Stage3-Thresholded",
        help="Root directory for filtered output (one subdir per lang pair).",
    )
    parser.add_argument(
        "--stats_output",
        default=str(Path(__file__).resolve().parent / "stats/filter_stats.csv"),
        help="Output CSV with per-pair row counts (per-task chunks go to {stats_output}.chunk{K}.csv).",
    )
    parser.add_argument(
        "--max_rows", type=int, default=10_000_000,
        help="Max rows per output parquet shard. Default 10M.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=500_000,
        help="Row-batch size used when streaming input shards. Default 500k.",
    )
    parser.add_argument(
        "--compression", default="zstd",
        help="Output parquet compression (zstd|snappy|gzip|none). Default zstd.",
    )
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument(
        "--include_same_lang", action="store_true",
        help="Also process dirs where src == tgt. Off by default.",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip pairs that already have a _DONE sentinel in their output dir "
             "or are already recorded in the stats CSV.",
    )
    args = parser.parse_args()

    scored_dir = Path(args.scored_dir)
    thresholds_csv = Path(args.thresholds_csv)
    out_root = Path(args.out_dir)
    stats_output = Path(args.stats_output)

    if args.n_chunks > 1:
        stats_output = stats_output.with_suffix(f".chunk{args.chunk_id:04d}.csv")

    logger.info("=" * 70)
    logger.info(f"scored_dir      : {scored_dir}")
    logger.info(f"thresholds_csv  : {thresholds_csv}")
    logger.info(f"out_dir         : {out_root}")
    logger.info(f"stats_output    : {stats_output}")
    logger.info(f"max_rows        : {args.max_rows:,}")
    logger.info(f"batch_size      : {args.batch_size:,}")
    logger.info(f"compression     : {args.compression}")
    logger.info(f"chunk_id        : {args.chunk_id} / {args.n_chunks}")
    logger.info(f"include_same_lang: {args.include_same_lang}")
    logger.info(f"skip_existing   : {args.skip_existing}")
    logger.info("=" * 70)

    out_root.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(thresholds_csv)
    logger.info(f"Loaded thresholds for {len(thresholds):,} pairs.")

    pairs = enumerate_pairs_from_scored_dir(scored_dir, args.include_same_lang)
    assigned = [p for i, p in enumerate(pairs) if i % args.n_chunks == args.chunk_id]
    logger.info(f"Total pairs     : {len(pairs):,}")
    logger.info(f"Assigned pairs  : {len(assigned):,}")

    done_in_stats = read_existing_pairs(stats_output) if args.skip_existing else set()

    n_processed = 0
    n_skipped = 0
    n_missing_t = 0
    total_before = 0
    total_after = 0

    for i, (src, tgt) in enumerate(assigned, 1):
        lang_pair = f"{src}-{tgt}"
        out_dir = out_root / lang_pair

        if args.skip_existing and (
            lang_pair in done_in_stats
            or (out_dir / "_DONE").exists()
        ):
            logger.info(f"[{i}/{len(assigned)}] {lang_pair}: already done, skip.")
            n_skipped += 1
            continue

        row = thresholds.get((src, tgt))
        if row is None:
            logger.warning(
                f"[{i}/{len(assigned)}] {lang_pair}: no row in thresholds.csv, skip."
            )
            n_missing_t += 1
            continue
        T = float(row["T"])
        confidence = str(row["confidence"])

        logger.info(f"[{i}/{len(assigned)}] {lang_pair}: T={T:.4f} conf={confidence}")
        try:
            rb, ra, n_in, n_out = filter_pair(
                src, tgt, scored_dir, out_root,
                T=T,
                max_rows=args.max_rows,
                batch_size=args.batch_size,
                compression=args.compression,
            )
        except Exception as e:
            logger.error(f"  {lang_pair}: FAILED: {e}", exc_info=True)
            continue

        kept = (ra / rb) if rb > 0 else 0.0
        logger.info(
            f"  {lang_pair}: {rb:,} -> {ra:,}  ({kept:.1%} kept)  "
            f"shards {n_in} -> {n_out}"
        )

        append_stats_row(stats_output, {
            "lang_pair": lang_pair,
            "source_lang": src,
            "target_lang": tgt,
            "T": T,
            "confidence": confidence,
            "n_shards_in": n_in,
            "n_shards_out": n_out,
            "rows_before": rb,
            "rows_after": ra,
            "kept_fraction": f"{kept:.6f}",
        })
        n_processed += 1
        total_before += rb
        total_after += ra

    logger.info("=" * 70)
    logger.info(f"processed         : {n_processed:,}")
    logger.info(f"skipped (done)    : {n_skipped:,}")
    logger.info(f"skipped (no T)    : {n_missing_t:,}")
    if total_before > 0:
        logger.info(
            f"rows (processed)  : {total_before:,} -> {total_after:,}  "
            f"({total_after / total_before:.1%} kept)"
        )


if __name__ == "__main__":
    main()
