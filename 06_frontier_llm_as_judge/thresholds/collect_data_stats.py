#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per language-pair scan of llm_judge_score distributions on the already-scored
FineOPUS-Filtered-Stage4-LLMScored parquet shards.

Enumerates every {src}-{tgt} directory directly under --scored_dir so that it
covers all pairs actually scored, including those that don't appear in the
benchmark's best_model table. By default same-language directories (src == tgt)
are skipped; pass --include_same_lang to keep them.

For each language pair:
  - Streams the 'llm_judge_score' column (only) across all shards
  - Subsamples uniformly across shards up to --sample_rows
  - Computes {mean, std, min, max, p01, p05, p10, p25, p50, p75, p90, p95, p99}
  - Writes one row per pair to an output CSV

Designed to be run as a SLURM array job: each task processes
chunks[chunk_id] of language pairs (round-robin by pair index).

Usage (single machine):
  python collect_data_stats.py \
      --scored_dir /scratch/.../FineOPUS-Filtered-Stage4-LLMScored \
      --output stats/data_score_stats.csv \
      --sample_rows 300000

Usage (SLURM array, one task per chunk):
  python collect_data_stats.py \
      ... --n_chunks $SLURM_ARRAY_TASK_COUNT --chunk_id $SLURM_ARRAY_TASK_ID
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
QUANTILE_COLS = [f"p{int(q * 100):02d}" for q in QUANTILES]
OUTPUT_COLUMNS = [
    "lang_pair", "source_lang", "target_lang",
    "n_shards", "n_rows_total", "n_rows_sampled",
    "mean", "std", "min", "max",
    *QUANTILE_COLS,
]


def enumerate_pairs_from_scored_dir(
    scored_dir: Path, include_same_lang: bool
) -> List[Tuple[str, str]]:
    """List every '{src}-{tgt}' subdirectory under scored_dir that actually
    contains at least one parquet shard. Directory names are split on the
    first '-' occurrence in order to survive language codes with no dashes
    (our codes are always '<lang>_<script>', no dashes inside).
    """
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
        pairs.append((src, tgt))
    return pairs


def list_shards(pair_dir: Path, lang_pair: str) -> List[Path]:
    return sorted(pair_dir.glob(f"{lang_pair}_shard_*.parquet"))


def shard_row_counts(shards: List[Path]) -> List[int]:
    import pyarrow.parquet as pq
    counts = []
    for p in shards:
        try:
            counts.append(pq.read_metadata(p).num_rows)
        except Exception as e:
            logger.warning(f"  could not read metadata for {p.name}: {e}")
            counts.append(0)
    return counts


def sample_pair_scores(
    shards: List[Path], total_rows: int, row_counts: List[int], sample_rows: int
) -> Optional[np.ndarray]:
    """Stream llm_judge_score across shards and return a float32 sample array.

    Uses proportional sampling: each shard contributes roughly
    ceil(row_counts[i] / total_rows * sample_rows) rows via random selection
    over its batches. If total_rows <= sample_rows, returns all rows.
    """
    import pyarrow.parquet as pq

    if total_rows == 0:
        return None

    if total_rows <= sample_rows:
        # Read everything
        chunks: List[np.ndarray] = []
        for shard in shards:
            try:
                pf = pq.ParquetFile(shard)
                for batch in pf.iter_batches(columns=["llm_judge_score"], batch_size=200_000):
                    arr = np.asarray(batch.column("llm_judge_score"), dtype=np.float32)
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        chunks.append(arr)
            except Exception as e:
                logger.warning(f"  failed reading {shard.name}: {e}")
        if not chunks:
            return None
        return np.concatenate(chunks)

    # Otherwise subsample each shard
    rng = np.random.default_rng(seed=0xC0FFEE)
    chunks: List[np.ndarray] = []
    for shard, n_rows in zip(shards, row_counts):
        if n_rows == 0:
            continue
        target = max(1, int(round(n_rows / total_rows * sample_rows)))
        try:
            pf = pq.ParquetFile(shard)
            # Read in batches; within each batch, sample to stay within target
            remaining = target
            total_read = 0
            for batch in pf.iter_batches(columns=["llm_judge_score"], batch_size=500_000):
                if remaining <= 0:
                    break
                arr = np.asarray(batch.column("llm_judge_score"), dtype=np.float32)
                total_read += arr.size
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                # proportion of this batch to keep
                batch_target = min(remaining, max(1, int(round(arr.size / n_rows * target))))
                if batch_target >= arr.size:
                    chunks.append(arr)
                    remaining -= arr.size
                else:
                    idx = rng.choice(arr.size, size=batch_target, replace=False)
                    chunks.append(arr[idx])
                    remaining -= batch_target
        except Exception as e:
            logger.warning(f"  failed sampling {shard.name}: {e}")
    if not chunks:
        return None
    sample = np.concatenate(chunks)
    if sample.size > sample_rows:
        idx = rng.choice(sample.size, size=sample_rows, replace=False)
        sample = sample[idx]
    return sample


def compute_stats(sample: np.ndarray) -> dict:
    out = {
        "mean": float(sample.mean()),
        "std": float(sample.std()),
        "min": float(sample.min()),
        "max": float(sample.max()),
    }
    qs = np.quantile(sample, QUANTILES)
    for name, q in zip(QUANTILE_COLS, qs):
        out[name] = float(q)
    return out


def read_existing_pairs(out_csv: Path) -> set:
    if not out_csv.exists():
        return set()
    try:
        df = pd.read_csv(out_csv)
        return set(df["lang_pair"].astype(str).tolist())
    except Exception:
        return set()


def append_row(out_csv: Path, row: dict) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_csv.exists()
    with open(out_csv, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=OUTPUT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in OUTPUT_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scored_dir",
        default="/scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage4-LLMScored",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "stats/data_score_stats.csv"),
        help="Output CSV (per-task outputs go to {output}.chunk{K}.csv)",
    )
    parser.add_argument("--sample_rows", type=int, default=10_000_000)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument(
        "--include_same_lang",
        action="store_true",
        help="Also process dirs where src == tgt (e.g. abk_Cyrl-abk_Cyrl). Off by default.",
    )
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    scored_dir = Path(args.scored_dir)
    output = Path(args.output)

    if args.n_chunks > 1:
        output = output.with_suffix(f".chunk{args.chunk_id:04d}.csv")

    logger.info(f"scored_dir       : {scored_dir}")
    logger.info(f"output           : {output}")
    logger.info(f"sample_rows      : {args.sample_rows:,}")
    logger.info(f"chunk_id         : {args.chunk_id} / {args.n_chunks}")
    logger.info(f"include_same_lang: {args.include_same_lang}")

    pairs = enumerate_pairs_from_scored_dir(scored_dir, args.include_same_lang)
    # Round-robin slicing: stable and well-balanced regardless of per-pair cost
    assigned = [p for i, p in enumerate(pairs) if i % args.n_chunks == args.chunk_id]
    logger.info(f"Total pairs      : {len(pairs):,}")
    logger.info(f"Assigned pairs   : {len(assigned):,}")

    done = read_existing_pairs(output) if args.skip_existing else set()

    for i, (src, tgt) in enumerate(assigned, 1):
        lang_pair = f"{src}-{tgt}"
        if lang_pair in done:
            logger.info(f"[{i}/{len(assigned)}] {lang_pair} already processed, skip.")
            continue

        pair_dir = scored_dir / lang_pair
        if not pair_dir.exists():
            logger.warning(f"[{i}/{len(assigned)}] {lang_pair}: dir missing, skip.")
            continue

        shards = list_shards(pair_dir, lang_pair)
        if not shards:
            logger.warning(f"[{i}/{len(assigned)}] {lang_pair}: no shards, skip.")
            continue

        row_counts = shard_row_counts(shards)
        total_rows = sum(row_counts)
        if total_rows == 0:
            logger.warning(f"[{i}/{len(assigned)}] {lang_pair}: zero rows, skip.")
            continue

        sample = sample_pair_scores(shards, total_rows, row_counts, args.sample_rows)
        if sample is None or sample.size == 0:
            logger.warning(f"[{i}/{len(assigned)}] {lang_pair}: empty sample, skip.")
            continue

        stats = compute_stats(sample)
        row = {
            "lang_pair": lang_pair,
            "source_lang": src,
            "target_lang": tgt,
            "n_shards": len(shards),
            "n_rows_total": total_rows,
            "n_rows_sampled": int(sample.size),
            **stats,
        }
        append_row(output, row)
        logger.info(
            f"[{i}/{len(assigned)}] {lang_pair}: "
            f"shards={len(shards)}, rows={total_rows:,}, sampled={sample.size:,}, "
            f"mean={stats['mean']:.3f}, p05={stats['p05']:.3f}, p50={stats['p50']:.3f}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
