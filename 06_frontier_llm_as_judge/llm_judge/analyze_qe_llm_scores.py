#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream qe_score and llm_judge_score from FineOPUS-Filtered-Stage4-LLMScored
parquet shards and summarize their distributions (and agreement).

Usage:
  module load pytorch/2.5   # pyarrow + numpy + matplotlib
  python analyze_qe_llm_scores.py \
      --scored_dir /scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage4-LLMScored \
      --sample_rows 500000 \
      --plot_dir stats/score_dist_plots
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCORE_COLS = ("qe_score", "llm_judge_score")
QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
HIST_BINS = np.linspace(0.0, 1.0, 21)  # [0,1] in steps of 0.05


def list_pair_dirs(scored_dir: Path) -> List[Path]:
    return sorted(
        p for p in scored_dir.iterdir()
        if p.is_dir() and any(p.glob("*.parquet"))
    )


def list_shards(pair_dir: Path) -> List[Path]:
    return sorted(pair_dir.glob("*.parquet"))


def shard_row_counts(shards: List[Path]) -> List[int]:
    import pyarrow.parquet as pq

    counts = []
    for p in shards:
        try:
            counts.append(pq.read_metadata(p).num_rows)
        except Exception as e:
            logger.warning("metadata %s: %s", p.name, e)
            counts.append(0)
    return counts


def stream_pair_columns(
    shards: List[Path],
    total_rows: int,
    row_counts: List[int],
    sample_rows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned (qe_score, llm_judge_score) float32 samples."""
    import pyarrow.parquet as pq

    if total_rows == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    rng = np.random.default_rng(seed)
    read_all = total_rows <= sample_rows

    qe_chunks: List[np.ndarray] = []
    llm_chunks: List[np.ndarray] = []

    for shard, n_rows in zip(shards, row_counts):
        if n_rows == 0:
            continue
        target = n_rows if read_all else max(1, int(round(n_rows / total_rows * sample_rows)))
        remaining = target
        try:
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(columns=list(SCORE_COLS), batch_size=200_000):
                if not read_all and remaining <= 0:
                    break
                qe = np.asarray(batch.column("qe_score"), dtype=np.float32)
                llm = np.asarray(batch.column("llm_judge_score"), dtype=np.float32)
                n = qe.size
                if n == 0:
                    continue
                if read_all:
                    idx = np.arange(n)
                else:
                    batch_target = min(remaining, max(1, int(round(n / n_rows * target))))
                    if batch_target >= n:
                        idx = np.arange(n)
                    else:
                        idx = rng.choice(n, size=batch_target, replace=False)
                    remaining -= idx.size
                qe_chunks.append(qe[idx])
                llm_chunks.append(llm[idx])
        except Exception as e:
            logger.warning("read %s: %s", shard.name, e)

    if not qe_chunks:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    qe = np.concatenate(qe_chunks)
    llm = np.concatenate(llm_chunks)
    if qe.size > sample_rows:
        idx = rng.choice(qe.size, size=sample_rows, replace=False)
        qe, llm = qe[idx], llm[idx]
    return qe, llm


def finite_mask(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr)


def describe(name: str, arr: np.ndarray) -> None:
    n = arr.size
    finite = arr[finite_mask(arr)]
    n_fin = finite.size
    print(f"\n=== {name} (n={n:,}, finite={n_fin:,}, null/non-finite={n - n_fin:,}) ===")
    if n_fin == 0:
        print("  (no finite values)")
        return
    qs = np.quantile(finite, QUANTILES)
    print(f"  mean={finite.mean():.4f}  std={finite.std():.4f}")
    print(f"  min={finite.min():.4f}  max={finite.max():.4f}")
    for q, v in zip(QUANTILES, qs):
        print(f"  p{int(q * 100):02d}={v:.4f}")
    hist, _ = np.histogram(finite, bins=HIST_BINS)
    print("  histogram [0,1] step=0.05 (counts):")
    for i, c in enumerate(hist):
        lo, hi = HIST_BINS[i], HIST_BINS[i + 1]
        bar = "#" * max(1, int(40 * c / max(hist.max(), 1))) if c else ""
        print(f"    [{lo:.2f},{hi:.2f}): {c:8,} {bar}")


def describe_joint(qe: np.ndarray, llm: np.ndarray) -> None:
    both = finite_mask(qe) & finite_mask(llm)
    n = int(both.sum())
    print(f"\n=== joint (both finite, n={n:,}) ===")
    if n == 0:
        return
    q, l = qe[both], llm[both]
    pearson = float(np.corrcoef(q, l)[0, 1])
    # Spearman via rank (no scipy dependency)
    rq = np.argsort(np.argsort(q))
    rl = np.argsort(np.argsort(l))
    spearman = float(np.corrcoef(rq, rl)[0, 1])
    diff = l - q
    print(f"  pearson={pearson:.4f}  spearman={spearman:.4f}")
    print(f"  llm - qe: mean={diff.mean():.4f}  std={diff.std():.4f}")
    print(f"            p50={np.median(diff):.4f}")
    # agreement buckets on 0.05 grid
    agree = np.abs(diff) <= 0.05
    print(f"  |llm-qe| <= 0.05: {agree.sum():,} ({100 * agree.mean():.1f}%)")


def save_plots(
    plot_dir: Path,
    qe: np.ndarray,
    llm: np.ndarray,
    max_scatter: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, arr, title in zip(
        axes,
        (qe[finite_mask(qe)], llm[finite_mask(llm)]),
        ("qe_score", "llm_judge_score"),
    ):
        ax.hist(arr, bins=HIST_BINS, color="steelblue", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.set_xlim(0, 1)
    fig.tight_layout()
    hist_path = plot_dir / "marginal_histograms.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", hist_path)

    both = finite_mask(qe) & finite_mask(llm)
    if both.sum() == 0:
        return
    q, l = qe[both], llm[both]
    if q.size > max_scatter:
        idx = rng.choice(q.size, size=max_scatter, replace=False)
        q, l = q[idx], l[idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hexbin(q, l, gridsize=40, cmap="Blues", mincnt=1)
    ax.plot([0, 1], [0, 1], "r--", lw=1, label="y=x")
    ax.set_xlabel("qe_score")
    ax.set_ylabel("llm_judge_score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.set_title("qe vs llm (hexbin)")
    fig.tight_layout()
    scatter_path = plot_dir / "qe_vs_llm_hexbin.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", scatter_path)


def collect_global_sample(
    pair_dirs: Iterable[Path],
    sample_rows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Proportional subsample across all language-pair directories."""
    pair_dirs = list(pair_dirs)
    totals: List[Tuple[Path, List[Path], int]] = []
    grand_total = 0
    for d in pair_dirs:
        shards = list_shards(d)
        counts = shard_row_counts(shards)
        n = sum(counts)
        if n > 0:
            totals.append((d, shards, n))
            grand_total += n

    if grand_total == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0

    qe_all: List[np.ndarray] = []
    llm_all: List[np.ndarray] = []
    for i, (pair_dir, shards, n_rows) in enumerate(totals):
        row_counts = shard_row_counts(shards)
        pair_sample = max(1, int(round(n_rows / grand_total * sample_rows)))
        qe, llm = stream_pair_columns(
            shards, n_rows, row_counts, pair_sample, seed=seed + i
        )
        if qe.size:
            qe_all.append(qe)
            llm_all.append(llm)
        logger.info(
            "[%d/%d] %s rows=%s sampled=%s",
            i + 1,
            len(totals),
            pair_dir.name,
            f"{n_rows:,}",
            f"{qe.size:,}",
        )

    if not qe_all:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), grand_total

    qe = np.concatenate(qe_all)
    llm = np.concatenate(llm_all)
    if qe.size > sample_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(qe.size, size=sample_rows, replace=False)
        qe, llm = qe[idx], llm[idx]
    return qe, llm, grand_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scored_dir",
        type=Path,
        default=Path(
            "/scratch/project_462001069/FineOPUS/FineOPUS-Filtered-Stage4-LLMScored"
        ),
    )
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=500_000,
        help="Max aligned (qe, llm) rows to keep for global stats/plots",
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        default=None,
        help="If set, write marginal histograms and hexbin scatter PNGs here",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_scatter", type=int, default=200_000)
    args = parser.parse_args()

    scored_dir = args.scored_dir
    if not scored_dir.is_dir():
        raise FileNotFoundError(scored_dir)

    pair_dirs = list_pair_dirs(scored_dir)
    logger.info("scored_dir=%s  pair_dirs=%d", scored_dir, len(pair_dirs))

    qe, llm, total_rows = collect_global_sample(
        pair_dirs, args.sample_rows, args.seed
    )
    print(f"\nDataset: {scored_dir}")
    print(f"Language pairs with parquet: {len(pair_dirs):,}")
    print(f"Total rows (metadata sum): {total_rows:,}")
    print(f"Sampled aligned rows: {qe.size:,}")

    describe("qe_score", qe)
    describe("llm_judge_score", llm)
    describe_joint(qe, llm)

    if args.plot_dir is not None:
        save_plots(args.plot_dir, qe, llm, args.max_scatter, args.seed)


if __name__ == "__main__":
    main()
