#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure-data per-language-pair cleaning threshold on `similarity_score`.

For each language pair we already have a compact quantile summary of its
score distribution (min/max/mean + p01/p05/p10/p25/p50/p75/p90/p95/p99) in
`data_score_stats.csv`. This script turns those per-pair summaries into a
single threshold `T` plus an estimated kept fraction, with several safety
nets so no pair ends up filtering too aggressively.

Input:
  - data_stats : stats/data_score_stats.csv  (+ any .chunk*.csv siblings)
                 produced by collect_data_stats.py

Threshold formula (per language pair):

  T_raw     = data_{--data_quantile}                # default: p10
  T_clipped = clip(T_raw, --t_floor, --t_cap)       # default: [0.30, 0.95]

  # Last-line-of-defense caps (the tightest one wins).
  T_sanity  = data_{--data_sanity_quantile}         # default: p50
                                                    # (never filter > 50%)
  if --min_keep_fraction > 0:
      T_keep = data_quantile(1 - min_keep_fraction)
  else:
      T_keep = +inf

  T = min(T_clipped, T_sanity, T_keep)

Outputs:
  - thresholds.csv : per-pair final threshold + the intermediate components
                     + estimated kept fraction (interpolated from the
                     stored data quantiles).
  - plots/*.png    : summary + per-pair diagnostic figures.

Run:
  python compute_thresholds.py \
      --data_stats stats/data_score_stats.csv \
      --output stats/thresholds.csv \
      --plots_dir stats/plots \
      --data_quantile p10 \
      --t_floor 0.30 --t_cap 0.95 \
      --data_sanity_quantile p50 \
      --min_keep_fraction 0.5
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data_stats(path: Path) -> pd.DataFrame:
    """Load data stats, concatenating any .chunk*.csv siblings.

    Given e.g. `stats/data_score_stats.csv`, also picks up
    `stats/data_score_stats.chunk*.csv` (this is how collect_data_stats.py
    shards its output) and concatenates them. Duplicates by `lang_pair` are
    resolved by keeping the last seen row.
    """
    candidates: List[Path] = []
    if path.exists():
        candidates.append(path)
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    for p in sorted(parent.glob(f"{stem}.chunk*{suffix}")):
        candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No data stats found at {path} or its chunk siblings.")
    dfs = []
    for p in candidates:
        logger.info(f"  loading data stats: {p}")
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["lang_pair"], keep="last").reset_index(drop=True)
    logger.info(f"Data stats rows: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Kept-fraction estimation from a quantile summary
# ---------------------------------------------------------------------------

def _quantile_points(row: pd.Series) -> List[Tuple[float, float]]:
    """Return [(score, cumulative_fraction)] sorted by score, built from the
    stored (min, p01, ..., p99, max) summary."""
    xs: List[Tuple[float, float]] = [(float(row["min"]), 0.0)]
    for q, col in zip(QUANTILES, QUANTILE_COLS):
        xs.append((float(row[col]), q))
    xs.append((float(row["max"]), 1.0))
    xs.sort(key=lambda t: t[0])
    return xs


def estimate_kept_fraction(row: pd.Series, T: float) -> float:
    """Linear interpolation over the stored quantiles to estimate
    P(similarity_score >= T)."""
    if not np.isfinite(T):
        return float("nan")
    xs = _quantile_points(row)
    if T <= xs[0][0]:
        return 1.0
    if T >= xs[-1][0]:
        return 0.0
    for i in range(1, len(xs)):
        if T <= xs[i][0]:
            x0, y0 = xs[i - 1]
            x1, y1 = xs[i]
            if x1 == x0:
                cdf_at_T = y1
            else:
                cdf_at_T = y0 + (y1 - y0) * (T - x0) / (x1 - x0)
            return float(max(0.0, min(1.0, 1.0 - cdf_at_T)))
    return 0.0


def threshold_for_kept_fraction(row: pd.Series, min_keep: float) -> float:
    """Inverse of estimate_kept_fraction(): return the score T at which the
    estimated kept fraction equals `min_keep` (i.e. CDF == 1 - min_keep)."""
    if not np.isfinite(min_keep) or min_keep <= 0.0:
        return float("inf")
    if min_keep >= 1.0:
        return float(row["min"])

    target_cdf = 1.0 - min_keep
    xs = _quantile_points(row)
    if target_cdf <= xs[0][1]:
        return xs[0][0]
    if target_cdf >= xs[-1][1]:
        return xs[-1][0]
    for i in range(1, len(xs)):
        if target_cdf <= xs[i][1]:
            x0, y0 = xs[i - 1]
            x1, y1 = xs[i]
            if y1 == y0:
                return x0
            return float(x0 + (x1 - x0) * (target_cdf - y0) / (y1 - y0))
    return xs[-1][0]


# ---------------------------------------------------------------------------
# Threshold derivation (per pair)
# ---------------------------------------------------------------------------

def derive_threshold(
    data_row: pd.Series,
    *,
    data_quantile: str = "p10",
    t_floor: float = 0.30,
    t_cap: float = 0.95,
    min_keep_fraction: float = 0.0,
    data_sanity_quantile: str = "p50",
) -> Dict[str, float]:
    """Apply the data-only formula and return all intermediate components."""
    T_raw = float(data_row[data_quantile])

    eff_cap = max(t_cap, t_floor)  # never invert
    T_clipped = float(min(max(T_raw, t_floor), eff_cap))

    # Hard last-line-of-defense cap. With p50 default this means
    # "never throw away more than half the data for a single pair".
    if data_sanity_quantile and data_sanity_quantile != "none":
        data_sanity_cap = float(data_row[data_sanity_quantile])
    else:
        data_sanity_cap = float("inf")

    # Per-pair "keep >= X%" cap: cap T at the data quantile whose tail mass
    # equals min_keep_fraction.
    if min_keep_fraction and min_keep_fraction > 0.0:
        T_keep_cap = threshold_for_kept_fraction(data_row, min_keep_fraction)
    else:
        T_keep_cap = float("inf")

    T = float(min(T_clipped, data_sanity_cap, T_keep_cap))

    return {
        "T_raw": float(T_raw),
        "T_clipped": T_clipped,
        "data_sanity_cap": float(data_sanity_cap) if np.isfinite(data_sanity_cap) else float("nan"),
        "T_keep_cap": float(T_keep_cap) if np.isfinite(T_keep_cap) else float("nan"),
        "T": T,
        "eff_cap": float(eff_cap),
    }


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def build_thresholds(data_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    data_quantile = kwargs.get("data_quantile", "p10")

    rows = []
    for _, dr in data_df.iterrows():
        res = derive_threshold(dr, **kwargs)
        kept = estimate_kept_fraction(dr, res["T"])
        rows.append({
            "lang_pair": dr["lang_pair"],
            "source_lang": dr["source_lang"],
            "target_lang": dr["target_lang"],
            "n_rows_total": int(dr["n_rows_total"]),
            "n_shards": int(dr["n_shards"]),
            "data_mean": float(dr["mean"]),
            "data_p05": float(dr["p05"]),
            "data_p10": float(dr["p10"]),
            "data_p50": float(dr["p50"]),
            "data_quantile": data_quantile,
            "T_raw": res["T_raw"],
            "T_clipped": res["T_clipped"],
            "data_sanity_cap": res["data_sanity_cap"],
            "T_keep_cap": res["T_keep_cap"],
            "T": res["T"],
            "eff_cap": res["eff_cap"],
            "kept_fraction_est": kept,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _try_import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable; skipping plots: {e}")
        return None


def plot_summary(thresholds: pd.DataFrame, plots_dir: Path) -> None:
    plt = _try_import_mpl()
    if plt is None:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) T histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(thresholds["T"].dropna(), bins=50, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Threshold T")
    ax.set_ylabel("# language pairs")
    ax.set_title("Per-pair threshold T distribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_threshold_hist.png", dpi=130)
    plt.close(fig)

    # 2) Kept-fraction histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(thresholds["kept_fraction_est"].dropna(), bins=50,
            color="tab:green", alpha=0.8)
    ax.set_xlabel("Estimated kept fraction")
    ax.set_ylabel("# language pairs")
    ax.set_title("Estimated kept-fraction distribution across language pairs")
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_kept_fraction.png", dpi=130)
    plt.close(fig)

    # 3) T vs data volume (n_rows_total, log scale)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(thresholds["n_rows_total"], thresholds["T"],
               s=8, alpha=0.4, color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel("n_rows_total (log scale)")
    ax.set_ylabel("Threshold T")
    ax.set_title("Per-pair threshold vs data volume")
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_threshold_vs_volume.png", dpi=130)
    plt.close(fig)

    # 4) T vs data_p50 (where the median sits relative to the threshold)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(thresholds["data_p50"], thresholds["T"],
               s=8, alpha=0.4, color="tab:purple")
    lo = min(thresholds["data_p50"].min(), thresholds["T"].min())
    hi = max(thresholds["data_p50"].max(), thresholds["T"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.5, label="y = x")
    ax.set_xlabel("data p50")
    ax.set_ylabel("Threshold T")
    ax.set_title("Per-pair: T vs data median (p50)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_threshold_vs_p50.png", dpi=130)
    plt.close(fig)


def plot_pair_candlestick(
    thresholds: pd.DataFrame,
    data_df: pd.DataFrame,
    plots_dir: Path,
    selection: str,
    pairs: List[str],
) -> None:
    """Render per-pair candlestick plots (min/p05/p25/p50/p75/p95/max) for
    the data distribution, with the threshold as a vertical marker. Pairs
    are drawn 10-at-a-time per figure.
    """
    plt = _try_import_mpl()
    if plt is None or not pairs:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    data_by_pair = {r["lang_pair"]: r for _, r in data_df.iterrows()}
    th_by_pair = {r["lang_pair"]: r for _, r in thresholds.iterrows()}

    per_fig = 10
    for fi, start in enumerate(range(0, len(pairs), per_fig)):
        chunk = pairs[start:start + per_fig]
        fig, ax = plt.subplots(figsize=(11, 0.55 * len(chunk) + 1.2))
        y_labels = []
        for yi, pair in enumerate(chunk):
            th = th_by_pair.get(pair)
            d = data_by_pair.get(pair)
            if th is None or d is None:
                continue
            y = yi * 2
            ax.hlines(y, d["p05"], d["p95"], color="tab:blue", lw=2)
            ax.hlines(y, d["p25"], d["p75"], color="tab:blue", lw=6, alpha=0.6)
            ax.plot(d["p50"], y, "|", color="white", markersize=10, mew=2)
            ax.plot([d["min"], d["max"]], [y, y], "|", color="tab:blue",
                    markersize=6, alpha=0.5)

            ax.axvline(th["T"], color="red", lw=0.3, alpha=0.1)
            ax.plot(th["T"], y + 0.4, "v", color="red", markersize=6)

            label = (f"{pair}  [n={int(th['n_rows_total']):,}  "
                     f"T={th['T']:.3f}  keep≈{th['kept_fraction_est']:.1%}]")
            y_labels.append((y + 0.4, label))

        ax.set_yticks([p[0] for p in y_labels])
        ax.set_yticklabels([p[1] for p in y_labels], fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("similarity_score")
        ax.set_title(
            f"Diagnostic candlesticks — {selection}  ({start + 1}-{start + len(chunk)} / {len(pairs)})\n"
            "blue = data (min · p05 · [p25–p75] · p50 · p95 · max), red ▼ = T"
        )
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        out = plots_dir / f"diag_{selection}_{fi:02d}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        logger.info(f"  wrote {out.name}")


def pick_diagnostic_pairs(
    thresholds: pd.DataFrame,
    n_biggest: int,
    n_smallest: int,
    n_lowest_keep: int,
    n_random: int,
) -> Dict[str, List[str]]:
    """Pick interesting subsets of pairs for the candlestick diagnostics."""
    selections: Dict[str, List[str]] = {}
    if n_biggest > 0:
        sub = thresholds.sort_values("n_rows_total", ascending=False).head(n_biggest)
        selections["biggest_volume"] = sub["lang_pair"].tolist()
    if n_smallest > 0:
        sub = thresholds.sort_values("n_rows_total", ascending=True).head(n_smallest)
        selections["smallest_volume"] = sub["lang_pair"].tolist()
    if n_lowest_keep > 0:
        sub = thresholds.sort_values("kept_fraction_est",
                                     ascending=True,
                                     na_position="last").head(n_lowest_keep)
        selections["lowest_keep"] = sub["lang_pair"].tolist()
    if n_random > 0:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(thresholds), size=min(n_random, len(thresholds)), replace=False)
        selections["random_sample"] = thresholds.iloc[idx]["lang_pair"].tolist()
    return selections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    here = Path(__file__).resolve().parent
    parser.add_argument("--data_stats", default=str(here / "stats/data_score_stats.csv"),
                        help="Base path; sibling .chunk*.csv files are auto-discovered.")
    parser.add_argument("--output", default=str(here / "stats/thresholds.csv"))
    parser.add_argument("--plots_dir", default=str(here / "stats/plots"))

    parser.add_argument(
        "--data_quantile", default="p10", choices=QUANTILE_COLS,
        help="Data-side percentile used as the raw threshold T_raw "
             "(default: p10, i.e. cut the lowest ~10%% of each pair).",
    )
    parser.add_argument("--t_floor", type=float, default=0.30,
                        help="Hard lower bound on T. Default 0.30.")
    parser.add_argument("--t_cap", type=float, default=0.95,
                        help="Hard upper bound on T. Default 0.95.")
    parser.add_argument(
        "--data_sanity_quantile", default="p50",
        choices=QUANTILE_COLS + ["none"],
        help="Hard last-line-of-defense cap: T <= data_{quantile}. Default "
             "p50 (never filter more than half a pair's data). "
             "Use 'none' to disable.",
    )
    parser.add_argument(
        "--min_keep_fraction", type=float, default=0.0,
        help="Per-pair lower bound on retained data. If > 0, T is capped so "
             "that each pair keeps at least this fraction of its rows "
             "(interpolated from the recorded data quantiles). "
             "E.g. 0.85 => every pair keeps >= 85%%. Default 0.0 (disabled).",
    )

    parser.add_argument("--n_diag_biggest", type=int, default=30)
    parser.add_argument("--n_diag_smallest", type=int, default=20)
    parser.add_argument("--n_diag_lowest_keep", type=int, default=30)
    parser.add_argument("--n_diag_random", type=int, default=20)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"data_stats      : {args.data_stats}")
    logger.info(f"output          : {args.output}")
    logger.info(f"plots_dir       : {args.plots_dir}")
    logger.info(f"data_quantile   : {args.data_quantile}")
    logger.info(f"data_sanity_q   : {args.data_sanity_quantile}")
    logger.info(f"T floor/cap     : {args.t_floor} / {args.t_cap}")
    logger.info(f"min_keep        : {args.min_keep_fraction:.2%}"
                if args.min_keep_fraction > 0 else
                "min_keep        : disabled")
    logger.info("=" * 70)

    data_df = load_data_stats(Path(args.data_stats))

    thresholds = build_thresholds(
        data_df,
        data_quantile=args.data_quantile,
        t_floor=args.t_floor,
        t_cap=args.t_cap,
        min_keep_fraction=args.min_keep_fraction,
        data_sanity_quantile=args.data_sanity_quantile,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(thresholds):,} thresholds → {out_path}")

    logger.info("")
    logger.info("Threshold summary:")
    logger.info(f"  median T             : {thresholds['T'].median():.3f}")
    logger.info(f"  mean   T             : {thresholds['T'].mean():.3f}")
    logger.info(f"  P10 / P90 of T       : "
                f"{thresholds['T'].quantile(0.10):.3f} / "
                f"{thresholds['T'].quantile(0.90):.3f}")
    logger.info(f"  median kept fraction : {thresholds['kept_fraction_est'].median():.1%}")
    logger.info(f"  pairs keeping <50%   : {int((thresholds['kept_fraction_est'] < 0.5).sum()):,}")
    logger.info(f"  pairs keeping <10%   : {int((thresholds['kept_fraction_est'] < 0.1).sum()):,}")
    logger.info(f"  pairs hitting t_cap  : {int((thresholds['T'] >= args.t_cap - 1e-9).sum()):,}")
    logger.info(f"  pairs hitting t_floor: {int((thresholds['T'] <= args.t_floor + 1e-9).sum()):,}")

    if args.no_plots:
        return

    plots_dir = Path(args.plots_dir)
    plot_summary(thresholds, plots_dir)
    logger.info("Wrote summary plots.")

    selections = pick_diagnostic_pairs(
        thresholds,
        args.n_diag_biggest,
        args.n_diag_smallest,
        args.n_diag_lowest_keep,
        args.n_diag_random,
    )
    for name, pairs in selections.items():
        plot_pair_candlestick(thresholds, data_df, plots_dir, name, pairs)


if __name__ == "__main__":
    main()
