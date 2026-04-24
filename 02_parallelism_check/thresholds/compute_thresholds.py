#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine per-pair data stats, gold stats, and benchmark MRR into a per-language-
pair cleaning threshold on `similarity_score`, plus diagnostic plots.

Pipeline inputs (all produced by this repository):
  - best_model_csv       : ../results/best_model_per_lang_pair_*_selected_models.csv
                           columns: source_lang, target_lang, model, MRR
  - data_stats           : stats/data_score_stats.csv  (+ any .chunk*.csv siblings)
                           produced by collect_data_stats.py
  - gold_stats  (opt.)   : stats/gold_score_stats.csv
                           produced by collect_gold_stats.py

Threshold formula (per language pair):

  # Anchor from gold distribution (if available), else data only
  T_gold = max(gold_q_gold, gold_mean - gold_std_k * gold_std)
           # gold_q_gold = gold_{--gold_quantile}, default: gold_p01
  T_data = data_q_data                                  # data_{--data_quantile}, default: data_p10

  if MRR unavailable (pair not in best_model_csv):      # no benchmark anchor
      # confidence = "no_benchmark"; pure data-only fallback
      T_raw = data_p01
  elif MRR >= high_mrr:                                 # very trustworthy
      T_raw = max(T_gold, T_data)
  elif MRR >= low_mrr:                                  # moderate
      T_raw = max(T_gold - 0.02, T_data)                # small relaxation
  else:                                                 # low confidence
      # Model struggles on this pair; don't trust the anchors. Fall back
      # to a pure percentile of the data and flag the row as 'low_mrr'.
      T_raw = data_p01

  # Sanity bounds. eff_cap uses gold_{--gold_cap_quantile} so the cap is
  # "never filter higher than (100-K)% of gold-parallel passes", which is
  # both more interpretable and more robust than the old gold_mean−0.5σ rule.
  eff_cap   = min(t_cap, gold_{gold_cap_quantile})      # default: gold_p25
  T_clipped = clip(T_raw, t_floor, eff_cap)

  # Last-line-of-defense caps (applied as min, so the tightest one wins):
  data_sanity_cap = data_{--data_sanity_quantile}       # default: data_p50
                                                         # (never filter > 50%)
  if --min_keep_fraction > 0:
      T_keep_cap = data_quantile(1 - min_keep_fraction) # inverse of kept-est
  else:
      T_keep_cap = +inf
  T = min(T_clipped, data_sanity_cap, T_keep_cap)

Outputs:
  - thresholds.csv    : per-pair final threshold + the intermediate components
                        + estimated kept fraction (interpolated from data
                        quantiles) so you can see at a glance how aggressive
                        each threshold is.
  - plots/*.png       : summary + per-model + per-pair diagnostic figures.

Run:
  python compute_thresholds.py \
      --data_stats stats/data_score_stats.csv \
      --gold_stats stats/gold_score_stats.csv \
      --best_model_csv ../results/best_model_per_lang_pair_by_flores_bouquet_combined_selected_models.csv \
      --output stats/thresholds.csv \
      --plots_dir stats/plots \
      --low_mrr 0.80 --high_mrr 0.95 \
      --data_quantile p10 \
      --gold_quantile p01 \
      --gold_std_k 2.0 \
      --t_floor 0.30 --t_cap 0.95 \
      --n_diag_worst 30 --n_diag_biggest 30 --n_diag_random 20
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """Load data stats, concatenating any .chunk*.csv siblings."""
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
    # If chunks duplicated any pair, keep the last
    df = df.drop_duplicates(subset=["lang_pair"], keep="last").reset_index(drop=True)
    logger.info(f"Data stats rows: {len(df):,}")
    return df


def load_gold_stats(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning(f"Gold stats not found at {path}; proceeding data-only.")
        return None
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["model", "source_lang", "target_lang"],
                            keep="last").reset_index(drop=True)
    logger.info(f"Gold stats rows: {len(df):,}")
    return df


def load_best_model(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"Best-model CSV not found at {path}; all pairs will be no_benchmark.")
        return pd.DataFrame(columns=["source_lang", "target_lang", "model", "MRR"])
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["source_lang", "target_lang"],
                            keep="last").reset_index(drop=True)
    logger.info(f"Best-model rows: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Kept-fraction estimation from a quantile summary
# ---------------------------------------------------------------------------

def estimate_kept_fraction(row: pd.Series, T: float) -> float:
    """Linear interpolation over the stored quantiles to estimate
    P(similarity_score >= T).

    Uses the fixed grid QUANTILES defined above, which matches what
    collect_data_stats.py records.
    """
    if not np.isfinite(T):
        return float("nan")
    # Build (score, cumulative_fraction) sorted by score
    xs: List[Tuple[float, float]] = [(row["min"], 0.0)]
    for q, col in zip(QUANTILES, QUANTILE_COLS):
        xs.append((float(row[col]), q))
    xs.append((row["max"], 1.0))
    xs.sort(key=lambda t: t[0])

    if T <= xs[0][0]:
        return 1.0
    if T >= xs[-1][0]:
        return 0.0
    # Find bracket
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
    estimated kept fraction equals `min_keep` (i.e. CDF == 1 - min_keep),
    using the same linear interpolation over stored quantiles.

    Used to enforce a per-pair lower bound on the kept fraction: the caller
    caps the final T at this value so that each pair retains >= min_keep
    of its data.
    """
    if not np.isfinite(min_keep) or min_keep <= 0.0:
        return float("inf")
    if min_keep >= 1.0:
        return float(row["min"])

    target_cdf = 1.0 - min_keep
    xs: List[Tuple[float, float]] = [(row["min"], 0.0)]
    for q, col in zip(QUANTILES, QUANTILE_COLS):
        xs.append((float(row[col]), q))
    xs.append((row["max"], 1.0))
    xs.sort(key=lambda t: t[0])

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
# Threshold derivation
# ---------------------------------------------------------------------------

def derive_threshold(
    data_row: pd.Series,
    gold_row: Optional[pd.Series],
    mrr: Optional[float],
    *,
    data_quantile: str = "p10",
    gold_quantile: str = "p01",
    gold_cap_quantile: str = "p25",
    gold_std_k: float = 2.0,
    low_mrr: float = 0.80,
    high_mrr: float = 0.95,
    t_floor: float = 0.30,
    t_cap: float = 0.95,
    min_keep_fraction: float = 0.0,
    data_sanity_quantile: str = "p50",
) -> Dict[str, float]:
    """Apply the hybrid formula and return all the intermediate components."""
    data_q_value = float(data_row[data_quantile])
    data_p01 = float(data_row["p01"])
    data_p50 = float(data_row["p50"])

    if gold_row is not None:
        gold_q_value = float(gold_row[gold_quantile])
        gold_mean = float(gold_row["mean"])
        gold_std = float(gold_row["std"])
        T_gold = max(gold_q_value, gold_mean - gold_std_k * gold_std)
        has_gold = True
    else:
        gold_q_value = float("nan")
        gold_mean = float("nan")
        gold_std = float("nan")
        T_gold = float("nan")
        has_gold = False

    # Choose confidence tier from MRR
    if mrr is None or not np.isfinite(mrr):
        confidence = "no_benchmark"
    elif mrr >= high_mrr:
        confidence = "high"
    elif mrr >= low_mrr:
        confidence = "mid"
    else:
        confidence = "low"

    # Formula
    if confidence in ("low", "no_benchmark"):
        # Either we know the model struggles (low MRR), or we have no
        # benchmark signal at all. Fall back to a very conservative
        # data-only floor.
        T_raw = data_p01
    elif has_gold:
        if confidence == "high":
            T_raw = max(T_gold, data_q_value)
        else:  # mid
            T_raw = max(T_gold - 0.02, data_q_value)
    else:
        # High/mid confidence but no gold stats recorded for this pair.
        T_raw = data_q_value

    # Dynamic gold-side cap: never filter higher than gold_{gold_cap_quantile},
    # i.e. always let at least (100 - K)% of gold-parallel sentences pass.
    if has_gold:
        gold_cap_value = float(gold_row[gold_cap_quantile])
        eff_cap = min(t_cap, gold_cap_value) if np.isfinite(gold_cap_value) else t_cap
    else:
        gold_cap_value = float("nan")
        eff_cap = t_cap
    eff_cap = max(eff_cap, t_floor)  # never invert

    T_clipped = float(min(max(T_raw, t_floor), eff_cap))

    # Data-side sanity cap: T must not exceed this percentile of the actual
    # data. With the default p50 this means "never throw away more than half
    # the data for a single pair" -- a last-line-of-defense against broken
    # gold stats or MRR noise. Set data_sanity_quantile='none' to disable.
    if data_sanity_quantile and data_sanity_quantile != "none":
        data_sanity_cap = float(data_row[data_sanity_quantile])
    else:
        data_sanity_cap = float("inf")

    # Per-pair "keep >= X%" cap. We invert the data CDF to find the score at
    # which kept_fraction == min_keep_fraction, and force T <= that value.
    # This intentionally wins over t_floor so the promise is preserved even
    # when the pair's distribution is very low.
    if min_keep_fraction and min_keep_fraction > 0.0:
        T_keep_cap = threshold_for_kept_fraction(data_row, min_keep_fraction)
    else:
        T_keep_cap = float("inf")

    T = float(min(T_clipped, data_sanity_cap, T_keep_cap))

    return {
        "confidence": confidence,
        "T_gold": T_gold,
        "T_data": data_q_value,
        "T_raw": float(T_raw),
        "T_clipped": T_clipped,
        "data_sanity_cap": float(data_sanity_cap) if np.isfinite(data_sanity_cap) else float("nan"),
        "T_keep_cap": float(T_keep_cap) if np.isfinite(T_keep_cap) else float("nan"),
        "T": T,
        "eff_cap": float(eff_cap),
        "gold_cap_value": gold_cap_value,
        "gold_mean": gold_mean,
        "gold_std": gold_std,
        "gold_q_value": gold_q_value,
        "data_p50": data_p50,
    }


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def build_thresholds(
    data_df: pd.DataFrame,
    gold_df: Optional[pd.DataFrame],
    best_df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """data_stats is the primary table. best_model_csv and gold_stats are
    left-joined in; if either is missing, the pair falls back to a pure
    data-only threshold with confidence='no_benchmark' (or 'no_gold' if we
    have MRR but no gold stats)."""
    data_quantile = kwargs.get("data_quantile", "p10")
    gold_quantile = kwargs.get("gold_quantile", "p01")

    # Build lookups keyed by language pair / by (model, src, tgt)
    best_by_pair = {
        (r["source_lang"], r["target_lang"]): r for _, r in best_df.iterrows()
    }
    if gold_df is not None:
        gold_by_key = {
            (r["model"], r["source_lang"], r["target_lang"]): r
            for _, r in gold_df.iterrows()
        }
    else:
        gold_by_key = {}

    rows = []
    for _, dr in data_df.iterrows():
        src = dr["source_lang"]
        tgt = dr["target_lang"]
        lang_pair = dr["lang_pair"]

        bm = best_by_pair.get((src, tgt))
        if bm is not None:
            model = bm["model"]
            mrr_raw = bm.get("MRR", float("nan"))
            try:
                mrr = float(mrr_raw)
            except (TypeError, ValueError):
                mrr = float("nan")
        else:
            model = ""
            mrr = float("nan")

        gold_row = gold_by_key.get((model, src, tgt)) if model else None

        res = derive_threshold(dr, gold_row, mrr, **kwargs)

        kept = estimate_kept_fraction(dr, res["T"])

        rows.append({
            "lang_pair": lang_pair,
            "source_lang": src,
            "target_lang": tgt,
            "model": model,
            "MRR": mrr,
            "n_rows_total": int(dr["n_rows_total"]),
            "n_shards": int(dr["n_shards"]),
            "data_mean": float(dr["mean"]),
            "data_p05": float(dr["p05"]),
            "data_p10": float(dr["p10"]),
            "data_p50": float(dr["p50"]),
            "gold_mean": res["gold_mean"],
            "gold_std": res["gold_std"],
            "gold_q_value": res["gold_q_value"],
            "gold_quantile": gold_quantile,
            "data_quantile": data_quantile,
            "T_gold": res["T_gold"],
            "T_data": res["T_data"],
            "T_raw": res["T_raw"],
            "T_clipped": res["T_clipped"],
            "data_sanity_cap": res["data_sanity_cap"],
            "T_keep_cap": res["T_keep_cap"],
            "T": res["T"],
            "eff_cap": res["eff_cap"],
            "gold_cap_value": res["gold_cap_value"],
            "confidence": res["confidence"],
            "has_benchmark": bm is not None,
            "has_gold": gold_row is not None,
            "kept_fraction_est": kept,
        })

    n_no_bm = sum(1 for r in rows if not r["has_benchmark"])
    n_no_gold = sum(1 for r in rows if r["has_benchmark"] and not r["has_gold"])
    if n_no_bm:
        logger.info(f"  {n_no_bm:,} pairs without benchmark (data-only fallback).")
    if n_no_gold:
        logger.info(f"  {n_no_gold:,} pairs with benchmark but no gold stats (data-only anchor).")

    out = pd.DataFrame(rows)
    return out


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

    # 1) gold_q vs data_q scatter colored by MRR
    if thresholds["has_gold"].any():
        sub = thresholds[thresholds["has_gold"]].copy()
        # Look up which quantiles were used (recorded per-row; pick any)
        gold_q_name = str(sub["gold_quantile"].iloc[0])
        # The data axis uses the actual data percentile column (always present)
        data_axis_col = "data_p05"
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(
            sub["gold_q_value"], sub[data_axis_col],
            c=sub["MRR"], cmap="viridis", s=10, alpha=0.6
        )
        lo = min(sub["gold_q_value"].min(), sub[data_axis_col].min())
        hi = max(sub["gold_q_value"].max(), sub[data_axis_col].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.5, label="y = x")
        ax.set_xlabel(f"gold {gold_q_name}")
        ax.set_ylabel("data p05")
        ax.set_title(
            f"Per-pair: gold {gold_q_name} vs data p05 (color = MRR)"
        )
        ax.legend(loc="best")
        fig.colorbar(sc, ax=ax, label="MRR")
        fig.tight_layout()
        fig.savefig(plots_dir / "summary_gold_vs_data.png", dpi=130)
        plt.close(fig)

    # 2) Threshold vs MRR scatter (color = confidence tier)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"high": "tab:green", "mid": "tab:orange",
              "low": "tab:red", "no_benchmark": "tab:gray"}
    for tier, sub in thresholds.groupby("confidence"):
        ax.scatter(
            sub["MRR"], sub["T"],
            s=8, alpha=0.5, c=colors.get(tier, "tab:gray"), label=tier
        )
    ax.set_xlabel("MRR (benchmark)")
    ax.set_ylabel("Threshold T")
    ax.set_title("Per-pair threshold vs MRR")
    ax.legend(title="confidence")
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_threshold_vs_mrr.png", dpi=130)
    plt.close(fig)

    # 3) Kept-fraction histogram (global)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(thresholds["kept_fraction_est"].dropna(), bins=50, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Estimated kept fraction")
    ax.set_ylabel("# language pairs")
    ax.set_title("Estimated kept-fraction distribution across language pairs")
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_kept_fraction.png", dpi=130)
    plt.close(fig)

    # 4) Per-model boxplot of thresholds
    fig, ax = plt.subplots(figsize=(12, 5))
    by_model = [sub["T"].values for _, sub in thresholds.groupby("model")]
    labels = [m for m, _ in thresholds.groupby("model")]
    try:
        ax.boxplot(by_model, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(by_model, labels=labels, showfliers=False)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Threshold T")
    ax.set_title("Threshold distribution per embedding model")
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_threshold_per_model.png", dpi=130)
    plt.close(fig)


def plot_pair_candlestick(
    thresholds: pd.DataFrame,
    data_df: pd.DataFrame,
    gold_df: Optional[pd.DataFrame],
    plots_dir: Path,
    selection: str,
    pairs: List[str],
) -> None:
    """Render per-pair 'candlestick' plots (min/p05/p25/p50/p75/p95/max)
    for data and gold distributions, with the threshold as a vertical line.
    Pairs are drawn 10-at-a-time per figure.
    """
    plt = _try_import_mpl()
    if plt is None or not pairs:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    data_by_pair = {r["lang_pair"]: r for _, r in data_df.iterrows()}
    if gold_df is not None:
        gold_by_key = {
            (r["model"], r["source_lang"], r["target_lang"]): r
            for _, r in gold_df.iterrows()
        }
    else:
        gold_by_key = {}
    th_by_pair = {r["lang_pair"]: r for _, r in thresholds.iterrows()}

    per_fig = 10
    for fi, start in enumerate(range(0, len(pairs), per_fig)):
        chunk = pairs[start:start + per_fig]
        fig, ax = plt.subplots(figsize=(11, 0.55 * len(chunk) + 1.2))
        y_labels = []
        for yi, pair in enumerate(chunk):
            th = th_by_pair.get(pair)
            if th is None:
                continue
            d = data_by_pair.get(pair)
            if d is None:
                continue
            # Draw data candle in blue
            y = yi * 2
            ax.hlines(y, d["p05"], d["p95"], color="tab:blue", lw=2)
            ax.hlines(y, d["p25"], d["p75"], color="tab:blue", lw=6, alpha=0.6)
            ax.plot(d["p50"], y, "|", color="white", markersize=10, mew=2)
            # Markers for min/max
            ax.plot([d["min"], d["max"]], [y, y], "|", color="tab:blue",
                    markersize=6, alpha=0.5)

            # Gold candle in orange below data
            g = gold_by_key.get((th["model"], th["source_lang"], th["target_lang"]))
            if g is not None:
                yg = y + 0.8
                ax.hlines(yg, g["p05"], g["p95"], color="tab:orange", lw=2)
                ax.hlines(yg, g["p25"], g["p75"], color="tab:orange", lw=6, alpha=0.6)
                ax.plot(g["p50"], yg, "|", color="white", markersize=10, mew=2)

            # Threshold line
            ax.axvline(th["T"], color="red", lw=0.3, alpha=0.1)
            ax.plot(th["T"], y + 0.4, "v", color="red", markersize=6)

            label = (f"{pair}  [MRR={th['MRR']:.2f}  "
                     f"conf={th['confidence']}  T={th['T']:.3f}  "
                     f"keep≈{th['kept_fraction_est']:.1%}]")
            y_labels.append((y + 0.4, label))

        ax.set_yticks([p[0] for p in y_labels])
        ax.set_yticklabels([p[1] for p in y_labels], fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("similarity_score")
        ax.set_title(
            f"Diagnostic candlesticks — {selection}  ({start + 1}-{start + len(chunk)} / {len(pairs)})\n"
            "blue = data, orange = gold, red ▼ = T"
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
    n_worst: int,
    n_biggest: int,
    n_random: int,
    n_no_benchmark: int,
) -> Dict[str, List[str]]:
    selections: Dict[str, List[str]] = {}
    benched = thresholds[thresholds["has_benchmark"]]
    if n_worst > 0 and len(benched):
        worst = benched.sort_values("MRR", ascending=True, na_position="last").head(n_worst)
        selections["worst_mrr"] = worst["lang_pair"].tolist()
    if n_biggest > 0:
        biggest = thresholds.sort_values("n_rows_total", ascending=False).head(n_biggest)
        selections["biggest_volume"] = biggest["lang_pair"].tolist()
    if n_random > 0:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(thresholds), size=min(n_random, len(thresholds)), replace=False)
        selections["random_sample"] = thresholds.iloc[idx]["lang_pair"].tolist()
    # Dedicated sample of data-only fallback pairs so we can eyeball them
    no_bm = thresholds[~thresholds["has_benchmark"]]
    if n_no_benchmark > 0 and len(no_bm):
        biggest_no_bm = no_bm.sort_values("n_rows_total", ascending=False).head(n_no_benchmark)
        selections["no_benchmark"] = biggest_no_bm["lang_pair"].tolist()
    return selections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--best_model_csv",
        default=str(here.parent
                    / "results/best_model_per_lang_pair_by_flores_bouquet_combined_selected_models.csv"),
    )
    parser.add_argument("--data_stats", default=str(here / "stats/data_score_stats.csv"))
    parser.add_argument("--gold_stats", default=str(here / "stats/gold_score_stats.csv"))
    parser.add_argument("--output", default=str(here / "stats/thresholds.csv"))
    parser.add_argument("--plots_dir", default=str(here / "stats/plots"))

    parser.add_argument(
        "--data_quantile", default="p10", choices=QUANTILE_COLS,
        help="Data-side percentile used as T_data (default: p10)",
    )
    parser.add_argument(
        "--gold_quantile", default="p01", choices=QUANTILE_COLS,
        help="Gold-side empirical percentile used in T_gold = max(gold_{q}, μ−kσ). "
             "Higher (e.g. p10) → stricter threshold. Default: p01.",
    )
    parser.add_argument(
        "--gold_cap_quantile", default="p25", choices=QUANTILE_COLS,
        help="Gold-side percentile used as eff_cap (hard upper bound on T). "
             "Guarantees at least (100-K)%% of gold-parallel sentences "
             "survive the threshold. Lower (e.g. p10) => more permissive "
             "ceiling. Default: p25.",
    )
    parser.add_argument("--gold_std_k", type=float, default=2.0)
    parser.add_argument("--low_mrr", type=float, default=0.80)
    parser.add_argument("--high_mrr", type=float, default=0.95)
    parser.add_argument("--t_floor", type=float, default=0.30)
    parser.add_argument("--t_cap", type=float, default=0.95)
    parser.add_argument(
        "--min_keep_fraction", type=float, default=0.0,
        help="Per-pair lower bound on retained data. If > 0, T is capped so "
             "that each pair keeps at least this fraction of its rows "
             "(interpolated from the recorded data quantiles). "
             "E.g. 0.85 => every pair keeps >= 85%%. Default 0.0 (disabled).",
    )
    parser.add_argument(
        "--data_sanity_quantile", default="p50",
        choices=QUANTILE_COLS + ["none"],
        help="Hard last-line-of-defense cap: T <= data_{quantile}. Default "
             "p50 (never filter more than half a pair's data, regardless of "
             "gold/MRR). Use 'none' to disable.",
    )
    parser.add_argument("--n_diag_worst", type=int, default=30)
    parser.add_argument("--n_diag_biggest", type=int, default=30)
    parser.add_argument("--n_diag_random", type=int, default=20)
    parser.add_argument("--n_diag_no_benchmark", type=int, default=30)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"best_model_csv : {args.best_model_csv}")
    logger.info(f"data_stats     : {args.data_stats}")
    logger.info(f"gold_stats     : {args.gold_stats}")
    logger.info(f"output         : {args.output}")
    logger.info(f"plots_dir      : {args.plots_dir}")
    logger.info(f"data_quantile  : {args.data_quantile}")
    logger.info(f"gold_quantile  : {args.gold_quantile}")
    logger.info(f"gold_cap_q     : {args.gold_cap_quantile}")
    logger.info(f"data_sanity_q  : {args.data_sanity_quantile}")
    logger.info(f"gold_std_k     : {args.gold_std_k}")
    logger.info(f"low/high MRR   : {args.low_mrr} / {args.high_mrr}")
    logger.info(f"T floor/cap    : {args.t_floor} / {args.t_cap}")
    logger.info(f"min_keep       : {args.min_keep_fraction:.2%}"
                if args.min_keep_fraction > 0 else
                "min_keep       : disabled")
    logger.info("=" * 70)

    data_df = load_data_stats(Path(args.data_stats))
    gold_df = load_gold_stats(Path(args.gold_stats))
    best_df = load_best_model(Path(args.best_model_csv))

    thresholds = build_thresholds(
        data_df, gold_df, best_df,
        data_quantile=args.data_quantile,
        gold_quantile=args.gold_quantile,
        gold_cap_quantile=args.gold_cap_quantile,
        gold_std_k=args.gold_std_k,
        low_mrr=args.low_mrr,
        high_mrr=args.high_mrr,
        t_floor=args.t_floor,
        t_cap=args.t_cap,
        min_keep_fraction=args.min_keep_fraction,
        data_sanity_quantile=args.data_sanity_quantile,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(thresholds):,} thresholds → {out_path}")

    # Quick summary
    logger.info("")
    logger.info("Confidence tier distribution:")
    for tier, sub in thresholds.groupby("confidence"):
        logger.info(
            f"  {tier:<8}: {len(sub):>6}  "
            f"T median={sub['T'].median():.3f}  "
            f"keep median={sub['kept_fraction_est'].median():.1%}"
        )
    logger.info("")
    logger.info(f"  pairs with benchmark   : {int(thresholds['has_benchmark'].sum()):,}")
    logger.info(f"  pairs without benchmark: {int((~thresholds['has_benchmark']).sum()):,}")
    logger.info(f"  pairs with gold anchor : {int(thresholds['has_gold'].sum()):,}")
    logger.info(f"  pairs without gold     : {int((~thresholds['has_gold']).sum()):,}")
    logger.info(f"  median T               : {thresholds['T'].median():.3f}")
    logger.info(f"  median kept fraction   : {thresholds['kept_fraction_est'].median():.1%}")
    logger.info(f"  pairs keeping <50%     : {int((thresholds['kept_fraction_est'] < 0.5).sum()):,}")

    if args.no_plots:
        return

    plots_dir = Path(args.plots_dir)
    plot_summary(thresholds, plots_dir)
    logger.info("Wrote summary plots.")

    selections = pick_diagnostic_pairs(
        thresholds,
        args.n_diag_worst,
        args.n_diag_biggest,
        args.n_diag_random,
        args.n_diag_no_benchmark,
    )
    for name, pairs in selections.items():
        plot_pair_candlestick(thresholds, data_df, gold_df, plots_dir, name, pairs)


if __name__ == "__main__":
    main()
