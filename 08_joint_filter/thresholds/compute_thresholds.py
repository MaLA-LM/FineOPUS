#!/usr/bin/env python3
"""Generate multiple token-budget thresholds from one completed statistics run."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from common import HERE, atomic_json, digest, engine_digest, file_digest, read_json, write_csv


def parse_profiles(value):
    result = {}
    for item in value.split(","):
        name, amount = item.split("=", 1)
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in result:
            raise ValueError(f"Invalid/duplicate profile: {name}")
        target = float(amount)
        if not np.isfinite(target) or target <= 0:
            raise ValueError("Token targets must be finite and positive")
        result[name] = target
    return result


def allocate_budgets(original, eligible, target, config):
    """Water filling: min(eligible_i, c * original_i * rate_i)."""
    original, eligible = np.asarray(original, float), np.asarray(eligible, float)
    rates = np.asarray(config["volume_keep_rates"])[np.searchsorted(config["volume_edges"], original, side="right")]
    base = original * rates
    achievable = min(target, float(eligible.sum()))
    if achievable <= 0:
        return np.zeros_like(base), 0.0
    lo, hi = 0.0, 1.0
    while np.minimum(eligible, hi * base).sum() < achievable:
        hi *= 2
        if not np.isfinite(hi):
            raise ValueError("Cannot allocate budget: inconsistent token totals")
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.minimum(eligible, mid * base).sum() < achievable:
            lo = mid
        else:
            hi = mid
    return np.minimum(eligible, hi * base), hi


def boundary(hist, budget):
    """Return a score bucket and token fraction to retain within that bucket."""
    total = float(hist.sum())
    if budget <= 0 or total <= 0:
        return len(hist), 0.0
    if budget >= total * (1 - 1e-12):
        return -1, 1.0
    cumulative = np.cumsum(hist[::-1])
    reverse_bin = int(np.searchsorted(cumulative, budget, side="left"))
    b = len(hist) - 1 - reverse_bin
    higher = float(cumulative[reverse_bin - 1]) if reverse_bin else 0.0
    return b, float(np.clip((budget - higher) / hist[b], 0, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stats_dir", default=str(HERE / "thresholds/stats"))
    p.add_argument("--out_dir", help="Default: <stats_dir>/profiles")
    p.add_argument("--profiles", default="relaxed=1.2e12,balanced=1e12,strict=8e11",
                   help="Comma-separated NAME=TARGET_TOKENS")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(args.stats_dir).resolve()
    run = read_json(root / "run.json")
    if run["run_id"] != digest({k: v for k, v in run.items() if k != "run_id"}):
        raise ValueError("Statistics run metadata checksum mismatch")
    if run["engine_digest"] != engine_digest():
        raise ValueError("Scoring code changed since collection; regenerate statistics")
    summaries = []
    # Fail before publishing any threshold set when an array task is incomplete.
    for pair in run["pairs"]:
        directory = root / "pairs" / pair
        s = read_json(directory / "summary.json")
        if s["identity"]["run_id"] != run["run_id"]:
            raise ValueError(f"Mixed statistics runs: {pair}")
        for name, checksum in s["checksums"].items():
            if file_digest(directory / name) != checksum:
                raise ValueError(f"Corrupt statistics: {directory / name}")
        summaries.append(s)
    profiles = parse_profiles(args.profiles)
    out = Path(args.out_dir).resolve() if args.out_dir else root / "profiles"
    original = np.asarray([s["input_tokens"] for s in summaries])
    eligible = np.asarray([s["eligible_tokens"] for s in summaries])
    for name, target in profiles.items():
        budgets, coefficient = allocate_budgets(original, eligible, target, run["config"])
        records = []
        for s, budget in zip(summaries, budgets):
            pair = s["lang_pair"]
            with np.load(root / "pairs" / pair / "histogram.npz") as data:
                hist_tokens, hist_rows = data["tokens"], data["rows"]
            for cohort in (0, 1):
                group_tokens = float(hist_tokens[cohort].sum())
                # Equal token retention ratio for present/missing LLM cohorts.
                group_budget = budget * group_tokens / max(s["eligible_tokens"], 1e-300)
                b, fraction = boundary(hist_tokens[cohort], group_budget)
                row_count = float(hist_rows[cohort, b + 1:].sum())
                if 0 <= b < run["bins"]:
                    row_count += float(hist_rows[cohort, b]) * fraction
                records.append({"lang_pair": pair, "source_lang": pair.split("-")[0],
                                "target_lang": pair.split("-")[1], "llm_present": cohort,
                                "T": max(0, b) / run["bins"], "boundary_bin": b,
                                "boundary_fraction": fraction, "bins": run["bins"],
                                "input_tokens": s["input_tokens"], "eligible_tokens": group_tokens,
                                "budget_tokens": group_budget, "kept_rows_est": row_count,
                                "token_mode": run["token_mode"],
                                "calibration_digest": s["checksums"]["calibration.json"],
                                "summary_digest": file_digest(root / "pairs" / pair / "summary.json")})
        metadata = {"profile": name, "target_tokens": target, "allocated_tokens": float(budgets.sum()),
                    "input_tokens": float(original.sum()), "eligible_tokens": float(eligible.sum()),
                    "shortfall_tokens": max(0.0, target - float(eligible.sum())),
                    "budget_coefficient": coefficient, "token_mode": run["token_mode"],
                    "run_id": run["run_id"], "engine_digest": run["engine_digest"],
                    "stats_dir": str(root), "seed": run["config"]["seed"], "bins": run["bins"]}
        # Thresholds and their companion metadata must be kept together.
        csv_path = out / f"thresholds_{name}.csv"
        meta_path = csv_path.with_suffix(".json")
        if csv_path.exists() or meta_path.exists():
            raise FileExistsError(f"Profile already exists: {csv_path}; use a new --out_dir/profile name")
        write_csv(csv_path, records)
        metadata["csv_digest"] = file_digest(csv_path)
        metadata["profile_id"] = digest(metadata)
        atomic_json(meta_path, metadata)
        logging.info("%s: target %.4g, selected %.4g tokens (%s), shortfall %.4g -> %s",
                     name, target, budgets.sum(), run["token_mode"], metadata["shortfall_tokens"], csv_path)


if __name__ == "__main__":
    main()
