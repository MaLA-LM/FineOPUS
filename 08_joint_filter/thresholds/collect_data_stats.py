#!/usr/bin/env python3
"""Fit per-pair score transforms, then build full-corpus joint-score histograms."""
from __future__ import annotations

import argparse
import fcntl
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import (HERE, DEFAULT_INPUT, DEFAULT_TOKENS, DEFAULT_MRR, DEFAULT_TOKENIZER,
                    atomic_json, digest, engine_digest, enumerate_pairs, file_digest, inventory,
                    load_mrr, load_token_totals, read_json, sample_rows, score_bins, TokenCounter,
                    validate_chunks)
from scoring import FEATURE_COLUMNS, REASONS, extract, fit_calibration, score, validate_config

LOG = logging.getLogger(__name__)


def collect_pair(pair, args, run, tokens, mrr, tokenizer):
    pair_dir = Path(args.scored_dir) / pair
    files = inventory(pair_dir)
    rows = sum(m["rows"] for m in files)
    if args.token_mode == "estimated":
        if pair not in tokens or tokens[pair][0] != rows:
            raise ValueError(f"{pair}: token CSV missing or n_lines does not match input ({rows}); regenerate CSV or use --token_mode exact")
    identity = {"run_id": run["run_id"], "inventory": files}
    destination = Path(args.stats_dir) / "pairs" / pair
    if destination.exists():
        complete = destination / "summary.json"
        if not args.skip_existing or not complete.exists():
            raise FileExistsError(f"Output already exists: {destination}; use --skip_existing for a completed matching run")
        old = read_json(complete)
        if old["identity"] != identity:
            raise ValueError(f"{pair}: existing statistics are stale; use a new --stats_dir")
        for name in ("calibration.json", "histogram.npz"):
            if file_digest(destination / name) != old["checksums"][name]:
                raise ValueError(f"Corrupt statistics: {destination / name}")
        LOG.info("%s: already complete", pair)
        return
    sample_seed = int(digest([run["config"]["seed"], pair])[:16], 16)
    sampled = sample_rows(pair_dir, files, args.sample_rows, sample_seed, FEATURE_COLUMNS)
    calibration = fit_calibration(extract(sampled, pair, run["config"]), run["config"], mrr.get(pair, {"MRR": None, "model": ""}))
    calibration["pair"] = pair
    count_hist = np.zeros((2, args.bins), dtype=np.int64)
    src_hist = np.zeros((2, args.bins), dtype=np.float64)
    tgt_hist = np.zeros_like(src_hist)
    total_src = total_tgt = 0.0
    reason_counts = {name: 0 for name in REASONS.values()}
    coverage = {k: 0 for k in ("P", "Q", "J")}
    processed = eligible_rows = 0
    for m in files:
        pf = pq.ParquetFile(pair_dir / m["file"])
        cols = [c for c in FEATURE_COLUMNS if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
            if not batch.num_rows:
                continue
            f = extract(pa.Table.from_batches([batch]), pair, run["config"])
            values = score(f, calibration)
            eligible = f["reasons"] == 0
            for bit, name in REASONS.items():
                reason_counts[name] += int(np.count_nonzero(f["reasons"] & bit))
            for key in coverage:
                coverage[key] += int(np.isfinite(f[key]).sum())
            if tokenizer is not None:
                src_weights, tgt_weights = [tokenizer.lengths(t.tolist()) for t in f["texts"]]
            else:
                src_weights, tgt_weights = f["chars_src"], f["chars_tgt"]
            total_src += float(src_weights.sum())
            total_tgt += float(tgt_weights.sum())
            codes = score_bins(values, args.bins)
            for cohort in (0, 1):
                mask = eligible & (f["cohort"] == cohort)
                count_hist[cohort] += np.bincount(codes[mask], minlength=args.bins)
                src_hist[cohort] += np.bincount(codes[mask], weights=src_weights[mask], minlength=args.bins)
                tgt_hist[cohort] += np.bincount(codes[mask], weights=tgt_weights[mask], minlength=args.bins)
            processed += batch.num_rows
            eligible_rows += int(eligible.sum())
        LOG.info("%s: %s/%s rows", pair, f"{processed:,}", f"{rows:,}")
    if processed != rows or inventory(pair_dir) != files:
        raise ValueError(f"{pair}: input changed during scan")
    if tokenizer is None:
        _, true_src, true_tgt = tokens[pair]
        if (not total_src and true_src) or (not total_tgt and true_tgt):
            raise ValueError(f"{pair}: token/character totals inconsistent")
        src_scale, tgt_scale = true_src / max(total_src, 1), true_tgt / max(total_tgt, 1)
        input_tokens = true_src + true_tgt
    else:
        src_scale = tgt_scale = 1.0
        input_tokens = total_src + total_tgt
    token_hist = src_hist * src_scale + tgt_hist * tgt_scale
    calibration["token_scaling"] = {"src": src_scale, "tgt": tgt_scale}
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{pair}.", dir=destination.parent))
    try:
        atomic_json(work / "calibration.json", calibration)
        np.savez_compressed(work / "histogram.npz", rows=count_hist, tokens=token_hist)
        summary = {"lang_pair": pair, "identity": identity, "sample_rows": sampled.num_rows,
                   "rows_before": rows, "rows_eligible": eligible_rows,
                   "input_tokens": input_tokens, "eligible_tokens": float(token_hist.sum()),
                   "token_mode": args.token_mode, "reason_counts": reason_counts, "coverage": coverage,
                   "checksums": {name: file_digest(work / name) for name in ("calibration.json", "histogram.npz")}}
        atomic_json(work / "summary.json", summary)
        os.rename(work, destination)
    finally:
        if work.exists():
            shutil.rmtree(work)
    LOG.info("%s complete: eligible %.2f%%, tokens %.3g (%s)", pair, 100 * eligible_rows / max(rows, 1), token_hist.sum(), args.token_mode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored_dir", default=DEFAULT_INPUT)
    p.add_argument("--stats_dir", default=str(HERE / "thresholds/stats"))
    p.add_argument("--config", default=str(HERE / "config.json"))
    p.add_argument("--token_stats_csv", default=str(DEFAULT_TOKENS))
    p.add_argument("--mrr_csv", default=str(DEFAULT_MRR), help="Empty string disables MRR metadata")
    p.add_argument("--token_mode", choices=["estimated", "exact"], default="estimated")
    p.add_argument("--tokenizer_json", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--sample_rows", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=32_768)
    p.add_argument("--bins", type=int, default=10_000)
    p.add_argument("--pairs", help="Comma-separated subset for a pilot; saved in run metadata")
    p.add_argument("--include_same_lang", action="store_true")
    p.add_argument("--n_chunks", type=int, default=1)
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()
    validate_chunks(args)
    if min(args.sample_rows, args.batch_size, args.bins) < 1:
        p.error("sample_rows, batch_size, bins must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = read_json(args.config)
    validate_config(config)
    pairs = enumerate_pairs(args.scored_dir, args.pairs, args.include_same_lang)
    tokens = load_token_totals(args.token_stats_csv) if args.token_mode == "estimated" else {}
    mrr = load_mrr(args.mrr_csv)
    run = {"format_version": 1, "input_root": str(Path(args.scored_dir).resolve()), "pairs": pairs,
           "config": config, "engine_digest": engine_digest(), "collector_digest": file_digest(__file__),
           "sample_rows": args.sample_rows, "bins": args.bins, "token_mode": args.token_mode,
           "token_csv_digest": file_digest(args.token_stats_csv) if tokens else None,
           "mrr_digest": file_digest(args.mrr_csv) if args.mrr_csv else None,
           "tokenizer_json": str(Path(args.tokenizer_json).resolve()) if args.token_mode == "exact" else None,
           "tokenizer_digest": file_digest(args.tokenizer_json) if args.token_mode == "exact" else None}
    run["run_id"] = digest(run)
    root = Path(args.stats_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Lock is held only while registering metadata, not during the scan.
    with open(root / ".run.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "run.json").exists():
            if read_json(root / "run.json") != run:
                raise ValueError("Statistics configuration/input/code changed; use a new --stats_dir")
        else:
            atomic_json(root / "run.json", run)
    tokenizer = TokenCounter(args.tokenizer_json) if args.token_mode == "exact" else None
    # Deterministic assignment; large pairs can be split into separate pilot runs
    # only with their own stats directory. No concurrent writers for one pair.
    for pair in pairs[args.chunk_id::args.n_chunks]:
        collect_pair(pair, args, run, tokens, mrr, tokenizer)


if __name__ == "__main__":
    main()
