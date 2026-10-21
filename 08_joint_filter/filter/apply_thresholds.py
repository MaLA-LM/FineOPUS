#!/usr/bin/env python3
"""Apply one immutable joint-threshold profile to Stage5, streaming by pair."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
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

from common import (DEFAULT_INPUT, DEFAULT_TOKENIZER, atomic_json, digest, engine_digest, file_digest,
                    inventory, read_json, RotatingWriter, selection_mask, TokenCounter, validate_chunks)
from scoring import REASONS, extract, score

LOG = logging.getLogger(__name__)


def output_inventory(directory):
    return [{"file": p.name, "bytes": p.stat().st_size, "rows": pq.read_metadata(p).num_rows}
            for p in sorted(Path(directory).glob("*.parquet"))]


def load_profile(path, stats_dir=None):
    path = Path(path).resolve()
    meta = read_json(path.with_suffix(".json"))
    if meta["profile_id"] != digest({k: v for k, v in meta.items() if k != "profile_id"}):
        raise ValueError("Threshold metadata checksum mismatch")
    if file_digest(path) != meta["csv_digest"]:
        raise ValueError("Threshold CSV changed; regenerate it with compute_thresholds.py")
    if meta["engine_digest"] != engine_digest():
        raise ValueError("Scoring code changed since statistics collection")
    root = Path(stats_dir or meta["stats_dir"]).resolve()
    run = read_json(root / "run.json")
    if run["run_id"] != digest({k: v for k, v in run.items() if k != "run_id"}):
        raise ValueError("Statistics run metadata checksum mismatch")
    if run["run_id"] != meta["run_id"] or run["engine_digest"] != meta["engine_digest"]:
        raise ValueError("Thresholds and statistics belong to different runs")
    thresholds = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            pair, cohort = row["lang_pair"], int(row["llm_present"])
            if cohort not in (0, 1) or cohort in thresholds[pair]:
                raise ValueError(f"Duplicate/invalid threshold row: {pair}/{cohort}")
            b, frac = int(row["boundary_bin"]), float(row["boundary_fraction"])
            if not -1 <= b <= meta["bins"] or not 0 <= frac <= 1 or int(row["bins"]) != meta["bins"]:
                raise ValueError(f"Invalid boundary: {pair}/{cohort}")
            thresholds[pair][cohort] = row
    if set(thresholds) != set(run["pairs"]) or any(set(v) != {0, 1} for v in thresholds.values()):
        raise ValueError("Threshold coverage is incomplete")
    return meta, run, root, thresholds


def filter_pair(pair, args, profile, run, root, thresholds, tokenizer, output_run_id):
    pair_dir = Path(args.scored_dir) / pair
    stats_dir = root / "pairs" / pair
    summary_path = stats_dir / "summary.json"
    summary = read_json(summary_path)
    calibration_path = stats_dir / "calibration.json"
    for row in thresholds.values():
        if file_digest(summary_path) != row["summary_digest"] or file_digest(calibration_path) != row["calibration_digest"]:
            raise ValueError(f"Statistics changed after threshold generation: {pair}")
    files = inventory(pair_dir)
    if summary["identity"] != {"run_id": run["run_id"], "inventory": files}:
        raise ValueError(f"Input files changed since statistics collection: {pair}")
    calibration = read_json(calibration_path)
    destination = Path(args.out_dir) / pair
    if destination.exists():
        done = destination / "_DONE.json"
        if not args.skip_existing or not done.exists():
            raise FileExistsError(f"Existing output: {destination}; use --skip_existing for matching completed output")
        old = read_json(done)
        if old["output_run_id"] != output_run_id or old["input_inventory"] != files:
            raise ValueError(f"Cannot resume different configuration/input: {pair}")
        if old["output_inventory"] != output_inventory(destination):
            raise ValueError(f"Incomplete/corrupt output: {destination}")
        LOG.info("%s: already complete", pair)
        return
    schemas = [pq.ParquetFile(pair_dir / m["file"]).schema_arrow for m in files]
    schema = pa.unify_schemas(schemas, promote_options="permissive")
    if {"joint_score", "joint_llm_present"} & set(schema.names):
        raise ValueError("Input already contains joint-filter output fields")
    schema = schema.append(pa.field("joint_score", pa.float32())).append(pa.field("joint_llm_present", pa.bool_()))
    work = Path(tempfile.mkdtemp(prefix=f".{pair}.", dir=args.out_dir))
    writer = RotatingWriter(work, pair, schema, args.max_rows, args.compression)
    rows_before = rows_after = rows_eligible = 0
    kept_tokens = 0.0
    reasons = {name: 0 for name in REASONS.values()}
    corpora = defaultdict(lambda: {"rows_before": 0, "rows_after": 0, "kept_tokens": 0.0})
    try:
        for m in files:
            offset = 0
            for batch in pq.ParquetFile(pair_dir / m["file"]).iter_batches(batch_size=args.batch_size):
                if not batch.num_rows:
                    continue
                table = pa.Table.from_batches([batch])
                features = extract(table, pair, calibration["config"])
                values = score(features, calibration)
                eligible = features["reasons"] == 0
                keep = selection_mask(values, eligible, features["cohort"], thresholds, pair, m["file"],
                                      offset, profile["seed"], profile["bins"])
                selected = table.filter(pa.array(keep))
                selected = selected.append_column("joint_score", pa.array(values[keep], type=pa.float32()))
                selected = selected.append_column("joint_llm_present", pa.array(features["cohort"][keep].astype(bool)))
                # Union nullable input schemas without losing optional columns.
                for field in schema:
                    if field.name not in selected.column_names:
                        selected = selected.append_column(field.name, pa.nulls(selected.num_rows, type=field.type))
                selected = selected.select(schema.names)
                writer.write(selected)
                if tokenizer is not None:
                    weights = sum(tokenizer.lengths(t[keep].tolist()) for t in features["texts"])
                else:
                    scale = calibration["token_scaling"]
                    weights = (features["chars_src"][keep] * scale["src"] + features["chars_tgt"][keep] * scale["tgt"])
                kept_tokens += float(weights.sum())
                corpus = np.asarray([str(x) if x is not None else "unknown" for x in table["corpus"].to_pylist()], dtype=object) if "corpus" in table.column_names else np.full(batch.num_rows, "unknown", dtype=object)
                kept_corpus = corpus[keep]
                for name in set(corpus):
                    corpora[name]["rows_before"] += int(np.count_nonzero(corpus == name))
                    mask = kept_corpus == name
                    corpora[name]["rows_after"] += int(mask.sum())
                    corpora[name]["kept_tokens"] += float(weights[mask].sum())
                for bit, name in REASONS.items():
                    reasons[name] += int(np.count_nonzero(features["reasons"] & bit))
                rows_before += batch.num_rows
                rows_after += int(keep.sum())
                rows_eligible += int(eligible.sum())
                offset += batch.num_rows
            LOG.info("%s: read %s, retained %s rows", pair, f"{rows_before:,}", f"{rows_after:,}")
        writer.close()
        if rows_before != summary["rows_before"] or rows_eligible != summary["rows_eligible"] or inventory(pair_dir) != files:
            raise ValueError(f"Input or scoring drift detected while filtering {pair}")
        outputs = output_inventory(work)
        if sum(m["rows"] for m in outputs) != rows_after:
            raise ValueError(f"Output row conservation failed: {pair}")
        done = {"lang_pair": pair, "output_run_id": output_run_id, "profile_id": profile["profile_id"],
                "input_inventory": files, "output_inventory": outputs,
                "rows_before": rows_before, "rows_eligible": rows_eligible, "rows_after": rows_after,
                "rows_hard_rejected_or_missing": rows_before - rows_eligible,
                "rows_budget_rejected": rows_eligible - rows_after,
                "kept_fraction": rows_after / max(rows_before, 1),
                "input_tokens": summary["input_tokens"], "kept_tokens": kept_tokens,
                "kept_tokens_kind": "exact" if tokenizer is not None else "estimated",
                "budget_tokens": sum(float(r["budget_tokens"]) for r in thresholds.values()),
                "reason_counts": reasons, "corpora": dict(corpora)}
        atomic_json(work / "_DONE.json", done)
        os.rename(work, destination)
    finally:
        writer.close()
        if work.exists():
            shutil.rmtree(work)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored_dir", default=DEFAULT_INPUT)
    p.add_argument("--thresholds_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--stats_dir", help="Override stored statistics path after relocating artifacts")
    p.add_argument("--max_rows", type=int, default=10_000_000)
    p.add_argument("--batch_size", type=int, default=32_768)
    p.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "none"])
    p.add_argument("--count_tokens", action="store_true", help="Count selected tokens exactly even if thresholds used estimates")
    p.add_argument("--tokenizer_json", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--n_chunks", type=int, default=1)
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()
    validate_chunks(args)
    if min(args.max_rows, args.batch_size) < 1:
        p.error("max_rows and batch_size must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta, run, root, thresholds = load_profile(args.thresholds_csv, args.stats_dir)
    source, output = Path(args.scored_dir).resolve(), Path(args.out_dir).resolve()
    if source != Path(run["input_root"]):
        raise ValueError("Input root differs from the statistics snapshot")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input/output roots must be separate, non-nested directories")
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Data output and statistics must be separate, non-nested directories")
    tokenizer_path = run["tokenizer_json"] if run["token_mode"] == "exact" else args.tokenizer_json
    exact = args.count_tokens or run["token_mode"] == "exact"
    tokenizer_hash = file_digest(tokenizer_path) if exact else None
    if run["token_mode"] == "exact" and tokenizer_hash != run["tokenizer_digest"]:
        raise ValueError("Exact-count tokenizer changed")
    tokenizer = TokenCounter(tokenizer_path) if exact else None
    output_run = {"profile_id": meta["profile_id"], "run_id": run["run_id"], "input_root": str(source),
                  "pairs": run["pairs"], "engine_digest": engine_digest(),
                  "filter_digest": file_digest(__file__), "max_rows": args.max_rows,
                  "compression": args.compression, "count_tokens": exact, "tokenizer_digest": tokenizer_hash}
    output_run["output_run_id"] = digest(output_run)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / ".run.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (output / "run.json").exists():
            if read_json(output / "run.json") != output_run:
                raise ValueError("Output belongs to a different profile/configuration; choose another --out_dir")
        else:
            if any(p.name != ".run.lock" for p in output.iterdir()):
                raise ValueError("Refusing to reuse a nonempty, unowned output directory")
            atomic_json(output / "run.json", output_run)
    for pair in run["pairs"][args.chunk_id::args.n_chunks]:
        filter_pair(pair, args, meta, run, root, thresholds[pair], tokenizer, output_run["output_run_id"])


if __name__ == "__main__":
    main()
