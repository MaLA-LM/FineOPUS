#!/usr/bin/env python3
"""Extract a token-budgeted multilingual parallel mix from three source corpora.

For every target language pair (row) in the quota CSV, extract approximately
``quota_Qwen3_5_tokens`` tokens (source+target tokens, Qwen/Qwen3.5-9B tokenizer)
from each of the three datasets (NLLB, MaLA_Bi, FineOPUS).

The exact number of rows to sample is derived from the *pre-computed* per-source-pair
token statistics (token_stats/*.csv): rows = allocated_tokens / (tokens_per_line).
This avoids re-tokenizing tens of billions of tokens while still hitting the
token budget in expectation. Random Bernoulli sampling (seeded) is used.

If a target pair maps to several source pairs (``a|b|...``) the token budget is
split as evenly as possible across them (water-filling, respecting each pair's
available tokens).

Every output pair folder keeps a single parquet with two columns
``source_text`` (always eng_Latn) and ``target_text`` (the foreign side), so a
reversed source direction (e.g. ``xxx-eng_Latn``) is swapped on write.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

STATS_DIR = Path("/scratch/project_462001427/FineOPUS/tools/token_stats")

DATASETS = {
    "NLLB": {
        "data_dir": Path("/scratch/project_462001069/nllb/nllb-conversion"),
        "stats_csv": STATS_DIR / "NLLB_Qwen3_5.csv",
        "csv_col": "NLLB_csv_lang_pair_rows",
        "src_col": "source_text",
        "tgt_col": "target_text",
    },
    "MaLA_Bi": {
        "data_dir": Path("/scratch/project_462001069/mala-bilingual-translation-corpus"),
        "stats_csv": STATS_DIR / "MaLA-Bilingual-Translation-Corpus_Qwen3_5.csv",
        "csv_col": "MaLA_Bi_csv_lang_pair_rows",
        "src_col": "src_text",
        "tgt_col": "tgt_text",
    },
    "FineOPUS": {
        "data_dir": Path("/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage5"),
        "stats_csv": STATS_DIR / "FineOPUS-Filtered-Stage5_Qwen3_5.csv",
        "csv_col": "FineOPUS_csv_lang_pair_rows",
        "src_col": "source_text",
        "tgt_col": "target_text",
    },
}

ENG = "eng_Latn"
TOK_COLS = ("n_src_tokens_Qwen3_5", "n_tgt_tokens_Qwen3_5")
FLUSH_ROWS = 200_000


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def load_stats(stats_csv: Path) -> dict[str, tuple[int, int]]:
    """lang_pair -> (n_lines, total_tokens=src+tgt)."""
    stats: dict[str, tuple[int, int]] = {}
    with stats_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lp = (row.get("lang_pair") or "").strip()
            if not lp:
                continue
            n_lines = int(row["n_lines"])
            toks = int(row[TOK_COLS[0]]) + int(row[TOK_COLS[1]])
            stats[lp] = (n_lines, toks)
    return stats


# ---------------------------------------------------------------------------
# Water-filling: allocate quota across source pairs as evenly as possible
# ---------------------------------------------------------------------------

def water_fill(quota: int, caps: list[int]) -> list[int]:
    """Split ``quota`` tokens across pairs, as even as possible, each <= cap."""
    n = len(caps)
    alloc = [0] * n
    order = sorted(range(n), key=lambda i: caps[i])
    remaining = quota
    left = n
    for idx in order:
        if left == 0:
            break
        even = remaining / left
        if caps[idx] <= even:
            alloc[idx] = caps[idx]
            remaining -= caps[idx]
        else:
            alloc[idx] = int(round(even))
            remaining -= alloc[idx]
        left -= 1
    return alloc


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class SourcePlan:
    source_pair: str          # e.g. "deu_Latn-eng_Latn"
    n_lines: int
    total_tokens: int
    alloc_tokens: int
    rows: int
    eng_is_source: bool       # True if source_text column holds eng_Latn


@dataclass
class PairPlan:
    target_pair: str          # eng_Latn-xxx
    quota: int
    sources: list[SourcePlan] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def eng_is_source_side(source_pair: str) -> bool | None:
    parts = source_pair.split("-")
    if len(parts) != 2:
        return None
    a, b = parts
    if a == ENG:
        return True
    if b == ENG:
        return False
    return None


def plan_pair(target_pair: str, quota: int, source_pairs: list[str],
              stats: dict[str, tuple[int, int]], data_dir: Path) -> PairPlan:
    plan = PairPlan(target_pair=target_pair, quota=quota)

    valid = []
    for sp in source_pairs:
        if sp not in stats:
            plan.warnings.append(f"missing stats for {sp}")
            continue
        if not (data_dir / sp).is_dir():
            plan.warnings.append(f"missing folder for {sp}")
            continue
        eis = eng_is_source_side(sp)
        if eis is None:
            plan.warnings.append(f"no eng side in {sp}")
            continue
        n_lines, toks = stats[sp]
        if n_lines <= 0 or toks <= 0:
            plan.warnings.append(f"empty stats for {sp}")
            continue
        valid.append((sp, n_lines, toks, eis))

    if not valid:
        plan.warnings.append("no valid source pairs")
        return plan

    caps = [toks for _, _, toks, _ in valid]
    allocs = water_fill(quota, caps)

    for (sp, n_lines, toks, eis), alloc in zip(valid, allocs):
        avg = toks / n_lines
        rows = min(n_lines, int(round(alloc / avg))) if avg > 0 else 0
        plan.sources.append(SourcePlan(
            source_pair=sp, n_lines=n_lines, total_tokens=toks,
            alloc_tokens=alloc, rows=rows, eng_is_source=eis,
        ))
    return plan


# ---------------------------------------------------------------------------
# Sampling / writing
# ---------------------------------------------------------------------------

def list_parquets(pair_dir: Path) -> list[Path]:
    return sorted(p for p in pair_dir.iterdir()
                  if p.is_file() and p.suffix == ".parquet")


def total_rows(files: list[Path]) -> int:
    return sum(pq.read_metadata(p).num_rows for p in files)


OUT_SCHEMA = pa.schema([
    ("source_text", pa.string()),
    ("target_text", pa.string()),
])


class PairWriter:
    def __init__(self, out_file: Path):
        self.out_file = out_file
        self._writer: pq.ParquetWriter | None = None
        self._src: list[str] = []
        self._tgt: list[str] = []
        self.written = 0

    def add(self, src_vals: list, tgt_vals: list):
        self._src.extend("" if v is None else str(v) for v in src_vals)
        self._tgt.extend("" if v is None else str(v) for v in tgt_vals)
        if len(self._src) >= FLUSH_ROWS:
            self._flush()

    def _flush(self):
        if not self._src:
            return
        batch = pa.record_batch([pa.array(self._src, type=pa.string()),
                                 pa.array(self._tgt, type=pa.string())],
                                schema=OUT_SCHEMA)
        if self._writer is None:
            self.out_file.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.out_file, OUT_SCHEMA,
                                            compression="snappy")
        self._writer.write_batch(batch)
        self.written += len(self._src)
        self._src.clear()
        self._tgt.clear()

    def close(self):
        self._flush()
        if self._writer is not None:
            self._writer.close()


def sample_source(sp_plan: SourcePlan, data_dir: Path, cfg: dict,
                  writer: PairWriter, seed: int, batch_size: int) -> int:
    files = list_parquets(data_dir / sp_plan.source_pair)
    if not files:
        return 0
    tot = total_rows(files)
    if tot == 0 or sp_plan.rows <= 0:
        return 0

    p = min(1.0, sp_plan.rows / tot)
    rng = np.random.default_rng(seed)

    src_col, tgt_col = cfg["src_col"], cfg["tgt_col"]
    # Output source_text must be eng. If eng is on source side, keep; else swap.
    read_cols = [src_col, tgt_col]

    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
            n = batch.num_rows
            if n == 0:
                continue
            mask = rng.random(n) < p
            if not mask.any():
                continue
            idx = np.nonzero(mask)[0]
            col_src = batch.column(0).take(pa.array(idx)).to_pylist()
            col_tgt = batch.column(1).take(pa.array(idx)).to_pylist()
            if sp_plan.eng_is_source:
                writer.add(col_src, col_tgt)
            else:
                writer.add(col_tgt, col_src)
    return writer.written


# ---------------------------------------------------------------------------
# CSV reading & driver
# ---------------------------------------------------------------------------

def read_quota_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_targets(args, rows: list[dict]) -> list[dict]:
    if args.lang_pairs:
        want = set(args.lang_pairs)
        rows = [r for r in rows if r["pair_key"] in want]
    if args.chunk is not None and args.total_chunks is not None:
        rows = [r for i, r in enumerate(rows)
                if i % args.total_chunks == args.chunk]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--output_root", required=True, type=Path,
                    help="multilingual_mix root; writes <root>/<dataset>/<pair>/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=50_000)
    ap.add_argument("--lang_pairs", nargs="+",
                    help="Restrict to these target pair_keys.")
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--total_chunks", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--plan_csv", type=Path,
                    help="Write the per-source plan to this CSV (dry-run).")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    data_dir = cfg["data_dir"]
    stats = load_stats(cfg["stats_csv"])
    rows = collect_targets(args, read_quota_rows(args.csv))
    out_dataset_dir = args.output_root / args.dataset

    plan_records = []
    for r in rows:
        target_pair = r["pair_key"]
        quota = int(r["quota_Qwen3_5_tokens"])
        source_pairs = [s for s in r[cfg["csv_col"]].split("|") if s]
        plan = plan_pair(target_pair, quota, source_pairs, stats, data_dir)

        for w in plan.warnings:
            print(f"[WARN] {args.dataset} {target_pair}: {w}", file=sys.stderr)

        planned_tokens = sum(s.alloc_tokens for s in plan.sources)
        planned_rows = sum(s.rows for s in plan.sources)
        print(f"[PLAN] {args.dataset} {target_pair}: quota={quota:,} "
              f"alloc={planned_tokens:,} rows={planned_rows:,} "
              f"sources={len(plan.sources)}")
        for s in plan.sources:
            plan_records.append({
                "dataset": args.dataset,
                "target_pair": target_pair,
                "source_pair": s.source_pair,
                "eng_is_source": s.eng_is_source,
                "quota_tokens": quota,
                "alloc_tokens": s.alloc_tokens,
                "avail_tokens": s.total_tokens,
                "avail_lines": s.n_lines,
                "rows_to_sample": s.rows,
            })

        if args.dry_run:
            continue

        out_file = out_dataset_dir / target_pair / f"{target_pair}.parquet"
        if out_file.exists() and not args.overwrite:
            print(f"[SKIP] exists: {out_file}")
            continue

        writer = PairWriter(out_file)
        try:
            for s in plan.sources:
                # Distinct seed per (target, source) for reproducibility.
                sp_seed = args.seed + (hash((target_pair, s.source_pair)) & 0xFFFF)
                sample_source(s, data_dir, cfg, writer, sp_seed, args.batch_size)
        finally:
            writer.close()
        print(f"[DONE] {out_file} rows={writer.written:,}")

    if args.plan_csv and plan_records:
        args.plan_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.plan_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(plan_records[0].keys()))
            w.writeheader()
            w.writerows(plan_records)
        print(f"Wrote plan: {args.plan_csv}")


if __name__ == "__main__":
    main()
