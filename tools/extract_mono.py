#!/usr/bin/env python3
"""Extract a token-budgeted monolingual mix as JSONL.

The quota CSV provides ``monolingual_quota_tokens`` for each language.
English (``eng_Latn``, ``eng_Latn_45B``, ``eng_Latn_46_5B``, or
``eng_Latn_47_5B``) is sampled from Nemotron-CC's ``eng_Latn`` data; every
other language is sampled from FineWeb-2.  Sampling is based on pre-computed
token statistics:
rows_to_sample ~= quota_tokens / average_tokens_per_row.  If the quota is
larger than the available tokens, the source rows are upsampled by writing
complete repeats plus a randomly sampled remainder.

Output is one JSONL file per language:
    <output_dir>/<lang>.jsonl

Each JSONL line has one field by default:
    {"text": "..."}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ENG = {"eng_Latn", "eng_Latn_45B", "eng_Latn_46_5B", "eng_Latn_47_5B"}
TOKEN_COL = "n_tokens_Qwen3_5"
FLUSH_LINES = 50_000

DATASETS = {
    "FineWeb-2": {
        "data_dir": Path("/scratch/project_462001069/fineweb-2"),
        "stats_csv": Path("/scratch/project_462001427/FineOPUS/tools/token_stats/FineWeb-2.csv"),
    },
    "Nemotron-CC": {
        "data_dir": Path("/scratch/project_462001069/nemotron-cc"),
        "stats_csv": Path("/scratch/project_462001427/FineOPUS/tools/token_stats/Nemotron-CC.csv"),
    },
}


@dataclass
class LangPlan:
    lang: str
    source_lang: str
    dataset: str
    data_dir: Path
    quota_tokens: int
    available_lines: int
    available_tokens: int
    rows_to_sample: int
    sample_prob: float
    warnings: list[str]


def parse_int(value: str | None) -> int:
    value = (value or "").strip().replace(",", "")
    return int(value) if value else 0


def stable_seed(seed: int, *parts: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(part.encode("utf-8"))
    return int.from_bytes(h.digest(), "little") & 0xFFFFFFFF


def load_stats(path: Path) -> dict[str, tuple[int, int]]:
    stats: dict[str, tuple[int, int]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = (row.get("lang") or "").strip()
            if not lang:
                continue
            stats[lang] = (parse_int(row.get("n_lines")),
                           parse_int(row.get(TOKEN_COL)))
    return stats


def read_quota_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = (row.get("language") or "").strip()
            quota = parse_int(row.get("monolingual_quota_tokens"))
            if not lang or quota <= 0:
                continue
            rows.append({"language": lang, "quota": quota})
    return rows


def select_rows(args, rows: list[dict]) -> list[dict]:
    if args.langs:
        wanted = set(args.langs)
        rows = [r for r in rows if r["language"] in wanted]
    if args.chunk is not None and args.total_chunks is not None:
        rows = [
            r for i, r in enumerate(rows)
            if i % args.total_chunks == args.chunk
        ]
    return rows


def list_parquets(lang_dir: Path) -> list[Path]:
    return sorted(
        p for p in lang_dir.iterdir()
        if p.is_file() and p.suffix == ".parquet"
    )


def build_plan(lang: str, quota: int, fineweb_stats, nemotron_stats,
               fineweb_dir: Path, nemotron_dir: Path) -> LangPlan:
    is_english = lang in ENG
    source_lang = "eng_Latn" if is_english else lang
    dataset = "Nemotron-CC" if is_english else "FineWeb-2"
    data_dir = nemotron_dir if is_english else fineweb_dir
    stats = nemotron_stats if is_english else fineweb_stats
    warnings: list[str] = []

    n_lines, n_tokens = stats.get(source_lang, (0, 0))
    lang_dir = data_dir / source_lang
    if n_lines <= 0 or n_tokens <= 0:
        warnings.append(f"missing or empty stats for {source_lang}")
    if not lang_dir.is_dir():
        warnings.append(f"missing folder: {lang_dir}")

    if n_lines <= 0 or n_tokens <= 0:
        rows = 0
        prob = 0.0
    else:
        avg_tokens = n_tokens / n_lines
        # Do not cap at n_lines: quotas above the available token count are
        # fulfilled by repeating the corpus in sample_language().
        rows = int(round(quota / avg_tokens))
        prob = min(1.0, rows / n_lines)

    return LangPlan(
        lang=lang,
        source_lang=source_lang,
        dataset=dataset,
        data_dir=data_dir,
        quota_tokens=quota,
        available_lines=n_lines,
        available_tokens=n_tokens,
        rows_to_sample=rows,
        sample_prob=prob,
        warnings=warnings,
    )


class JsonlWriter:
    def __init__(self, path: Path, text_key: str, include_lang: bool, lang: str):
        self.path = path
        self.tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.text_key = text_key
        self.include_lang = include_lang
        self.lang = lang
        self._fh = None
        self._buf: list[str] = []
        self.written = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.tmp_path.open("w", encoding="utf-8")
        return self

    def add_texts(self, texts: list):
        for value in texts:
            item = {self.text_key: "" if value is None else str(value)}
            if self.include_lang:
                item["language"] = self.lang
            self._buf.append(json.dumps(item, ensure_ascii=False))
        if len(self._buf) >= FLUSH_LINES:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        assert self._fh is not None
        self._fh.write("\n".join(self._buf))
        self._fh.write("\n")
        self.written += len(self._buf)
        self._buf.clear()

    def __exit__(self, exc_type, exc, tb):
        self.flush()
        if self._fh is not None:
            self._fh.close()
        if exc_type is None:
            self.tmp_path.replace(self.path)
        elif self.tmp_path.exists():
            self.tmp_path.unlink()


def sample_language(plan: LangPlan, out_file: Path, seed: int, batch_size: int,
                    input_text_column: str, output_text_key: str,
                    include_lang: bool) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if plan.rows_to_sample <= 0:
        return 0

    files = list_parquets(plan.data_dir / plan.source_lang)
    if not files:
        return 0

    # Stats determine the target rows via average tokens/row, but parquet
    # metadata is authoritative for the number of rows currently on disk.
    total_rows = sum(pq.ParquetFile(fp).metadata.num_rows for fp in files)
    if total_rows <= 0:
        return 0
    full_repeats, remainder_rows = divmod(plan.rows_to_sample, total_rows)
    remainder_prob = remainder_rows / total_rows

    rng = np.random.default_rng(stable_seed(seed, plan.dataset, plan.lang))
    with JsonlWriter(out_file, output_text_key, include_lang, plan.lang) as writer:
        for fp in files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(
                batch_size=batch_size,
                columns=[input_text_column],
            ):
                n = batch.num_rows
                if n == 0:
                    continue
                col = batch.column(batch.schema.get_field_index(input_text_column))
                if full_repeats:
                    texts = col.to_pylist()
                    for _ in range(full_repeats):
                        writer.add_texts(texts)

                if remainder_rows <= 0:
                    continue
                mask = rng.random(n) < remainder_prob
                if not mask.any():
                    continue
                idx = np.nonzero(mask)[0]
                selected = col.take(pa.array(idx)).to_pylist()
                writer.add_texts(selected)
    return writer.written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--fineweb_dir", type=Path,
                    default=DATASETS["FineWeb-2"]["data_dir"])
    ap.add_argument("--nemotron_dir", type=Path,
                    default=DATASETS["Nemotron-CC"]["data_dir"])
    ap.add_argument("--fineweb_stats", type=Path,
                    default=DATASETS["FineWeb-2"]["stats_csv"])
    ap.add_argument("--nemotron_stats", type=Path,
                    default=DATASETS["Nemotron-CC"]["stats_csv"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=50_000)
    ap.add_argument("--input_text_column", default="text")
    ap.add_argument("--output_text_key", default="text")
    ap.add_argument("--include_lang", action="store_true",
                    help="Add a language field to each JSON object.")
    ap.add_argument("--langs", nargs="+",
                    help="Restrict extraction to these languages.")
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--total_chunks", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--plan_csv", type=Path,
                    help="Write the extraction plan to this CSV.")
    args = ap.parse_args()

    if (args.chunk is None) != (args.total_chunks is None):
        print("Error: --chunk and --total_chunks must be provided together.",
              file=sys.stderr)
        sys.exit(1)
    if args.chunk is not None and not (0 <= args.chunk < args.total_chunks):
        print("Error: --chunk must satisfy 0 <= chunk < total_chunks.",
              file=sys.stderr)
        sys.exit(1)

    fineweb_stats = load_stats(args.fineweb_stats)
    nemotron_stats = load_stats(args.nemotron_stats)
    quota_rows = select_rows(args, read_quota_rows(args.csv))

    plan_records = []
    for row in quota_rows:
        plan = build_plan(
            row["language"], row["quota"], fineweb_stats, nemotron_stats,
            args.fineweb_dir, args.nemotron_dir,
        )
        for warning in plan.warnings:
            print(f"[WARN] {plan.lang}: {warning}", file=sys.stderr)

        planned_tokens = int(round(plan.rows_to_sample *
                                   (plan.available_tokens / plan.available_lines))) \
            if plan.available_lines else 0
        upsample_factor = (plan.rows_to_sample / plan.available_lines
                           if plan.available_lines else 0)
        print(f"[PLAN] {plan.lang} dataset={plan.dataset} "
              f"quota={plan.quota_tokens:,} avail={plan.available_tokens:,} "
              f"rows={plan.rows_to_sample:,}/{plan.available_lines:,} "
              f"p={plan.sample_prob:.8f} "
              f"upsample={upsample_factor:.4f}x")

        plan_records.append({
            "language": plan.lang,
            "source_language": plan.source_lang,
            "dataset": plan.dataset,
            "data_dir": str(plan.data_dir),
            "quota_tokens": plan.quota_tokens,
            "available_tokens": plan.available_tokens,
            "available_lines": plan.available_lines,
            "rows_to_sample": plan.rows_to_sample,
            "sample_prob": f"{plan.sample_prob:.12g}",
            "upsample_factor": upsample_factor,
            "planned_tokens": planned_tokens,
        })

        if args.dry_run:
            continue

        out_file = args.output_dir / f"{plan.lang}.jsonl"
        if out_file.exists() and not args.overwrite:
            print(f"[SKIP] exists: {out_file}")
            continue
        if plan.warnings:
            print(f"[SKIP] {plan.lang}: unresolved warnings", file=sys.stderr)
            continue

        written = sample_language(
            plan, out_file, args.seed, args.batch_size,
            args.input_text_column, args.output_text_key, args.include_lang,
        )
        print(f"[DONE] {out_file} rows={written:,}")

    if args.plan_csv and plan_records:
        args.plan_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.plan_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(plan_records[0].keys()))
            writer.writeheader()
            writer.writerows(plan_records)
        print(f"Wrote plan: {args.plan_csv}")


if __name__ == "__main__":
    main()
