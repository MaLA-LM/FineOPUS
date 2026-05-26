#!/usr/bin/env python3
"""
Check duplicate rows in Parquet files (all content/score columns below).
Optionally deduplicate in place (keep first occurrence).

Usage:
    python check_duplicate_rows.py /path/to/a.parquet /path/to/b.parquet
    python check_duplicate_rows.py /path/to/dir --recursive
    python check_duplicate_rows.py file.parquet --dedupe
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Columns used for duplicate detection (full row match on these fields).
DEDUPE_COLUMNS = [
    "corpus",
    "version",
    "url",
    "orig_src_lang",
    "orig_tgt_lang",
    "conv_src_lang",
    "conv_tgt_lang",
    "source_text",
    "target_text",
    "src_predlang_id_glotlid",
    "src_predlang_conf_glotlid",
    "tgt_predlang_id_glotlid",
    "tgt_predlang_conf_glotlid",
    "src_predlang_id_conlid",
    "src_predlang_conf_conlid",
    "tgt_predlang_id_conlid",
    "tgt_predlang_conf_conlid",
    "src_lang",
    "tgt_lang",
    "src_char_len",
    "trg_char_len",
    "src_word_len",
    "trg_word_len",
    "src_max_word_len",
    "trg_max_word_len",
    "src_avg_word_len",
    "trg_avg_word_len",
    "char_len_ratio",
    "word_len_ratio",
    "filter_html_src",
    "filter_html_trg",
    "score_term_punct",
    "score_numerals",
    "score_lcs_ratio",
    "score_levenshtein",
    "score_repeat_src",
    "score_repeat_trg",
    "filter_regex_src",
    "similarity_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check duplicate rows by all content/score columns "
            f"({len(DEDUPE_COLUMNS)} fields, see DEDUPE_COLUMNS in script)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more parquet files or directories containing .parquet files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When a path is a directory, search subdirectories for .parquet files.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate rows in source files (keep first row per key).",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print DEDUPE_COLUMNS and exit.",
    )
    return parser.parse_args()


def collect_parquet_files(paths: list[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            if path.suffix != ".parquet":
                raise ValueError(f"Not a parquet file: {path}")
            files.append(path.resolve())
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Path does not exist: {path}")

        pattern = "**/*.parquet" if recursive else "*.parquet"
        found = sorted(path.glob(pattern))
        if not found:
            raise FileNotFoundError(f"No .parquet files found under {path}")
        files.extend(p.resolve() for p in found)

    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def _resolve_subset(df: pd.DataFrame, path: Path) -> list[str]:
    missing = [c for c in DEDUPE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} required column(s) in {path}: {missing}"
        )
    return list(DEDUPE_COLUMNS)


def _dup_stats(df: pd.DataFrame, subset: list[str]) -> dict:
    n_extra = int(df.duplicated(subset=subset).sum())
    n_involved = int(df.duplicated(subset=subset, keep=False).sum())
    n_unique = int(df.drop_duplicates(subset=subset).shape[0])
    n_groups = 0
    max_copies = 1
    size_dist: dict[int, int] = {}

    if n_extra > 0:
        sizes = df.groupby(subset, dropna=False).size()
        dup_sizes = sizes[sizes > 1]
        n_groups = len(dup_sizes)
        max_copies = int(dup_sizes.max())
        size_dist = dup_sizes.value_counts().sort_index().to_dict()

    return {
        "total": len(df),
        "unique_rows": n_unique,
        "extra_rows": n_extra,
        "involved_rows": n_involved,
        "dup_groups": n_groups,
        "max_copies": max_copies,
        "size_dist": size_dist,
        "has_duplicates": n_extra > 0,
        "subset": subset,
    }


def check_file(path: Path) -> dict:
    meta_rows = pq.ParquetFile(path).metadata.num_rows
    df = pd.read_parquet(path, columns=DEDUPE_COLUMNS)
    subset = _resolve_subset(df, path)

    if len(df) != meta_rows:
        print(
            f"Warning: metadata rows ({meta_rows:,}) != loaded rows ({len(df):,}) "
            f"in {path}",
            file=sys.stderr,
        )

    stats = _dup_stats(df, subset)
    return {"path": path, "deduped": False, "rows_removed": 0, **stats}


def dedupe_file(path: Path) -> dict:
    df = pd.read_parquet(path)
    subset = _resolve_subset(df, path)

    before = len(df)
    stats_before = _dup_stats(df[subset], subset)
    rows_removed = stats_before["extra_rows"]

    if rows_removed == 0:
        return {
            "path": path,
            "deduped": False,
            "rows_removed": 0,
            **stats_before,
        }

    df_out = df.drop_duplicates(subset=subset, keep="first")
    tmp_path = path.with_name(f".{path.name}.dedupe.tmp")
    try:
        df_out.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    stats_after = _dup_stats(df_out[subset], subset)
    return {
        "path": path,
        "deduped": True,
        "rows_removed": rows_removed,
        "rows_before": before,
        "rows_after": len(df_out),
        **stats_after,
    }


def print_report(result: dict) -> None:
    subset = result["subset"]
    print(f"\n=== {result['path']} ===")
    print(f"Key columns ({len(subset)}): {', '.join(subset)}")
    if result.get("deduped"):
        print(f"Rows before dedupe:   {result['rows_before']:,}")
        print(f"Rows removed:         {result['rows_removed']:,}")
        print(f"Rows after dedupe:    {result['rows_after']:,}")
    print(f"Total rows:           {result['total']:,}")
    print(f"Unique rows:          {result['unique_rows']:,}")
    print(f"Extra duplicate rows: {result['extra_rows']:,}")
    print(f"Rows in dup groups:   {result['involved_rows']:,}")
    print(f"Duplicate groups:     {result['dup_groups']:,}")
    if result["has_duplicates"]:
        print(f"Max copies per group: {result['max_copies']}")
        print("Group size distribution (copies -> count):")
        for size, count in sorted(result["size_dist"].items()):
            print(f"  {size}: {count:,}")
    if result.get("deduped"):
        print("Deduped in place:     YES")
    print(f"Has duplicates:       {'YES' if result['has_duplicates'] else 'NO'}")


def main() -> int:
    args = parse_args()
    if args.list_columns:
        for col in DEDUPE_COLUMNS:
            print(col)
        return 0

    files = collect_parquet_files(args.paths, args.recursive)

    any_dup = False
    total_rows = 0
    total_extra = 0
    total_removed = 0

    for f in files:
        result = dedupe_file(f) if args.dedupe else check_file(f)
        print_report(result)
        total_rows += result["total"]
        total_extra += result["extra_rows"]
        total_removed += result.get("rows_removed", 0)
        any_dup = any_dup or result.get("rows_removed", 0) > 0 or result["has_duplicates"]

    if len(files) > 1:
        print("\n=== Summary (all files) ===")
        print(f"Files checked:        {len(files)}")
        print(f"Key columns:          {len(DEDUPE_COLUMNS)}")
        print(f"Total rows:           {total_rows:,}")
        print(f"Extra duplicate rows: {total_extra:,}")
        if args.dedupe:
            print(f"Rows removed:         {total_removed:,}")
        print(f"Has duplicates:       {'YES' if any_dup else 'NO'}")

    if args.dedupe:
        return 1 if total_extra > 0 else 0
    return 1 if any_dup else 0


if __name__ == "__main__":
    sys.exit(main())
