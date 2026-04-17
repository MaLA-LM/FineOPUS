from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from dataset.flores200.langcode_mapping import build_model_language_mapping
from models.language_data.metricx24 import METRICX24_SUPPORTED_LANGUAGES
from stand_alone_modules.lookup_table.opus_data_loader import (
    count_all_directions,
    load_flores_lookup,
    load_model_runtime,
    scan_opus_directions,
)
from stand_alone_modules.lookup_table.opus_matcher import (
    HIGH_RESOURCE_SENTENCE_THRESHOLD,
    METRICX_MODEL,
    build_opus_rows,
)

DEFAULT_OPUS_PATH = "/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2"
DEFAULT_TABLES_DIR = Path(__file__).resolve().parent / "tables"
DEFAULT_OUTPUT = Path("data/lookups/lookup_OPUS.csv")

OUTPUT_COLUMNS = [
    "direction_key",
    "winner_model",
    "winner_avg_score",
    "num_sentences",
    "rate_per_hour",
    "est_hours",
]

__all__ = ["main", "parse_args"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate lookup_OPUS.csv from OPUS direction directories."
    )
    parser.add_argument(
        "--opus-path",
        default=DEFAULT_OPUS_PATH,
        help="Root directory containing OPUS translation-direction sub-dirs "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--tables-dir",
        default=str(DEFAULT_TABLES_DIR),
        help="Directory holding lookup_flores.csv and model_runtime.csv "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--default-model",
        choices=["qwen3", "metricx-24", "both"],
        default="both",
        help="Default model for unseen directions: "
        "qwen3 = always qwen3-4b, "
        "metricx-24 = always metricx-24, "
        "both = metricx-24 where either source or target lang is supported "
        "else qwen3-4b (default: %(default)s)",
    )
    return parser.parse_args(argv)


def _build_metricx_supported_codes():
    mapping = build_model_language_mapping(METRICX24_SUPPORTED_LANGUAGES)
    return frozenset(
        code for code, (is_supported, _) in mapping.items() if is_supported
    )


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_report(matched, unmatched, rows, strategy, stats):
    total = len(matched) + len(unmatched)
    final_metricx = sum(1 for row in rows if row["winner_model"] == METRICX_MODEL)
    qwen_total = stats["matched_qwen_winners"]
    qwen_high_resource = stats["matched_qwen_high_resource"]
    qwen_shifted_to_metricx = stats["matched_qwen_reassigned_to_metricx"]
    qwen_shifted_support_only = stats["matched_qwen_reassigned_support_only"]
    qwen_shifted_gap_only = stats["matched_qwen_reassigned_gap_only"]
    qwen_shifted_both = stats["matched_qwen_reassigned_both"]
    qwen_kept = qwen_total - qwen_shifted_to_metricx
    qwen_high_resource_kept = qwen_high_resource - qwen_shifted_to_metricx

    print(f"\n{'=' * 55}")
    print("  OPUS -> FLORES matching report")
    print(f"{'=' * 55}")
    print(f"  Default-model strategy    : {strategy}")
    print(f"  Total directions scanned  : {total}")
    print(f"  Matched with FLORES       : {len(matched)}")
    if qwen_total:
        print(f"  Matched Qwen winners      : {qwen_total}")
        print(
            "    high-resource qwen winners"
            f" (>{HIGH_RESOURCE_SENTENCE_THRESHOLD:,}): {qwen_high_resource}"
        )
        print(f"    reassigned -> metricx24 : {qwen_shifted_to_metricx}")
        if qwen_shifted_to_metricx:
            print(f"      support only          : {qwen_shifted_support_only}")
            print(f"      score-gap only        : {qwen_shifted_gap_only}")
            print(f"      support + score-gap   : {qwen_shifted_both}")
        print(f"    kept on qwen3-4b        : {qwen_kept}")
        if qwen_high_resource:
            shifted_share = qwen_shifted_to_metricx / qwen_high_resource * 100
            print(
                "    high-resource kept on qwen3-4b:"
                f" {qwen_high_resource_kept}"
            )
            print(
                "    qwen-split trigger rate :"
                f" {shifted_share:.1f}% of high-resource qwen winners"
            )
        if final_metricx:
            metricx_share = qwen_shifted_to_metricx / final_metricx * 100
            print(
                "    contribution to metricx24:"
                f" +{qwen_shifted_to_metricx} directions"
                f" ({metricx_share:.1f}% of all metricx24 assignments)"
            )

    if strategy == "both":
        unmatched_set = set(unmatched)
        n_metricx = sum(
            1
            for r in rows
            if r["direction_key"] in unmatched_set
            and r["winner_model"] == "metricx24"
        )
        n_qwen = len(unmatched) - n_metricx
        print(f"  Unmatched (default models): {len(unmatched)}")
        print(f"    metricx24 (src/tgt supported): {n_metricx}")
        print(f"    qwen3-4b (default):           {n_qwen}")
    else:
        label = "metricx24" if strategy == "metricx-24" else "qwen3-4b"
        print(f"  Unmatched (-> {label})      : {len(unmatched)}")

    if total:
        print(f"  Match rate                : {len(matched) / total * 100:.1f}%")
    if unmatched:
        print("\n  First 10 unmatched directions:")
        for direction in unmatched[:10]:
            print(f"    - {direction}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")
    print(f"{'=' * 55}\n")


def main(argv=None):
    args = parse_args(argv)
    tables_dir = Path(args.tables_dir)
    output_path = Path(args.output)

    flores_csv = tables_dir / "lookup_flores.csv"
    runtime_csv = tables_dir / "model_runtime.csv"

    if not os.path.isdir(args.opus_path):
        sys.exit(f"Error: OPUS path does not exist: {args.opus_path}")
    for path in (flores_csv, runtime_csv):
        if not path.is_file():
            sys.exit(f"Error: required table not found: {path}")

    flores_lookup = load_flores_lookup(str(flores_csv))
    runtime_lookup = load_model_runtime(str(runtime_csv))
    opus_dirs = scan_opus_directions(args.opus_path)

    if not opus_dirs:
        sys.exit(f"Error: no sub-directories found under {args.opus_path}")

    strategy = args.default_model
    metricx_supported_codes = frozenset()
    if strategy in ("metricx-24", "both"):
        metricx_supported_codes = _build_metricx_supported_codes()
        print(f"MetricX-24 covers {len(metricx_supported_codes)} FLORES codes")
    print(f"Default-model strategy: {strategy}")

    print(f"Counting sentences across {len(opus_dirs)} directions (parallel)...")
    sentence_counts = count_all_directions(args.opus_path, opus_dirs)

    rows, matched, unmatched, stats = build_opus_rows(
        opus_dirs,
        sentence_counts,
        flores_lookup,
        runtime_lookup,
        default_strategy=strategy,
        metricx_supported_codes=metricx_supported_codes,
    )

    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")

    print_report(matched, unmatched, rows, strategy, stats)
