from __future__ import annotations

import argparse
from pathlib import Path

from stand_alone_modules.opus_stats.summarize import (
    DEFAULT_MERGED_BASE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_THRESHOLDS,
    DEFAULT_TMP_DIR,
    OpusStatsConfig,
    run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize OPUS scored merged parquet files with DuckDB SQL."
        )
    )
    parser.add_argument(
        "--merged-base",
        default=str(DEFAULT_MERGED_BASE),
        help="Merged OPUS output root containing one directory per model.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Restrict to one model directory under --merged-base. "
            "Can be passed multiple times. Default: summarize all models."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory for CSV and Markdown outputs. "
            "Default: <repo root>/data/stats."
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="QE score thresholds for pooled retention percentages.",
    )
    parser.add_argument(
        "--tmp-dir",
        default=str(DEFAULT_TMP_DIR),
        help="DuckDB temporary directory, preferably on scratch storage.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="DuckDB worker threads.",
    )
    parser.add_argument(
        "--memory-limit",
        default=None,
        help="Optional DuckDB memory limit, for example 180GB.",
    )
    parser.add_argument(
        "--max-temp-size",
        default=None,
        help="Optional DuckDB max temp directory size, for example 500GiB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_base = Path(args.merged_base).expanduser()
    run(
        OpusStatsConfig(
            merged_base=merged_base,
            models=None if args.model is None else tuple(args.model),
            output_dir=Path(args.output_dir).expanduser(),
            thresholds=tuple(args.thresholds),
            tmp_dir=Path(args.tmp_dir).expanduser(),
            threads=args.threads,
            memory_limit=args.memory_limit,
            max_temp_size=args.max_temp_size,
        )
    )
