from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact stage outputs into bucketed parquet files."
    )
    parser.add_argument("--output-base", required=True, help="Base output directory.")
    parser.add_argument("--dataset", required=True, help="Dataset id.")
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=32,
        help="Number of hash buckets in final output layout.",
    )
    parser.add_argument(
        "--target-part-bytes",
        type=int,
        default=268_435_456,
        help="Best-effort target size per compacted output file.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id for compacted output filenames.",
    )
    return parser.parse_args()
