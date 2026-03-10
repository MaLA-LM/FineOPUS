from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact stage JSONL outputs into per-model parquet buckets."
    )
    parser.add_argument("--output-base", required=True, help="Base output directory.")
    parser.add_argument("--dataset", required=True, help="Dataset id.")
    parser.add_argument(
        "--target-part-bytes",
        type=int,
        default=268_435_456,  # 256 MiB
        help="Best-effort target size per compacted output file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=9,
        help="Max parallel threads (one model per thread).",
    )
    return parser.parse_args()
