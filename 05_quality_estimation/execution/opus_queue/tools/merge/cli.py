from __future__ import annotations

import argparse

from execution.opus_queue.tools.merge.runner import run

__all__ = ["main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge completed per-shard OPUS outputs into one Parquet file per direction."
    )
    parser.add_argument("--db", required=True, help="Shared jobs SQLite file.")
    parser.add_argument(
        "--output-base",
        required=True,
        help="Base dir of shard JSONLs (<model>/<direction>/shard_*.jsonl).",
    )
    parser.add_argument(
        "--merged-base",
        required=True,
        help="Destination dir; merged files go to <merged-base>/<model>/<direction>.parquet.",
    )
    parser.add_argument(
        "--model", default=None, help="Optional: restrict merge to a single model."
    )
    parser.add_argument(
        "--delete-shards",
        action="store_true",
        help="Remove the source shard files after a successful merge.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-merge directions even if the output file exists.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())
