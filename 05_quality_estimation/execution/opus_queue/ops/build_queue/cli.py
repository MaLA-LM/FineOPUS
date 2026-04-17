from __future__ import annotations

import argparse

from execution.opus_queue.ops.build_queue.runner import run

__all__ = ["main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate the OPUS queue SQLite DB from lookup_OPUS.csv."
    )
    parser.add_argument(
        "--lookup", required=True, help="Path to lookup_OPUS.csv."
    )
    parser.add_argument(
        "--opus-root",
        required=True,
        help="Root directory that holds OPUS direction subdirs.",
    )
    parser.add_argument(
        "--db", required=True, help="SQLite database file (shared storage)."
    )
    parser.add_argument(
        "--shard-size-override",
        action="append",
        default=[],
        help="Override seed shard sizes: --shard-size-override model:int.",
    )
    parser.add_argument(
        "--reset-pending-for-model",
        default=None,
        help=(
            "Rebuild non-'done' jobs for a single model. Refuses to run if "
            "any 'done' rows exist unless --force is also given."
        ),
    )
    parser.add_argument(
        "--reassign",
        action="store_true",
        help=(
            "For rows whose model in the lookup CSV differs from the DB, delete "
            "only the old non-'done' jobs and insert the new. Refuses to run "
            "if 'done' rows would be discarded unless --force is given."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow destructive rebuilds that discard 'done' rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the summary without writing.",
    )
    parser.add_argument(
        "--count-cache",
        default=None,
        help=(
            "Path to the parquet row-count cache JSON (defaults to "
            "FineOPUS_test/.row_counts.json next to --opus-root)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())
