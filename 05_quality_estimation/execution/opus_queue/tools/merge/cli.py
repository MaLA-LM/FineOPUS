from __future__ import annotations

import argparse

from execution.opus_queue.tools.merge.runner import run

__all__ = ["main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge completed per-shard OPUS outputs into one or more Parquet files per direction."
    )
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "Legacy shared jobs SQLite file. If manifest trace args are also "
            "provided, DB done rows and trace done rows are combined."
        ),
    )
    parser.add_argument(
        "--manifest-root",
        default=None,
        help=(
            "Manifest root containing <build_tag>/manifest.jsonl. Provide with "
            "--build-tag and --trace-root; can be combined with --db."
        ),
    )
    parser.add_argument(
        "--build-tag",
        default=None,
        help="Manifest build tag to merge from.",
    )
    parser.add_argument(
        "--trace-root",
        default=None,
        help="Trace root containing <build_tag>/<worker_slot>/state.jsonl.",
    )
    parser.add_argument(
        "--output-base",
        required=True,
        help="Base dir of shard JSONLs / part files (<model>/<direction>/*.jsonl).",
    )
    parser.add_argument(
        "--merged-base",
        required=True,
        help=(
            "Destination dir; merged files go to "
            "<merged-base>/<model>/<direction>/<direction>.part-0000.parquet "
            "(and more parts if the direction exceeds 5 GB)."
        ),
    )
    parser.add_argument(
        "--model", default=None, help="Optional: restrict merge to a single model."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-merge directions even if merged parquet outputs already exist.",
    )
    parser.add_argument(
        "--delete-merged-directions",
        "--cleanup-merged-inputs",
        dest="cleanup_merged_inputs",
        action="store_true",
        help=(
            "Cleanup mode: delete source direction directories from output-base "
            "when matching merged parquet outputs already exist, then exit. "
            "This does not run merging."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --delete-merged-directions, report what would be deleted.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of directions to merge in parallel. Default: 1.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())
