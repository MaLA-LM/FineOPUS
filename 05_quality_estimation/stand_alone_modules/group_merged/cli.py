from __future__ import annotations

import argparse
from pathlib import Path

from stand_alone_modules.group_merged.grouper import group_merged_outputs
from utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move flat OPUS merged parquet/meta files into per-direction "
            "subdirectories."
        )
    )
    parser.add_argument(
        "--merged-base",
        required=True,
        help="Merged output base containing model directories.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Restrict to one model directory. Can be passed multiple times. "
            "Default: process every model directory under --merged-base."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    args = parser.parse_args()
    args.merged_base = Path(args.merged_base).expanduser().resolve()
    if not args.merged_base.is_dir():
        raise SystemExit(
            f"Merged base does not exist or is not a directory: {args.merged_base}"
        )
    return args


def main() -> None:
    args = parse_args()
    summary = group_merged_outputs(
        args.merged_base,
        models=args.model,
        dry_run=args.dry_run,
        force=args.force,
    )
    if summary.conflicts:
        logger.error(
            "Found %d destination conflict(s). Re-run with --force to overwrite.",
            len(summary.conflicts),
        )
        for path in summary.conflicts[:20]:
            logger.error("Conflict: %s", path)
        raise SystemExit(1)
    logger.info(
        "Grouped merged outputs: models=%d directions=%d planned=%d moved=%d",
        summary.models_seen,
        summary.directions_seen,
        summary.files_planned,
        summary.files_moved,
    )
