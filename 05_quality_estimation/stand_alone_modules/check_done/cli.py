from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stand_alone_modules.check_done.checkpoint_status import (
    build_report,
    discover_splits,
    format_report,
    load_completed,
    load_expected,
)

DEFAULT_PATH = "/scratch/project_2008161/QE_flores200_scores/dataset=flores200"
DEFAULT_MODEL = "bicleaner-ai"
DEFAULT_DATASET = "flores200"

__all__ = ["main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check shard completion by comparing a manifest TSV with checkpoint.jsonl "
            "summary rows under model split folders."
        )
    )
    parser.add_argument(
        "--tsv",
        required=True,
        help="Path to TSV with columns: src_lang, tgt_lang, split, shard_id.",
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"Dataset output root path. Default: {DEFAULT_PATH}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model tag under the dataset path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset id used to resolve split values. Default: {DEFAULT_DATASET}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_root = Path(args.path) / f"model={args.model}"
    expected = load_expected(args.tsv)
    discovered_splits = discover_splits(args.dataset)
    completed = load_completed(model_root, discovered_splits)
    report = build_report(expected, completed)
    print(format_report(report))

    if not model_root.exists():
        print(
            f"warning: model directory does not exist: {model_root}",
            file=sys.stderr,
        )
