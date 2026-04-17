from __future__ import annotations

import argparse
from collections.abc import Sequence

from dataset.mediator import DEFAULT_DATASET_ID
from execution import get_executor, list_executor_names

_MISSING = object()


def add_common_scoring_args(
    parser: argparse.ArgumentParser,
    *,
    batch_size_default: int | None | object = _MISSING,
    batch_size_help: str | None = None,
    gpus_default: int | None | object = _MISSING,
    gpus_help: str | None = None,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_ID,
        help="Dataset id to use.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory of the dataset files (defaults to dataset root).",
    )
    parser.add_argument(
        "--output-base",
        required=True,
        help="Base output directory for stage outputs and compacted outputs.",
    )
    parser.add_argument(
        "--execution",
        default="flores_array",
        choices=list_executor_names(),
        help="Execution strategy used to schedule and write work.",
    )
    if batch_size_default is not _MISSING:
        parser.add_argument(
            "--batch-size",
            type=int,
            default=batch_size_default,
            help=batch_size_help or "Batch size for model prediction.",
        )
    if gpus_default is not _MISSING:
        parser.add_argument(
            "--gpus",
            type=int,
            default=gpus_default,
            help=gpus_help or "Number of GPUs to use (set 0 for CPU).",
        )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for number of rows (for debugging).",
    )
    return parser


def add_selected_executor_args(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None = None
) -> argparse.ArgumentParser:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--execution", default="flores_array")
    known_args, _ = bootstrap.parse_known_args(argv)
    get_executor(known_args.execution).add_cli_args(parser)
    return parser
