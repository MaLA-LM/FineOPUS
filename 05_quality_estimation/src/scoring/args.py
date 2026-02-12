from __future__ import annotations

import argparse

from dataset.mediator import DEFAULT_DATASET_ID

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
    parser.add_argument("--src-lang", help="Source language code.")
    parser.add_argument("--tgt-lang", help="Target language code.")
    parser.add_argument(
        "--split",
        default="devtest",
        help="Dataset split to score.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory of the dataset files (defaults to dataset root).",
    )
    parser.add_argument("--output", default=None, help="Output Parquet path.")
    parser.add_argument(
        "--output-base",
        default=None,
        help="Base output directory for partitioned Parquet dataset output.",
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
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip completed outputs when they are valid.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Recompute outputs even if they already exist.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="TSV manifest with columns: src_lang, tgt_lang, split.",
    )
    parser.add_argument(
        "--discover-all",
        action="store_true",
        help="Discover all directions under --root and score each.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Continuously claim and score free directions from the manifest.",
    )
    parser.add_argument(
        "--worker-max-files",
        type=int,
        default=200,
        help="Max outputs to write before worker exits (0 for unlimited).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for number of rows (for debugging).",
    )
    return parser
