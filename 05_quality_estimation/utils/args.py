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
        "--manifest",
        required=True,
        help="TSV manifest with columns: src_lang, tgt_lang, split[, shard_id].",
    )
    parser.add_argument(
        "--max-directions-per-part",
        type=int,
        default=25,
        help="Close and commit a part after this many directions.",
    )
    parser.add_argument(
        "--target-part-bytes",
        type=int,
        default=67_108_864,  # 64 MiB
        help="Best-effort target bytes before rotating a part file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for number of rows (for debugging).",
    )
    return parser
