from __future__ import annotations

import argparse

from dataset.mediator import DatasetAdapter


def ensure_dataset_ready(args: argparse.Namespace, dataset: DatasetAdapter) -> None:
    if args.root is None:
        args.root = dataset.default_root


def load_examples(entry, args: argparse.Namespace, dataset: DatasetAdapter):
    examples = dataset.load_parallel(
        entry.src_lang, entry.tgt_lang, split=entry.split, root=args.root
    )
    return dataset.limit_rows(examples, args.max_rows)
