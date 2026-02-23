from __future__ import annotations

import argparse
import math
import re
import statistics

from dataset.mediator import DatasetAdapter


def ensure_dataset_ready(args: argparse.Namespace, dataset: DatasetAdapter) -> None:
    if args.root is None:
        args.root = dataset.default_root


def summarize_scores(scores: list[float]) -> tuple[float | None, float | None]:
    valid = [s for s in scores if not math.isnan(s)]
    if not valid:
        return None, None
    return float(statistics.fmean(valid)), float(statistics.median(valid))


def sanitize_model_tag(name: str) -> str:
    tag = name.strip().lower()
    tag = re.sub(r"[^a-z0-9._-]+", "-", tag)
    tag = tag.strip("-")
    return tag or "model"


def load_examples(entry, args: argparse.Namespace, dataset: DatasetAdapter):
    examples = dataset.load_parallel(
        entry.src_lang, entry.tgt_lang, split=entry.split, root=args.root
    )
    return dataset.limit_rows(examples, args.max_rows)
