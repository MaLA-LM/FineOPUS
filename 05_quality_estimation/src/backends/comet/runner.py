from __future__ import annotations

import argparse

from execution.flores_array.manifest import ManifestEntry
from models.language_support import CometLanguages
from src.backends.comet.backend import score_comet
from src.common import build_scored_frames, load_examples

__all__ = ["score_entry"]


def score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    model,
    model_id: str,
    language_support: CometLanguages,
    dataset,
):
    examples = load_examples(entry, args, dataset)
    scores = score_comet(examples, model, args.batch_size, args.gpus)
    return build_scored_frames(
        dataset, entry, model_id, scores, examples, language_support
    )
