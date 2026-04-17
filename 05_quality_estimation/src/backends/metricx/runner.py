from __future__ import annotations

import argparse

from execution.flores_array.manifest import ManifestEntry
from models.language_support import MetricX24Languages
from src.backends.metricx.backend import apply_score_adjustment, score_metricx
from src.common import build_scored_frames, load_examples

__all__ = ["score_entry"]


def score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    spec,
    tokenizer,
    model,
    device,
    language_support: MetricX24Languages,
    dataset,
):
    examples = load_examples(entry, args, dataset)
    scores = score_metricx(examples, spec, tokenizer, model, args.batch_size, device)
    scores = apply_score_adjustment(scores, spec)
    return build_scored_frames(
        dataset, entry, spec.model_id, scores, examples, language_support
    )
