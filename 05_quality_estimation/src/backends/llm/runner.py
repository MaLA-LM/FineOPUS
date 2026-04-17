from __future__ import annotations

import argparse

from dataset.mediator import DatasetAdapter
from execution.flores_array.manifest import ManifestEntry
from models.language_support import BaseLanguageSupport
from src.backends.llm.backend import score_llm_offline
from src.common import build_scored_frames, load_examples

__all__ = ["score_entry"]


def score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    engine,
    model_name: str,
    language_support: BaseLanguageSupport,
    dataset: DatasetAdapter,
):
    examples = load_examples(entry, args, dataset)
    src_lang_name = language_support.get_full_language_name(entry.src_lang)
    tgt_lang_name = language_support.get_full_language_name(entry.tgt_lang)

    scores = score_llm_offline(
        engine,
        examples,
        src_lang_name,
        tgt_lang_name,
        prompt_mode=args.prompt_mode,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        response_format=args.response_format,
    )
    return build_scored_frames(
        dataset, entry, model_name, scores, examples, language_support
    )
