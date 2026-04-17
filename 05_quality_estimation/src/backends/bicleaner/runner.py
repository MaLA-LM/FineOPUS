from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from execution.flores_array.manifest import ManifestEntry
from models.language_support import CometLanguages
from src.backends.bicleaner.backend import (
    BICLEANER_MODEL_IDS,
    iso639_1_from_dataset,
    read_scores,
    run_bicleaner,
    select_model_id,
    write_tsv,
)
from src.common import build_scored_frames, load_examples

__all__ = ["score_entry"]


def score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    model_selector: str,
    language_support: CometLanguages,
    dataset,
):
    examples = load_examples(entry, args, dataset)

    src_iso = iso639_1_from_dataset(entry.src_lang, dataset)
    tgt_iso = iso639_1_from_dataset(entry.tgt_lang, dataset)
    model_id = select_model_id(model_selector, src_iso, tgt_iso, BICLEANER_MODEL_IDS)

    tmp_parent = Path(args.output_base) / ".tmp_bicleaner"
    tmp_parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=tmp_parent) as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_path = tmp_root / "input.tsv"
        output_path = tmp_root / "output.tsv"
        write_tsv(examples, input_path)

        run_bicleaner(
            input_path,
            output_path,
            model_id,
            src_iso,
            tgt_iso,
        )
        scores = read_scores(output_path)

    if len(scores) != len(examples):
        raise RuntimeError(
            f"bicleaner-ai returned {len(scores)} scores for {len(examples)} rows."
        )

    return build_scored_frames(
        dataset, entry, model_id, scores, examples, language_support
    )
