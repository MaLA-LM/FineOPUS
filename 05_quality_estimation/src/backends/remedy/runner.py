from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from execution.flores_array.manifest import ManifestEntry
from models.language_support import RemedyLanguages
from src.backends.remedy.backend import (
    read_calibration_scores,
    resolve_calibration_scores_path,
    run_remedy,
    write_parallel_files,
)
from src.backends.remedy.lang_mapping import map_lang_codes_to_iso
from src.common import build_scored_frames, load_examples

__all__ = ["score_entry"]


def score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    spec,
    dataset,
    language_support: RemedyLanguages,
    cache_dir: Path,
    iteration: int = 0,
):
    examples = load_examples(entry, args, dataset)
    remedy_src_lang, remedy_tgt_lang = map_lang_codes_to_iso(
        entry.src_lang, entry.tgt_lang, language_support
    )

    tmp_parent = Path(args.output_base) / ".tmp_remedy"
    tmp_parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=tmp_parent) as tmp_dir:
        tmp_root = Path(tmp_dir)
        src_file, mt_file = write_parallel_files(
            examples, tmp_root, remedy_src_lang, remedy_tgt_lang
        )
        run_remedy(
            model_id=spec.model_id,
            src_file=src_file,
            mt_file=mt_file,
            src_lang=remedy_src_lang,
            tgt_lang=remedy_tgt_lang,
            cache_dir=cache_dir,
            save_dir=tmp_root,
            num_gpus=args.gpus,
            gpu_memory_utilization=args.gpu_memory_utilization,
            iteration=iteration,
        )
        scores_path = resolve_calibration_scores_path(
            save_dir=tmp_root,
            model_id=spec.model_id,
            src_lang=remedy_src_lang,
            tgt_lang=remedy_tgt_lang,
        )
        scores = read_calibration_scores(scores_path)

    if len(scores) != len(examples):
        raise RuntimeError(
            f"remedy-score returned {len(scores)} scores for {len(examples)} rows."
        )

    return build_scored_frames(
        dataset, entry, spec.model_id, scores, examples, language_support
    )
