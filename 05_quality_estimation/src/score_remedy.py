from __future__ import annotations

import argparse
import os
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory

from dataset.manifest import ManifestEntry
from dataset.mediator import get_dataset
from models.language_support import RemedyLanguages
from models.model_registry import resolve_model_spec, supported_model_keys
from src.common import (
    ensure_dataset_ready,
    load_examples,
    sanitize_model_tag,
    summarize_scores,
)
from src.remedy_backend import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    read_calibration_scores,
    resolve_calibration_scores_path,
    run_remedy,
    write_parallel_files,
)
from utils.args import add_common_scoring_args
from utils.cli import validate_args
from utils.frames import build_frames
from utils.logger import logger
from utils.runner import collect_directions, run_scoring

DEFAULT_REMEDY_MODEL = "remedy"


def default_cache_dir() -> Path:
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.getenv(key)
        if value:
            return Path(value)
    return Path(".cache") / "huggingface" / "hub"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a dataset split with ReMedy via remedy-score CLI."
    )
    add_common_scoring_args(
        parser,
        batch_size_default=None,
        batch_size_help="Ignored (kept for CLI compatibility).",
        gpus_default=1,
        gpus_help="Number of GPUs to pass to remedy-score (--num_gpus).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_REMEDY_MODEL,
        help=(
            "Model name or HF repo id. "
            f"Supported: {', '.join(supported_model_keys('remedy'))}."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(default_cache_dir()),
        help=(
            "Cache directory passed to remedy-score --cache_dir. "
            "Defaults to HF_HUB_CACHE/HUGGINGFACE_HUB_CACHE, then .cache/huggingface/hub."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Value passed to remedy-score --gpu_memory_utilization.",
    )
    return parser.parse_args()


def resolve_model(name: str):
    # If --model points to a local directory, build a spec on the fly
    # so remedy-score receives the full path directly.
    model_path = Path(name)
    if model_path.is_dir():
        from models.model_registry import ModelSpec

        key = model_path.name.lower().replace(" ", "-")
        spec = ModelSpec(
            key=key,
            backend="remedy",
            model_id=str(model_path),
        )
        return spec, key
    return resolve_model_spec(name, "remedy")


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

    src_lang_seen = language_support.support_status(entry.src_lang)
    tgt_lang_seen = language_support.support_status(entry.tgt_lang)

    src_lang_name = language_support.get_full_language_name(entry.src_lang)
    tgt_lang_name = language_support.get_full_language_name(entry.tgt_lang)
    remedy_src_lang = language_support._iso639_1_from_language(src_lang_name)
    remedy_tgt_lang = language_support._iso639_1_from_language(tgt_lang_name)

    # default to 'en' if language not recognized by ReMedy
    if not remedy_src_lang:
        logger.warning(
            f"Source language '{entry.src_lang}' not recognized by ReMedy. "
            "Defaulting to 'en' for scoring. Scores may be inaccurate."
        )
        remedy_src_lang = "en"
    if not remedy_tgt_lang:
        logger.warning(
            f"Target language '{entry.tgt_lang}' not recognized by ReMedy. "
            "Defaulting to 'en' for scoring. Scores may be inaccurate."
        )
        remedy_tgt_lang = "en"

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

    mean_score, median_score = summarize_scores(scores)
    return build_frames(
        spec.model_id,
        dataset.id,
        entry.split,
        entry.src_lang,
        entry.tgt_lang,
        scores,
        examples,
        src_lang_seen=src_lang_seen,
        tgt_lang_seen=tgt_lang_seen,
        mean=mean_score,
        median=median_score,
    )


def main() -> None:
    args = parse_args()
    if args.gpus <= 0:
        raise SystemExit("--gpus must be >= 1 for remedy-score.")
    if args.gpu_memory_utilization <= 0 or args.gpu_memory_utilization > 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1].")
    try:
        spec, model_key = resolve_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    dataset = get_dataset(args.dataset)
    ensure_dataset_ready(args, dataset)
    validate_args(args)

    directions = collect_directions(args, dataset)
    if not directions:
        logger.info("No directions found.")
        return

    model_tag = sanitize_model_tag(model_key)
    language_support = RemedyLanguages(dataset=dataset)
    cache_dir = Path(args.cache_dir)
    iteration_counter = count()

    run_scoring(
        args,
        dataset,
        directions,
        model_tag,
        lambda entry: score_entry(
            entry,
            args,
            spec,
            dataset,
            language_support,
            cache_dir,
            iteration=next(iteration_counter),
        ),
    )


if __name__ == "__main__":
    main()
