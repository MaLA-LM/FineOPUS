from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from dataset.manifest import ManifestEntry
from dataset.mediator import get_dataset
from models.language_support import CometLanguages
from models.model_registry import resolve_model_spec, supported_model_keys
from src.bicleaner_backend import (
    ISO639_1_BY_BASE_NAME,
    iso639_1_from_dataset,
    read_scores,
    run_bicleaner,
    select_model_id,
    write_tsv,
)
from src.common import (
    ensure_dataset_ready,
    load_examples,
    sanitize_model_tag,
    summarize_scores,
)
from utils.args import add_common_scoring_args
from utils.cli import validate_args
from utils.frames import build_frames
from utils.logger import logger
from utils.runner import collect_directions, run_scoring

BICLEANER_MODEL_IDS = {
    key: resolve_model_spec(key, "bicleaner")[0].model_id
    for key in ("en-xx", "es-xx", "de-xx")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a dataset split with bicleaner-ai via CLI."
    )
    add_common_scoring_args(
        parser,
        batch_size_default=None,
        batch_size_help="Ignored (kept for CLI compatibility).",
        gpus_default=None,
        gpus_help="Ignored (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help=(
            "Model selector or full bitextor repo. "
            f"Supported: {', '.join(supported_model_keys('bicleaner'))}."
        ),
    )
    return parser.parse_args()


def resolve_model(name: str) -> tuple[str, str]:
    _, key = resolve_model_spec(name, "bicleaner")
    model_key = "bicleaner-ai" if key == "auto" else key
    return key, model_key


def score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    model_selector: str,
    language_support: CometLanguages,
    dataset,
):
    examples = load_examples(entry, args, dataset)

    src_iso = iso639_1_from_dataset(entry.src_lang, dataset, ISO639_1_BY_BASE_NAME)
    tgt_iso = iso639_1_from_dataset(entry.tgt_lang, dataset, ISO639_1_BY_BASE_NAME)
    model_id = select_model_id(model_selector, src_iso, tgt_iso, BICLEANER_MODEL_IDS)

    # Use scratch (output_base) for temp files instead of /tmp, which is
    # too small on Mahti compute nodes and fills up across array tasks.
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

    src_lang_seen = language_support.support_status(entry.src_lang)
    tgt_lang_seen = language_support.support_status(entry.tgt_lang)
    mean_score, median_score = summarize_scores(scores)

    return build_frames(
        model_id,
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
    try:
        model_selector, model_key = resolve_model(args.model)
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
    language_support = CometLanguages(dataset=dataset)

    run_scoring(
        args,
        dataset,
        directions,
        model_tag,
        lambda entry: score_entry(
            entry, args, model_selector, language_support, dataset
        ),
    )


if __name__ == "__main__":
    main()
