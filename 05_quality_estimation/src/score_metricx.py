from __future__ import annotations

import argparse

from dataset.manifest import ManifestEntry
from dataset.mediator import get_dataset
from models.language_support import MetricX24Languages
from models.model_registry import resolve_model_spec, supported_model_keys
from src.common import (
    ensure_dataset_ready,
    load_examples,
    sanitize_model_tag,
    summarize_scores,
)
from src.metricx_backend import (
    apply_score_adjustment,
    load_metricx,
    score_metricx,
    select_device,
)
from utils.args import add_common_scoring_args
from utils.cli import validate_args
from utils.frames import build_frames
from utils.logger import logger
from utils.runner import collect_directions, run_scoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a dataset split with MetricX-24 QE."
    )
    add_common_scoring_args(
        parser,
        batch_size_default=8,
        gpus_default=1,
    )
    parser.add_argument(
        "--model",
        default="metricx24",
        help=(
            "Model name or HF repo. "
            f"Supported: {', '.join(supported_model_keys('metricx'))}."
        ),
    )
    return parser.parse_args()


def resolve_model(name: str):
    return resolve_model_spec(name, "metricx")


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

    src_lang_seen = language_support.support_status(entry.src_lang)
    tgt_lang_seen = language_support.support_status(entry.tgt_lang)
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
    device = select_device(args.gpus)
    tokenizer, model = load_metricx(spec, device)
    language_support = MetricX24Languages(dataset=dataset)

    run_scoring(
        args,
        dataset,
        directions,
        model_tag,
        lambda entry: score_entry(
            entry, args, spec, tokenizer, model, device, language_support, dataset
        ),
    )


if __name__ == "__main__":
    main()
