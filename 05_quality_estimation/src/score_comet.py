from __future__ import annotations

import argparse

from dataset.manifest import ManifestEntry
from dataset.mediator import Example, get_dataset
from models.language_support import CometLanguages
from models.model_registry import resolve_model_spec, supported_model_keys
from src.common import (
    ensure_dataset_ready,
    load_examples,
    sanitize_model_tag,
    summarize_scores,
)
from utils.args import add_common_scoring_args
from utils.cli import validate_args
from utils.frames import build_frames
from utils.runner import collect_directions, run_scoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a dataset split with COMET-style QE models."
    )
    add_common_scoring_args(
        parser,
        batch_size_default=8,
        gpus_default=1,
    )
    parser.add_argument(
        "--model",
        default="wmt22-cometkiwi-da",
        help=(
            "Model name or HF repo. "
            f"Supported: {', '.join(supported_model_keys('comet'))}."
        ),
    )
    return parser.parse_args()


def resolve_model(name: str):
    return resolve_model_spec(name, "comet")


def load_comet_model(model_id: str):
    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_id)
    return load_from_checkpoint(model_path)


def score_comet(
    examples: list[Example],
    model,
    batch_size: int,
    gpus: int,
) -> list[float]:
    samples = [{"src": ex["src"], "mt": ex["tgt"]} for ex in examples]
    prediction = model.predict(samples, batch_size=batch_size, gpus=gpus)
    return [float(score) for score in prediction["scores"]]


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
        spec, model_key = resolve_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = get_dataset(args.dataset)
    ensure_dataset_ready(args, dataset)
    validate_args(args)

    directions = collect_directions(args, dataset)
    if not directions:
        print("No directions found.")
        return

    model_tag = sanitize_model_tag(model_key)
    model = load_comet_model(spec.model_id)
    language_support = CometLanguages(dataset=dataset)

    run_scoring(
        args,
        dataset,
        directions,
        model_tag,
        lambda entry: score_entry(
            entry, args, model, spec.model_id, language_support, dataset
        ),
    )


if __name__ == "__main__":
    main()
