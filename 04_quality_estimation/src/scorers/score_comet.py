from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from comet import download_model, load_from_checkpoint

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dataset.manifest import ManifestEntry
from dataset.mediator import DEFAULT_DATASET_ID, Example, get_dataset
from models.language_support import CometLanguages
from src.scoring.cli import resolve_output_path, validate_args
from src.scoring.frames import build_frames
from src.scoring.output_path import sanitize_model_tag
from src.scoring.runner import collect_directions, run_scoring

COMET_MODELS = {
    "wmt22-cometkiwi-da": "Unbabel/wmt22-cometkiwi-da",
    "wmt23-cometkiwi-da-xl": "Unbabel/wmt23-cometkiwi-da-xl",
    "xcomet-xl": "Unbabel/XCOMET-XL",
}

ALIASES = {
    "wmt22-comet": "wmt22-cometkiwi-da",
    "wmt23-comet": "wmt23-cometkiwi-da-xl",
    "xcomet": "xcomet-xl",
    "Unbabel/wmt22-cometkiwi-da": "wmt22-cometkiwi-da",
    "Unbabel/wmt23-cometkiwi-da-xl": "wmt23-cometkiwi-da-xl",
    "Unbabel/XCOMET-XL": "xcomet-xl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a dataset split with COMET-style QE models."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_ID,
        help="Dataset id to use.",
    )
    parser.add_argument("--src-lang", help="Source language code.")
    parser.add_argument("--tgt-lang", help="Target language code.")
    parser.add_argument(
        "--split",
        default="devtest",
        help="Dataset split to score.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory of the dataset files (defaults to dataset root).",
    )
    parser.add_argument("--output", default=None, help="Output Parquet path.")
    parser.add_argument(
        "--output-base",
        default=None,
        help="Base output directory for partitioned Parquet dataset output.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for model prediction.",
    )
    parser.add_argument(
        "--model",
        default="wmt22-cometkiwi-da",
        help=(
            "Model name or HF repo. "
            f"Supported: {', '.join(sorted(COMET_MODELS.keys()))}."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (set 0 for CPU).",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip completed outputs when they are valid.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Recompute outputs even if they already exist.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="TSV manifest with columns: src_lang, tgt_lang, split.",
    )
    parser.add_argument(
        "--discover-all",
        action="store_true",
        help="Discover all directions under --root and score each.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Continuously claim and score free directions from the manifest.",
    )
    parser.add_argument(
        "--worker-max-files",
        type=int,
        default=200,
        help="Max outputs to write before worker exits (0 for unlimited).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for number of rows (for debugging).",
    )
    return parser.parse_args()


def resolve_model(name: str) -> tuple[str, str]:
    key = ALIASES.get(name, name)
    if key in COMET_MODELS:
        return COMET_MODELS[key], key
    supported = ", ".join(sorted(COMET_MODELS.keys()))
    raise ValueError(f"Unknown COMET model '{name}'. Supported: {supported}.")


def load_comet_model(model_id: str):
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


def summarize_scores(scores: list[float]) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    return float(statistics.fmean(scores)), float(statistics.median(scores))


def score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    model,
    model_id: str,
    language_support: CometLanguages,
    dataset,
):
    examples = dataset.load_parallel(
        entry.src_lang, entry.tgt_lang, split=entry.split, root=args.root
    )
    examples = dataset.limit_rows(examples, args.max_rows)

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
        model_id, model_key = resolve_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = get_dataset(args.dataset)
    if args.root is None:
        args.root = dataset.default_root
    if args.split not in dataset.split_values:
        supported = ", ".join(dataset.split_values)
        raise SystemExit(
            f"Unsupported split '{args.split}' for dataset '{dataset.id}'. "
            f"Supported: {supported}."
        )
    validate_args(args)

    directions = collect_directions(args, dataset)
    if not directions:
        print("No directions found.")
        return

    model_tag = sanitize_model_tag(model_key)
    model = load_comet_model(model_id)
    language_support = CometLanguages(dataset=dataset)

    run_scoring(
        args,
        dataset,
        directions,
        model_tag,
        resolve_output_path,
        lambda entry: score_entry(
            entry, args, model, model_id, language_support, dataset
        ),
    )


if __name__ == "__main__":
    main()
