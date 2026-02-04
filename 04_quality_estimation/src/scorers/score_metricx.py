from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import torch
from mbrs.metrics.metricx import MT5ForRegression
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dataset.manifest import ManifestEntry
from dataset.mediator import DEFAULT_DATASET_ID, Example, get_dataset
from models.metricx_spec import ModelSpec
from models.language_support import MetricX24Languages
from src.scoring.cli import resolve_output_path, validate_args
from src.scoring.runner import collect_directions, run_scoring
from src.scoring.frames import build_frames
from src.scoring.output_path import sanitize_model_tag


def metricx_adjust(score: float) -> float:
    """Convert MetricX-24 (lower is better) to a 0-1 higher-is-better score."""
    return 1.0 - (score / 25.0)


METRICX_MODELS = {
    "metricx24": ModelSpec(
        model_id="google/metricx-24-hybrid-xl-v2p6",
        tokenizer_id="google/mt5-xl",
        max_length=1536,
        score_adjuster=metricx_adjust,
    )
}

ALIASES = {
    "metricx": "metricx24",
    "metricx-24": "metricx24",
    "metricx-24-hybrid-xl-v2p6": "metricx24",
    "google/metricx-24-hybrid-xl-v2p6": "metricx24",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a dataset split with MetricX-24 QE."
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
        default="metricx24",
        help=(
            "Model name or HF repo. "
            f"Supported: {', '.join(sorted(METRICX_MODELS.keys()))}."
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


def resolve_model(name: str) -> tuple[ModelSpec, str]:
    key = ALIASES.get(name, name)
    if key in METRICX_MODELS:
        return METRICX_MODELS[key], key
    supported = ", ".join(sorted(METRICX_MODELS.keys()))
    raise ValueError(f"Unknown MetricX model '{name}'. Supported: {supported}.")


def select_device(gpus: int) -> torch.device:
    if gpus > 0 and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def metricx_input_ids(tokenizer, source: str, candidate: str) -> list[int]:
    # MetricX-24 QE expects an empty reference but keeps the "reference:" prefix.
    return (
        tokenizer.encode(f"source: {source}", add_special_tokens=False)
        + tokenizer.encode(f" candidate: {candidate}", add_special_tokens=False)
        + tokenizer.encode(" reference: ", add_special_tokens=False)
    )


def load_metricx(spec: ModelSpec, device: torch.device):
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    tokenizer = AutoTokenizer.from_pretrained(
        spec.tokenizer_id, legacy=False, use_fast=False
    )
    model = MT5ForRegression.from_pretrained(spec.model_id).to(device).eval()
    return tokenizer, model


def score_metricx(
    examples: list[Example],
    spec: ModelSpec,
    tokenizer,
    model,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    with torch.no_grad():
        for idx in range(0, len(examples), batch_size):
            batch = examples[idx : idx + batch_size]
            input_ids = []
            for ex in batch:
                ids = metricx_input_ids(tokenizer, ex["src"], ex["tgt"])
                prepared = tokenizer.prepare_for_model(
                    ids,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=spec.max_length,
                    return_attention_mask=False,
                )
                input_ids.append(prepared["input_ids"])

            batch_inputs = tokenizer.pad(
                {"input_ids": input_ids}, return_tensors="pt"
            ).to(device)

            outputs = model(**batch_inputs)
            predictions = outputs.predictions
            if isinstance(predictions, torch.Tensor):
                batch_scores = predictions.detach().cpu().flatten().tolist()
            else:
                batch_scores = torch.as_tensor(predictions).flatten().tolist()
            scores.extend(float(score) for score in batch_scores)

    return scores


def apply_score_adjustment(scores: list[float], spec: ModelSpec) -> list[float]:
    if spec.score_adjuster is None:
        return [float(score) for score in scores]
    return [float(spec.score_adjuster(score)) for score in scores]


def summarize_scores(scores: list[float]) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    return float(statistics.fmean(scores)), float(statistics.median(scores))


def score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    spec: ModelSpec,
    tokenizer,
    model,
    device: torch.device,
    language_support: MetricX24Languages,
    dataset,
):
    examples = dataset.load_parallel(
        entry.src_lang, entry.tgt_lang, split=entry.split, root=args.root
    )
    examples = dataset.limit_rows(examples, args.max_rows)

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
    device = select_device(args.gpus)
    tokenizer, model = load_metricx(spec, device)
    language_support = MetricX24Languages(dataset=dataset)

    run_scoring(
        args,
        dataset,
        directions,
        model_tag,
        resolve_output_path,
        lambda entry: score_entry(
            entry, args, spec, tokenizer, model, device, language_support, dataset
        ),
    )


if __name__ == "__main__":
    main()
