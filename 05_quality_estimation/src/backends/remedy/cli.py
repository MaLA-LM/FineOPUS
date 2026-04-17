from __future__ import annotations

import argparse
import os
from itertools import count
from pathlib import Path

from dataset.mediator import get_dataset
from execution import get_executor
from models.language_support import RemedyLanguages
from models.model_registry import resolve_model_spec, supported_model_keys
from src.backends.remedy.backend import DEFAULT_GPU_MEMORY_UTILIZATION
from src.backends.remedy.runner import score_entry
from src.common import ensure_dataset_ready, sanitize_model_tag
from utils.args import add_common_scoring_args, add_selected_executor_args
from utils.cli import validate_args

DEFAULT_REMEDY_MODEL = "remedy"

__all__ = [
    "DEFAULT_REMEDY_MODEL",
    "default_cache_dir",
    "parse_args",
    "resolve_model",
    "main",
]


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
    add_selected_executor_args(parser)
    return parser.parse_args()


def resolve_model(name: str):
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

    executor = get_executor(args.execution)
    model_tag = sanitize_model_tag(model_key)
    language_support = RemedyLanguages(dataset=dataset)
    cache_dir = Path(args.cache_dir)
    iteration_counter = count()

    executor.run(
        args,
        dataset,
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
