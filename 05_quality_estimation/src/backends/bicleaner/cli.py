from __future__ import annotations

import argparse

from dataset.mediator import get_dataset
from execution import get_executor
from models.language_support import CometLanguages
from models.model_registry import resolve_model_spec, supported_model_keys
from src.backends.bicleaner.backend import BICLEANER_MODEL_IDS
from src.backends.bicleaner.runner import score_entry
from src.common import ensure_dataset_ready, sanitize_model_tag
from utils.args import add_common_scoring_args, add_selected_executor_args
from utils.cli import validate_args

__all__ = ["BICLEANER_MODEL_IDS", "parse_args", "resolve_model", "main"]


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
    add_selected_executor_args(parser)
    return parser.parse_args()


def resolve_model(name: str) -> tuple[str, str]:
    _, key = resolve_model_spec(name, "bicleaner")
    model_key = "bicleaner-ai" if key == "auto" else key
    return key, model_key


def main() -> None:
    args = parse_args()
    try:
        model_selector, model_key = resolve_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = get_dataset(args.dataset)
    ensure_dataset_ready(args, dataset)
    validate_args(args)

    executor = get_executor(args.execution)
    model_tag = sanitize_model_tag(model_key)
    language_support = CometLanguages(dataset=dataset)

    executor.run(
        args,
        dataset,
        model_tag,
        lambda entry: score_entry(
            entry, args, model_selector, language_support, dataset
        ),
    )
