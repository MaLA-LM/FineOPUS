from __future__ import annotations

import argparse

from dataset.mediator import get_dataset
from execution import get_executor
from execution.flores_array.manifest import ManifestEntry
from models.language_support import CometLanguages
from models.model_registry import resolve_model_spec, supported_model_keys
from src.backends.comet.backend import load_comet_model
from src.backends.comet.runner import score_entry
from src.common import (
    ensure_dataset_ready,
    sanitize_model_tag,
)
from utils.args import add_common_scoring_args, add_selected_executor_args
from utils.cli import validate_args

__all__ = ["parse_args", "resolve_model", "main"]


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
    add_selected_executor_args(parser)
    return parser.parse_args()


def resolve_model(name: str):
    return resolve_model_spec(name, "comet")


def main() -> None:
    args = parse_args()
    try:
        spec, model_key = resolve_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = get_dataset(args.dataset)
    ensure_dataset_ready(args, dataset)
    validate_args(args)

    executor = get_executor(args.execution)
    model_tag = sanitize_model_tag(model_key)
    language_support = CometLanguages(dataset=dataset)
    model_holder: dict[str, object] = {}

    def run_entry(entry: ManifestEntry):
        model = model_holder.get("model")
        if model is None:
            model = load_comet_model(spec.model_id)
            model_holder["model"] = model
        return score_entry(entry, args, model, spec.model_id, language_support, dataset)

    executor.run(
        args,
        dataset,
        model_tag,
        run_entry,
    )
