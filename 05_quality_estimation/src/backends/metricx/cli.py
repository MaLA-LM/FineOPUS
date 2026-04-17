from __future__ import annotations

import argparse

from dataset.mediator import get_dataset
from execution import get_executor
from execution.flores_array.manifest import ManifestEntry
from models.language_support import MetricX24Languages
from models.model_registry import resolve_model_spec, supported_model_keys
from src.backends.metricx.backend import load_metricx, select_device
from src.backends.metricx.runner import score_entry
from src.common import ensure_dataset_ready, sanitize_model_tag
from utils.args import add_common_scoring_args, add_selected_executor_args
from utils.cli import validate_args

__all__ = ["parse_args", "resolve_model", "main"]


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
    add_selected_executor_args(parser)
    return parser.parse_args()


def resolve_model(name: str):
    return resolve_model_spec(name, "metricx")


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
    device = select_device(args.gpus)
    language_support = MetricX24Languages(dataset=dataset)
    model_state: dict[str, object] = {}

    def run_entry(entry: ManifestEntry):
        tokenizer = model_state.get("tokenizer")
        model = model_state.get("model")
        if tokenizer is None or model is None:
            tokenizer, model = load_metricx(spec, device)
            model_state["tokenizer"] = tokenizer
            model_state["model"] = model
        return score_entry(
            entry, args, spec, tokenizer, model, device, language_support, dataset
        )

    executor.run(
        args,
        dataset,
        model_tag,
        run_entry,
    )
