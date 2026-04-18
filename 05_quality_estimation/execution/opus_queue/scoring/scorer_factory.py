from __future__ import annotations

import argparse
from collections.abc import Callable
from itertools import count
from pathlib import Path

from dataset.mediator import DatasetAdapter
from execution.flores_array.manifest import ManifestEntry
from models.model_registry import resolve_model_spec
from src.common import sanitize_model_tag
from utils.logger import logger

ScoreEntry = Callable[[ManifestEntry], object]
BuildFn = Callable[[argparse.Namespace, DatasetAdapter], tuple[ScoreEntry, str]]


def _build_comet(args, dataset):
    from models.language_support import CometLanguages
    from src.backends.comet.backend import load_comet_model
    from src.backends.comet.runner import score_entry as comet_score_entry

    spec, model_key = resolve_model_spec(args.model, "comet")
    language_support = CometLanguages(dataset=dataset)
    model = load_comet_model(spec.model_id)
    model_tag = sanitize_model_tag(model_key)
    logger.info("COMET scorer ready: model_id=%s key=%s", spec.model_id, model_key)

    def run_entry(entry: ManifestEntry):
        return comet_score_entry(
            entry, args, model, spec.model_id, language_support, dataset
        )

    return run_entry, model_tag


def _build_metricx(args, dataset):
    from models.language_support import MetricX24Languages
    from src.backends.metricx.backend import load_metricx, select_device
    from src.backends.metricx.runner import score_entry as metricx_score_entry

    spec, model_key = resolve_model_spec(args.model, "metricx")
    device = select_device(args.gpus)
    language_support = MetricX24Languages(dataset=dataset)
    tokenizer, model = load_metricx(spec, device)
    model_tag = sanitize_model_tag(model_key)
    logger.info("MetricX scorer ready: key=%s", model_key)

    def run_entry(entry: ManifestEntry):
        return metricx_score_entry(
            entry, args, spec, tokenizer, model, device, language_support, dataset
        )

    return run_entry, model_tag


def _build_llm(args, dataset):
    from models.language_support import (
        BaseLanguageSupport,
        PrometheusLanguages,
        QwenLanguages,
    )
    from src.backends.llm.backend import RESPONSE_FORMAT_JSON_SCHEMA, build_engine
    from src.backends.llm.runner import score_entry as llm_score_entry

    try:
        spec, model_key = resolve_model_spec(args.model, "llm")
        model_id = spec.model_id
    except ValueError:
        model_id = args.model
        model_key = args.model

    prompt_mode = getattr(args, "prompt_mode", "detailed")
    model_tag = sanitize_model_tag(f"{model_key}-{prompt_mode}")
    model_name = f"{model_key}-{prompt_mode}"

    if model_key.startswith("m-prometheus-"):
        language_support: BaseLanguageSupport = PrometheusLanguages(dataset=dataset)
    else:
        language_support = QwenLanguages(dataset=dataset)

    if getattr(args, "response_format", None) is None:
        args.response_format = RESPONSE_FORMAT_JSON_SCHEMA

    engine = build_engine(
        model=model_id,
        dtype=getattr(args, "dtype", "bfloat16"),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.90),
        enforce_eager=getattr(args, "enforce_eager", False),
        max_num_batched_tokens=getattr(args, "max_num_batched_tokens", 16384),
        max_num_seqs=getattr(args, "max_num_seqs", 128),
        max_model_len=getattr(args, "max_model_len", None),
    )
    logger.info("LLM scorer ready: model_id=%s prompt_mode=%s", model_id, prompt_mode)

    def run_entry(entry: ManifestEntry):
        return llm_score_entry(
            entry, args, engine, model_name, language_support, dataset
        )

    return run_entry, model_tag


def _build_remedy(args, dataset):
    from models.language_support import RemedyLanguages
    from src.backends.remedy.cli import resolve_model as resolve_remedy_model
    from src.backends.remedy.runner import score_entry as remedy_score_entry

    spec, model_key = resolve_remedy_model(args.model)
    cache_dir = Path(getattr(args, "cache_dir", ".cache/huggingface/hub"))
    language_support = RemedyLanguages(dataset=dataset)
    iteration = count()
    model_tag = sanitize_model_tag(model_key)
    logger.info("ReMedy scorer ready: key=%s", model_key)

    def run_entry(entry: ManifestEntry):
        return remedy_score_entry(
            entry,
            args,
            spec,
            dataset,
            language_support,
            cache_dir,
            iteration=next(iteration),
        )

    return run_entry, model_tag


def _build_bicleaner(args, dataset):
    from models.language_support import CometLanguages
    from src.backends.bicleaner.cli import (
        BICLEANER_MODEL_IDS,
        resolve_model as resolve_bicleaner_model,
    )
    from src.backends.bicleaner.runner import score_entry as bicleaner_score_entry

    model_selector, model_key = resolve_bicleaner_model(args.model)
    language_support = CometLanguages(dataset=dataset)
    model_tag = sanitize_model_tag(model_key)
    logger.info("Bicleaner scorer ready: selector=%s key=%s", model_selector, model_key)

    def run_entry(entry: ManifestEntry):
        return bicleaner_score_entry(
            entry, args, model_selector, language_support, dataset
        )

    # ensure the resolved bicleaner model ids dict is initialized
    _ = BICLEANER_MODEL_IDS
    return run_entry, model_tag


_BUILDERS: dict[str, BuildFn] = {
    "comet": _build_comet,
    "metricx": _build_metricx,
    "llm": _build_llm,
    "remedy": _build_remedy,
    "bicleaner": _build_bicleaner,
}


def resolve_backend(model_name: str) -> str:
    """Identify which scorer backend handles a given model name."""
    normalized = model_name.strip().lower()
    for backend in ("comet", "metricx", "llm", "remedy", "bicleaner"):
        try:
            resolve_model_spec(normalized, backend)  # type: ignore[arg-type]
            return backend
        except ValueError:
            continue
    raise SystemExit(
        f"Cannot resolve scorer backend for model '{model_name}'. "
        "Supported backends: comet, metricx, llm, remedy, bicleaner."
    )


def build_scorer(
    backend: str, args: argparse.Namespace, dataset: DatasetAdapter
) -> tuple[ScoreEntry, str]:
    if backend not in _BUILDERS:
        raise SystemExit(
            f"Unknown backend '{backend}'. Supported: {sorted(_BUILDERS)}."
        )
    return _BUILDERS[backend](args, dataset)
