from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


ScoreAdjuster = Callable[[float], float]
Backend = Literal["comet", "metricx", "llm", "bicleaner", "remedy"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    backend: Backend
    model_id: str
    aliases: tuple[str, ...] = ()
    tokenizer_id: str | None = None
    max_length: int | None = None
    score_adjuster: ScoreAdjuster | None = None


def metricx_adjust(score: float) -> float:
    return 1.0 - (score / 25.0)


_SPECS: dict[Backend, dict[str, ModelSpec]] = {
    "comet": {
        "wmt22-cometkiwi-da": ModelSpec(
            key="wmt22-cometkiwi-da",
            backend="comet",
            model_id="Unbabel/wmt22-cometkiwi-da",
            aliases=("wmt22-comet", "Unbabel/wmt22-cometkiwi-da"),
        ),
        "wmt23-cometkiwi-da-xl": ModelSpec(
            key="wmt23-cometkiwi-da-xl",
            backend="comet",
            model_id="Unbabel/wmt23-cometkiwi-da-xl",
            aliases=("wmt23-comet", "Unbabel/wmt23-cometkiwi-da-xl"),
        ),
        "xcomet-xl": ModelSpec(
            key="xcomet-xl",
            backend="comet",
            model_id="Unbabel/XCOMET-XL",
            aliases=("xcomet", "Unbabel/XCOMET-XL"),
        ),
    },
    "metricx": {
        "metricx24": ModelSpec(
            key="metricx24",
            backend="metricx",
            model_id="google/metricx-24-hybrid-xl-v2p6",
            aliases=(
                "metricx",
                "metricx-24",
                "metricx-24-hybrid-xl-v2p6",
                "google/metricx-24-hybrid-xl-v2p6",
            ),
            tokenizer_id="google/mt5-xl",
            max_length=1536,
            score_adjuster=metricx_adjust,
        ),
    },
    "llm": {
        "qwen3-14b": ModelSpec(
            key="qwen3-14b",
            backend="llm",
            model_id="Qwen/Qwen3-14B",
            aliases=(
                "qwen/qwen3-14b",
                "hosted_vllm/qwen3-14b",
                "hosted_vllm/qwen/qwen3-14b",
            ),
        ),
        "qwen3-8b": ModelSpec(
            key="qwen3-8b",
            backend="llm",
            model_id="Qwen/Qwen3-8B",
            aliases=(
                "qwen/qwen3-8b",
                "hosted_vllm/qwen3-8b",
                "hosted_vllm/qwen/qwen3-8b",
            ),
        ),
        "qwen3-4b-instruct-2507": ModelSpec(
            key="qwen3-4b-instruct-2507",
            backend="llm",
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            aliases=(
                "qwen3-4b",
                "qwen/qwen3-4b-instruct-2507",
                "hosted_vllm/qwen3-4b-instruct-2507",
                "hosted_vllm/qwen/qwen3-4b-instruct-2507",
            ),
        ),
        "m-prometheus-7b": ModelSpec(
            key="m-prometheus-7b",
            backend="llm",
            model_id="Unbabel/M-Prometheus-7B",
            aliases=(
                "unbabel/m-prometheus-7b",
                "m-prometheus-7b",
                "mprometheus-7b",
                "hosted_vllm/unbabel/m-prometheus-7b",
                "hosted_vllm/m-prometheus-7b",
            ),
        ),
        "m-prometheus-3b": ModelSpec(
            key="m-prometheus-3b",
            backend="llm",
            model_id="Unbabel/M-Prometheus-3B",
            aliases=(
                "unbabel/m-prometheus-3b",
                "m-prometheus-3b",
                "mprometheus-3b",
                "hosted_vllm/unbabel/m-prometheus-3b",
                "hosted_vllm/m-prometheus-3b",
            ),
        ),
    },
    "bicleaner": {
        "auto": ModelSpec(
            key="auto",
            backend="bicleaner",
            model_id="auto",
            aliases=("bicleaner", "bicleaner-ai"),
        ),
        "en-xx": ModelSpec(
            key="en-xx",
            backend="bicleaner",
            model_id="bitextor/bicleaner-ai-full-en-xx",
            aliases=("bitextor/bicleaner-ai-full-en-xx",),
        ),
        "es-xx": ModelSpec(
            key="es-xx",
            backend="bicleaner",
            model_id="bitextor/bicleaner-ai-full-es-xx",
            aliases=("bitextor/bicleaner-ai-full-es-xx",),
        ),
        "de-xx": ModelSpec(
            key="de-xx",
            backend="bicleaner",
            model_id="bitextor/bicleaner-ai-full-de-xx",
            aliases=("bitextor/bicleaner-ai-full-de-xx",),
        ),
    },
    "remedy": {
        "remedy": ModelSpec(
            key="remedy",
            backend="remedy",
            model_id="ShaomuTan/ReMedy-9B-22",
            aliases=(
                "remedy",
                "remedy-9b-22",
                "shaomutan/remedy-9b-22",
                "ShaomuTan/ReMedy-9B-22",
            ),
        ),
    },
}


def supported_model_keys(backend: Backend) -> list[str]:
    return sorted(_SPECS[backend].keys())


def resolve_model_spec(name: str, backend: Backend) -> tuple[ModelSpec, str]:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("--model cannot be empty.")
    for key, spec in _SPECS[backend].items():
        if normalized in (
            key,
            spec.model_id.lower(),
            *(a.lower() for a in spec.aliases),
        ):
            return spec, key
    supported = ", ".join(supported_model_keys(backend))
    raise ValueError(f"Unknown {backend} model '{name}'. Supported: {supported}.")
