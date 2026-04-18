from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from prompts.batch import render_batch_prompt
from prompts.detailed import render_prompt
from prompts.simple import render_simple_prompt
from utils.logger import logger

from src.backends.llm.backend.constants import (
    _RETRY_TEMPERATURE,
    PROMPT_MODE_SIMPLE,
    RESPONSE_FORMAT_JSON_OBJECT,
    RESPONSE_FORMAT_JSON_SCHEMA,
    RESPONSE_FORMAT_NONE,
    PromptMode,
    ResponseFormat,
)
from src.backends.llm.backend.parsing import _get_result_model

if TYPE_CHECKING:
    from vllm import LLM

__all__ = [
    "_build_sampling_params",
    "_build_structured_outputs",
    "_resolve_structured_outputs_backend",
    "_render_prompt",
    "_retry_temperature",
    "build_engine",
]


def _retry_temperature(initial_temperature: float) -> float:
    return max(initial_temperature, _RETRY_TEMPERATURE)


def _build_sampling_params(
    *, temperature: float, max_tokens: int, structured: Any | None = None
) -> Any:
    from vllm import SamplingParams

    sampling_kwargs: dict[str, Any] = dict(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if structured is not None:
        sampling_kwargs["structured_outputs"] = structured
    return SamplingParams(**sampling_kwargs)


def _render_prompt(
    src_lang: str,
    src_seg: str,
    tgt_lang: str,
    tgt_seg: str,
    prompt_mode: PromptMode,
) -> str:
    if prompt_mode == PROMPT_MODE_SIMPLE:
        return render_simple_prompt(src_lang, src_seg, tgt_lang, tgt_seg)
    return render_prompt(src_lang, src_seg, tgt_lang, tgt_seg)


def _build_structured_outputs(
    prompt_mode: PromptMode, response_format: ResponseFormat
) -> Any:
    if response_format == RESPONSE_FORMAT_NONE:
        return None

    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as exc:
        import vllm

        vllm_ver = getattr(vllm, "__version__", "unknown")
        raise RuntimeError(
            f"vLLM structured outputs API not found (vLLM {vllm_ver}). "
            f"This code expects vllm.sampling_params.StructuredOutputsParams "
            f"(available in vLLM >=0.12). "
            f"Use --response-format none or upgrade vLLM."
        ) from exc

    if response_format == RESPONSE_FORMAT_JSON_OBJECT:
        return StructuredOutputsParams(json_object=True)

    if response_format != RESPONSE_FORMAT_JSON_SCHEMA:
        return None

    model_cls = _get_result_model(prompt_mode)
    return StructuredOutputsParams(json=model_cls.model_json_schema())


def _resolve_structured_outputs_backend(
    response_format: ResponseFormat,
    explicit_backend: str | None = None,
) -> str | None:
    if response_format == RESPONSE_FORMAT_NONE:
        return None

    backend = explicit_backend
    if backend is None:
        backend = os.environ.get("VLLM_STRUCTURED_OUTPUTS_BACKEND")
    if backend is None:
        backend = os.environ.get("STRUCTURED_OUTPUTS_BACKEND")

    if backend is not None:
        backend = backend.strip()
    if backend:
        return backend
    return "outlines"


def build_engine(
    model: str,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = False,
    max_num_batched_tokens: int = 16384,
    max_num_seqs: int = 128,
    max_model_len: int | None = None,
    response_format: ResponseFormat = RESPONSE_FORMAT_NONE,
    extra_kwargs: dict[str, Any] | None = None,
) -> LLM:
    from vllm import LLM

    kwargs: dict[str, Any] = dict(
        model=model,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    if "structured_outputs_config" not in kwargs:
        structured_backend = _resolve_structured_outputs_backend(response_format)
        if structured_backend is not None:
            try:
                from vllm.config.structured_outputs import StructuredOutputsConfig
            except ImportError as exc:
                import vllm

                vllm_ver = getattr(vllm, "__version__", "unknown")
                raise RuntimeError(
                    f"vLLM structured outputs config API not found (vLLM {vllm_ver}). "
                    "This code expects vllm.config.structured_outputs.StructuredOutputsConfig."
                ) from exc

            kwargs["structured_outputs_config"] = StructuredOutputsConfig(
                backend=structured_backend
            )

    logger.info(
        "Loading vLLM engine: model=%s dtype=%s enforce_eager=%s max_num_batched_tokens=%d max_num_seqs=%d max_model_len=%s structured_backend=%s",
        model,
        dtype,
        enforce_eager,
        max_num_batched_tokens,
        max_num_seqs,
        max_model_len or "auto",
        _resolve_structured_outputs_backend(response_format),
    )
    return LLM(**kwargs)
