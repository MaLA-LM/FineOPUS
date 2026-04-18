from __future__ import annotations

import math
from typing import Any

from dataset.mediator import Example
from utils.logger import logger

from src.backends.llm.backend.batch import _score_batch_mode
from src.backends.llm.backend.constants import (
    _SIMPLE_MAX_TOKENS,
    PROMPT_MODE_BATCH,
    PROMPT_MODE_SIMPLE,
    RESPONSE_FORMAT_NONE,
    PromptMode,
    ResponseFormat,
)
from src.backends.llm.backend.engine import (
    _build_sampling_params,
    _build_structured_outputs,
    _render_prompt,
    _retry_temperature,
)
from src.backends.llm.backend.parsing import _parse_response

__all__ = ["score_llm_offline"]


def score_llm_offline(
    engine,
    examples: list[Example],
    src_lang: str,
    tgt_lang: str,
    *,
    prompt_mode: PromptMode,
    batch_size: int = 8,
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_retries: int = 5,
    response_format: ResponseFormat = RESPONSE_FORMAT_NONE,
) -> list[float]:
    if not examples:
        return []

    if prompt_mode == PROMPT_MODE_BATCH:
        return _score_batch_mode(
            engine,
            examples,
            src_lang,
            tgt_lang,
            batch_size=batch_size,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            response_format=response_format,
        )

    if prompt_mode == PROMPT_MODE_SIMPLE:
        max_tokens = min(max_tokens, _SIMPLE_MAX_TOKENS)

    structured = _build_structured_outputs(prompt_mode, response_format)
    sampling = _build_sampling_params(
        temperature=temperature,
        max_tokens=max_tokens,
        structured=structured,
    )

    conversations: list[list[dict[str, str]]] = []
    for ex in examples:
        prompt_text = _render_prompt(
            src_lang, ex["src"], tgt_lang, ex["tgt"], prompt_mode
        )
        conversations.append([{"role": "user", "content": prompt_text}])

    logger.info(
        "Scoring %d segments (prompt_mode=%s, max_tokens=%d, response_format=%s)",
        len(examples),
        prompt_mode,
        max_tokens,
        response_format,
    )

    chat_kwargs: dict[str, Any] = {}
    chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    outputs = engine.chat(
        conversations,
        sampling_params=sampling,
        **chat_kwargs,
    )

    scores: list[float] = []
    failed_indices: list[int] = []
    for idx, output in enumerate(outputs):
        score = _parse_response(output.outputs[0].text, prompt_mode)
        if score is not None:
            scores.append(score)
        else:
            scores.append(float("nan"))
            failed_indices.append(idx)

    if failed_indices:
        retry_temperature = _retry_temperature(temperature)
        retry_sampling = _build_sampling_params(
            temperature=retry_temperature,
            max_tokens=max_tokens,
            structured=structured,
        )
        for retry_round in range(max_retries):
            if not failed_indices:
                break
            retry_conversations = [conversations[i] for i in failed_indices]
            logger.info(
                "Retry round %d: %d failed segments (temperature=%s)",
                retry_round + 1,
                len(retry_conversations),
                retry_temperature,
            )
            retry_outputs = engine.chat(
                retry_conversations,
                sampling_params=retry_sampling,
                **chat_kwargs,
            )
            still_failed: list[int] = []
            for orig_idx, output in zip(failed_indices, retry_outputs):
                score = _parse_response(output.outputs[0].text, prompt_mode)
                if score is not None:
                    scores[orig_idx] = score
                else:
                    still_failed.append(orig_idx)
            failed_indices = still_failed

    n_failed = sum(1 for s in scores if math.isnan(s))
    if n_failed:
        logger.warning("Total failed segments: %d / %d", n_failed, len(scores))
    else:
        logger.info("All %d segments scored successfully.", len(scores))
    return scores
