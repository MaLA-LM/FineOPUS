from __future__ import annotations

import math
from typing import Any

from dataset.mediator import Example
from prompts.batch import render_batch_prompt
from utils.logger import logger

from src.backends.llm.backend.constants import (
    _BATCH_TOKENS_PER_ITEM,
    PROMPT_MODE_BATCH,
    RESPONSE_FORMAT_NONE,
    ResponseFormat,
)
from src.backends.llm.backend.engine import (
    _build_sampling_params,
    _build_structured_outputs,
    _retry_temperature,
)
from src.backends.llm.backend.parsing import _parse_batch_response

__all__ = ["_score_batch_mode"]


def _score_batch_mode(
    engine,
    examples: list[Example],
    src_lang: str,
    tgt_lang: str,
    *,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    response_format: ResponseFormat = RESPONSE_FORMAT_NONE,
) -> list[float]:
    batch_max_tokens = max(max_tokens, batch_size * _BATCH_TOKENS_PER_ITEM)
    structured = _build_structured_outputs(PROMPT_MODE_BATCH, response_format)
    sampling = _build_sampling_params(
        temperature=temperature,
        max_tokens=batch_max_tokens,
        structured=structured,
    )

    chat_kwargs: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": False},
    }

    batches = [
        examples[i : i + batch_size] for i in range(0, len(examples), batch_size)
    ]
    conversations = [
        [{"role": "user", "content": render_batch_prompt(batch, src_lang, tgt_lang)}]
        for batch in batches
    ]

    logger.info(
        "Scoring %d segments in %d batches (batch_size=%d, max_tokens=%d, response_format=%s)",
        len(examples),
        len(batches),
        batch_size,
        batch_max_tokens,
        response_format,
    )

    outputs = engine.chat(conversations, sampling_params=sampling, **chat_kwargs)

    batch_scores: list[list[float | None]] = []
    failed_indices: list[int] = []
    for idx, output in enumerate(outputs):
        parsed = _parse_batch_response(output.outputs[0].text, len(batches[idx]))
        batch_scores.append(parsed)
        if any(s is None for s in parsed):
            failed_indices.append(idx)

    if failed_indices:
        retry_temperature = _retry_temperature(temperature)
        retry_sampling = _build_sampling_params(
            temperature=retry_temperature,
            max_tokens=batch_max_tokens,
            structured=structured,
        )
        for retry_round in range(max_retries):
            if not failed_indices:
                break
            retry_convs = [conversations[i] for i in failed_indices]
            logger.info(
                "Retry round %d: %d failed batches (temperature=%s)",
                retry_round + 1,
                len(retry_convs),
                retry_temperature,
            )
            retry_outputs = engine.chat(
                retry_convs,
                sampling_params=retry_sampling,
                **chat_kwargs,
            )
            still_failed: list[int] = []
            for batch_idx, output in zip(failed_indices, retry_outputs):
                new_parsed = _parse_batch_response(
                    output.outputs[0].text, len(batches[batch_idx])
                )
                for seg_i, (old, new) in enumerate(
                    zip(batch_scores[batch_idx], new_parsed)
                ):
                    if old is None and new is not None:
                        batch_scores[batch_idx][seg_i] = new
                if any(s is None for s in batch_scores[batch_idx]):
                    still_failed.append(batch_idx)
            failed_indices = still_failed

    scores = [
        s if s is not None else float("nan") for batch in batch_scores for s in batch
    ]

    n_failed = sum(1 for s in scores if math.isnan(s))
    if n_failed:
        logger.warning("Total failed segments: %d / %d", n_failed, len(scores))
    else:
        logger.info("All %d segments scored successfully.", len(scores))
    return scores
