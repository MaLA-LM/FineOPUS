from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from models.llm_qe import QEBatchResult, QEResult, QESimpleResult
from pydantic import ValidationError
from utils.logger import logger

from src.backends.llm.backend.constants import (
    PROMPT_MODE_BATCH,
    PROMPT_MODE_SIMPLE,
    PromptMode,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

__all__ = [
    "_get_result_model",
    "_overall_to_unit",
    "_parse_batch_response",
    "_parse_response",
    "_strip_thinking",
]


def _get_result_model(prompt_mode: PromptMode) -> type[BaseModel]:
    if prompt_mode == PROMPT_MODE_SIMPLE:
        return QESimpleResult
    if prompt_mode == PROMPT_MODE_BATCH:
        return QEBatchResult
    return QEResult


def _overall_to_unit(score_0_to_100: int) -> float:
    if not 0 <= score_0_to_100 <= 100:
        raise ValueError(f"overall_0to100 out of range: {score_0_to_100}")
    return score_0_to_100 / 100.0


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _parse_response(
    text: str, prompt_mode: PromptMode
) -> tuple[float | None, bool]:
    cleaned = _strip_thinking(text)
    model_cls = _get_result_model(prompt_mode)
    try:
        payload = json.loads(cleaned)
        parsed = model_cls.model_validate(payload)
    except json.JSONDecodeError as exc:
        logger.debug("JSONDecodeError: %s", exc)
        return None, True
    except ValidationError as exc:
        logger.debug("ValidationError: %s", exc)
        return None, True
    return _overall_to_unit(int(parsed.overall_0to100)), False


def _parse_batch_response(
    text: str, expected_count: int
) -> tuple[list[float | None], bool]:
    cleaned = _strip_thinking(text)
    try:
        payload = json.loads(cleaned)
        parsed = QEBatchResult.model_validate(payload)
    except json.JSONDecodeError as exc:
        logger.debug("Batch JSONDecodeError: %s", exc)
        return [None] * expected_count, True
    except ValidationError as exc:
        logger.debug("Batch ValidationError: %s", exc)
        return [None] * expected_count, True

    scores_by_id: dict[int, float] = {}
    for item in parsed.results:
        try:
            scores_by_id[item.id] = _overall_to_unit(int(item.overall_0to100))
        except ValueError:
            pass

    return [scores_by_id.get(i) for i in range(expected_count)], False
