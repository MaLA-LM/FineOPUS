from __future__ import annotations

import json
import re

from dataset.mediator import Example
from prompts.llm_prompt import render_prompt
from utils.logger import logger

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def overall_to_unit(score_0_to_100: int) -> float:
    if not 0 <= score_0_to_100 <= 100:
        raise ValueError(f"overall_0to100 out of range: {score_0_to_100}")
    return score_0_to_100 / 100.0


def build_lm(
    api_base: str, model: str, api_key: str | None, temperature: float, max_tokens: int
):
    import dspy

    api_key = api_key or "EMPTY"
    model_name = model if model.startswith("openai/") else f"openai/{model}"
    return dspy.LM(
        model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def score_llm(
    examples: list[Example],
    lm,
    max_retries: int,
    src_lang: str,
    tgt_lang: str,
    continue_on_error: bool = False,
) -> list[float]:
    from models.llm_qe import QEResult
    from pydantic import ValidationError

    scores: list[float] = []
    invalid_rows: list[int] = []
    for idx, ex in enumerate(examples):
        prompt = render_prompt(src_lang, ex["src"], tgt_lang, ex["tgt"])
        last_error: Exception | None = None
        for _attempt in range(max_retries + 1):
            try:
                completions = lm(messages=[{"role": "user", "content": prompt}])
                text = strip_thinking(completions[0])
                payload = json.loads(text)
                parsed = QEResult.model_validate(payload)
                scores.append(overall_to_unit(int(parsed.overall_0to100)))
                last_error = None
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                continue
        if last_error is not None:
            if continue_on_error:
                scores.append(float("nan"))
                invalid_rows.append(idx)
                continue
            raise RuntimeError(
                f"Row {idx} failed validation: {last_error}"
            ) from last_error
    if continue_on_error and invalid_rows:
        sample = ", ".join(str(idx) for idx in invalid_rows[:5])
        suffix = f" Example indices: {sample}." if sample else ""
        logger.info("Invalid JSON rows: %s.%s", len(invalid_rows), suffix)
    return scores
