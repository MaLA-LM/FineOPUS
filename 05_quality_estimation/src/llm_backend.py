from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any

from dataset.mediator import Example
from models.llm_qe import QEBatchResult
from prompts.llm_prompt import render_prompt
from prompts.llm_prompt_batch import render_batch_prompt
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
    """Build a DSPy LM client (used only by the legacy single-segment path)."""
    import dspy

    api_key = api_key or "EMPTY"
    model_name = model if model.startswith("hosted_vllm/") else f"hosted_vllm/{model}"
    return dspy.LM(
        model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


# ── single-segment scoring (legacy, kept for backwards compat) ─────────


def score_llm(
    examples: list[Example],
    lm,
    max_retries: int,
    src_lang: str,
    tgt_lang: str,
    continue_on_error: bool = True,
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


# ── micro-batched scoring with async concurrency ───────────────────────


def _parse_batch_response(text: str, expected_count: int) -> list[float] | None:
    """Parse a batched JSON response and return unit-scaled scores.

    Returns None on any validation failure so the caller can retry.
    """
    from models.llm_qe import QEBatchResult
    from pydantic import ValidationError

    try:
        payload = json.loads(text)
        parsed = QEBatchResult.model_validate(payload)
        if len(parsed.results) != expected_count:
            logger.debug(
                f"Batch size mismatch: expected {expected_count} results, "
                f"got {len(parsed.results)}"
            )
            return None
    except json.JSONDecodeError as exc:
        logger.debug("JSONDecodeError while parsing batch response: %s", exc)
        return None
    except ValidationError as exc:
        logger.debug("ValidationError while parsing batch response: %s", exc)
        return None

    # Sort by id to ensure correct ordering
    items_by_id = {item.id: item for item in parsed.results}
    scores: list[float] = []
    for i in range(expected_count):
        if i not in items_by_id:
            logger.debug(f"Missing result for id={i}")
            return None
        scores.append(overall_to_unit(int(items_by_id[i].overall_0to100)))
    return scores


# ── async HTTP helpers ──────────────────────────────────────────────────


async def _call_vllm(
    client: Any,
    url: str,
    model: str,
    prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """POST a single chat-completion request to the vLLM endpoint."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "QEBatchResult",
                "schema": QEBatchResult.model_json_schema(),
            },
        },
        # Qwen3-style: disable thinking tokens in the response
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = await client.post(url, json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def _score_one_batch(
    batch_idx: int,
    batch: list[Example],
    sem: asyncio.Semaphore,
    client: Any,
    url: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    src_lang: str,
    tgt_lang: str,
    total_batches: int,
) -> tuple[int, list[float] | None]:
    """Score one micro-batch, bounded by *sem*.

    Returns ``(batch_idx, scores)`` on success, or
    ``(batch_idx, None)`` when all retries are exhausted.
    """
    current_size = len(batch)
    prompt = render_batch_prompt(batch, src_lang, tgt_lang, start_id=0)

    async with sem:
        last_error: Exception | None = None
        for attempt in range(max_retries + 3):
            text = await _call_vllm(
                client,
                url,
                model,
                prompt,
                api_key,
                temperature,
                max_tokens,
            )
            scores = _parse_batch_response(text, current_size)
            if scores is None:
                logger.debug(
                    "Batch %d/%d attempt %d: parse failed, retrying",
                    batch_idx + 1,
                    total_batches,
                    attempt + 1,
                )
                continue
            return batch_idx, scores

        # All retries exhausted
        logger.warning(
            "Batch %d/%d FAILED after %d attempts (last error: %s). "
            "Filling %d scores with NaN.",
            batch_idx + 1,
            total_batches,
            max_retries + 3,
            last_error,
            current_size,
        )
        return batch_idx, None


async def _run_batches_async(
    batches: list[list[Example]],
    concurrency: int,
    max_retries: int,
    src_lang: str,
    tgt_lang: str,
    api_base: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> list[float]:
    """Fire all batches concurrently (bounded by *concurrency*) and
    return scores in original segment order."""
    import httpx

    url = f"{api_base}/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    total = len(batches)

    # Generous timeout: large batches may take a while to generate
    timeout = httpx.Timeout(600.0, connect=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _score_one_batch(
                i,
                b,
                sem,
                client,
                url,
                model,
                api_key,
                temperature,
                max_tokens,
                max_retries,
                src_lang,
                tgt_lang,
                total,
            )
            for i, b in enumerate(batches)
        ]
        results = await asyncio.gather(*tasks)

    # Reassemble in order (gather preserves input order, but be explicit)
    results_sorted = sorted(results, key=lambda r: r[0])  # sort by batch_idx
    all_scores: list[float] = []
    failed = 0
    for batch_idx, scores in results_sorted:
        if scores is not None:
            all_scores.extend(scores)
        else:
            all_scores.extend([float("nan")] * len(batches[batch_idx]))
            failed += 1

    if failed:
        logger.warning("Total failed batches: %d / %d", failed, total)
    else:
        n_segs = sum(len(b) for b in batches)
        logger.info(
            "All %d batches scored successfully (%d segments total).",
            total,
            n_segs,
        )
    return all_scores


def score_llm_batched(
    examples: list[Example],
    max_retries: int,
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 32,
    concurrency: int = 16,
    *,
    api_base: str,
    model: str,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> list[float]:
    """Score *examples* in micro-batches, with up to *concurrency* batches
    in flight at once via async HTTP to the vLLM endpoint."""

    if not examples:
        return []

    api_key = api_key or "EMPTY"

    # Split examples into micro-batches
    batches = [
        examples[i : i + batch_size] for i in range(0, len(examples), batch_size)
    ]

    logger.info(
        "Scoring %d segments in %d batches "
        "(batch_size=%d, concurrency=%d, max_tokens=%d)",
        len(examples),
        len(batches),
        batch_size,
        concurrency,
        max_tokens,
    )

    return asyncio.run(
        _run_batches_async(
            batches,
            concurrency,
            max_retries,
            src_lang,
            tgt_lang,
            api_base,
            model,
            api_key,
            temperature,
            max_tokens,
        )
    )
