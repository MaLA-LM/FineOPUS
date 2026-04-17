from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dataset.mediator import Example
from models.model_registry import ModelSpec

if TYPE_CHECKING:
    import torch

__all__ = [
    "apply_score_adjustment",
    "load_metricx",
    "metricx_input_ids",
    "score_metricx",
    "select_device",
]

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "full": "float32",
}


def select_device(gpus: int) -> torch.device:
    import torch

    if gpus > 0 and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(device: torch.device):
    import torch

    if device.type != "cuda":
        return torch.float32
    name = os.environ.get("METRICX_DTYPE", "bf16").strip().lower()
    canonical = _DTYPE_ALIASES.get(name, "bfloat16")
    return getattr(torch, canonical)


def metricx_input_ids(tokenizer, source: str, candidate: str) -> list[int]:
    return (
        tokenizer.encode(f"source: {source}", add_special_tokens=False)
        + tokenizer.encode(f" candidate: {candidate}", add_special_tokens=False)
        + tokenizer.encode(" reference: ", add_special_tokens=False)
    )


def load_metricx(spec: ModelSpec, device: torch.device):
    import logging

    from mbrs.metrics.metricx import MT5ForRegression
    from transformers import AutoTokenizer

    log = logging.getLogger(__name__)

    if spec.tokenizer_id is None or spec.max_length is None:
        raise ValueError(
            f"MetricX model '{spec.key}' is missing tokenizer_id/max_length metadata."
        )
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    tokenizer = AutoTokenizer.from_pretrained(
        spec.tokenizer_id, legacy=False, use_fast=False
    )

    torch_dtype = _resolve_dtype(device)

    model = (
        MT5ForRegression.from_pretrained(spec.model_id, torch_dtype=torch_dtype)
        .to(device)
        .eval()
    )

    log.info("MetricX loaded: dtype=%s, device=%s", torch_dtype, device)

    return tokenizer, model


def _prepare_token_lists(
    examples: list[Example], tokenizer, max_length: int
) -> list[list[int]]:
    token_lists: list[list[int]] = []
    for ex in examples:
        ids = metricx_input_ids(tokenizer, ex["src"], ex["tgt"])
        prepared = tokenizer.prepare_for_model(
            ids,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
        )
        token_lists.append(prepared["input_ids"])
    return token_lists


def score_metricx(
    examples: list[Example],
    spec: ModelSpec,
    tokenizer,
    model,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    import torch

    if spec.max_length is None:
        raise ValueError(f"MetricX model '{spec.key}' is missing max_length metadata.")
    if not examples:
        return []

    token_lists = _prepare_token_lists(examples, tokenizer, spec.max_length)
    order = sorted(range(len(examples)), key=lambda i: len(token_lists[i]))
    scores: list[float] = [0.0] * len(examples)
    use_pinned = device.type == "cuda"

    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch_ids = [token_lists[i] for i in batch_indices]
            batch_inputs = tokenizer.pad({"input_ids": batch_ids}, return_tensors="pt")
            if use_pinned:
                batch_inputs = {
                    k: v.pin_memory().to(device, non_blocking=True)
                    for k, v in batch_inputs.items()
                }
            else:
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

            outputs = model(**batch_inputs)
            predictions = outputs.predictions
            if isinstance(predictions, torch.Tensor):
                batch_scores = predictions.detach().float().cpu().flatten().tolist()
            else:
                batch_scores = torch.as_tensor(predictions).float().flatten().tolist()
            for idx, score in zip(batch_indices, batch_scores):
                scores[idx] = float(score)

    return scores


def apply_score_adjustment(scores: list[float], spec: ModelSpec) -> list[float]:
    if spec.score_adjuster is None:
        return [float(score) for score in scores]
    return [float(spec.score_adjuster(score)) for score in scores]
