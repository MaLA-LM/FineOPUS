from __future__ import annotations

import os
from typing import TYPE_CHECKING

from models.model_registry import ModelSpec

from dataset.mediator import Example

if TYPE_CHECKING:
    import torch


def select_device(gpus: int) -> torch.device:
    import torch

    if gpus > 0 and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def metricx_input_ids(tokenizer, source: str, candidate: str) -> list[int]:
    return (
        tokenizer.encode(f"source: {source}", add_special_tokens=False)
        + tokenizer.encode(f" candidate: {candidate}", add_special_tokens=False)
        + tokenizer.encode(" reference: ", add_special_tokens=False)
    )


def load_metricx(spec: ModelSpec, device: torch.device):
    from mbrs.metrics.metricx import MT5ForRegression
    from transformers import AutoTokenizer

    if spec.tokenizer_id is None or spec.max_length is None:
        raise ValueError(
            f"MetricX model '{spec.key}' is missing tokenizer_id/max_length metadata."
        )
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    tokenizer = AutoTokenizer.from_pretrained(
        spec.tokenizer_id, legacy=False, use_fast=False
    )
    model = MT5ForRegression.from_pretrained(spec.model_id).to(device).eval()
    return tokenizer, model


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
    scores: list[float] = []
    with torch.no_grad():
        for idx in range(0, len(examples), batch_size):
            batch = examples[idx : idx + batch_size]
            input_ids = []
            for ex in batch:
                ids = metricx_input_ids(tokenizer, ex["src"], ex["tgt"])
                prepared = tokenizer.prepare_for_model(
                    ids,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=spec.max_length,
                    return_attention_mask=False,
                )
                input_ids.append(prepared["input_ids"])

            batch_inputs = tokenizer.pad(
                {"input_ids": input_ids}, return_tensors="pt"
            ).to(device)
            outputs = model(**batch_inputs)
            predictions = outputs.predictions
            if isinstance(predictions, torch.Tensor):
                batch_scores = predictions.detach().cpu().flatten().tolist()
            else:
                batch_scores = torch.as_tensor(predictions).flatten().tolist()
            scores.extend(float(score) for score in batch_scores)

    return scores


def apply_score_adjustment(scores: list[float], spec: ModelSpec) -> list[float]:
    if spec.score_adjuster is None:
        return [float(score) for score in scores]
    return [float(spec.score_adjuster(score)) for score in scores]
