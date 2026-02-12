from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

ScoreAdjuster = Callable[[float], float]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    tokenizer_id: str
    max_length: int
    score_adjuster: ScoreAdjuster | None = None
