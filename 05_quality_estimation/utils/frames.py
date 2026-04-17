from __future__ import annotations

import math


def sanitize_scores(scores: list[float]) -> list[float]:
    """Replace NaN with 0.0 and coerce to float.

    Failed segments (LLM timeouts, validation errors, backend exceptions
    that surface as NaN) are treated as worst-case 0 scores rather than
    null, so downstream aggregations don't need to special-case missing
    values. ``summarize_scores`` already applies the same NaN->0 rule when
    computing mean/median, so detail rows and summary rows stay consistent.
    """
    sanitized: list[float] = []
    for score in scores:
        value = float(score)
        if math.isnan(value):
            value = 0.0
        sanitized.append(value)
    return sanitized
