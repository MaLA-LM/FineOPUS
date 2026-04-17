from __future__ import annotations

import statistics

from utils.frames import sanitize_scores


def summarize_scores(scores: list[float]) -> tuple[float | None, float | None]:
    valid = sanitize_scores(scores)
    if not valid:
        return None, None
    return float(statistics.fmean(valid)), float(statistics.median(valid))
