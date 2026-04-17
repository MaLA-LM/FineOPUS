from __future__ import annotations

import math
import time

__all__ = ["remaining_seconds"]


def remaining_seconds(start_ts: float, walltime: int | None) -> float:
    if walltime is None:
        return math.inf
    elapsed = time.time() - start_ts
    return max(0.0, walltime - elapsed)
