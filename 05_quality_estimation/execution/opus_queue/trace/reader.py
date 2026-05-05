from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from execution.opus_queue.trace.state import TraceState

__all__ = ["read_events", "replay_events"]


def read_events(events_path: str | Path) -> list[dict[str, Any]]:
    path = Path(events_path)
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid trace event at {path}:{lineno}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"Invalid trace event at {path}:{lineno}: expected object")
            events.append(event)
    return events


def replay_events(events_path: str | Path | Iterable[str | Path]) -> TraceState:
    if isinstance(events_path, (str, Path)):
        paths = [events_path]
    else:
        paths = list(events_path)

    state = TraceState()
    for path in paths:
        for event in read_events(path):
            state.apply_event(event)
    return state
