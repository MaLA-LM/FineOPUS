from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from execution.opus_queue.manifest.reader import ManifestRow

__all__ = ["ShardProgress", "TraceState", "event_key", "key_to_string"]

TraceKey = tuple[str, str, int]


def event_key(event: dict[str, Any]) -> TraceKey:
    return (
        str(event["model"]),
        str(event["direction_key"]),
        int(event["shard_id"]),
    )


def key_to_string(key: TraceKey) -> str:
    model, direction_key, shard_id = key
    return f"{model}/{direction_key}/{shard_id}"


@dataclass
class ShardProgress:
    status: str = "pending"
    finished_at: int | None = None
    gpu_count: int = 1
    gpu_seconds_total: float = 0.0
    out_path: str | None = None
    worker_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "finished_at": self.finished_at,
            "gpu_count": self.gpu_count,
            "gpu_seconds_total": self.gpu_seconds_total,
            "out_path": self.out_path,
            "worker_run_id": self.worker_run_id,
        }


class TraceState:
    def __init__(self) -> None:
        self._progress: dict[TraceKey, ShardProgress] = {}

    def get(self, row: ManifestRow) -> ShardProgress:
        progress = self._progress.get(row.key)
        if progress is None:
            return ShardProgress()
        return progress

    def is_done(self, row: ManifestRow) -> bool:
        return self.get(row).status == "done"

    def can_process(self, row: ManifestRow) -> bool:
        return not self.is_done(row)

    def apply_event(self, event: dict[str, Any]) -> None:
        event_name = str(event.get("event", ""))
        if event_name != "done":
            return

        key = event_key(event)
        progress = self._progress.setdefault(key, ShardProgress())

        progress.status = "done"
        progress.finished_at = _optional_int(event.get("finished_at"))
        progress.out_path = _optional_str(event.get("out_path"))
        progress.gpu_seconds_total = float(event.get("gpu_seconds_delta") or 0.0)
        gpu_count = event.get("gpu_count", event.get("claim_gpu_count"))
        if gpu_count is not None:
            progress.gpu_count = max(1, int(gpu_count))
        if event.get("worker_run_id") is not None:
            progress.worker_run_id = str(event["worker_run_id"])

    def counts(self, assignments: list[ManifestRow]) -> dict[str, int]:
        counts: dict[str, int] = {
            "pending": 0,
            "done": 0,
        }
        for row in assignments:
            progress = self.get(row)
            status = progress.status
            counts[status] = counts.get(status, 0) + 1
        return counts

    def snapshot(self, assignments: list[ManifestRow]) -> dict[str, Any]:
        shards: dict[str, dict[str, Any]] = {}
        for row in assignments:
            shards[key_to_string(row.key)] = {
                "assignment_seq": row.assignment_seq,
                "worker_slot_id": row.worker_slot_id,
                **self.get(row).to_dict(),
            }
        return {
            "counts": self.counts(assignments),
            "shards": shards,
        }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
