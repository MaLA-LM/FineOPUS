from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from execution.opus_queue.manifest.reader import ManifestRow
from utils.logger import logger

__all__ = ["TraceWriter"]


class TraceWriter:
    def __init__(
        self,
        trace_root: str | Path,
        build_tag: str,
        worker_slot_id: str,
    ) -> None:
        self.trace_dir = Path(trace_root).expanduser() / build_tag / worker_slot_id
        self.completed_path = self.trace_dir / "state.jsonl"
        self.state_path = self.trace_dir / "state.json"
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def append_completion(self, event: dict[str, Any]) -> dict[str, Any]:
        if event.get("event") != "done":
            raise ValueError("manifest trace only accepts successful completion events")
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with self.completed_path.open("ab") as handle:
            line = json.dumps(
                event,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            handle.write(line)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        return self.append_completion(event)

    def write_state(self, snapshot: dict[str, Any]) -> None:
        payload = {
            "materialized_at": int(time.time()),
            **snapshot,
        }
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_path, self.state_path)
        except OSError as exc:
            logger.warning("Could not materialize trace state cache %s: %s", self.state_path, exc)

    def done_event(
        self,
        row: ManifestRow,
        *,
        worker_run_id: str,
        finished_at: int | None,
        gpu_count: int,
        gpu_seconds_delta: float,
        out_path: str | Path | None,
    ) -> dict[str, Any]:
        return self._base_event(
            row,
            event="done",
            worker_run_id=worker_run_id,
            finished_at=finished_at,
            gpu_count=gpu_count,
            gpu_seconds_delta=gpu_seconds_delta,
            out_path=str(out_path) if out_path is not None else None,
        )

    @staticmethod
    def _base_event(
        row: ManifestRow,
        *,
        event: str,
        worker_run_id: str,
        finished_at: int | None,
        gpu_count: int,
        gpu_seconds_delta: float,
        out_path: str | None,
    ) -> dict[str, Any]:
        return {
            "ts": int(time.time()),
            "event": event,
            "worker_slot_id": row.worker_slot_id,
            "worker_run_id": worker_run_id,
            "model": row.model,
            "direction_key": row.direction_key,
            "shard_id": row.shard_id,
            "start_idx": row.start_idx,
            "end_idx": row.end_idx,
            "finished_at": finished_at,
            "gpu_count": max(1, int(gpu_count)),
            "gpu_seconds_delta": float(gpu_seconds_delta),
            "out_path": out_path,
        }
