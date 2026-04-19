"""Shard I/O helpers for legacy shard files and worker-owned part files."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from utils.io import ROW_TYPE_SUMMARY
from utils.logger import logger


def shard_output_path(
    output_base: Path, model_name: str, direction_key: str, shard_id: int
) -> Path:
    return (
        output_base
        / model_name
        / direction_key
        / f"shard_{shard_id:05d}.jsonl"
    )


def direction_dir(output_base: Path, model_name: str, direction_key: str) -> Path:
    return output_base / model_name / direction_key


def next_part_path(direction_path: Path, worker_id_safe: str, seq: int) -> Path:
    return direction_path / f"part-{worker_id_safe}-{seq:04d}.jsonl"


def sanitize_worker_id(worker_id: str, *, max_length: int = 180) -> str:
    safe_worker_id = (
        worker_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    )
    if len(safe_worker_id) <= max_length:
        return safe_worker_id
    digest = hashlib.blake2b(worker_id.encode("utf-8"), digest_size=8).hexdigest()
    head_budget = max_length - len(digest) - 1
    if head_budget <= 0:
        return digest[:max_length]
    return f"{safe_worker_id[:head_budget]}-{digest}"


def _normalize_value(value):
    """Coerce numpy scalars and NaN/Inf to JSON-safe Python types."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def frame_to_jsonl_bytes(frame) -> bytes:
    """Serialize a pandas DataFrame to newline-delimited JSON bytes."""
    columns = list(frame.columns)
    buf = bytearray()
    for row_values in frame.itertuples(index=False, name=None):
        record = {
            col: _normalize_value(val) for col, val in zip(columns, row_values)
        }
        line = json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        buf.extend(line.encode("utf-8"))
        buf.append(0x0A)
    return bytes(buf)


def write_temp_payload(path: Path, payload: bytes, worker_id: str) -> Path:
    """Write payload to a worker-specific temp file next to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_worker_id = sanitize_worker_id(worker_id)
    tmp = path.with_suffix(path.suffix + f".{safe_worker_id}.tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return tmp


def cleanup_temp_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        logger.warning("Failed to remove stale temp shard file: %s", path)


def count_detail_rows(frame) -> int:
    if "row_type" not in frame.columns:
        return int(len(frame))
    return int((frame["row_type"] != ROW_TYPE_SUMMARY).sum())


@dataclass
class _DirectionPartState:
    output_dir: Path
    next_seq: int = 0
    current_path: Path | None = None
    handle: BinaryIO | None = None
    bytes_written: int = 0
    shards_in_file: int = 0


class DirectionPartWriter:

    def __init__(
        self,
        output_base: Path | str,
        model: str,
        worker_id: str,
        *,
        max_bytes: int,
        max_shards_per_part: int,
    ) -> None:
        self._output_base = Path(output_base)
        self._model = model
        self._worker_id_safe = sanitize_worker_id(worker_id)
        self._max_bytes = int(max_bytes)
        self._max_shards_per_part = int(max_shards_per_part)
        if self._max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        if self._max_shards_per_part < 1:
            raise ValueError("max_shards_per_part must be >= 1")
        self._states: dict[str, _DirectionPartState] = {}

    def _state_for(self, direction_key: str) -> _DirectionPartState:
        state = self._states.get(direction_key)
        if state is not None:
            return state
        state = _DirectionPartState(
            output_dir=direction_dir(self._output_base, self._model, direction_key)
        )
        self._states[direction_key] = state
        return state

    def _open_handle(self, state: _DirectionPartState) -> None:
        if state.handle is not None:
            return
        state.output_dir.mkdir(parents=True, exist_ok=True)
        if state.current_path is None:
            while True:
                candidate = next_part_path(
                    state.output_dir, self._worker_id_safe, state.next_seq
                )
                state.next_seq += 1
                if candidate.exists():
                    continue
                state.current_path = candidate
                state.bytes_written = 0
                state.shards_in_file = 0
                break
        else:
            state.bytes_written = (
                state.current_path.stat().st_size if state.current_path.exists() else 0
            )
        state.handle = state.current_path.open("ab")

    @staticmethod
    def _close_state(state: _DirectionPartState) -> None:
        if state.handle is None:
            return
        state.handle.close()
        state.handle = None

    def _rotate_state(self, state: _DirectionPartState) -> None:
        self._close_state(state)
        state.current_path = None
        state.bytes_written = 0
        state.shards_in_file = 0

    def append_shard(self, frame, direction_key: str, shard_id: int) -> Path:
        del shard_id  # payload rows already carry shard_id; writer only tracks files.
        payload = frame_to_jsonl_bytes(frame)
        state = self._state_for(direction_key)
        if (
            state.current_path is not None
            and state.bytes_written > 0
            and (
                state.bytes_written + len(payload) > self._max_bytes
                or state.shards_in_file >= self._max_shards_per_part
            )
        ):
            self._rotate_state(state)
        self._open_handle(state)
        assert state.handle is not None
        assert state.current_path is not None
        state.handle.write(payload)
        state.handle.flush()
        os.fsync(state.handle.fileno())
        state.bytes_written += len(payload)
        state.shards_in_file += 1
        return state.current_path

    def close_direction(self, direction_key: str) -> None:
        state = self._states.get(direction_key)
        if state is None:
            return
        self._close_state(state)

    def close_all(self) -> None:
        for state in self._states.values():
            self._close_state(state)
