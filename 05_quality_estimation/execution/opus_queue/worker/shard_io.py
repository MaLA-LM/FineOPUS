"""Shard I/O: JSONL serialization, temp-file management, output paths.

Used by ``worker.py`` to write scored shard data to disk with
crash-safe atomic rename.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

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
    safe_worker_id = (
        worker_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    )
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
