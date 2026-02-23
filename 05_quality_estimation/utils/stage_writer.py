from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    import pandas as pd

from utils.io import ROW_TYPE_SUMMARY
from utils.logger import logger


class ShardStageWriter:

    def __init__(
        self,
        *,
        output_base: str | Path,
        dataset: str,
        model_tag: str,
        split: str,
        shard_id: int,
        run_id: str,
        max_directions_per_part: int,
        target_part_bytes: int,
    ) -> None:
        self.output_dir = (
            Path(output_base)
            / f"dataset={dataset}"
            / f"model={model_tag}"
            / f"split={split}"
            / f"shard={shard_id:03d}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.jsonl"
        self.committed_direction_keys = self._load_checkpoint()

        self.run_id = run_id
        self.max_directions_per_part = max_directions_per_part
        self.target_part_bytes = target_part_bytes

        self._part_seq = 0
        self._current_part_handle: BinaryIO | None = None
        self._current_part_path: Path | None = None
        self._current_part_bytes = 0
        self._current_part_directions = 0

    def _load_checkpoint(self) -> set[str]:
        if not self.checkpoint_path.exists():
            return set()
        committed: set[str] = set()
        with self.checkpoint_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("row_type") == ROW_TYPE_SUMMARY:
                    committed.add(record["direction_key"])
        return committed

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None:
            return None

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None

        return value

    @classmethod
    def _encode_json_line(cls, record: dict[str, Any]) -> bytes:
        payload = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return payload + b"\n"

    def _next_part_path(self) -> Path:
        while True:
            filename = f"part-{self.run_id}-{self._part_seq:06d}.jsonl"
            self._part_seq += 1
            candidate = self.output_dir / filename
            if not candidate.exists():
                return candidate

    def _open_part_if_needed(self) -> None:
        if self._current_part_handle is not None:
            return
        path = self._next_part_path()
        self._current_part_path = path
        self._current_part_handle = path.open("ab")
        self._current_part_bytes = path.stat().st_size
        self._current_part_directions = 0

    def _close_current_part(self) -> None:
        if self._current_part_handle is not None:
            self._current_part_handle.close()
        self._current_part_handle = None
        self._current_part_path = None
        self._current_part_bytes = 0
        self._current_part_directions = 0

    def _rotate_before_next_direction(self) -> None:
        if self._current_part_handle is None:
            return
        if (
            self._current_part_bytes >= self.target_part_bytes
            or self._current_part_directions >= self.max_directions_per_part
        ):
            self._close_current_part()

    def _append_checkpoint(self, summary_record: dict[str, Any]) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with self.checkpoint_path.open("ab") as handle:
            handle.write(self._encode_json_line(summary_record))
            handle.flush()
            os.fsync(handle.fileno())

    def add_direction(self, frame: pd.DataFrame, direction_key: str) -> None:
        self._rotate_before_next_direction()
        self._open_part_if_needed()

        columns = list(frame.columns)
        summary_record: dict[str, Any] | None = None

        for row_values in frame.itertuples(index=False, name=None):
            row = {
                column: self._normalize_value(value)
                for column, value in zip(columns, row_values)
            }
            if row.get("row_type") == ROW_TYPE_SUMMARY:
                summary_record = row

            encoded = self._encode_json_line(row)
            self._current_part_handle.write(encoded)
            self._current_part_bytes += len(encoded)

        self._current_part_handle.flush()
        os.fsync(self._current_part_handle.fileno())
        self._append_checkpoint(summary_record)
        self.committed_direction_keys.add(direction_key)
        self._current_part_directions += 1

    def flush(self) -> bool:
        if self._current_part_handle is None:
            return False
        self._close_current_part()
        return True

    def close(self) -> None:
        self.flush()
