from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


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
        max_seconds_per_part: int,
        target_part_bytes: int,
    ) -> None:
        self.output_dir = (
            Path(output_base)
            / f"dataset={dataset}"
            / f"model={model_tag}"
            / f"split={split}"
            / "stage"
            / f"shard={shard_id:03d}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.done"
        self.committed_direction_keys = self._load_checkpoint()

        self.run_id = run_id
        self.max_directions_per_part = max_directions_per_part
        self.max_seconds_per_part = max_seconds_per_part
        self.target_part_bytes = target_part_bytes

        self._part_seq = 0
        self._frames: list[pd.DataFrame] = []
        self._pending_direction_keys: list[str] = []
        self._pending_direction_set: set[str] = set()
        self._pending_direction_count = 0
        self._pending_bytes_estimate = 0
        self._opened_at: float | None = None

    def _load_checkpoint(self) -> set[str]:
        if not self.checkpoint_path.exists():
            return set()
        with self.checkpoint_path.open("r", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}

    def _estimate_frame_bytes(self, frame: pd.DataFrame) -> int:
        return int(frame.memory_usage(index=True, deep=True).sum())

    def _elapsed_seconds(self) -> float:
        if self._opened_at is None:
            return 0.0
        return time.time() - self._opened_at

    def _should_flush(self) -> bool:
        if self._pending_direction_count >= self.max_directions_per_part:
            return True
        if self._pending_bytes_estimate >= self.target_part_bytes:
            return True
        return self._elapsed_seconds() >= self.max_seconds_per_part

    def _next_part_path(self) -> Path:
        while True:
            filename = f"part-{self.run_id}-{self._part_seq:06d}.parquet"
            self._part_seq += 1
            candidate = self.output_dir / filename
            if not candidate.exists():
                return candidate

    def add_direction(self, frame: pd.DataFrame, direction_key: str) -> None:
        if self._opened_at is None:
            self._opened_at = time.time()

        self._frames.append(frame)
        self._pending_bytes_estimate += self._estimate_frame_bytes(frame)
        self._pending_direction_count += 1
        if direction_key not in self._pending_direction_set:
            self._pending_direction_set.add(direction_key)
            self._pending_direction_keys.append(direction_key)

        if self._should_flush():
            self.flush()

    def _append_checkpoint(self, direction_keys: list[str]) -> None:
        if not direction_keys:
            return
        new_keys = [
            key for key in direction_keys if key not in self.committed_direction_keys
        ]
        if not new_keys:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with self.checkpoint_path.open("a", encoding="utf-8") as handle:
            for key in new_keys:
                handle.write(f"{key}\n")
            handle.flush()
            os.fsync(handle.fileno())

    def flush(self) -> bool:
        if not self._frames:
            return False

        import pandas as pd

        frame = pd.concat(self._frames, ignore_index=True, sort=False)
        final_path = self._next_part_path()
        tmp_path = Path(f"{final_path}.tmp")
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, final_path)

        self._append_checkpoint(self._pending_direction_keys)
        self.committed_direction_keys.update(self._pending_direction_keys)

        self._frames.clear()
        self._pending_direction_keys.clear()
        self._pending_direction_set.clear()
        self._pending_direction_count = 0
        self._pending_bytes_estimate = 0
        self._opened_at = None
        return True

    def close(self) -> None:
        self.flush()
