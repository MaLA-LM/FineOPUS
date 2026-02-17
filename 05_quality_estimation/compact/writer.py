from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa


class BucketPartWriter:
    def __init__(
        self,
        *,
        output_base: str | Path,
        dataset: str,
        num_buckets: int,
        target_part_bytes: int,
        run_id: str,
    ) -> None:
        self.output_base = Path(output_base)
        self.dataset = dataset
        self.num_buckets = num_buckets
        self.target_part_bytes = target_part_bytes
        self.run_id = run_id

        self._buffers: dict[int, list[pa.Table]] = defaultdict(list)
        self._buffer_bytes: dict[int, int] = defaultdict(int)
        self._seq_by_bucket: dict[int, int] = defaultdict(int)
        self._pending_keys: dict[int, set[tuple[str, str, str]]] = defaultdict(set)

        self._checkpoint_path = (
            self.output_base / f"dataset={self.dataset}" / "buckets" / "checkpoint.done"
        )
        self.committed_keys: set[tuple[str, str, str]] = self._load_checkpoint()

    def _load_checkpoint(self) -> set[tuple[str, str, str]]:
        if not self._checkpoint_path.exists():
            return set()
        keys: set[tuple[str, str, str]] = set()
        with self._checkpoint_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 3:
                    keys.add((parts[0], parts[1], parts[2]))
        return keys

    def _append_checkpoint(self, keys: set[tuple[str, str, str]]) -> None:
        if not keys:
            return
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with self._checkpoint_path.open("a", encoding="utf-8") as fh:
            for dk, model, split in sorted(keys):
                fh.write(f"{dk}\t{model}\t{split}\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _bucket_dir(self, bucket_id: int) -> Path:
        return (
            self.output_base
            / f"dataset={self.dataset}"
            / "buckets"
            / f"bucket={bucket_id:03d}"
        )

    def _next_bucket_path(self, bucket_id: int) -> Path:
        bucket_dir = self._bucket_dir(bucket_id)
        bucket_dir.mkdir(parents=True, exist_ok=True)
        while True:
            seq = self._seq_by_bucket[bucket_id]
            self._seq_by_bucket[bucket_id] += 1
            candidate = bucket_dir / f"part-{self.run_id}-{seq:06d}.parquet"
            if not candidate.exists():
                return candidate

    def append(
        self,
        bucket_id: int,
        table: pa.Table,
        summary_keys: set[tuple[str, str, str]] | None = None,
    ) -> None:
        self._buffers[bucket_id].append(table)
        self._buffer_bytes[bucket_id] += int(table.nbytes)
        if summary_keys:
            self._pending_keys[bucket_id].update(summary_keys)
        if self._buffer_bytes[bucket_id] >= self.target_part_bytes:
            self.flush_bucket(bucket_id)

    def flush_bucket(self, bucket_id: int) -> bool:
        tables = self._buffers.get(bucket_id)
        if not tables:
            return False
        import pyarrow as pa
        import pyarrow.parquet as pq

        merged = pa.concat_tables(tables)
        output_path = self._next_bucket_path(bucket_id)
        tmp_path = Path(f"{output_path}.tmp")
        pq.write_table(merged, tmp_path)
        os.replace(tmp_path, output_path)

        # Checkpoint the newly committed direction/model/split combos.
        pending = self._pending_keys.pop(bucket_id, set())
        new_keys = pending - self.committed_keys
        self._append_checkpoint(new_keys)
        self.committed_keys.update(new_keys)

        self._buffers[bucket_id].clear()
        self._buffer_bytes[bucket_id] = 0
        return True

    def flush_all(self) -> None:
        for bucket_id in list(self._buffers.keys()):
            self.flush_bucket(bucket_id)
