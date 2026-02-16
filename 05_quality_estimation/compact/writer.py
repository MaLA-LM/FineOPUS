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
        model_tag: str,
        num_buckets: int,
        target_part_bytes: int,
        run_id: str,
    ) -> None:
        self.output_base = Path(output_base)
        self.dataset = dataset
        self.model_tag = model_tag
        self.num_buckets = num_buckets
        self.target_part_bytes = target_part_bytes
        self.run_id = run_id

        self._buffers: dict[int, list[pa.Table]] = defaultdict(list)
        self._buffer_bytes: dict[int, int] = defaultdict(int)
        self._seq_by_bucket: dict[int, int] = defaultdict(int)

    def _bucket_dir(self, bucket_id: int) -> Path:
        return (
            self.output_base
            / f"dataset={self.dataset}"
            / f"model={self.model_tag}"
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

    def append(self, bucket_id: int, table: pa.Table) -> None:
        self._buffers[bucket_id].append(table)
        self._buffer_bytes[bucket_id] += int(table.nbytes)
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
        self._buffers[bucket_id].clear()
        self._buffer_bytes[bucket_id] = 0
        return True

    def flush_all(self) -> None:
        for bucket_id in list(self._buffers.keys()):
            self.flush_bucket(bucket_id)
