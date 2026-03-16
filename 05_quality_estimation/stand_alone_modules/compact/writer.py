from __future__ import annotations

import os
from pathlib import Path

from utils.io import ROW_TYPE_SUMMARY


class ModelPartWriter:
    """Writes compacted Parquet files for a single model bucket.

    Each bucket owns its own output directory and checkpoint file.
    """

    def __init__(
        self,
        *,
        output_base: str | Path,
        dataset: str,
        model: str,
        target_part_bytes: int,
        run_id: str,
        bucket_name: str,
        bucket_id: int,
    ) -> None:
        self.output_base = Path(output_base)
        self.dataset = dataset
        self.model = model
        self.target_part_bytes = target_part_bytes
        self.run_id = run_id
        self.bucket_name = bucket_name
        self.bucket_id = bucket_id

        self._bucket_dir = (
            self.output_base
            / f"dataset={self.dataset}"
            / f"buckets={self.bucket_name}"
            / f"model={self.model}"
            / f"bucket={self.bucket_id:03d}"
        )
        self._checkpoint_path = self._bucket_dir / "checkpoint.parquet"

        self._buffer_tables: list = []
        self._buffer_bytes: int = 0
        self._seq: int = 0

        self.committed_splits: set[tuple[str, str]] = self._load_checkpoint()

    # -- checkpoint ---------------------------------------------------------

    def _load_checkpoint(self) -> set[tuple[str, str]]:
        """Load (direction_key, split) pairs already committed for this bucket."""
        if not self._checkpoint_path.exists():
            return set()
        import pyarrow.parquet as pq

        table = pq.read_table(self._checkpoint_path, columns=["direction_key", "split"])
        dk = table.column("direction_key").to_pylist()
        sp = table.column("split").to_pylist()
        return {(dk[i], sp[i]) for i in range(len(dk))}

    def _save_checkpoint(self) -> None:
        """Rewrite the checkpoint Parquet with all committed splits."""
        if not self.committed_splits:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        dks, sps = zip(*sorted(self.committed_splits))
        table = pa.table(
            {
                "direction_key": pa.array(dks, type=pa.utf8()),
                "split": pa.array(sps, type=pa.utf8()),
            }
        )
        self._bucket_dir.mkdir(parents=True, exist_ok=True)
        tmp = Path(f"{self._checkpoint_path}.tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, self._checkpoint_path)

    # -- writing ------------------------------------------------------------

    def _next_path(self) -> Path:
        self._bucket_dir.mkdir(parents=True, exist_ok=True)
        while True:
            candidate = self._bucket_dir / f"part-{self.run_id}-{self._seq:06d}.parquet"
            self._seq += 1
            if not candidate.exists():
                return candidate

    def append_new_rows(self, table) -> tuple[int, int]:
        """Append only rows that are not already committed for this bucket."""
        import pyarrow as pa

        skipped = 0
        if self.committed_splits:
            dk = table.column("direction_key").to_pylist()
            sp = table.column("split").to_pylist()
            keep = [
                i
                for i in range(table.num_rows)
                if (dk[i], sp[i]) not in self.committed_splits
            ]
            skipped = table.num_rows - len(keep)
            if not keep:
                return 0, skipped
            table = table.take(pa.array(keep, type=pa.int64()))

        rt = table.column("row_type").to_pylist()
        dk = table.column("direction_key").to_pylist()
        sp = table.column("split").to_pylist()
        new_splits = {
            (dk[i], sp[i]) for i in range(table.num_rows) if rt[i] == ROW_TYPE_SUMMARY
        }

        self._buffer_tables.append(table)
        self._buffer_bytes += int(table.nbytes)
        if new_splits:
            self.committed_splits.update(new_splits)
        if self._buffer_bytes >= self.target_part_bytes:
            self._flush_buffer()
        return table.num_rows, skipped

    def _flush_buffer(self) -> None:
        if not self._buffer_tables:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        merged = pa.concat_tables(self._buffer_tables)
        output_path = self._next_path()
        tmp = Path(f"{output_path}.tmp")
        pq.write_table(merged, tmp)
        os.replace(tmp, output_path)

        self._buffer_tables.clear()
        self._buffer_bytes = 0
        self._save_checkpoint()

    def flush_all(self) -> None:
        """Flush remaining buffer and persist the checkpoint."""
        self._flush_buffer()
        self._save_checkpoint()
