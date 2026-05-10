from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from execution.opus_queue.tools.merge.collect import (
    CompletedShard,
    done_jobs_for_direction,
    sorted_part_files,
)
from utils.logger import logger

__all__ = ["merge_direction"]

# Rows per Arrow batch when streaming shards into the ParquetWriter. Large
# enough to keep row-group overhead low; small enough that peak memory is
# bounded (~tens of MB per batch of detail rows) regardless of shard count.
_MERGE_BATCH_ROWS = 20_000
# Split merged outputs once the current parquet part reaches roughly 5 GB.
# Rotation is checked after each written batch, so a single file can exceed the
# cap by up to one merge batch.
_MAX_PARQUET_FILE_BYTES = 10_000_000_000
_DROP_OUTPUT_COLUMNS = ("shard_id", "worker_id", "worker_run_id", "direction_key")


@dataclass
class _MergeScanStats:
    rows_kept: int = 0
    rows_dropped: int = 0
    matched_shard_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class _CompletedPart:
    tmp_path: Path
    final_path: Path
    rows: int


def _drop_output_columns(record: dict) -> dict:
    for column in _DROP_OUTPUT_COLUMNS:
        record.pop(column, None)
    return record


def _merged_legacy_file(merged_model_dir: Path, direction_key: str) -> Path:
    return merged_model_dir / f"{direction_key}.parquet"


def _merged_direction_dir(merged_model_dir: Path, direction_key: str) -> Path:
    return merged_model_dir / direction_key


def _temporary_direction_dir(merged_model_dir: Path, direction_key: str) -> Path:
    return merged_model_dir / f".{direction_key}.tmp-{os.getpid()}-{uuid.uuid4().hex}"


def _merged_meta_file(direction_dir: Path, direction_key: str) -> Path:
    return direction_dir / f"{direction_key}.meta.json"


def _merged_part_file(direction_dir: Path, direction_key: str, part_idx: int) -> Path:
    return direction_dir / f"{direction_key}.part-{part_idx:04d}.parquet"


def _existing_flat_part_files(merged_model_dir: Path, direction_key: str) -> list[Path]:
    return sorted(
        path
        for path in merged_model_dir.glob(f"{direction_key}.part-*.parquet")
        if path.is_file()
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _install_completed_direction(
    tmp_direction_dir: Path, final_direction_dir: Path, *, force: bool
) -> None:
    if final_direction_dir.exists():
        if not force:
            raise FileExistsError(
                f"merged direction directory already exists: {final_direction_dir}"
            )
        _remove_path(final_direction_dir)
    os.replace(tmp_direction_dir, final_direction_dir)


def _remove_flat_outputs(merged_model_dir: Path, direction_key: str) -> None:
    stale_paths = _existing_flat_part_files(merged_model_dir, direction_key)
    legacy_out_file = _merged_legacy_file(merged_model_dir, direction_key)
    legacy_meta_file = merged_model_dir / f"{direction_key}.meta.json"
    if legacy_out_file.exists():
        stale_paths.append(legacy_out_file)
    if legacy_meta_file.exists():
        stale_paths.append(legacy_meta_file)
    for stale_path in stale_paths:
        try:
            stale_path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning("Failed to remove stale flat merged output: %s", stale_path)


class _SplitParquetWriter:

    def __init__(self, pq, direction_dir: Path, direction_key: str) -> None:
        self._pq = pq
        self._direction_dir = direction_dir
        self._direction_key = direction_key
        self._schema = None
        self._writer = None
        self._handle = None
        self._current_tmp_path: Path | None = None
        self._current_final_path: Path | None = None
        self._rows_in_current = 0
        self._next_part_idx = 0
        self._completed_parts: list[_CompletedPart] = []

    def _open_part(self) -> None:
        if self._writer is not None:
            return
        if self._schema is None:
            raise ValueError("schema must be initialized before opening a parquet part")
        final_path = _merged_part_file(
            self._direction_dir, self._direction_key, self._next_part_idx
        )
        self._next_part_idx += 1
        self._current_final_path = final_path
        self._current_tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        self._handle = self._current_tmp_path.open("wb")
        self._writer = self._pq.ParquetWriter(self._handle, self._schema)
        self._rows_in_current = 0

    def write_batch(self, pa, batch: list[dict]) -> int:
        if not batch:
            return 0
        if self._schema is None:
            table = pa.Table.from_pylist(batch)
            self._schema = table.schema
        else:
            table = pa.Table.from_pylist(batch, schema=self._schema)
        self._open_part()
        assert self._writer is not None
        assert self._handle is not None
        self._writer.write_table(table)
        self._handle.flush()
        self._rows_in_current += table.num_rows
        if self._handle.tell() >= _MAX_PARQUET_FILE_BYTES:
            self._close_current_part()
        return table.num_rows

    def _close_current_part(self) -> None:
        if self._writer is None:
            return
        assert self._handle is not None
        assert self._current_tmp_path is not None
        assert self._current_final_path is not None
        self._writer.close()
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._completed_parts.append(
            _CompletedPart(
                tmp_path=self._current_tmp_path,
                final_path=self._current_final_path,
                rows=self._rows_in_current,
            )
        )
        self._writer = None
        self._handle = None
        self._current_tmp_path = None
        self._current_final_path = None
        self._rows_in_current = 0

    def finalize(self) -> list[_CompletedPart]:
        self._close_current_part()
        return list(self._completed_parts)

    def cleanup(self) -> None:
        current_tmp_path = self._current_tmp_path
        if self._writer is not None:
            self._writer.close()
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._writer = None
        self._handle = None
        self._current_tmp_path = None
        self._current_final_path = None
        self._rows_in_current = 0
        paths_to_remove = [part.tmp_path for part in self._completed_parts]
        if current_tmp_path is not None:
            paths_to_remove.append(current_tmp_path)
        for path in paths_to_remove:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                logger.warning("Failed to remove merge temp file: %s", path)


def _iter_filtered_records(
    path: Path,
    *,
    winners: dict[int, CompletedShard] | None = None,
    legacy_shard_id: int | None = None,
    stats: _MergeScanStats | None = None,
) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as reader:
        for line_no, raw_line in enumerate(reader, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s:%s", path, line_no)
                if stats is not None:
                    stats.rows_dropped += 1
                continue

            if legacy_shard_id is not None:
                record = _drop_output_columns(record)
                if stats is not None:
                    stats.rows_kept += 1
                yield record
                continue

            if winners is None:
                raise ValueError("part-file filtering requires winners")

            raw_shard_id = record.get("shard_id")
            raw_worker_id = record.get("worker_id")
            if raw_shard_id is None or raw_worker_id is None:
                logger.warning(
                    "Skipping malformed part row missing shard_id/worker_id at %s:%s",
                    path,
                    line_no,
                )
                if stats is not None:
                    stats.rows_dropped += 1
                continue
            try:
                shard_id = int(raw_shard_id)
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping malformed part row with non-integer shard_id at %s:%s",
                    path,
                    line_no,
                )
                if stats is not None:
                    stats.rows_dropped += 1
                continue
            worker_id = str(raw_worker_id)
            winner = winners.get(shard_id)
            if winner is None or winner.worker_id != worker_id:
                if stats is not None:
                    stats.rows_dropped += 1
                continue
            if (
                winner.worker_run_id is not None
                and record.get("worker_run_id") != winner.worker_run_id
            ):
                if stats is not None:
                    stats.rows_dropped += 1
                continue
            record = _drop_output_columns(record)
            if stats is not None:
                stats.rows_kept += 1
                stats.matched_shard_ids.add(shard_id)
            yield record


def merge_direction(
    conn,
    output_base: Path,
    merged_base: Path,
    source_ref: Path | str,
    direction_key: str,
    model: str,
    *,
    force: bool,
    winners: dict[int, CompletedShard] | None = None,
) -> tuple[bool, int, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_dir = output_base / model / direction_key
    if not shard_dir.exists():
        logger.warning("No shard dir for %s/%s at %s", model, direction_key, shard_dir)
        return False, 0, 0

    if winners is None:
        if conn is None:
            raise ValueError(
                "merge_direction requires conn when winners are not provided"
            )
        winners = done_jobs_for_direction(conn, direction_key, model)
    if not winners:
        logger.warning(
            "No completion markers found for %s/%s; skipping merge.",
            model,
            direction_key,
        )
        return False, 0, 0

    part_files = sorted_part_files(shard_dir)
    if not part_files:
        logger.warning("No shard files in %s", shard_dir)
        return False, 0, 0

    merged_model_dir = merged_base / model
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    final_direction_dir = _merged_direction_dir(merged_model_dir, direction_key)

    if final_direction_dir.is_dir() and not force:
        logger.info(
            "Skipping merged direction directory (exists): %s", final_direction_dir
        )
        return False, 0, 0
    if final_direction_dir.exists() and not final_direction_dir.is_dir():
        raise NotADirectoryError(
            f"merged direction path exists but is not a directory: {final_direction_dir}"
        )

    tmp_direction_dir = _temporary_direction_dir(merged_model_dir, direction_key)
    tmp_direction_dir.mkdir()
    writer = _SplitParquetWriter(pq, tmp_direction_dir, direction_key)
    total_rows = 0
    rows_kept = 0
    rows_dropped = 0
    part_file_count = sum(1 for item in part_files if item.kind == "part")
    legacy_file_count = len(part_files) - part_file_count
    seen_shard_ids_from_new_layout: set[int] = set()
    batch: list[dict] = []

    def flush_batch() -> None:
        nonlocal total_rows
        if not batch:
            return
        total_rows += writer.write_batch(pa, batch)
        batch.clear()

    try:
        for item in part_files:
            if item.kind == "part":
                stats = _MergeScanStats()
                for record in _iter_filtered_records(
                    item.path, winners=winners, stats=stats
                ):
                    batch.append(record)
                    if len(batch) >= _MERGE_BATCH_ROWS:
                        flush_batch()
                seen_shard_ids_from_new_layout.update(stats.matched_shard_ids)
                rows_kept += stats.rows_kept
                rows_dropped += stats.rows_dropped
                continue

            assert item.shard_id is not None
            if item.shard_id in seen_shard_ids_from_new_layout:
                logger.info(
                    "Skipping legacy shard file shadowed by winning part rows: %s",
                    item.path,
                )
                continue
            if item.shard_id not in winners:
                logger.warning(
                    "Skipping legacy shard file not present in completion set: %s",
                    item.path,
                )
                continue

            stats = _MergeScanStats()
            for record in _iter_filtered_records(
                item.path,
                legacy_shard_id=item.shard_id,
                stats=stats,
            ):
                batch.append(record)
                if len(batch) >= _MERGE_BATCH_ROWS:
                    flush_batch()
            rows_kept += stats.rows_kept
            rows_dropped += stats.rows_dropped
        flush_batch()
        completed_parts = writer.finalize()
    except Exception:
        writer.cleanup()
        shutil.rmtree(tmp_direction_dir, ignore_errors=True)
        raise

    if not completed_parts:
        # No records across any shard: nothing to write.
        logger.warning("No records found in %s; skipping merge.", shard_dir)
        shutil.rmtree(tmp_direction_dir, ignore_errors=True)
        return False, len(winners), 0

    final_output_files = [part.final_path for part in completed_parts]
    for part in completed_parts:
        os.replace(part.tmp_path, part.final_path)

    meta = {
        "n_shards": len(winners),
        "n_rows": total_rows,
        "source": str(source_ref),
        "source_db": str(source_ref),
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "direction_key": direction_key,
        "model": model,
        "n_parquet_files": len(final_output_files),
        "parquet_files": [path.name for path in final_output_files],
    }
    meta_file = _merged_meta_file(tmp_direction_dir, direction_key)
    _write_json_atomic(meta_file, meta)

    try:
        _install_completed_direction(
            tmp_direction_dir, final_direction_dir, force=force
        )
    except Exception:
        shutil.rmtree(tmp_direction_dir, ignore_errors=True)
        raise
    _remove_flat_outputs(merged_model_dir, direction_key)

    logger.info(
        (
            "Merged %s/%s: winners=%d rows=%d parquet_files=%d part_inputs=%d "
            "legacy_inputs=%d rows_kept=%d rows_dropped=%d -> %s"
        ),
        model,
        direction_key,
        len(winners),
        total_rows,
        len(final_output_files),
        part_file_count,
        legacy_file_count,
        rows_kept,
        rows_dropped,
        final_direction_dir / final_output_files[0].name,
    )
    return True, len(winners), total_rows
