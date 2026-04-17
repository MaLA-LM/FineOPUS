from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset.opus import OPUS_ADAPTER
from dataset.mediator import DatasetAdapter
from dataset.opus.builder import REQUIRED_COLUMNS
from utils.logger import logger

Example = dict[str, Any]


@dataclass
class ShardBounds:
    """Per-call bounds for the sharding OPUS adapter.

    The worker sets ``start`` and ``end`` before invoking a scorer's
    ``run_entry(entry)`` so the wrapping ``load_parallel`` returns only
    the requested slice.
    """

    start: int = 0
    end: int | None = None
    active: bool = False

    def clear(self) -> None:
        self.start = 0
        self.end = None
        self.active = False


@dataclass
class ShardingOpusContext:
    adapter: DatasetAdapter
    bounds: ShardBounds = field(default_factory=ShardBounds)


def _stream_parquet_slice(
    direction_dir: Path,
    start_idx: int,
    end_idx: int,
) -> list[Example]:
    import pyarrow.parquet as pq

    if start_idx < 0 or end_idx <= start_idx:
        return []

    shard_files = sorted(p for p in direction_dir.glob("*.parquet") if p.is_file())
    if not shard_files:
        raise FileNotFoundError(f"No parquet files under: {direction_dir}")

    rows_seen = 0
    examples: list[Example] = []

    for path in shard_files:
        parquet_file = pq.ParquetFile(str(path))
        schema_names = set(parquet_file.schema_arrow.names)
        missing = [col for col in REQUIRED_COLUMNS if col not in schema_names]
        if missing:
            raise ValueError(
                f"OPUS parquet {path} is missing required columns: {missing}"
            )

        file_rows = int(parquet_file.metadata.num_rows)
        if rows_seen + file_rows <= start_idx:
            rows_seen += file_rows
            continue

        for batch in parquet_file.iter_batches(
            batch_size=8192,
            columns=parquet_file.schema_arrow.names,
        ):
            batch_rows = batch.to_pylist()
            batch_size = len(batch_rows)
            batch_start = rows_seen
            batch_end = rows_seen + batch_size
            if batch_end <= start_idx:
                rows_seen = batch_end
                continue
            if batch_start >= end_idx:
                return examples

            slice_start = max(0, start_idx - batch_start)
            slice_end = min(batch_size, end_idx - batch_start)
            for row in batch_rows[slice_start:slice_end]:
                src = row.get("source_text")
                tgt = row.get("target_text")
                if src is None or tgt is None:
                    continue
                row["src"] = src
                row["tgt"] = tgt
                examples.append(row)
            rows_seen = batch_end
            if batch_end >= end_idx:
                return examples

    return examples


def build_sharding_opus_adapter() -> ShardingOpusContext:
    """Wrap OPUS_ADAPTER with a stateful slice of load_parallel.

    The returned context's ``bounds`` attribute is mutated by the worker
    before each shard. ``load_parallel`` consults it; when ``active`` is
    False it falls through to the default full-direction loader.
    """
    bounds = ShardBounds()

    def load_parallel(src_lang, tgt_lang, *, split="all", root=None) -> list[Example]:
        if not bounds.active:
            return OPUS_ADAPTER.load_parallel(
                src_lang, tgt_lang, split=split, root=root
            )
        if root is None:
            root = OPUS_ADAPTER.default_root
        direction_dir = Path(root) / f"{src_lang}-{tgt_lang}"
        if not direction_dir.exists():
            raise FileNotFoundError(f"OPUS direction dir not found: {direction_dir}")
        end = bounds.end if bounds.end is not None else bounds.start
        examples = _stream_parquet_slice(direction_dir, bounds.start, end)
        logger.info(
            "OPUS shard loaded %s-%s range=[%d,%d) rows=%d",
            src_lang,
            tgt_lang,
            bounds.start,
            end,
            len(examples),
        )
        return examples

    def limit_rows(examples, max_rows):
        # Shard already bounds the slice; preserve max_rows for debug caps.
        return OPUS_ADAPTER.limit_rows(examples, max_rows)

    adapter = DatasetAdapter(
        id=OPUS_ADAPTER.id,
        default_root=OPUS_ADAPTER.default_root,
        split_values=OPUS_ADAPTER.split_values,
        load_parallel=load_parallel,
        limit_rows=limit_rows,
        discover_directions=OPUS_ADAPTER.discover_directions,
        expected_detail_rows=OPUS_ADAPTER.expected_detail_rows,
        language_codes=OPUS_ADAPTER.language_codes,
        langcode_to_name=OPUS_ADAPTER.langcode_to_name,
        build_frames=OPUS_ADAPTER.build_frames,
    )
    return ShardingOpusContext(adapter=adapter, bounds=bounds)
