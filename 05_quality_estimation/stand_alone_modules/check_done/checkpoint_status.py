from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from execution.flores_array.manifest import ManifestEntry
from dataset.mediator import get_dataset

SplitShardKey = tuple[str, int]
LangPair = tuple[str, str]
DirectionSet = set[LangPair]
StatusMap = dict[SplitShardKey, DirectionSet]


@dataclass(frozen=True)
class SplitProgress:
    split: str
    done: int
    total: int


@dataclass(frozen=True)
class ShardReport:
    shard_id: int
    done_total: int
    expected_total: int
    status: str
    split_progress: tuple[SplitProgress, ...]


def load_expected(tsv_path: str | Path) -> StatusMap:
    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"TSV file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV missing header: {path}")

        required_columns = {"src_lang", "tgt_lang", "split", "shard_id"}
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"TSV missing required columns {sorted(missing)}: {path}")

        expected: StatusMap = {}
        for row_number, row in enumerate(reader, start=2):
            src_lang = (row.get("src_lang") or "").strip()
            tgt_lang = (row.get("tgt_lang") or "").strip()
            split = (row.get("split") or "").strip()
            shard_raw = (row.get("shard_id") or "").strip()

            if not src_lang or not tgt_lang or not split or not shard_raw:
                raise ValueError(
                    f"TSV row {row_number} has empty required values: {path}"
                )

            try:
                entry = ManifestEntry(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    split=split,
                    shard_id=int(shard_raw),
                )
            except ValueError as exc:
                raise ValueError(
                    f"TSV row {row_number} has invalid shard_id='{shard_raw}': {path}"
                ) from exc

            if entry.shard_id is None or entry.shard_id < 0:
                raise ValueError(
                    f"TSV row {row_number} has negative shard_id={entry.shard_id}: {path}"
                )

            key = (entry.split, entry.shard_id)
            expected.setdefault(key, set()).add((entry.src_lang, entry.tgt_lang))

    return expected


def discover_splits(dataset_id: str) -> list[str]:
    dataset = get_dataset(dataset_id)
    return sorted(dataset.split_values)


def _parse_shard_id(folder_name: str) -> int | None:
    if not folder_name.startswith("shard="):
        return None

    raw = folder_name.split("=", 1)[1].strip()
    if not raw:
        return None

    try:
        shard_id = int(raw)
    except ValueError:
        return None
    if shard_id < 0:
        return None
    return shard_id


def load_completed(
    model_root: str | Path, discovered_splits: Sequence[str]
) -> StatusMap:
    root = Path(model_root)
    completed: StatusMap = {}

    for split in discovered_splits:
        split_dir = root / f"split={split}"
        if not split_dir.exists() or not split_dir.is_dir():
            continue

        for shard_dir in split_dir.iterdir():
            if not shard_dir.is_dir():
                continue

            shard_id = _parse_shard_id(shard_dir.name)
            if shard_id is None:
                continue

            checkpoint_path = shard_dir / "checkpoint.jsonl"
            if not checkpoint_path.exists() or not checkpoint_path.is_file():
                continue

            key = (split, shard_id)
            done_set = completed.setdefault(key, set())

            with checkpoint_path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue

                    record = json.loads(line)
                    done_set.add(
                        (
                            str(record["src_lang"]).strip(),
                            str(record["tgt_lang"]).strip(),
                        )
                    )

    return completed


def build_report(expected: StatusMap, completed: StatusMap) -> list[ShardReport]:
    per_shard: dict[int, dict[str, tuple[int, int]]] = {}

    for (split, shard_id), expected_pairs in expected.items():
        done_pairs = completed.get((split, shard_id), set())
        done_count = len(expected_pairs & done_pairs)
        total_count = len(expected_pairs)

        shard_entry = per_shard.setdefault(shard_id, {})
        shard_entry[split] = (done_count, total_count)

    reports: list[ShardReport] = []
    for shard_id in sorted(per_shard):
        split_parts = per_shard[shard_id]
        ordered_splits = sorted(split_parts)

        split_progress = tuple(
            SplitProgress(
                split=split,
                done=split_parts[split][0],
                total=split_parts[split][1],
            )
            for split in ordered_splits
        )

        done_total = sum(progress.done for progress in split_progress)
        expected_total = sum(progress.total for progress in split_progress)
        status = "COMPLETE" if done_total == expected_total else "INCOMPLETE"

        reports.append(
            ShardReport(
                shard_id=shard_id,
                done_total=done_total,
                expected_total=expected_total,
                status=status,
                split_progress=split_progress,
            )
        )

    return reports


def format_report(rows: Sequence[ShardReport]) -> str:
    lines: list[str] = []
    for row in rows:
        line = (
            f"shard={row.shard_id:03d} total={row.done_total}/{row.expected_total} "
            f"status={row.status}"
        )
        for split_progress in row.split_progress:
            line += (
                f" split={split_progress.split} "
                f"{split_progress.done}/{split_progress.total}"
            )
        lines.append(line)

    total_shards = len(rows)
    complete_shards = sum(1 for row in rows if row.status == "COMPLETE")
    incomplete_shards = total_shards - complete_shards
    lines.append(
        f"summary shards={total_shards} complete={complete_shards} "
        f"incomplete={incomplete_shards}"
    )
    return "\n".join(lines)
