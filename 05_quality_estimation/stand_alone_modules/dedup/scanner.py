from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from create_spreadsheet.discover import discover_checkpoint_files
from utils.io import ROW_TYPE_SUMMARY

DuplicateKey = tuple[str, str, str]  # (direction_key, model_name, split)


@dataclass(frozen=True)
class DuplicateRecord:
    shard_path: str
    direction_key: str
    model_name: str
    split: str
    total_occurrences: int
    duplicates_to_remove: int


@dataclass(frozen=True)
class ScanResult:
    dataset_path: str
    scan_timestamp: str
    shards_scanned: int
    duplicates: list[DuplicateRecord]


def _scan_single_checkpoint(checkpoint_path: Path) -> list[DuplicateRecord]:
    """Scan one checkpoint.jsonl and return duplicate records found in it."""
    shard_path = str(checkpoint_path.parent)
    key_counts: dict[DuplicateKey, int] = {}

    with checkpoint_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {checkpoint_path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Expected JSON object at {checkpoint_path}:{line_no}, "
                    f"got {type(record).__name__}"
                )

            row_type = record.get("row_type")
            if row_type != ROW_TYPE_SUMMARY:
                raise RuntimeError(
                    f"Expected row_type={ROW_TYPE_SUMMARY!r} at "
                    f"{checkpoint_path}:{line_no}, got row_type={row_type!r}"
                )

            for field_name in ("direction_key", "model_name", "split"):
                if field_name not in record:
                    raise RuntimeError(
                        f"Missing required field {field_name!r} at "
                        f"{checkpoint_path}:{line_no}"
                    )

            key: DuplicateKey = (
                str(record["direction_key"]),
                str(record["model_name"]),
                str(record["split"]),
            )
            key_counts[key] = key_counts.get(key, 0) + 1

    return [
        DuplicateRecord(
            shard_path=shard_path,
            direction_key=dk,
            model_name=mn,
            split=sp,
            total_occurrences=count,
            duplicates_to_remove=count - 1,
        )
        for (dk, mn, sp), count in sorted(key_counts.items())
        if count > 1
    ]


def scan_dataset(dataset_path: Path) -> ScanResult:
    """Walk every model/split/shard under *dataset_path* and detect duplicates."""
    if not dataset_path.exists():
        raise SystemExit(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise SystemExit(f"Dataset path is not a directory: {dataset_path}")

    checkpoint_files = discover_checkpoint_files(dataset_path)
    if not checkpoint_files:
        raise SystemExit(f"No checkpoint.jsonl files found under: {dataset_path}")

    all_duplicates: list[DuplicateRecord] = []
    for checkpoint in checkpoint_files:
        all_duplicates.extend(_scan_single_checkpoint(checkpoint.path))

    return ScanResult(
        dataset_path=str(dataset_path),
        scan_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        shards_scanned=len(checkpoint_files),
        duplicates=all_duplicates,
    )


def scan_result_to_plan(result: ScanResult) -> dict:
    """Convert a ScanResult into a JSON-serialisable plan dictionary."""
    return {
        "dataset_path": result.dataset_path,
        "scan_timestamp": result.scan_timestamp,
        "shards_scanned": result.shards_scanned,
        "total_duplicates": len(result.duplicates),
        "duplicates": [
            {
                "shard_path": d.shard_path,
                "direction_key": d.direction_key,
                "model_name": d.model_name,
                "split": d.split,
                "total_occurrences": d.total_occurrences,
                "duplicates_to_remove": d.duplicates_to_remove,
            }
            for d in result.duplicates
        ],
    }


def write_plan(plan: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
