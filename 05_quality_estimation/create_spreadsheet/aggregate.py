from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from create_spreadsheet.discover import CheckpointFile, discover_checkpoint_files

SUMMARY_ROW_TYPE = "summary"
CSV_COLUMNS = [
    "direction_key",
    "dataset_name",
    "split",
    "model_name",
    "src_lang_seen",
    "tgt_lang_seen",
    "mean",
    "median",
]


@dataclass(frozen=True)
class AggregateStats:
    checkpoint_files_found: int
    rows_written: int
    invalid_json_lines_skipped: int


def _dataset_name_from_root(results_path: Path) -> str | None:
    name = results_path.name
    if name.startswith("dataset="):
        dataset_name = name.split("=", 1)[1]
        if dataset_name:
            return dataset_name
    return None


def _with_fallback(value: object, fallback: str | None) -> object:
    if value is None:
        return fallback if fallback is not None else ""
    if isinstance(value, str) and value == "":
        return fallback if fallback is not None else ""
    return value


def _to_csv_row(
    record: dict[str, object],
    checkpoint: CheckpointFile,
    dataset_fallback: str | None,
) -> list[object]:
    return [
        _with_fallback(record.get("direction_key"), None),
        _with_fallback(record.get("dataset"), dataset_fallback),
        _with_fallback(record.get("split"), checkpoint.split),
        _with_fallback(record.get("model_name"), checkpoint.model_name),
        record.get("src_lang_seen"),
        record.get("tgt_lang_seen"),
        record.get("mean"),
        record.get("median"),
    ]


def aggregate_checkpoints_to_csv(
    results_path: Path, output_path: Path
) -> AggregateStats:
    checkpoint_files = discover_checkpoint_files(results_path)
    if not checkpoint_files:
        raise SystemExit(f"No checkpoint.jsonl files found under: {results_path}")

    dataset_fallback = _dataset_name_from_root(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    invalid_json_lines_skipped = 0

    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.writer(output_handle)
        writer.writerow(CSV_COLUMNS)

        for checkpoint in checkpoint_files:
            with checkpoint.path.open("r", encoding="utf-8-sig") as checkpoint_handle:
                for raw_line in checkpoint_handle:
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_json_lines_skipped += 1
                        continue

                    if not isinstance(record, dict):
                        continue

                    row_type = record.get("row_type")
                    if row_type is not None and row_type != SUMMARY_ROW_TYPE:
                        continue

                    writer.writerow(
                        _to_csv_row(
                            record=record,
                            checkpoint=checkpoint,
                            dataset_fallback=dataset_fallback,
                        )
                    )
                    rows_written += 1

    return AggregateStats(
        checkpoint_files_found=len(checkpoint_files),
        rows_written=rows_written,
        invalid_json_lines_skipped=invalid_json_lines_skipped,
    )
