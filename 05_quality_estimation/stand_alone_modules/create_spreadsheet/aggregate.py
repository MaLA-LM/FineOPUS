from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from create_spreadsheet.discover import CheckpointFile, discover_checkpoint_files

SUMMARY_ROW_TYPE = "summary"
BASE_COLUMNS = [
    "direction_key",
    "dataset_name",
    "split",
]
MODEL_COLUMN_SUFFIXES = ("mean", "median", "src_lang_seen", "tgt_lang_seen")
LOGGER = logging.getLogger(__name__)


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


def _to_text(value: object, fallback: str | None) -> str:
    resolved = _with_fallback(value, fallback)
    if resolved is None:
        return ""
    if isinstance(resolved, str):
        return resolved
    return str(resolved)


def _sanitize_model_token(model_name: str) -> str:
    token = model_name.strip()
    if "/" in token:
        token = token.rsplit("/", 1)[-1]
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"[^0-9A-Za-z_.-]", "_", token)
    token = re.sub(r"_+", "_", token).strip("._-")
    return token or "model"


def _extract_summary_values(
    record: dict[str, object],
    checkpoint: CheckpointFile,
    dataset_fallback: str | None,
) -> tuple[tuple[str, str, str], str, dict[str, object]]:
    direction_key = _to_text(record.get("direction_key"), None)
    dataset_name = _to_text(record.get("dataset"), dataset_fallback)
    split_name = _to_text(record.get("split"), checkpoint.split)
    model_name = _to_text(record.get("model_name"), checkpoint.model_name)
    return (
        (direction_key, dataset_name, split_name),
        model_name,
        {
            "mean": record.get("mean"),
            "median": record.get("median"),
            "src_lang_seen": record.get("src_lang_seen"),
            "tgt_lang_seen": record.get("tgt_lang_seen"),
        },
    )


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
    rows_by_key: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    model_tokens: set[str] = set()
    token_to_model_name: dict[str, str] = {}
    duplicate_locations: set[tuple[str, str, str, str]] = set()

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

                base_key, model_name, model_values = _extract_summary_values(
                    record=record,
                    checkpoint=checkpoint,
                    dataset_fallback=dataset_fallback,
                )
                model_token = _sanitize_model_token(model_name)

                existing_model_name = token_to_model_name.get(model_token)
                if existing_model_name is None:
                    token_to_model_name[model_token] = model_name
                elif existing_model_name != model_name:
                    raise SystemExit(
                        "Model token collision: "
                        f"{model_name!r} and {existing_model_name!r} both map to "
                        f"{model_token!r}"
                    )

                model_tokens.add(model_token)
                model_rows = rows_by_key.setdefault(base_key, {})
                if model_token in model_rows:
                    direction_key, dataset_name, split_name = base_key
                    duplicate_locations.add(
                        (direction_key, dataset_name, split_name, model_name)
                    )
                    # Keep the first-seen value for duplicate records.
                    continue
                model_rows[model_token] = model_values

    ordered_model_tokens = sorted(model_tokens)
    csv_columns = list(BASE_COLUMNS)
    for model_token in ordered_model_tokens:
        for suffix in MODEL_COLUMN_SUFFIXES:
            csv_columns.append(f"{model_token}_{suffix}")

    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.writer(output_handle)
        writer.writerow(csv_columns)

        for direction_key, dataset_name, split_name in sorted(rows_by_key):
            row = [direction_key, dataset_name, split_name]
            model_rows = rows_by_key[(direction_key, dataset_name, split_name)]
            for model_token in ordered_model_tokens:
                values = model_rows.get(model_token)
                if values is None:
                    row.extend(["", "", "", ""])
                    continue
                row.extend(
                    [
                        values["mean"],
                        values["median"],
                        values["src_lang_seen"],
                        values["tgt_lang_seen"],
                    ]
                )
            writer.writerow(row)
            rows_written += 1

    if duplicate_locations:
        duplicate_values = [
            (
                f"direction_key={direction_key!r}, "
                f"dataset_name={dataset_name!r}, "
                f"split={split_name!r}, "
                f"model={model_name!r}"
            )
            for direction_key, dataset_name, split_name, model_name in sorted(
                duplicate_locations
            )
        ]
        LOGGER.warning("duplicate values at: %s", duplicate_values)

    return AggregateStats(
        checkpoint_files_found=len(checkpoint_files),
        rows_written=rows_written,
        invalid_json_lines_skipped=invalid_json_lines_skipped,
    )
