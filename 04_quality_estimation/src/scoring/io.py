from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

ROW_TYPE_SUMMARY = "summary"
ROW_TYPE_DETAIL = "detail"

OUTPUT_COLUMNS = [
    "row_type",
    "model_name",
    "dataset",
    "split",
    "src_lang",
    "tgt_lang",
    "src_lang_seen",
    "tgt_lang_seen",
    "mean",
    "median",
    "score",
    "src_txt",
    "tgt_txt",
]

REQUIRED_COLUMNS = set(OUTPUT_COLUMNS)


def write_parquet_atomic(frame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(output_path) + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)


def is_complete_parquet(
    path: Path,
    expected_detail_count: int | None,
    required_columns: set[str] | None = None,
) -> bool:
    if not path.exists():
        return False
    required = required_columns or REQUIRED_COLUMNS
    try:

        parquet_file = pq.ParquetFile(path)
        schema_columns = set(parquet_file.schema.names)
        if not required.issubset(schema_columns):
            return False
        row_count = parquet_file.metadata.num_rows
        if row_count is None or row_count <= 1:
            return False
        table = parquet_file.read(columns=["row_type"])
        row_types = table["row_type"].to_pylist()
        summary_count = sum(1 for value in row_types if value == ROW_TYPE_SUMMARY)
        detail_count = sum(1 for value in row_types if value == ROW_TYPE_DETAIL)
        if summary_count < 1 or detail_count < 1:
            return False
        if expected_detail_count is not None and detail_count != expected_detail_count:
            return False
        return True
    except Exception:
        return False
