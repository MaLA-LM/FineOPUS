from __future__ import annotations

from pathlib import Path

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
