from __future__ import annotations

import csv
from pathlib import Path

Example = dict[str, str]

DEFAULT_FLORES_ROOT = "/scratch/project_2008161/downstream_benchmarks/flores200"


def load_flores200_parallel(
    src_lang: str,
    tgt_lang: str,
    *, # keyword-only arguments
    split: str = "devtest",
    root: str | Path = DEFAULT_FLORES_ROOT,
    include_metadata: bool = False,
) -> list[Example]:
    split = split.strip()
    if split not in {"dev", "devtest"}:
        msg = "Unsupported split: {!r} (expected 'dev' or 'devtest')".format(split)
        raise ValueError(msg)

    root_path = Path(root)
    src_path = root_path / split / f"{src_lang}.{split}"
    tgt_path = root_path / split / f"{tgt_lang}.{split}"
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Target file not found: {tgt_path}")

    src_lines = _read_lines(src_path)
    tgt_lines = _read_lines(tgt_path)
    if len(src_lines) != len(tgt_lines):
        msg = (
            f"Line count mismatch: {src_path} has {len(src_lines)} lines; "
            f"{tgt_path} has {len(tgt_lines)} lines"
        )
        raise ValueError(msg)

    if not include_metadata:
        return [{"src": src, "tgt": tgt} for src, tgt in zip(src_lines, tgt_lines)]

    metadata_path = root_path / f"metadata_{split}.tsv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata_rows = _read_metadata(metadata_path)
    if len(metadata_rows) != len(src_lines):
        msg = (
            f"Metadata row count mismatch: {metadata_path} has {len(metadata_rows)} rows; "
            f"expected {len(src_lines)}"
        )
        raise ValueError(msg)

    records: list[Example] = []
    for idx, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
        record = {"src": src, "tgt": tgt}
        record.update(metadata_rows[idx])
        records.append(record)

    return records


def limit_rows(examples: list[Example], max_rows: int | None) -> list[Example]:
    if max_rows is None:
        return examples
    return examples[: min(max_rows, len(examples))]


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)
