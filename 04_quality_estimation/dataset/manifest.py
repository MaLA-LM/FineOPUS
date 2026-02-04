from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    src_lang: str
    tgt_lang: str
    split: str
    lock_id: str


def _sanitize_lock_id(value: str) -> str:
    safe = value.strip()
    safe = safe.replace("/", "_").replace("\\", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", safe)
    safe = safe.strip("_")
    return safe or "direction"


def make_lock_id(src_lang: str, tgt_lang: str, split: str) -> str:
    raw = f"{split}__{src_lang}__{tgt_lang}"
    return _sanitize_lock_id(raw)


def read_manifest_entries(
    manifest_path: str | Path,
) -> list[ManifestEntry]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Manifest missing header: {path}")
        required = {"src_lang", "tgt_lang", "split"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing columns {sorted(missing)}: {path}")
        directions: list[ManifestEntry] = []
        for idx, row in enumerate(reader, start=2):
            src = (row.get("src_lang") or "").strip()
            tgt = (row.get("tgt_lang") or "").strip()
            split = (row.get("split") or "").strip()
            if not src or not tgt or not split:
                raise ValueError(f"Manifest row {idx} missing values: {path}")
            lock_id = (row.get("lock_id") or "").strip()
            if not lock_id:
                lock_id = make_lock_id(src, tgt, split)
            else:
                lock_id = _sanitize_lock_id(lock_id)
            directions.append(
                ManifestEntry(
                    src_lang=src,
                    tgt_lang=tgt,
                    split=split,
                    lock_id=lock_id,
                )
            )
    return directions


def read_manifest(manifest_path: str | Path) -> list[tuple[str, str, str]]:
    return [
        (entry.src_lang, entry.tgt_lang, entry.split)
        for entry in read_manifest_entries(manifest_path)
    ]


def write_manifest(
    directions: Iterable[tuple[str, str, str] | ManifestEntry],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["src_lang", "tgt_lang", "split", "lock_id"])
        for item in directions:
            if isinstance(item, ManifestEntry):
                src_lang = item.src_lang
                tgt_lang = item.tgt_lang
                split = item.split
                lock_id = item.lock_id
            else:
                src_lang, tgt_lang, split = item
                lock_id = make_lock_id(src_lang, tgt_lang, split)
            writer.writerow([src_lang, tgt_lang, split, lock_id])
