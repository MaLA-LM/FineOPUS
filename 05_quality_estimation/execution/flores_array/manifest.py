from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from execution.flores_array.hashing import compute_shard_id
from utils.hashing import direction_key


@dataclass(frozen=True)
class ManifestEntry:
    src_lang: str
    tgt_lang: str
    split: str
    shard_id: int | None = None


def read_manifest_entries(manifest_path: str | Path) -> list[ManifestEntry]:
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

            shard_id: int | None = None
            shard_raw = row.get("shard_id")
            if shard_raw is not None and str(shard_raw).strip():
                try:
                    shard_id = int(str(shard_raw).strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Manifest row {idx} has invalid shard_id={shard_raw!r}: {path}"
                    ) from exc

            directions.append(
                ManifestEntry(
                    src_lang=src, tgt_lang=tgt, split=split, shard_id=shard_id
                )
            )
    return directions


def read_manifest(manifest_path: str | Path) -> list[tuple[str, str, str]]:
    return [
        (entry.src_lang, entry.tgt_lang, entry.split)
        for entry in read_manifest_entries(manifest_path)
    ]


def write_manifest(
    directions: Iterable[tuple[str, str, str]],
    output_path: str | Path,
    *,
    num_shards: int,
) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["src_lang", "tgt_lang", "split", "shard_id"])
        for item in directions:
            src_lang, tgt_lang, split = item
            shard_id = compute_shard_id(direction_key(src_lang, tgt_lang), num_shards)
            if shard_id < 0 or shard_id >= num_shards:
                raise ValueError(
                    f"shard_id out of range for row {src_lang}->{tgt_lang} split={split}: "
                    f"{shard_id} not in [0, {num_shards - 1}]"
                )
            writer.writerow([src_lang, tgt_lang, split, shard_id])
