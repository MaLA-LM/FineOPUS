from __future__ import annotations

import argparse

from dataset.mediator import DatasetAdapter
from execution.flores_array.manifest import ManifestEntry, read_manifest_entries

__all__ = ["collect_directions", "validate_flores_args"]


def validate_flores_args(args: argparse.Namespace, dataset: DatasetAdapter) -> None:
    manifest = getattr(args, "manifest", None)
    shard_id = getattr(args, "shard_id", None)
    num_shards = getattr(args, "num_shards", None)
    max_directions_per_part = getattr(args, "max_directions_per_part", None)
    target_part_bytes = getattr(args, "target_part_bytes", None)

    if dataset.id == "opus":
        raise SystemExit(
            "Execution 'flores_array' is incompatible with dataset 'opus'. "
            "Use --execution opus_queue."
        )
    if not manifest:
        raise SystemExit("--manifest is required for execution='flores_array'.")
    if shard_id is not None and shard_id < 0:
        raise SystemExit("--shard-id must be >= 0.")
    if num_shards is not None and num_shards <= 0:
        raise SystemExit("--num-shards must be > 0.")
    if max_directions_per_part is None or max_directions_per_part <= 0:
        raise SystemExit("--max-directions-per-part must be > 0.")
    if target_part_bytes is None or target_part_bytes <= 0:
        raise SystemExit("--target-part-bytes must be > 0.")


def collect_directions(args, dataset: DatasetAdapter) -> list[ManifestEntry]:
    directions = read_manifest_entries(args.manifest)
    valid_splits = set(dataset.split_values)
    for entry in directions:
        if entry.split not in valid_splits:
            supported = ", ".join(dataset.split_values)
            raise SystemExit(
                f"Unsupported split '{entry.split}' for dataset '{dataset.id}'. "
                f"Supported: {supported}."
            )
    return directions
