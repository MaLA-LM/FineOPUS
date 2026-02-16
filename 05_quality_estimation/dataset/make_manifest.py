from __future__ import annotations

import argparse
from pathlib import Path

from dataset.mediator import DEFAULT_DATASET_ID, get_dataset
from dataset.manifest import write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a TSV manifest for dataset directions."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_ID,
        help="Dataset id to use.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory of the dataset files (defaults to dataset root).",
    )
    parser.add_argument(
        "--split",
        default="devtest",
        help="Split to include in the manifest (or 'all').",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Number of deterministic shards used to precompute shard_id.",
    )
    parser.add_argument("--out", required=True, help="Output TSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be > 0.")

    dataset = get_dataset(args.dataset)
    root = args.root or dataset.default_root

    split = None if args.split == "all" else args.split
    if split is not None and split not in dataset.split_values:
        supported = ", ".join(dataset.split_values)
        raise SystemExit(
            f"Unsupported split '{split}' for dataset '{dataset.id}'. "
            f"Supported: {supported}."
        )

    directions = dataset.discover_directions(root, split=split)
    rows = sorted(directions, key=lambda item: (item[2], item[0], item[1]))
    write_manifest(
        ((src, tgt, split) for src, tgt, split, _path in rows),
        args.out,
        num_shards=args.num_shards,
    )
    print(f"Wrote manifest: {Path(args.out)}")


if __name__ == "__main__":
    main()
