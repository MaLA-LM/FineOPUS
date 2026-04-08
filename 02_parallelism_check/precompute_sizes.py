#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan FineOPUS-Filtered-Stage2 to get total parquet bytes per language pair,
then rewrite model_to_language_pairs.json so each entry is
  [pair_name, total_bytes]
instead of just the pair name string.

Run once before submitting jobs:
  python precompute_sizes.py [--input_dir DIR] [--json PATH]
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage2",
    )
    parser.add_argument(
        "--json",
        default="/scratch/project_462000941/members/zihao/OPUS2410/02_parallelism_check/model_to_language_pairs.json",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    with open(args.json) as f:
        data = json.load(f)

    updated: dict = {}
    for model, pairs in data.items():
        print(f"\nModel: {model}  ({len(pairs)} pairs)")
        pair_sizes = []
        missing = 0
        for entry in pairs:
            # Support both old format (str) and already-updated format ([str, int])
            pair = entry if isinstance(entry, str) else entry[0]
            pair_dir = input_dir / pair
            shards = list(pair_dir.glob("*.parquet"))
            if not shards:
                print(f"  WARNING: no parquet files found for {pair}")
                total_bytes = 0
                missing += 1
            else:
                total_bytes = sum(s.stat().st_size for s in shards)
            pair_sizes.append([pair, total_bytes])

        total_gb = sum(b for _, b in pair_sizes) / 1e9
        print(f"  Total data: {total_gb:.2f} GB  |  Missing dirs: {missing}")
        updated[model] = pair_sizes

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated: {args.json}")


if __name__ == "__main__":
    main()
