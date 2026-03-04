from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate checkpoint.jsonl summary rows into a single CSV for visual "
            "analysis."
        )
    )
    parser.add_argument(
        "results_path",
        help=(
            "Dataset output root path containing model=*/split=*/shard=* folders."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: <results_path>/checkpoint_summary.csv",
    )
    args = parser.parse_args()

    results_path = Path(args.results_path)
    if not results_path.exists() or not results_path.is_dir():
        raise SystemExit(f"results_path must be an existing directory: {results_path}")

    output_path = (
        results_path / "checkpoint_summary.csv"
        if args.output is None
        else Path(args.output)
    )

    args.results_path = results_path
    args.output = output_path
    return args
