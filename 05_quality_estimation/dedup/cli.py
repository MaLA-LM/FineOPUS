from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate checkpoint and part files in dataset output directories.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan for duplicates and generate a plan + report (no modifications).",
    )
    scan_parser.add_argument(
        "--dataset-path",
        required=True,
        help=(
            "Root path of the dataset output directory "
            "(e.g. /scratch/.../dataset=flores200)."
        ),
    )
    scan_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory for the plan JSON and report markdown. "
            "Defaults to --dataset-path."
        ),
    )

    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply deletions described in a plan file produced by 'scan'.",
    )
    apply_parser.add_argument(
        "--plan",
        required=True,
        help="Path to the dedup_plan.json file produced by 'scan'.",
    )

    args = parser.parse_args()

    if args.command == "scan":
        args.dataset_path = Path(args.dataset_path)
        if not args.dataset_path.exists():
            raise SystemExit(f"Dataset path does not exist: {args.dataset_path}")
        if not args.dataset_path.is_dir():
            raise SystemExit(f"Dataset path is not a directory: {args.dataset_path}")
        args.output = args.dataset_path if args.output is None else Path(args.output)

    elif args.command == "apply":
        args.plan = Path(args.plan)
        if not args.plan.exists():
            raise SystemExit(f"Plan file does not exist: {args.plan}")
        if not args.plan.is_file():
            raise SystemExit(f"Plan path is not a file: {args.plan}")

    return args
