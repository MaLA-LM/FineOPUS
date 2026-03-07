from __future__ import annotations

import argparse
from pathlib import Path

_DEFAULT_MODEL_PATH = (
    "/scratch/project_462001050/QE_flores200_scores/"
    "dataset=flores200/model=m-prometheus-7b"
)


def _add_model_path_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-path",
        default=_DEFAULT_MODEL_PATH,
        help=(
            "Root path of the model output directory "
            "(e.g. /scratch/.../model=m-prometheus-7b)."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch and manage checkpoint / part files in dataset output directories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- patch --- #
    patch_parser = subparsers.add_parser(
        "patch",
        help="Rewrite checkpoint and part files with corrected scores and seen flags.",
    )
    _add_model_path_arg(patch_parser)

    # --- replace --- #
    replace_parser = subparsers.add_parser(
        "replace",
        help="Swap original files with their *-patched.jsonl counterparts.",
    )
    _add_model_path_arg(replace_parser)

    args = parser.parse_args()
    args.model_path = Path(args.model_path)
    if not args.model_path.is_dir():
        raise SystemExit(
            f"Model path does not exist or is not a directory: {args.model_path}"
        )
    return args
