"""
cli.py
------
Argument parsing and subcommand dispatch.

Commands
--------
  pivot    — compute mean/median ensembles from normalized scores
  weighted — compute weighted-average ensembles from mean/median ensembles

Usage
-----
  python -m mean_median_ensemble pivot    [--src SRC] [--dst DST] [--tmp TMP] [--preview]
  python -m mean_median_ensemble weighted [--norm NORM] [--ens ENS] [--dst DST] [--tmp TMP] [--preview]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from .config import (
    DEFAULT_ENS_ROOT,
    DEFAULT_NORM_ROOT,
    DEFAULT_SCRATCH_TMP,
    DEFAULT_WAVG_ROOT,
    MODELS,
)
from .pivot import run_pivot
from .weighted import run_weighted


def _filter_models(
    exclude: List[str],
) -> List[Tuple[str, str]]:
    """Return MODELS with any matching entries removed.

    Each name in *exclude* is matched against both the hive partition name
    and the safe SQL column name (case-insensitive).
    """
    if not exclude:
        return list(MODELS)

    exclude_lower = {e.lower() for e in exclude}
    filtered = [
        (h, s)
        for h, s in MODELS
        if h.lower() not in exclude_lower and s.lower() not in exclude_lower
    ]

    removed = len(MODELS) - len(filtered)
    if removed == 0:
        known = [h for h, _ in MODELS]
        print(
            f"WARNING: --exclude matched 0 models. Known models: {known}",
            file=sys.stderr,
        )
    else:
        print(f"Excluding {removed} model(s): {sorted(exclude_lower)}")

    if not filtered:
        print("ERROR: all models excluded — nothing to compute.", file=sys.stderr)
        sys.exit(1)

    return filtered


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mean_median_ensemble",
        description="Compute mean/median and weighted-average QE ensembles over FLORES-200.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── pivot ──────────────────────────────────────────────────────────────
    p = sub.add_parser("pivot", help="Compute mean/median ensembles (Steps 1–3)")
    p.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_NORM_ROOT,
        metavar="PATH",
        help="Root of normalized_scores parquet tree (default: %(default)s)",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_ENS_ROOT,
        metavar="PATH",
        help="Output root for mean_median_ensembles (default: %(default)s)",
    )
    p.add_argument(
        "--tmp",
        type=Path,
        default=DEFAULT_SCRATCH_TMP,
        metavar="PATH",
        help="DuckDB scratch temp directory (default: %(default)s)",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Print 5 rows to stdout without writing any output",
    )
    p.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="MODEL",
        help="Model name(s) to exclude (hive or SQL name, case-insensitive)",
    )

    # ── weighted ───────────────────────────────────────────────────────────
    w = sub.add_parser(
        "weighted", help="Compute weighted-average ensembles (Steps 6–7)"
    )
    w.add_argument(
        "--norm",
        type=Path,
        default=DEFAULT_NORM_ROOT,
        metavar="PATH",
        help="Root of normalized_scores parquet (used to derive weights)",
    )
    w.add_argument(
        "--ens",
        type=Path,
        default=DEFAULT_ENS_ROOT,
        metavar="PATH",
        help="Root of mean_median_ensembles parquet (pivot input)",
    )
    w.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_WAVG_ROOT,
        metavar="PATH",
        help="Output root for weighted_avg_ensembles",
    )
    w.add_argument(
        "--tmp",
        type=Path,
        default=DEFAULT_SCRATCH_TMP,
        metavar="PATH",
        help="DuckDB scratch temp directory",
    )
    w.add_argument(
        "--preview",
        action="store_true",
        help="Print 5 rows to stdout without writing any output",
    )
    w.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="MODEL",
        help="Model name(s) to exclude (hive or SQL name, case-insensitive)",
    )

    return parser


def main() -> None:
    args = create_parser().parse_args()
    models = _filter_models(args.exclude)

    if args.command == "pivot":
        run_pivot(
            src_root=args.src,
            dst_root=args.dst,
            scratch_tmp=args.tmp,
            preview=args.preview,
            models=models,
        )
    elif args.command == "weighted":
        run_weighted(
            norm_root=args.norm,
            ens_root=args.ens,
            dst_root=args.dst,
            scratch_tmp=args.tmp,
            preview=args.preview,
            models=models,
        )
