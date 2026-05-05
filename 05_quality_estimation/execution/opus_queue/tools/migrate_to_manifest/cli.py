from __future__ import annotations

import argparse

from execution.opus_queue.tools.migrate_to_manifest.runner import run

__all__ = ["main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pre-assigned OPUS QE worker manifest from jobs.db."
    )
    parser.add_argument("--db", required=True, help="Existing jobs.db path (read-only).")
    parser.add_argument(
        "--manifest-root",
        required=True,
        help="Output root for manifests/<build_tag>.",
    )
    parser.add_argument(
        "--build-tag",
        default=None,
        help="Manifest build tag; defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--walltime-seconds",
        type=int,
        default=144_000,
        help="Worker walltime budget in seconds (default: 40h).",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=0.85,
        help="Fraction of walltime available for scoring (default: 0.85).",
    )
    parser.add_argument(
        "--slots",
        action="append",
        default=[],
        help=(
            "Repeatable model:slots or model:slots:slots_per_array_task. "
            "Example: --slots metricx24:640:8 for standard-g."
        ),
    )
    parser.add_argument(
        "--slots-per-task",
        action="append",
        default=[],
        help=(
            "Repeatable model:int override for slots per array task. "
            "Equivalent to the third field in --slots."
        ),
    )
    parser.add_argument(
        "--include-status",
        default="pending,failed",
        help="Comma-separated job statuses to include (default: pending,failed).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary JSON without writing manifest files.",
    )
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))
