from __future__ import annotations

import argparse

from execution.opus_queue.tools.manifest_probe.runner import run

__all__ = ["main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an OPUS QE manifest before launching workers."
    )
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--build-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--array-spec", required=True)
    parser.add_argument("--slots-per-task", type=int, required=True)
    parser.add_argument("--trace-root", required=True)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))
