from __future__ import annotations


def validate_args(args) -> None:
    if not args.output_base:
        raise SystemExit("--output-base is required.")
    max_rows = getattr(args, "max_rows", None)
    if max_rows is not None and max_rows < 0:
        raise SystemExit("--max-rows must be >= 0.")
