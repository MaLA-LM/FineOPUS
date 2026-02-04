from __future__ import annotations

from pathlib import Path

from src.scoring.output_path import build_output_path


def resolve_output_path(
    args,
    model_tag: str,
    src_lang: str,
    tgt_lang: str,
    split: str,
) -> Path:
    if args.output:
        return Path(args.output)
    if not args.output_base:
        raise SystemExit("Provide --output or --output-base.")
    return build_output_path(
        args.output_base, args.dataset, model_tag, split, src_lang, tgt_lang
    )


def validate_args(args) -> None:
    if hasattr(args, "worker_max_files"):
        if args.worker_max_files is not None and args.worker_max_files < 0:
            raise SystemExit("--worker-max-files must be >= 0.")
    if args.manifest and args.discover_all:
        raise SystemExit("Use either --manifest or --discover-all, not both.")
    if args.worker:
        if not args.manifest:
            raise SystemExit("--worker requires --manifest.")
        if args.output:
            raise SystemExit("--output is only supported for single-direction runs.")
        if not args.output_base:
            raise SystemExit("--output-base is required for worker mode.")
        return
    if args.manifest or args.discover_all:
        if args.output:
            raise SystemExit("--output is only supported for single-direction runs.")
        if not args.output_base:
            raise SystemExit(
                "--output-base is required for manifest or discover-all runs."
            )
        return
    if not args.src_lang or not args.tgt_lang:
        raise SystemExit("Provide --src-lang and --tgt-lang for single runs.")
    if not args.output and not args.output_base:
        raise SystemExit("Provide --output or --output-base.")
