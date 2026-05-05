import argparse
import json
import sys
from pathlib import Path

from stand_alone_modules.opus_trace_summary.report import print_human, write_json
from stand_alone_modules.opus_trace_summary.summary import summarize


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize OPUS worker trace progress for one static-manifest model."
        )
    )
    parser.add_argument(
        "model_positional",
        nargs="?",
        help="Model key, for example metricx24.",
    )
    parser.add_argument(
        "--model",
        dest="model_option",
        help="Model key. Overrides the positional model when both are passed.",
    )
    parser.add_argument(
        "--trace-root",
        default=".",
        help=(
            "Trace root. This can be the directory containing worker folders, "
            "or the parent directory when --build-tag is passed. Default: ."
        ),
    )
    parser.add_argument(
        "--build-tag",
        default=None,
        help="Optional build tag subdirectory under --trace-root.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional manifest.jsonl path. When passed, remaining shards are "
            "computed against the full manifest for the model."
        ),
    )
    parser.add_argument(
        "--manifest-root",
        default=None,
        help=(
            "Optional root containing <build-tag>/manifest.jsonl. Used when "
            "--manifest is not passed."
        ),
    )
    parser.add_argument(
        "--manifest-summary",
        default=None,
        help=(
            "Optional manifest.summary.json path. Used to compare scanned "
            "trace folders against the manifest totals."
        ),
    )
    parser.add_argument(
        "--workers-limit",
        type=int,
        default=20,
        help="Number of worker rows to print in human output. Use 0 to hide.",
    )
    parser.add_argument(
        "--directions-limit",
        type=int,
        default=20,
        help="Number of direction rows to print in human output. Use 0 to hide.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full summary as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the full JSON summary.",
    )
    args = parser.parse_args(argv)
    args.model = args.model_option or args.model_positional
    if not args.model:
        parser.error("provide a model name either positionally or with --model")
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = resolve_manifest_path(args)
    summary_path = resolve_summary_path(args, manifest_path)
    result = summarize(
        args.model,
        args.trace_root,
        build_tag=args.build_tag,
        manifest_path=manifest_path,
        manifest_summary_path=summary_path,
    )
    if args.output_json:
        write_json(Path(args.output_json).expanduser(), result)
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print_human(result, args.workers_limit, args.directions_limit)
    return 0


def resolve_manifest_path(args):
    if args.manifest:
        return Path(args.manifest).expanduser()
    if args.manifest_root and args.build_tag:
        return Path(args.manifest_root).expanduser() / args.build_tag / "manifest.jsonl"
    return None


def resolve_summary_path(args, manifest_path):
    if args.manifest_summary:
        return Path(args.manifest_summary).expanduser()
    candidates = []
    if manifest_path is not None:
        candidates.append(manifest_path.parent / "manifest.summary.json")
    if args.manifest_root and args.build_tag:
        candidates.append(
            Path(args.manifest_root).expanduser()
            / args.build_tag
            / "manifest.summary.json"
        )
    if args.build_tag:
        candidates.append(Path(args.build_tag).expanduser() / "manifest.summary.json")
        candidates.append(
            Path(args.trace_root).expanduser()
            / args.build_tag
            / "manifest.summary.json"
        )
    for path in candidates:
        if path.is_file():
            return path
    return None
