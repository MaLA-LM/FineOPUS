from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dataset.manifest import ManifestEntry
from dataset.mediator import Example, get_dataset
from models.language_support import CometLanguages
from src.scoring.args import add_common_scoring_args
from src.scoring.cli import resolve_output_path, validate_args
from src.scoring.frames import build_frames
from src.scoring.output_path import sanitize_model_tag
from src.scoring.runner import collect_directions, run_scoring

BICLEANER_MODELS = {
    "en-xx": "bitextor/bicleaner-ai-full-en-xx",
    "es-xx": "bitextor/bicleaner-ai-full-es-xx",
    "de-xx": "bitextor/bicleaner-ai-full-de-xx",
}

ALIASES = {
    "auto": "auto",
    "bicleaner-ai": "auto",
    "bitextor/bicleaner-ai-full-en-xx": "en-xx",
    "bitextor/bicleaner-ai-full-es-xx": "es-xx",
    "bitextor/bicleaner-ai-full-de-xx": "de-xx",
}

ISO639_1_BY_BASE_NAME = {
    "english": "en",
    "spanish": "es",
    "german": "de",
}

# bicleaner-ai settings
BICLEANER_COMMAND = "bicleaner-ai-classify"
BICLEANER_PROCESSES: int | None = None
BICLEANER_DISABLE_HARDRULES = True


def _log(message: str) -> None:
    print(f"[bicleaner-ai] {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a dataset split with bicleaner-ai via CLI."
    )
    add_common_scoring_args(
        parser,
        batch_size_default=None,
        batch_size_help="Ignored (kept for CLI compatibility).",
        gpus_default=None,
        gpus_help="Ignored (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help=(
            "Model selector: auto, en-xx, es-xx, de-xx, or full bitextor repo. "
            f"Supported: auto, {', '.join(sorted(BICLEANER_MODELS.keys()))}."
        ),
    )
    parser.add_argument(
        "--bicleaner-command",
        default=BICLEANER_COMMAND,
        help="Path to bicleaner-ai-classify CLI.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=BICLEANER_PROCESSES,
        help="Number of bicleaner-ai worker processes (optional).",
    )
    parser.add_argument(
        "--disable-hardrules",
        "--disable_hardrules",
        dest="disable_hardrules",
        action="store_true",
        default=BICLEANER_DISABLE_HARDRULES,
        help="Disable bicleaner-ai hardrules.",
    )
    parser.add_argument(
        "--enable-hardrules",
        dest="disable_hardrules",
        action="store_false",
        help="Enable bicleaner-ai hardrules.",
    )
    return parser.parse_args()


def resolve_model(name: str) -> tuple[str, str]:
    key = ALIASES.get(name, name)
    if key == "auto":
        return "auto", "bicleaner-ai"
    if key in BICLEANER_MODELS:
        return key, key
    supported = ", ".join(sorted(BICLEANER_MODELS.keys()))
    raise ValueError(
        f"Unknown bicleaner-ai model '{name}'. Supported: auto, {supported}."
    )


def iso639_1_from_dataset(code: str, dataset, iso_map: dict[str, str]) -> str | None:
    if not code:
        _log("Missing dataset language code.")
        return None
    if dataset.iso639_1_from_code is None:
        _log(f"Dataset '{dataset.id}' does not provide ISO mapping.")
        return None
    alpha2 = dataset.iso639_1_from_code(code, iso_map)
    if not alpha2:
        _log(f"No ISO 639-1 mapping for {dataset.id} '{code}'.")
        return None
    return alpha2


def _model_for_iso(alpha2: str) -> str:
    if alpha2 == "es":
        return BICLEANER_MODELS["es-xx"]
    if alpha2 == "de":
        return BICLEANER_MODELS["de-xx"]
    return BICLEANER_MODELS["en-xx"]


def select_model_id(
    model_selector: str, src_iso: str | None, tgt_iso: str | None
) -> str:
    if model_selector == "auto":
        if src_iso in {"es", "en", "de"}:
            return _model_for_iso(src_iso)
        if tgt_iso in {"es", "en", "de"}:
            return _model_for_iso(tgt_iso)
        return BICLEANER_MODELS["en-xx"]
    return BICLEANER_MODELS[model_selector]


def write_tsv(examples: list[Example], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for ex in examples:
            writer.writerow([ex["src"], ex["tgt"]])


def read_scores(path: Path) -> list[float]:
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            try:
                scores.append(float(row[-1]))
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid score value in {path}: {row[-1]!r}"
                ) from exc
    return scores


def run_bicleaner(
    input_path: Path,
    output_path: Path,
    model_id: str,
    src_iso: str | None,
    tgt_iso: str | None,
    command: str,
    processes: int | None,
    disable_hardrules: bool,
) -> None:
    args = [
        command,
        "--scol",
        "1",
        "--tcol",
        "2",
    ]
    if src_iso and tgt_iso:
        args.extend(["-s", src_iso, "-t", tgt_iso])
    else:
        _log(
            "Missing ISO 639-1 mapping for one or both languages; "
            "running bicleaner-ai without -s/-t."
        )
    if processes is not None:
        if processes <= 0:
            raise ValueError("processes must be >= 1 when set.")
        args.extend(["--processes", str(processes)])
    if disable_hardrules:
        args.append("--disable_hardrules")
    args.extend([str(input_path), str(output_path), model_id])
    _log(
        f"Running bicleaner-ai with model '{model_id}' (src_iso={src_iso}, tgt_iso={tgt_iso})..."
    )
    subprocess.run(args, check=True)


def summarize_scores(scores: list[float]) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    return float(statistics.fmean(scores)), float(statistics.median(scores))


def score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    model_selector: str,
    language_support: CometLanguages,
    dataset,
):
    examples = dataset.load_parallel(
        entry.src_lang, entry.tgt_lang, split=entry.split, root=args.root
    )
    examples = dataset.limit_rows(examples, args.max_rows)

    src_iso = iso639_1_from_dataset(entry.src_lang, dataset, ISO639_1_BY_BASE_NAME)
    tgt_iso = iso639_1_from_dataset(entry.tgt_lang, dataset, ISO639_1_BY_BASE_NAME)
    model_id = select_model_id(model_selector, src_iso, tgt_iso)

    with TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_path = tmp_root / "input.tsv"
        output_path = tmp_root / "output.tsv"
        write_tsv(examples, input_path)
        run_bicleaner(
            input_path,
            output_path,
            model_id,
            src_iso,
            tgt_iso,
            args.bicleaner_command,
            args.processes,
            args.disable_hardrules,
        )
        scores = read_scores(output_path)

    if len(scores) != len(examples):
        raise RuntimeError(
            f"bicleaner-ai returned {len(scores)} scores for {len(examples)} rows."
        )

    src_lang_seen = language_support.support_status(entry.src_lang)
    tgt_lang_seen = language_support.support_status(entry.tgt_lang)
    mean_score, median_score = summarize_scores(scores)

    return build_frames(
        model_id,
        dataset.id,
        entry.split,
        entry.src_lang,
        entry.tgt_lang,
        scores,
        examples,
        src_lang_seen=src_lang_seen,
        tgt_lang_seen=tgt_lang_seen,
        mean=mean_score,
        median=median_score,
    )


def main() -> None:
    args = parse_args()
    try:
        model_selector, model_key = resolve_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = get_dataset(args.dataset)
    if args.root is None:
        args.root = dataset.default_root
    if args.split not in dataset.split_values:
        supported = ", ".join(dataset.split_values)
        raise SystemExit(
            f"Unsupported split '{args.split}' for dataset '{dataset.id}'. "
            f"Supported: {supported}."
        )
    validate_args(args)

    directions = collect_directions(args, dataset)
    if not directions:
        print("No directions found.")
        return

    model_tag = sanitize_model_tag(model_key)
    language_support = CometLanguages(dataset=dataset)

    run_scoring(
        args,
        dataset,
        directions,
        model_tag,
        resolve_output_path,
        lambda entry: score_entry(
            entry, args, model_selector, language_support, dataset
        ),
    )


if __name__ == "__main__":
    main()
