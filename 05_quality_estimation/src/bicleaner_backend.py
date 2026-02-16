from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from dataset.mediator import Example

ISO639_1_BY_BASE_NAME = {
    "english": "en",
    "spanish": "es",
    "german": "de",
}

DEFAULT_BICLEANER_COMMAND = "bicleaner-ai-classify"
DEFAULT_BICLEANER_PROCESSES: int | None = None
DEFAULT_DISABLE_HARDRULES = True


def _log(message: str) -> None:
    print(f"[bicleaner-ai] {message}", file=sys.stderr)


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


def _model_for_iso(alpha2: str, model_ids: dict[str, str]) -> str:
    if alpha2 == "es":
        return model_ids["es-xx"]
    if alpha2 == "de":
        return model_ids["de-xx"]
    return model_ids["en-xx"]


def select_model_id(
    model_key: str,
    src_iso: str | None,
    tgt_iso: str | None,
    model_ids: dict[str, str],
) -> str:
    if model_key == "auto":
        if src_iso in {"es", "en", "de"}:
            return _model_for_iso(src_iso, model_ids)
        if tgt_iso in {"es", "en", "de"}:
            return _model_for_iso(tgt_iso, model_ids)
        return model_ids["en-xx"]
    return model_ids[model_key]


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
    args = [command, "--scol", "1", "--tcol", "2"]
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
        f"Running bicleaner-ai with model '{model_id}' "
        f"(src_iso={src_iso}, tgt_iso={tgt_iso})..."
    )
    subprocess.run(args, check=True)
