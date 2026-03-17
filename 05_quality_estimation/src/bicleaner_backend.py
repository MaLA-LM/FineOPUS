from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from dataset.mediator import Example
from models.language_data.remedy import REMEDY_ISO_MAP
from utils.logger import logger

ISO639_1_BY_NAME = {name: code for code, name in REMEDY_ISO_MAP.items()}

DEFAULT_BICLEANER_COMMAND = "bicleaner-ai-classify"


def iso639_1_from_dataset(
    code: str, dataset, iso_map: dict[str, str] = ISO639_1_BY_NAME
) -> str | None:
    name = dataset.language_codes.get(code)
    if not name:
        return None
    return iso_map.get(name)


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


def _build_bicleaner_args(
    input_path: Path, output_path: Path, model_id: str
) -> list[str]:
    return [
        DEFAULT_BICLEANER_COMMAND,
        "--scol",
        "1",
        "--tcol",
        "2",
        "--disable_hardrules",
        str(input_path),
        str(output_path),
        model_id,
    ]


def run_bicleaner(
    input_path: Path,
    output_path: Path,
    model_id: str,
    src_iso: str | None,
    tgt_iso: str | None,
) -> None:
    base_args = _build_bicleaner_args(input_path, output_path, model_id)

    if src_iso and tgt_iso:
        args = base_args[:4] + ["-s", src_iso, "-t", tgt_iso] + base_args[4:]
        logger.warning(
            "[bicleaner-ai] Running with model '%s' (src_iso=%s, tgt_iso=%s)...",
            model_id,
            src_iso,
            tgt_iso,
        )
        try:
            subprocess.run(args, check=True)
            return
        except subprocess.CalledProcessError:
            logger.warning(
                "[bicleaner-ai] Failed with -s %s -t %s; retrying without language flags.",
                src_iso,
                tgt_iso,
            )

    logger.warning(
        "[bicleaner-ai] Running with model '%s' without -s/-t flags...",
        model_id,
    )
    subprocess.run(base_args, check=True)
