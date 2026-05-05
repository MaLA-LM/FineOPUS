from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from time import sleep

from dataset.mediator import Example
from models.language_data.remedy import REMEDY_ISO_MAP
from models.model_registry import resolve_model_spec
from utils.logger import logger

__all__ = [
    "DEFAULT_BICLEANER_COMMAND",
    "BICLEANER_MODEL_IDS",
    "ISO639_1_BY_NAME",
    "iso639_1_from_dataset",
    "select_model_id",
    "write_tsv",
    "read_scores",
    "run_bicleaner",
]

ISO639_1_BY_NAME = {name: code for code, name in REMEDY_ISO_MAP.items()}

DEFAULT_BICLEANER_COMMAND = "bicleaner-ai-classify"

BICLEANER_MODEL_IDS = {
    key: resolve_model_spec(key, "bicleaner")[0].model_id
    for key in ("en-xx", "es-xx", "de-xx")
}

_DATASET_ISO_CACHE: dict[int, dict[str, str]] = {}
_TSV_FIELD_TRANSLATION = str.maketrans({
    "\t": " ",
    "\r": " ",
    "\n": " ",
    "\x00": " ",
})


def _build_iso_lookup(dataset) -> dict[str, str]:
    mapping = dataset.langcode_to_name(set(REMEDY_ISO_MAP.values()))
    result: dict[str, str] = {}
    for lang_code, info in mapping.items():
        if not info or len(info) < 2:
            continue
        matched = info[1]
        if not matched:
            continue
        iso_code = ISO639_1_BY_NAME.get(matched)
        if iso_code:
            result[lang_code] = iso_code
    return result


def iso639_1_from_dataset(code: str, dataset) -> str | None:
    if not code:
        return None
    key = id(dataset)
    lookup = _DATASET_ISO_CACHE.get(key)
    if lookup is None:
        lookup = _build_iso_lookup(dataset)
        _DATASET_ISO_CACHE[key] = lookup
    return lookup.get(code)


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


def _tsv_field(value: object) -> str:
    return str(value).translate(_TSV_FIELD_TRANSLATION)


def write_tsv(examples: list[Example], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(
            handle,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
            doublequote=True,
            escapechar="\\",
            lineterminator="\n",
        )
        for ex in examples:
            writer.writerow([_tsv_field(ex["src"]), _tsv_field(ex["tgt"])])


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
    input_path: Path,
    output_path: Path,
    model_id: str,
    *,
    src_iso: str | None = None,
    tgt_iso: str | None = None,
) -> list[str]:
    args = [
        DEFAULT_BICLEANER_COMMAND,
        "--scol",
        "1",
        "--tcol",
        "2",
    ]
    if src_iso:
        args.extend(["-s", src_iso])
    if tgt_iso:
        args.extend(["-t", tgt_iso])
    args.extend(
        [
            "--disable_hardrules",
            str(input_path),
            str(output_path),
            model_id,
        ]
    )
    return args


def _language_flag_mode(src_iso: str | None, tgt_iso: str | None) -> str:
    if src_iso and tgt_iso:
        return "both"
    if src_iso:
        return "source-only"
    if tgt_iso:
        return "target-only"
    return "none"


def run_bicleaner(
    input_path: Path,
    output_path: Path,
    model_id: str,
    src_iso: str | None,
    tgt_iso: str | None,
) -> None:
    base_args = _build_bicleaner_args(input_path, output_path, model_id)
    args = _build_bicleaner_args(
        input_path,
        output_path,
        model_id,
        src_iso=src_iso,
        tgt_iso=tgt_iso,
    )
    mode = _language_flag_mode(src_iso, tgt_iso)

    sleep(5)

    if mode != "none":
        logger.warning(
            "[bicleaner-ai] Running with model '%s' using %s language flags (src_iso=%s, tgt_iso=%s)...",
            model_id,
            mode,
            src_iso,
            tgt_iso,
        )
        try:
            subprocess.run(args, check=True)
            return
        except subprocess.CalledProcessError:
            logger.warning(
                "[bicleaner-ai] Failed with %s language flags; retrying without language flags.",
                mode,
            )

    logger.warning(
        "[bicleaner-ai] Running with model '%s' without language flags...",
        model_id,
    )
    subprocess.run(base_args, check=True)
