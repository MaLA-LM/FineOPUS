#!/usr/bin/env python3
"""Shared paths and helpers for MT learning-curve plots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
CURRENT_PLOTS_DIR = SCRIPT_DIR / "plots" / "mt_eval_0.4B_flores200_current"
DATASET_ORDER = ["BOUQuET_Sentence", "FLORES-200", "NTREX-128"]
LANGUAGE_NAMES = {
    "ara_Arab": "Arabic",
    "bul_Cyrl": "Bulgarian",
    "deu_Latn": "German",
    "ell_Grek": "Greek",
    "fra_Latn": "French",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "ron_Latn": "Romanian",
    "rus_Cyrl": "Russian",
    "spa_Latn": "Spanish",
    "zho_Hans": "Chinese (Simplified)",
}
SUPPORTED_FORMATS = {"pdf", "png", "svg"}
METRIC_SPECS = (
    ("bleu", "spBLEU-200"),
    ("chrf", "chrF++"),
    ("comet", "COMET"),
)
METRIC_COLUMNS = tuple(metric for metric, _ in METRIC_SPECS)


def add_common_plot_arguments(
    parser: argparse.ArgumentParser,
    default_input: Path,
    default_output: Path,
) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Aggregated flores200 TSV (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Plot output directory (default: {default_output})",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats: png,pdf,svg (default: png,pdf)",
    )


def parse_formats(value: str) -> list[str]:
    formats = list(
        dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip())
    )
    if not formats:
        raise ValueError("At least one output format is required")
    unsupported = sorted(set(formats) - SUPPORTED_FORMATS)
    if unsupported:
        raise ValueError(f"Unsupported formats: {', '.join(unsupported)}")
    return formats


def iteration_from_checkpoint(checkpoint: str) -> int:
    match = re.fullmatch(r"iter_(\d+)", checkpoint)
    if match is None:
        raise ValueError(f"Invalid checkpoint name: {checkpoint}")
    return int(match.group(1))


def save_figure(fig: Any, stem: Path, formats: list[str], dpi: int) -> None:
    for extension in formats:
        kwargs = {"dpi": dpi} if extension == "png" else {}
        fig.savefig(stem.with_suffix("." + extension), bbox_inches="tight", **kwargs)
