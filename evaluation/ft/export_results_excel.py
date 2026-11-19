#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2,<3", "openpyxl>=3.1,<4"]
# ///
"""Export final FT results using the shared MT workbook tables and styling."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import export_mt_results_excel as shared


def variant_label(model: str) -> str:
    match = re.fullmatch(r"FT-FineOPUS-Filtered-Stage([1-6])-sample5k", model)
    if match:
        return f"FineOPUS Stage {match.group(1)}"
    return {
        "FT-mala-bilingual-translation-corpus-sample5k": "MaLA",
        "FT-nllb-sample5k": "NLLB",
    }.get(model, model)


def load_results(source: Path) -> tuple[list[dict], pd.DataFrame]:
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
        raise ValueError("Expected a nonempty list of result objects")
    raw = pd.DataFrame(records)
    required = {"model", "language", "checkpoint", "dataset", "source", "target", "status", *shared.METRICS}
    if missing := required - set(raw.columns):
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    if set(raw.columns) & {"variant", "iteration", "direction"}:
        raise ValueError("Input uses reserved derived fields")
    if not raw["checkpoint"].eq("final").all() or not raw["status"].eq("complete").all():
        raise ValueError("The FT export requires complete results from final checkpoints")
    raw["variant"] = raw["model"].map(variant_label)
    if raw.groupby("variant")["model"].nunique().gt(1).any():
        raise ValueError("Multiple models share the same variant label")
    # The shared summary functions require a sortable checkpoint key. This
    # internal sentinel is removed from every worksheet before export.
    raw["iteration"] = 0
    raw["direction"] = [shared.direction_label(s, t) for s, t in zip(raw["source"], raw["target"])]
    leading = ["variant", "language", "checkpoint", "dataset", "direction", "source", "target",
               *shared.METRICS, "num_examples", "status", "model"]
    columns = [c for c in leading if c in raw.columns]
    columns += [c for c in raw.columns if c not in columns]
    return records, raw[columns]


def overview(source: Path, raw: pd.DataFrame) -> pd.DataFrame:
    table = shared.overview(source, raw)
    replacements = {
        "Model exclusions": "None. All models in the source JSON are retained.",
        "RawData": "All source records and original fields in source order. Variant and direction are derived columns. Checkpoint remains final; no training iteration is inferred.",
        "FinalByLanguage": "Available dataset scores averaged per model and benchmark language code, using the exported final HF weights. No checkpoint selection is performed.",
        "FinalByDataset": "Dataset scores from the exported final HF weights, with both translation directions averaged equally.",
        "FinalByDirection": "Direction scores from the exported final HF weights. num_alias_pairs records the number of source rows averaged within a cell.",
        "Aggregation order": "Dataset aliases -> directions -> available datasets -> language codes; no weighting by sentence count. The same aggregation functions as evaluation/export_mt_results_excel.py are used.",
        "Regeneration": "Run uv run evaluation/ft/export_results_excel.py SOURCE_JSON --output WORKBOOK.xlsx after the source JSON changes.",
    }
    for field, description in replacements.items():
        table.loc[table["Field"].eq(field), "Description"] = description
    return pd.concat([table, pd.DataFrame([
        ("Checkpoint convention", "All source checkpoints are final. An internal ordering sentinel is used by the shared summary functions and omitted from every worksheet."),
        ("Language keys", "The source language field retains dataset-specific benchmark codes. Codes such as zho_Hant and cmn_Hant remain separate language keys across datasets; no additional cross-dataset normalization is performed. See evaluation/ft/reports/coverage.tsv for training-language mappings."),
        ("FT labels", "FineOPUS Stage 1–6, MaLA, and NLLB are display labels. Original model directory names are preserved in RawData and detailed tables."),
    ], columns=["Field", "Description"])], ignore_index=True)


def export_workbook(source: Path, output: Path) -> dict[str, int]:
    records, raw = load_results(source)
    sheets = {"Overview": overview(source, raw), **shared.summary_tables(raw)}
    sheets = {name: frame.drop(columns=["iteration", "first_iteration", "last_iteration"], errors="ignore")
              for name, frame in sheets.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=output.stem + "_", suffix=".tmp.xlsx",
                                     dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
            writer.book.properties.title = "FineOPUS FT translation evaluation"
            writer.book.properties.subject = "Final FT models: raw results, language averages, dataset averages and rankings"
            writer.book.properties.creator = "FineOPUS"
            for index, (name, frame) in enumerate(sheets.items(), 1):
                frame.to_excel(writer, sheet_name=name, index=False)
                shared.style_sheet(writer.sheets[name], list(frame.columns), index)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    counts = {name: len(frame) for name, frame in sheets.items()}
    assert counts["RawData"] == len(records)
    print(f"Wrote {output}: {counts}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="FT all_results.json")
    parser.add_argument("--output", type=Path, help="Destination workbook; defaults to ft_mt_results.xlsx beside the JSON")
    args = parser.parse_args()
    source = args.source.resolve()
    export_workbook(source, args.output or source.with_name("ft_mt_results.xlsx"))


if __name__ == "__main__":
    main()
