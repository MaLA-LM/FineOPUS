#!/usr/bin/env python3
"""Collect all per-direction metrics.json files into global TSV and JSON tables."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
from pathlib import Path


FIELDS = [
    "status", "model", "checkpoint", "dataset", "source", "target", "language",
    "num_examples", "bleu", "chrf", "comet", "seconds", "comet_seconds",
    "few_shot", "limit", "bleu_tokenizer", "bleu_signature", "chrf_word_order",
    "comet_model", "comet_num_examples", "result_dir",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-prefix", default="all_results")
    parser.add_argument("--model-glob", action="append", default=[])
    parser.add_argument("--bleu-tokenizer")
    parser.add_argument("--comet-model")
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in args.results_root.rglob("metrics.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping invalid {path}: {exc}")
            continue
        row.pop("bleu_13a", None)
        row.pop("bleu_13a_tokenizer", None)
        if row.get("status") != "complete":
            continue
        if not args.include_incomplete and not (path.parents[2] / "_SUCCESS").is_file():
            continue
        if args.model_glob and not any(
            fnmatch.fnmatchcase(str(row.get("model", "")), pattern)
            for pattern in args.model_glob
        ):
            continue
        if args.bleu_tokenizer and row.get("bleu_tokenizer") != args.bleu_tokenizer:
            continue
        if args.comet_model and row.get("comet_model") != args.comet_model:
            continue
        rows.append(row)
    rows.sort(key=lambda row: tuple(str(row.get(key, "")) for key in ("model", "checkpoint", "dataset", "source", "target")))
    if not rows:
        raise SystemExit(f"No completed metrics.json files under {args.results_root}")

    json_path = args.results_root / f"{args.output_prefix}.json"
    tsv_path = args.results_root / f"{args.output_prefix}.tsv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated {len(rows)} completed directions")
    print(tsv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
