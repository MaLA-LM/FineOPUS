#!/usr/bin/env python3
"""Split language pairs from a class-combo JSON into balanced parts by row count.

Excludes pairs already recorded in llm_judge_stats.csv. Row counts come from
fineopus-filtered-stage4-row-counts.xlsx (columns: language_pair, total_rows).

Example:
  python split_pair_combos.py --class-combo 1-2 --parts 14
  python split_pair_combos.py --class-combo 1-1 --parts 5 --dry-run
"""

import argparse
import csv
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "fineopus_pair_class_combinations.json"
DEFAULT_STATS = HERE / "stats" / "llm_judge_stats.csv"
DEFAULT_ROW_COUNTS = (
    HERE.parent
    / "tools"
    / "parquet_rows_per_pair_stats"
    / "fineopus-filtered-stage4-row-counts.xlsx"
)
XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_existing_pairs(stats_csv: Path) -> Set[str]:
    if not stats_csv.exists():
        return set()
    with stats_csv.open(newline="", encoding="utf-8") as fp:
        return {row["lang_pair"] for row in csv.DictReader(fp)}


def read_row_counts_xlsx(xlsx_path: Path) -> Dict[str, int]:
    """Parse row-count xlsx without third-party dependencies."""

    def cell_value(cell: ET.Element) -> str:
        cell_type = cell.get("t")
        if cell_type == "inlineStr":
            inline = cell.find("m:is", XLSX_NS)
            if inline is not None:
                text_el = inline.find("m:t", XLSX_NS)
                if text_el is not None:
                    return text_el.text or ""
                parts = [
                    (run.find("m:t", XLSX_NS).text or "")
                    for run in inline.findall("m:r", XLSX_NS)
                ]
                return "".join(parts)
        value_el = cell.find("m:v", XLSX_NS)
        return value_el.text if value_el is not None else ""

    with zipfile.ZipFile(xlsx_path) as archive:
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: List[List[str]] = []
    for row_el in sheet.findall(".//m:sheetData/m:row", XLSX_NS):
        rows.append([cell_value(cell) for cell in row_el.findall("m:c", XLSX_NS)])

    if not rows:
        raise ValueError(f"No rows found in {xlsx_path}")

    header = rows[0]
    try:
        pair_idx = header.index("language_pair")
        rows_idx = header.index("total_rows")
    except ValueError as exc:
        raise ValueError(
            f"Expected columns language_pair and total_rows in {xlsx_path}, got {header}"
        ) from exc

    counts: Dict[str, int] = {}
    for row in rows[1:]:
        if len(row) <= max(pair_idx, rows_idx):
            continue
        pair = row[pair_idx].strip()
        if not pair:
            continue
        counts[pair] = int(float(row[rows_idx]))
    return counts


def partition_by_rows(
    pairs: Sequence[str],
    row_counts: Dict[str, int],
    num_parts: int,
) -> List[List[str]]:
    """Greedy LPT: assign largest pairs to the currently lightest partition."""
    if num_parts < 1:
        raise ValueError("--parts must be >= 1")
    if not pairs:
        return [[] for _ in range(num_parts)]

    weighted = sorted(
        ((pair, row_counts.get(pair, 0)) for pair in pairs),
        key=lambda item: item[1],
        reverse=True,
    )

    parts: List[List[str]] = [[] for _ in range(num_parts)]
    part_totals = [0] * num_parts

    for pair, rows in weighted:
        idx = min(range(num_parts), key=lambda i: part_totals[i])
        parts[idx].append(pair)
        part_totals[idx] += rows

    return parts


def write_parts(
    class_combo: str,
    parts: Sequence[Sequence[str]],
    output_dir: Path,
    prefix: str,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for i, chunk in enumerate(parts, start=1):
        out_path = output_dir / f"{prefix}_part{i}.json"
        payload = {class_combo: list(chunk)}
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
        written.append(out_path)
    return written


def summarize_parts(
    parts: Sequence[Sequence[str]],
    row_counts: Dict[str, int],
) -> List[Tuple[int, int, int]]:
    """Return (n_pairs, total_rows, min_rows, max_rows) per part — actually 4-tuple."""
    summary = []
    for chunk in parts:
        rows_each = [row_counts.get(pair, 0) for pair in chunk]
        total = sum(rows_each)
        summary.append((len(chunk), total, min(rows_each) if rows_each else 0, max(rows_each) if rows_each else 0))
    return summary


def parse_args(argv=None):
    # type: (Optional[Sequence[str]]) -> argparse.Namespace
    parser = argparse.ArgumentParser(
        description="Split a class-combo JSON into row-balanced parts, skipping done pairs.",
    )
    parser.add_argument(
        "--class-combo",
        required=True,
        help='Resource-class combo key, e.g. "1-2".',
    )
    parser.add_argument(
        "--parts",
        type=int,
        required=True,
        help="Number of output JSON files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source pair-combo JSON (default: {DEFAULT_INPUT.name}).",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=DEFAULT_STATS,
        help=f"Stats CSV with completed lang_pair rows (default: {DEFAULT_STATS}).",
    )
    parser.add_argument(
        "--row-counts",
        type=Path,
        default=DEFAULT_ROW_COUNTS,
        help="XLSX with language_pair and total_rows columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE,
        help="Directory for output JSON files.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Output filename prefix (default: <input_stem>_<class-combo>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the split plan without writing files.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    # type: (Optional[Sequence[str]]) -> int
    args = parse_args(argv)

    if args.parts < 1:
        print("ERROR: --parts must be >= 1", file=sys.stderr)
        return 1

    with args.input.open(encoding="utf-8") as fp:
        combos = json.load(fp)

    class_combo = args.class_combo
    if class_combo not in combos:
        print(f"ERROR: class combo '{class_combo}' not found in {args.input}", file=sys.stderr)
        return 1

    all_pairs: List[str] = list(combos[class_combo])
    done_pairs = read_existing_pairs(args.stats)
    row_counts = read_row_counts_xlsx(args.row_counts)

    remaining = [pair for pair in all_pairs if pair not in done_pairs]
    skipped = [pair for pair in all_pairs if pair in done_pairs]

    missing_counts = [pair for pair in remaining if pair not in row_counts]
    if missing_counts:
        print(
            f"WARNING: {len(missing_counts)} pair(s) missing from row-count file; treating as 0 rows.",
            file=sys.stderr,
        )

    parts = partition_by_rows(remaining, row_counts, args.parts)
    prefix = args.prefix or f"{args.input.stem}_{class_combo}"
    summary = summarize_parts(parts, row_counts)

    total_rows = sum(row_counts.get(pair, 0) for pair in remaining)
    target_rows = total_rows / args.parts if args.parts else 0

    print(f"class combo     : {class_combo}")
    print(f"input pairs     : {len(all_pairs)}")
    print(f"already done    : {len(skipped)}")
    print(f"remaining       : {len(remaining)}")
    print(f"total rows      : {total_rows:,}")
    print(f"target rows/part: {target_rows:,.0f}")
    print(f"parts           : {args.parts}")
    print("-" * 72)
    for i, (n_pairs, part_rows, min_rows, max_rows) in enumerate(summary, start=1):
        drift = part_rows - target_rows
        print(
            f"part{i:2d}: {n_pairs:4d} pairs, {part_rows:12,} rows "
            f"(drift {drift:+12,.0f}, pair rows {min_rows:,}..{max_rows:,})"
        )

    if args.dry_run:
        print("-" * 72)
        print("dry-run: no files written")
        return 0

    written = write_parts(class_combo, parts, args.output_dir, prefix)
    print("-" * 72)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
