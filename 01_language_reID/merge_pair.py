import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def find_pair_dirs(root: Path) -> Dict[str, Path]:
    """Return mapping {pair_name: path} for subdirs like aaa_Bbbb-ccc_Dddd."""
    pairs = {}
    if not root.exists():
        return pairs
    for p in root.iterdir():
        if p.is_dir() and "-" in p.name:
            pairs[p.name] = p
    return pairs

def build_scanner_for_dirs(dirs: List[Optional[Path]]) -> Tuple[Optional[ds.Scanner], Optional[pa.Schema], List[str]]:
    """
    Build a dataset scanner from multiple directories (order preserved).
    Returns (scanner, unified_schema, file_list). If no files, (None, None, []).
    """
    files: List[str] = []
    for d in dirs:
        if d is None:
            continue
        fs = sorted(str(f) for f in d.glob("*.parquet"))
        files.extend(fs)

    if not files:
        return None, None, []

    dataset = ds.dataset(files, format="parquet")
    schema = dataset.schema
    scanner = dataset.scanner(batch_size=4096)
    return scanner, schema, files

def write_chunks_from_scanner(
    scanner: ds.Scanner,
    schema: pa.Schema,
    out_dir: Path,
    out_prefix: str,
    max_rows_per_file: int = 10_000,
):
    """Split scanner rows into chunks (<= max_rows_per_file) and write parquet parts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    buf_tables: List[pa.Table] = []
    buf_rows = 0
    part_idx = 0

    def flush():
        nonlocal buf_tables, buf_rows, part_idx
        if buf_rows == 0:
            return
        table = pa.concat_tables(buf_tables, promote=True)
        out_path = out_dir / f"{out_prefix}_part_{part_idx:03d}.parquet"
        pq.write_table(table, out_path, compression="snappy")
        logging.info(f"[WRITE] {out_path} rows={table.num_rows}")
        part_idx += 1
        buf_tables.clear()
        buf_rows = 0

    for batch in scanner.to_batches():
        tb = pa.Table.from_batches([batch])
        start = 0
        n = tb.num_rows
        while start < n:
            remain = max_rows_per_file - buf_rows
            take = min(remain, n - start)
            slice_tb = tb.slice(start, take)

            # Align to unified schema (fill missing columns with nulls; reorder columns)
            if schema is not None and slice_tb.schema != schema:
                cols = []
                for field in schema:
                    if field.name in slice_tb.schema.names:
                        cols.append(slice_tb.column(slice_tb.schema.get_field_index(field.name)))
                    else:
                        cols.append(pa.nulls(take, type=field.type))
                slice_tb = pa.Table.from_arrays(cols, schema=schema)

            buf_tables.append(slice_tb)
            buf_rows += take
            start += take

            if buf_rows >= max_rows_per_file:
                flush()

    flush()

def main():
    ap = argparse.ArgumentParser(
        description="Merge language-pair subdirs from two roots into chunked parquet files (union of pairs)."
    )
    ap.add_argument("--root1", required=True, type=Path)
    ap.add_argument("--root2", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--max-rows-per-file", type=int, default=10_000)
    args = ap.parse_args()

    pairs1 = find_pair_dirs(args.root1)
    pairs2 = find_pair_dirs(args.root2)

    # Use union of pairs; preserve deterministic order (common first by name)
    union_pairs = sorted(set(pairs1) | set(pairs2))
    if not union_pairs:
        logging.warning("No language-pair subdirectories found in either root.")
        return

    logging.info(f"Total language pairs to process (union): {len(union_pairs)}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    for pair in union_pairs:
        dir1 = pairs1.get(pair)
        dir2 = pairs2.get(pair)
        out_dir = args.out_root / pair
        out_prefix = pair

        # Log whether common or single-sided
        if dir1 and dir2:
            logging.info(f"[PAIR][COMMON] {pair}")
        elif dir1:
            logging.info(f"[PAIR][ONLY root1] {pair}")
        else:
            logging.info(f"[PAIR][ONLY root2] {pair}")

        scanner, schema, files = build_scanner_for_dirs([dir1, dir2])
        if scanner is None:
            logging.info(f"  Skipping (no parquet files): {pair}")
            continue

        write_chunks_from_scanner(
            scanner=scanner,
            schema=schema,
            out_dir=out_dir,
            out_prefix=out_prefix,
            max_rows_per_file=args.max_rows_per_file,
        )

    logging.info("All done.")

if __name__ == "__main__":
    main()
