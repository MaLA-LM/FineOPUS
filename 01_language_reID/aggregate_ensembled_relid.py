import re
import tarfile
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def list_existing_max_part_idx(pair_out_dir: Path, pair: str) -> int:
    max_idx = -1
    if pair_out_dir.exists():
        for p in pair_out_dir.glob(f"{pair}_part_*.parquet"):
            m = re.search(r"_part_(\d{3})\.parquet$", p.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx

def safe_move(src: Path, dst: Path, dry_run: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        logging.info(f"[DRY] move {src} -> {dst}")
        return
    shutil.move(str(src), str(dst))

def process_one_tar(
    tar_path: Path,
    output_root: Path,
    state_next_idx: Dict[str, int],
    extract_root: Path,
    dry_run: bool,
):
    logging.info(f"[TAR] {tar_path}")
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"extract_{tar_path.stem}_", dir=str(extract_root)))

    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(tmp_dir)

    candidates = [p for p in tmp_dir.iterdir() if p.is_dir()]
    root = candidates[0] if len(candidates) == 1 else tmp_dir

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        pair = entry.name
        if pair == "_mixed":
            continue

        out_pair_dir = output_root / pair
        if pair not in state_next_idx:
            max_existing = list_existing_max_part_idx(out_pair_dir, pair)
            state_next_idx[pair] = max_existing + 1

        parquet_files = sorted(entry.glob("*.parquet"))
        for src_file in parquet_files:
            part_idx = state_next_idx[pair]
            dst_name = f"{pair}_part_{part_idx:03d}.parquet"
            dst_path = out_pair_dir / dst_name

            while dst_path.exists():
                part_idx += 1
                dst_name = f"{pair}_part_{part_idx:03d}.parquet"
                dst_path = out_pair_dir / dst_name

            logging.info(f"[MOVE] {pair}: {src_file.name} -> {dst_name}")
            safe_move(src_file, dst_path, dry_run=dry_run)
            state_next_idx[pair] = part_idx + 1

    if dry_run:
        logging.info(f"[DRY] cleanup {tmp_dir}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    logging.info(f"[TAR] done {tar_path}")

def main():
    ap = argparse.ArgumentParser(description="Extract tmp_*.tar, regroup parquet by language pair, and renumber parts without reading parquet content.")
    ap.add_argument("--input_root", required=True, help="Folder containing tmp_*.tar")
    ap.add_argument("--output_root", required=True, help="Output folder to assemble per-pair directories")
    ap.add_argument("--extract_tmp_dir", default=None, help="Staging dir for extraction (recommend fast local/scratch)")
    ap.add_argument("--dry_run", action="store_true", help="Print actions only, no file moves")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    extract_root = Path(args.extract_tmp_dir) if args.extract_tmp_dir else Path(tempfile.gettempdir())
    extract_root.mkdir(parents=True, exist_ok=True)

    state_next_idx: Dict[str, int] = {}

    for pair_dir in sorted(output_root.iterdir()):
        if pair_dir.is_dir():
            pair = pair_dir.name
            max_existing = list_existing_max_part_idx(pair_dir, pair)
            state_next_idx[pair] = max_existing + 1

    tars = sorted(input_root.glob("tmp_*.tar"))
    if not tars:
        logging.warning(f"[WARN] no tmp_*.tar under {input_root}")
        return

    for tar_path in tars:
        process_one_tar(
            tar_path=tar_path,
            output_root=output_root,
            state_next_idx=state_next_idx,
            extract_root=extract_root,
            dry_run=args.dry_run,
        )

    logging.info("[DONE] regroup+rename finished.")
    for pair in sorted(state_next_idx.keys()):
        logging.info(f"  {pair}: next part idx -> {state_next_idx[pair]}")

if __name__ == "__main__":
    main()
