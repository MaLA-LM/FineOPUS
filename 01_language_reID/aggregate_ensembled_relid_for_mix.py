import os
import re
import tarfile
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def list_existing_max_part_idx(pair_out_dir: Path, pair: str) -> int:
    """返回已存在的最大 part 序号；不存在则返回 -1。"""
    max_idx = -1
    if pair_out_dir.exists():
        for p in pair_out_dir.glob(f"{pair}_part_*.parquet"):
            m = re.search(r"_part_(\d{3})\.parquet$", p.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx

def write_part(tables: List[pa.Table], out_dir: Path, pair: str, part_idx: int, compression: Optional[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pair}_part_{part_idx:03d}.parquet"
    if not tables:
        return None
    table = pa.concat_tables(tables, promote=True)
    pq.write_table(table, out_path, compression=None if compression == "none" else compression, use_dictionary=True)
    return out_path

class PairWriter:
    """为每个语言对维护缓冲与 part 编号，按需写盘。"""
    def __init__(self, output_root: Path, max_rows: int, compression: str, seed_next_idx: Dict[str, int]):
        self.output_root = output_root
        self.max_rows = max_rows
        self.compression = compression
        self.buffers: Dict[str, List[pa.Table]] = {}
        self.buf_rows: Dict[str, int] = {}
        self.next_idx: Dict[str, int] = dict(seed_next_idx)  # pair -> 下一个 part 序号

    def _ensure_pair(self, pair: str):
        if pair not in self.buffers:
            self.buffers[pair] = []
            self.buf_rows[pair] = 0
        if pair not in self.next_idx:
            # 若输出目录已有历史文件，接着编号
            out_dir = self.output_root / pair
            self.next_idx[pair] = list_existing_max_part_idx(out_dir, pair) + 1

    def append(self, pair: str, tbl: pa.Table):
        """向某个 pair 追加一块数据；必要时对块切片后写满一个 part。"""
        if tbl.num_rows == 0:
            return
        self._ensure_pair(pair)

        pending = tbl
        while pending.num_rows > 0:
            room = self.max_rows - self.buf_rows[pair]
            if room <= 0:
                self._flush(pair)
                room = self.max_rows
            if pending.num_rows <= room:
                # 全部放入缓冲
                self.buffers[pair].append(pending)
                self.buf_rows[pair] += pending.num_rows
                break
            else:
                # 只切一段填满
                head = pending.slice(0, room)
                tail = pending.slice(room)
                self.buffers[pair].append(head)
                self.buf_rows[pair] += head.num_rows
                self._flush(pair)
                pending = tail

    def _flush(self, pair: str):
        """把 pair 的缓冲写成一个新 part。"""
        rows = self.buf_rows.get(pair, 0)
        if rows == 0:
            return
        out_dir = self.output_root / pair
        part_idx = self.next_idx[pair]
        out = write_part(self.buffers[pair], out_dir, pair, part_idx, self.compression)
        logging.info(f"[WRITE] {pair} -> {out} ({rows} rows)")
        self.buffers[pair] = []
        self.buf_rows[pair] = 0
        self.next_idx[pair] = part_idx + 1

    def flush_all(self):
        for pair in list(self.buffers.keys()):
            if self.buf_rows.get(pair, 0) > 0:
                self._flush(pair)

def split_table_by_pair(tbl: pa.Table) -> List[Tuple[str, pa.Table]]:
    """
    将一个 table 按 (src_lang, tgt_lang) 拆分成多个 pair 子表。
    要求两列存在；会过滤掉空/Null 的记录。
    """
    cols = tbl.schema.names
    if "src_lang" not in cols or "tgt_lang" not in cols:
        raise ValueError("Table missing required columns 'src_lang' and 'tgt_lang'.")

    # 去掉缺失语言的信息
    mask_ok = pc.and_(pc.invert(pc.is_null(tbl["src_lang"])), pc.invert(pc.is_null(tbl["tgt_lang"])))
    tbl = tbl.filter(mask_ok)
    if tbl.num_rows == 0:
        return []

    # 生成 pair 键（字符串拼接）
    src = tbl["src_lang"].cast(pa.string())
    tgt = tbl["tgt_lang"].cast(pa.string())
    src = pc.utf8_trim_whitespace(src)
    tgt = pc.utf8_trim_whitespace(tgt)
    pair_arr = pa.array([f"{s}-{t}" for s, t in zip(src.to_pylist(), tgt.to_pylist())], type=pa.string())
    tbl = tbl.append_column("___pair_key", pair_arr)

    # 找出唯一 pair（使用 Arrow 原生 unique，避免对 ChunkedArray 访问 .dictionary）
    unique_arr = pc.unique(tbl["___pair_key"])  # pyarrow.Array
    unique_vals = list(map(str, unique_arr.to_pylist()))

    out: List[Tuple[str, pa.Table]] = []
    for val in sorted(unique_vals):
        mask = pc.equal(tbl["___pair_key"], pa.scalar(val))
        sub = tbl.filter(mask).drop_columns(["___pair_key"])
        out.append((val, sub))
    return out

def process_mixed_dir(mixed_dir: Path, writer: PairWriter):
    """处理某个解压出来的 _mixed 目录，读取所有 parquet 并按 pair 分发到 writer。"""
    files = sorted(mixed_dir.glob("*.parquet"))
    if not files:
        return
    for f in files:
        pf = pq.ParquetFile(f)
        logging.info(f"[READ] {f} ({pf.num_row_groups} row_groups)")
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)  # 读完整 row group（包含所有列）
            parts = split_table_by_pair(tbl)
            for pair, sub in parts:
                writer.append(pair, sub)

def process_one_tar_mixed(tar_path: Path, staging_root: Path, writer: PairWriter, keep_extracted: bool = False):
    """只解压 tar 内的 _mixed 目录（若存在），处理后可删除。"""
    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if m.isdir() and m.name.rstrip("/").endswith("/_mixed")]
        # 兼容不同层级：也允许直接匹配名为 _mixed 的目录
        members += [m for m in tf.getmembers() if m.isdir() and os.path.basename(m.name.rstrip("/")) == "_mixed"]
        if not members:
            # 没有 _mixed，直接返回
            return
        # 解压包含 _mixed 的整个上层目录更安全（parquet 在其内）
        mixed_parents = sorted({Path(m.name).parent for m in members})
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"mixed_{tar_path.stem}_", dir=str(staging_root)))
        for parent in mixed_parents:
            # 提取该 parent 下所有内容
            for m in tf.getmembers():
                p = Path(m.name)
                try:
                    p.relative_to(parent)
                except ValueError:
                    continue
                tf.extract(m, tmp_dir)

        # 在解压根下递归找 _mixed
        for mixed in tmp_dir.rglob("_mixed"):
            if (mixed.is_dir()):
                process_mixed_dir(Path(mixed), writer)

        if not keep_extracted:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logging.info(f"[CLEAN] {tmp_dir}")

def main():
    ap = argparse.ArgumentParser(description="Read all _mixed parquets from tmp_*.tar, group by (src_lang,tgt_lang), and append into output_root with ≤1,000,000 rows per part, continuing indices.")
    ap.add_argument("--input_root", required=True, help="Folder containing tmp_*.tar")
    ap.add_argument("--output_root", required=True, help="Output folder used previously for regrouped pairs")
    ap.add_argument("--max_rows_per_part", type=int, default=1_000_000)
    ap.add_argument("--compression", default="zstd", choices=["snappy","zstd","gzip","brotli","none"])
    ap.add_argument("--staging_dir", default=None, help="Staging dir for extraction (scratch/local recommended)")
    ap.add_argument("--keep_extracted", action="store_true", help="Keep extracted temp dirs (debug)")
    ap.add_argument("--start_idx", type=int, default=0, help="Start index (inclusive) in the sorted tmp_*.tar list")
    ap.add_argument("--end_idx", type=int, default=None, help="End index (inclusive) in the sorted tmp_*.tar list")
    args = ap.parse_args()

    logging.info(f"[ARGS] start_idx={args.start_idx}, end_idx={args.end_idx}")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    staging_root = Path(args.staging_dir) if args.staging_dir else Path(tempfile.gettempdir())
    staging_root.mkdir(parents=True, exist_ok=True)

    seed_next_idx: Dict[str, int] = {}
    for pair_dir in sorted(output_root.iterdir()):
        if pair_dir.is_dir():
            pair = pair_dir.name
            seed_next_idx[pair] = list_existing_max_part_idx(pair_dir, pair) + 1

    writer = PairWriter(output_root=output_root, max_rows=args.max_rows_per_part, compression=args.compression, seed_next_idx=seed_next_idx)

    tars = sorted(input_root.glob("tmp_*.tar"))
    if not tars:
        logging.warning(f"[WARN] No tmp_*.tar under {input_root}")
        return

    total = len(tars)
    start_idx = max(0, args.start_idx or 0)
    end_idx = total - 1 if args.end_idx is None else min(args.end_idx, total - 1)
    if start_idx > end_idx:
        logging.warning(f"[WARN] start_idx ({start_idx}) > end_idx ({end_idx}); nothing to process.")
        return

    selected = tars[start_idx:end_idx + 1]
    logging.info(f"[SELECT] total={total}, range=[{start_idx}:{end_idx}] -> {len(selected)} files")

    for tar_path in selected:
        logging.info(f"[TAR] {tar_path}")
        process_one_tar_mixed(tar_path, staging_root, writer, keep_extracted=args.keep_extracted)
        logging.info(f"[TAR] done {tar_path}")

    # flush all remaining data
    writer.flush_all()

    logging.info("[DONE] mixed integration finished.")
    for pair in sorted(writer.next_idx.keys()):
        logging.info(f"  {pair}: next part idx -> {writer.next_idx[pair]}")

if __name__ == "__main__":
    main()
