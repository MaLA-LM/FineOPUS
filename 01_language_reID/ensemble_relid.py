import os
import sys
import glob
import argparse
from typing import Dict, Iterable
import orjson
from datasets import load_dataset, Dataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_thr_json(path: str) -> Dict[str, float]:
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    out = {}
    for k, v in data.items():
        try:
            out[k] = float(v["thr"])
        except Exception:
            pass
    return out

def safe_mkdir(d: str):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def list_rel_jsonl_files(root: str) -> Iterable[str]:
    # {root}/{pair}/{pair}_*.jsonl -> relative path from root
    for p in sorted(glob.glob(os.path.join(root, "*", "*.jsonl"))):
        yield os.path.relpath(p, root)

def get_thr(thr_map: Dict[str, float], lang: str) -> float:
    return thr_map.get(lang, float("inf"))

def decide_final_lang(
    original_lang: str,
    pred_glotlid: str, conf_glotlid: float, thr_glotlid_map: Dict[str, float],
    pred_conlid: str, conf_conlid: float, thr_conlid_map: Dict[str, float],
) -> str:
    thr_g = get_thr(thr_glotlid_map, pred_glotlid)
    thr_c = get_thr(thr_conlid_map, pred_conlid)

    is_valid_g = conf_glotlid > thr_g
    is_valid_c = conf_conlid > thr_c

    # adj_g = conf_glotlid - thr_g
    # adj_c = conf_conlid - thr_c

    if is_valid_g and is_valid_c and (pred_glotlid == pred_conlid):
        return pred_glotlid
    # elif is_valid_g and is_valid_c and (pred_glotlid != pred_conlid):
    #     return pred_glotlid if adj_g > adj_c else pred_conlid
    # elif is_valid_g and not is_valid_c:
    #     return pred_glotlid
    # elif not is_valid_g and is_valid_c:
    #     return pred_conlid
    else:
        return original_lang

def write_shard(pair: str, out_root: str, shard_idx: int, buffer: list, compression: str = "snappy"):
    if not buffer:
        return
    out_dir = os.path.join(out_root, pair)
    safe_mkdir(out_dir)
    out_path = os.path.join(out_dir, f"{pair}_part_{shard_idx:03d}.parquet")
    ds = Dataset.from_list(buffer)
    ds.to_parquet(out_path, compression=compression)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glotlid_dir", required=True, help="Root of ReLID-by-GlotLID")
    ap.add_argument("--conlid_dir", required=True, help="Root of ReLID-by-ConLID")
    ap.add_argument("--glotlid_thr_json", required=True)
    ap.add_argument("--conlid_thr_json", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--max_rows_per_pair_shard", type=int, default=1_000_000,
                    help="Max rows per output parquet (default: 1000000)")
    ap.add_argument("--min_rows_per_pair_shard", type=int, default=100000,
                    help="Min rows per output parquet (default: 100000)")
    ap.add_argument("--filelist", type=str, help="Path to list of relative jsonl paths")
    ap.add_argument("--compression", default="snappy",
                    choices=["snappy", "zstd", "gzip", "brotli", "none"])
    ap.add_argument("--strict_check", action="store_true",
                    help="Check url/source_text/target_text equality per-line")
    args = ap.parse_args()

    thr_g = load_thr_json(args.glotlid_thr_json)
    thr_c = load_thr_json(args.conlid_thr_json)

    if args.filelist:
        with open(args.filelist, encoding="utf-8") as f:
            rel_files = [line.strip() for line in f if line.strip()]
    else:
        rel_files_g = set(list_rel_jsonl_files(args.glotlid_dir))
        rel_files_c = set(list_rel_jsonl_files(args.conlid_dir))
        missing_g = sorted(rel_files_c - rel_files_g)
        missing_c = sorted(rel_files_g - rel_files_c)
        if missing_g:
            logging.warning(f"[WARN] {len(missing_g)} files only in ConLID (missing in GlotLID), e.g. {missing_g[:3]}")
        if missing_c:
            logging.warning(f"[WARN] {len(missing_c)} files only in GlotLID (missing in ConLID), e.g. {missing_c[:3]}")

        rel_files = sorted(rel_files_g & rel_files_c)    

    safe_mkdir(args.out_root)

    # per-pair rolling buffers & shard indices
    buffers: Dict[str, list] = {}
    shard_idx: Dict[str, int] = {}
    written_counts: Dict[str, int] = {}

    total_rows = 0
    total_files = len(rel_files)
    
    logging.info(f"[START] Processing {total_files} files...")

    for file_idx, rel in enumerate(rel_files, start=1):
        logging.info(f"[PROGRESS] {file_idx}/{total_files} Processing file: {rel}")
        
        path_g = os.path.join(args.glotlid_dir, rel)
        path_c = os.path.join(args.conlid_dir, rel)

        ds_g = load_dataset("json", data_files=path_g, split="train")
        ds_c = load_dataset("json", data_files=path_c, split="train")

        pair_dir = os.path.dirname(rel)  # "{src}-{tgt}" of the original file
        file_row_count = 0

        for idx, (dg, dc) in enumerate(zip(ds_g, ds_c), start=1):
            if args.strict_check:
                if (dg.get("url") != dc.get("url") or
                    dg.get("source_text") != dc.get("source_text") or
                    dg.get("target_text") != dc.get("target_text")):
                    raise ValueError(f"Mismatch at {rel}:{idx}")

            src_orig = dg.get("source_lang") or dc.get("source_lang")
            tgt_orig = dg.get("target_lang") or dc.get("target_lang")

            # parse confidences as float safely
            def f(x, key):
                v = x.get(key, None)
                try:
                    return float(v)
                except Exception:
                    return float("nan")

            src_pred_g, src_conf_g = dg.get("source_predlang_id"), f(dg, "source_predlang_conf")
            src_pred_c, src_conf_c = dc.get("source_predlang_id"), f(dc, "source_predlang_conf")
            tgt_pred_g, tgt_conf_g = dg.get("target_predlang_id"), f(dg, "target_predlang_conf")
            tgt_pred_c, tgt_conf_c = dc.get("target_predlang_id"), f(dc, "target_predlang_conf")
            original_code = dg.get("original_code", dc.get("original_code"))

            final_src = decide_final_lang(src_orig, src_pred_g, src_conf_g, thr_g,
                                          src_pred_c, src_conf_c, thr_c)
            final_tgt = decide_final_lang(tgt_orig, tgt_pred_g, tgt_conf_g, thr_g,
                                          tgt_pred_c, tgt_conf_c, thr_c)

            out_pair = f"{final_src}-{final_tgt}"
            if out_pair not in buffers:
                buffers[out_pair] = []
                shard_idx[out_pair] = 0
                written_counts[out_pair] = 0

            rec = {
                # original/basic fields
                "url": dg.get("url", dc.get("url")),
                "collection": dg.get("collection", dc.get("collection")),
                "source": dg.get("source", dc.get("source")),
                # "original_code": original_code,
                "orig_src_lang": original_code.split("-")[0].strip() if original_code else "",
                "orig_tgt_lang": original_code.split("-")[1].strip() if original_code else "",
                "source_text": dg.get("source_text", dc.get("source_text")),
                "target_text": dg.get("target_text", dc.get("target_text")),
                "conv_src_lang": src_orig,
                "conv_tgt_lang": tgt_orig,

                # final langs
                "src_lang": final_src,
                "tgt_lang": final_tgt,

                # model preds & confs
                "src_predlang_id_glotlid": src_pred_g,
                "src_predlang_conf_glotlid": src_conf_g,
                "tgt_predlang_id_glotlid": tgt_pred_g,
                "tgt_predlang_conf_glotlid": tgt_conf_g,
                "src_predlang_id_conlid": src_pred_c,
                "src_predlang_conf_conlid": src_conf_c,
                "tgt_predlang_id_conlid": tgt_pred_c,
                "tgt_predlang_conf_conlid": tgt_conf_c,

                # traceability
                "original_pair_dir": pair_dir,
                # "rel_input_file": rel,
                # "lineno": idx,
            }

            buffers[out_pair].append(rec)
            total_rows += 1
            file_row_count += 1

            # shard flush by max_rows_per_pair_shard
            if len(buffers[out_pair]) >= args.max_rows_per_pair_shard:
                shard_idx[out_pair] += 1
                out_path = write_shard(out_pair, args.out_root, shard_idx[out_pair],
                                       buffers[out_pair],
                                       compression=None if args.compression == "none" else args.compression)
                written_counts[out_pair] += len(buffers[out_pair])
                buffers[out_pair].clear()
                if out_path:
                    logging.info(f"[FLUSH] {out_pair} -> {out_path}")
        
        # Log completion of current file
        logging.info(f"[COMPLETED] {file_idx}/{total_files} File: {rel} - Processed {file_row_count} rows")

    mixed_buf = []
    mixed_idx = 0    

    def flush_mixed(force=False):
        nonlocal mixed_buf, mixed_idx
        if not mixed_buf:
            return
        if force or len(mixed_buf) >= args.max_rows_per_pair_shard:
            mixed_idx += 1
            out_path = os.path.join(args.out_root, "_mixed")
            safe_mkdir(out_path)
            out_file = os.path.join(out_path, f"mixed_part_{mixed_idx:03d}.parquet")
            ds = Dataset.from_list(mixed_buf)
            ds.to_parquet(out_file, compression=None if args.compression == "none" else args.compression)
            logging.info(f"[FLUSH] _mixed -> {out_file} ({len(mixed_buf)} rows)")
            mixed_buf.clear()


    # flush remaining
    for pair, buf in buffers.items():
        if not buf:
            continue
        if shard_idx.get(pair, 0) > 0 or len(buf) >= args.min_rows_per_pair_shard:
            shard_idx[pair] += 1
            out_path = write_shard(pair, args.out_root, shard_idx[pair], buf,
                                   compression=None if args.compression == "none" else args.compression)
            written_counts[pair] += len(buf)
            buffers[pair].clear()
            if out_path:
                logging.info(f"[FLUSH] {pair} -> {out_path}")
        else:
            mixed_buf.extend(buf)
            written_counts[pair] += len(buf)
            buffers[pair].clear()

            if len(mixed_buf) >= args.max_rows_per_pair_shard:
                flush_mixed()

    flush_mixed(force=True)

    # summary
    pairs_written = sum(1 for v in shard_idx.values() if v > 0)
    logging.info(f"[DONE] total rows: {total_rows}, pairs with output: {pairs_written}")
    for pair in sorted(shard_idx.keys()):
        logging.info(f"  - {pair}: {shard_idx[pair]} shard(s), {written_counts[pair]} rows")

if __name__ == "__main__":
    main()
