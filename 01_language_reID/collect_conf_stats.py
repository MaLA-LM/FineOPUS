import os, gzip, orjson, argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def safe_mkdir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def lang_to_path(tmp_dir, lang):
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in lang)
    return os.path.join(tmp_dir, f"{safe}.bin")

def append_f16(path, arr):
    arr = np.asarray(arr, dtype=np.float16)
    with open(path, "ab") as f:
        arr.tofile(f)

def iter_conf_files(input_dir):
    files = [f for f in os.listdir(input_dir) if f.startswith("conf_stats_") and f.endswith(".json.gz")]
    files.sort()
    for fn in files:
        yield os.path.join(input_dir, fn)

def load_json_gz(fp):
    with gzip.open(fp, "rb") as f:
        return orjson.loads(f.read())

def collect(input_dir, tmp_dir, collect_stats_path):
    safe_mkdir(tmp_dir)
    sums   = defaultdict(float)
    sums2  = defaultdict(float)
    counts = defaultdict(int)
    mins   = defaultdict(lambda: 1.0)
    maxs   = defaultdict(lambda: 0.0)

    files = list(iter_conf_files(input_dir))
    logging.info(f"[collect] Found {len(files)} files")

    for fp in tqdm(files, desc="collect: reading"):
        try:
            data = load_json_gz(fp)
        except Exception as e:
            logging.info(f"[collect] skip {os.path.basename(fp)}: {e}")
            continue

        for lang, vals in data.items():
            if not isinstance(vals, list) or not vals:
                continue
            x = np.asarray(vals, dtype=np.float16)
            x = x[np.isfinite(x)]   # remove inf and nan
            if x.size == 0:
                continue
            np.clip(x, 0.0, 1.0, out=x)

            append_f16(lang_to_path(tmp_dir, lang), x.astype(np.float16, copy=False))

            n = int(x.size)
            s = float(x.sum())
            s2 = float(np.dot(x, x))
            counts[lang] += n
            sums[lang]   += s
            sums2[lang]  += s2
            x_min = float(x.min())
            x_max = float(x.max())
            if x_min < mins[lang]: mins[lang] = x_min
            if x_max > maxs[lang]: maxs[lang] = x_max

        del data

    payload = {
        "counts": counts,
        "sums": sums,
        "sums2": sums2,
        "mins": mins,
        "maxs": maxs,
        "tmp_dir": tmp_dir,
    }
    payload = {
        k: {kk: (float(vv) if isinstance(vv, float) else int(vv)) for kk, vv in d.items()}
        if isinstance(d, dict) else d
        for k, d in payload.items()
    }
    with open(collect_stats_path, "wb") as f:
        f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    logging.info(f"[collect] wrote {collect_stats_path}")



def main():
    parser = argparse.ArgumentParser(description="Collect confidence stats")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--tmp_dir", required=True)
    args = parser.parse_args()

    logging.info("Arguments:")
    logging.info(f"  Input Dir: {args.input_dir}")
    logging.info(f"  Temp Dir: {args.tmp_dir}")

    collect_stats = os.path.join(args.tmp_dir, "_collect_stats.json")
    collect(args.input_dir, args.tmp_dir, collect_stats)

if __name__ == "__main__":
    main()
