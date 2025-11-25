import math
import os
import orjson
import numpy as np
from tqdm import tqdm
import logging
import sys
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def exact_quantiles_from_sorted_array(arr, qs):
    n = int(arr.size)
    out = {}
    if n == 0:
        return {q: float("nan") for q in qs}

    for q in qs:
        if q <= 0:
            out[q] = float(arr[0]); continue
        if q >= 1:
            out[q] = float(arr[-1]); continue
        i = (n - 1) * q
        lo = int(math.floor(i))
        hi = int(math.ceil(i))
        if lo == hi:
            out[q] = float(arr[lo])
        else:
            w = i - lo
            out[q] = float((1.0 - w) * arr[lo] + w * arr[hi])
    return out


def quantiles(collect_stats_path, output_file, tmp_dir=None, keep_tmp=False):
    with open(collect_stats_path, "rb") as f:
        meta = orjson.loads(f.read())

    counts = {k: int(v) for k, v in meta["counts"].items()}
    sums   = {k: float(v) for k, v in meta["sums"].items()}
    sums2  = {k: float(v) for k, v in meta["sums2"].items()}
    mins   = {k: float(v) for k, v in meta["mins"].items()}
    maxs   = {k: float(v) for k, v in meta["maxs"].items()}

    langs = sorted(counts.keys(), key=lambda x: counts[x])
    logging.info(f"[quantiles] languages: {len(langs)}")

    results = {}
    for lang in tqdm(langs, desc="quantiles: per-lang"):
        logging.info(f"[quantiles] processing {lang}")
        n = counts[lang]
        if n <= 0:
            continue
        bin_path = f"{tmp_dir}/{lang}.bin"
        if not os.path.isfile(bin_path):
            logging.info(f"[quantiles] missing bin for {lang}, skip")
            continue
        arr = np.fromfile(bin_path, dtype=np.float16)

        if arr.size >= n:
            arr = arr[:n]
        else:
            n = arr.size

        if n == 0:
            del arr
            continue

        arr.sort(kind="quicksort")

        qs = exact_quantiles_from_sorted_array(arr, [0.5])

        mean = sums[lang] / counts[lang] if counts[lang] > 0 else float("nan")
        var  = (sums2[lang] - counts[lang] * mean * mean) / (counts[lang] - 1) if counts[lang] > 1 else 0.0
        std  = math.sqrt(var) if var > 0 else 0.0
        thr  = max(0.3, min(0.9, qs[0.5] - std))

        results[lang] = {
            "count": int(counts[lang]),
            "mean": float(mean),
            "median": float(qs[0.5]),
            "std": float(std),
            "variance": float(var),
            "min": float(mins[lang]),
            "max": float(maxs[lang]),
            "thr": float(thr)
        }

        del arr

        if not keep_tmp:
            try:
                os.remove(bin_path)
            except OSError:
                pass

    sorted_stats = dict(sorted(results.items(), key=lambda x: x[1]["count"], reverse=True))
    with open(output_file, "wb") as f:
        f.write(orjson.dumps(sorted_stats, option=orjson.OPT_INDENT_2))
    logging.info(f"[quantiles] wrote {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect_stats", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--tmp_dir", default=None)
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args()

    logging.info("Arguments:")
    logging.info(f"  Collect Stats: {args.collect_stats}")
    logging.info(f"  Output File: {args.output_file}")
    logging.info(f"  Temp Dir: {args.tmp_dir}")

    quantiles(
        collect_stats_path=args.collect_stats,
        output_file=args.output_file,
        tmp_dir=args.tmp_dir,
        keep_tmp=args.keep_tmp
    )

if __name__ == "__main__":
    main()