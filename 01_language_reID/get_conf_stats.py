import os
from glob import glob
import orjson
from datasets import load_dataset
from collections import defaultdict
import logging
import sys
import argparse
import gzip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="Directory containing .parquet files with predictions")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--filelist", type=str, help="Optional: Path to file containing list of files to process")
    parser.add_argument("--job_id", type=int, default=None, help="Optional: Job ID")
    args = parser.parse_args()

    logging.info("Arguments:")
    logging.info(f"  Source Directory: {args.source_dir}")
    logging.info(f"  Output Directory: {args.output_dir}")
    logging.info(f"  Filelist: {args.filelist}")

    if args.filelist:
        with open(args.filelist, encoding="utf-8") as f:
            all_files = [line.strip() for line in f if line.strip()]
    else:
        all_files = sorted(glob(f"{args.source_dir}/**/*.parquet", recursive=True))

    if not all_files:
        logging.warning("No files found to process.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.job_id is not None:
        stats_filename = f"conf_stats_{args.job_id}.json.gz"
    else:
        stats_filename = "conf_stats.json.gz"

    output_file = os.path.join(args.output_dir, stats_filename)

    if os.path.exists(output_file):
        logging.info(f"Output file {output_file} already exists. Skipping processing.")
        exit(0)

    lang_conf = defaultdict(list)
    total_files = len(all_files)

    for idx, input_path in enumerate(all_files, 1):
        logging.info(f"[{idx}/{total_files}] Processing file: {os.path.basename(input_path)}")

        try:
            ds = load_dataset("parquet", data_files=input_path, split="train").select_columns(
                ["source_predlang_id", "source_predlang_conf", "target_predlang_id", "target_predlang_conf"]
            )
            
            src_langs = ds["source_predlang_id"]
            src_confs = ds["source_predlang_conf"]
            tgt_langs = ds["target_predlang_id"]
            tgt_confs = ds["target_predlang_conf"]

            for lang, conf in zip(src_langs, src_confs):
                if lang is not None:
                    lang_conf[lang].append(conf)

            for lang, conf in zip(tgt_langs, tgt_confs):
                if lang is not None:
                    lang_conf[lang].append(conf)
        except Exception as e:
            logging.error(f"Error processing file {input_path}: {e}")
            continue

    with gzip.open(output_file, "wb") as f:
        f.write(orjson.dumps(lang_conf, option=orjson.OPT_INDENT_2))

    logging.info(f"Stats saved to {output_file}")
    exit(0)