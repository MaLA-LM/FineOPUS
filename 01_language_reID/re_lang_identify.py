import os
import argparse
from glob import glob
from datasets import load_dataset
import fasttext
from conlid import ConLID
import sys
import logging
import pyarrow.parquet as pq


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def count_lines(file_path):
    """Count number of rows in parquet file"""
    if file_path.endswith('.parquet'):
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows
    else:
        # Fallback for other file types
        logging.warning(f"Counting lines for non-parquet file: {file_path}")
        with open(file_path, 'rb') as f:
            return sum(1 for _ in f)
    

def get_lang_preds(source_text, target_text):
    source_pred = lid_model.predict(source_text, 1)
    target_pred = lid_model.predict(target_text, 1)
    return {
        "source_predlang_id": source_pred[0][0].replace("__label__", ""),
        "source_predlang_conf": source_pred[1][0],
        "target_predlang_id": target_pred[0][0].replace("__label__", ""),
        "target_predlang_conf": target_pred[1][0],
    }


def save_parquet(dataset, path):
    dataset.to_parquet(path, compression='zstd')
    logging.info(f"√ Saved to {path}")

def process_file(input_path, output_path, num_proc):
    try:
        ds = load_dataset("parquet", data_files=input_path, split="train")

        required_keys = {"source_text", "target_text", "conv_src_lang", "conv_tgt_lang"}
        if not required_keys.issubset(ds.column_names):
            raise ValueError(f"Missing required fields: {required_keys - set(ds.column_names)}")
        
        ds = ds.map(lambda x: get_lang_preds(x["source_text"], x["target_text"]), num_proc=num_proc)
        save_parquet(ds, output_path)

    except Exception as e:
        logging.error(f"[Error] Failed to process: {input_path}\n{type(e).__name__}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="Directory containing .parquet files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output .parquet files")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--model_path", default="model.bin", help="Path to fastText language ID model")
    parser.add_argument("--filelist", type=str, help="Optional: Path to file containing list of files to process")
    args = parser.parse_args()

    logging.info("Arguments:")
    logging.info(f"  Source Directory: {args.source_dir}")
    logging.info(f"  Output Directory: {args.output_dir}")
    logging.info(f"  Number of Processes: {args.num_proc}")
    logging.info(f"  Model Path: {args.model_path}")
    logging.info(f"  Filelist: {args.filelist}")
    logging.info(f"  HF_HOME: {os.environ.get('HF_HOME', 'default')}")

    os.makedirs(args.output_dir, exist_ok=True)
    if "glotlid" in args.model_path:
        lid_model = fasttext.load_model(args.model_path)
        logging.info("Loaded GlotLID model")
    elif "ConLID" in args.model_path:
        lid_model = ConLID.from_pretrained(args.model_path)
        logging.info("Loaded ConLID model")
    else:
        raise ValueError(f"Unknown model type in path: {args.model_path}")

    if args.filelist:
        with open(args.filelist, encoding="utf-8") as f:
            all_files = [line.strip() for line in f if line.strip()]
    else:
        all_files = sorted(glob(f"{args.source_dir}/**/*.parquet", recursive=True))

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for idx, input_path in enumerate(all_files, 1):
        logging.info(f"[{idx}/{len(all_files)}] Processing file: {os.path.basename(input_path)}")

        rel_path = os.path.relpath(input_path, args.source_dir)
        output_path = os.path.join(args.output_dir, rel_path) 
        
        skip = False
        if os.path.exists(output_path):
            try:
                input_lines = count_lines(input_path)
                output_lines = count_lines(output_path)
                if input_lines == output_lines:
                    logging.info(f"[Skip] {output_path} exists and line count matches ({input_lines})")
                    skip = True
                    skipped_count += 1
                else:
                    logging.warning(f"[Reprocess] {output_path} exists but line count mismatch (input={input_lines}, output={output_lines})")
            except Exception as e:
                logging.warning(f"[Reprocess] Failed to count lines for {input_path} or {output_path}: {e}")

        if skip:
            continue

        try:
            process_file(input_path, output_path, args.num_proc)
            processed_count += 1
        except Exception as e:
            logging.error(f"[Error] Failed during processing: {input_path}\n{e}")
            failed_count += 1
            continue
    
    logging.info("=" * 60)
    logging.info("Processing Summary:")
    logging.info(f"  Total files: {len(all_files)}")
    logging.info(f"  Successfully processed: {processed_count}")
    logging.info(f"  Skipped (already done): {skipped_count}")
    logging.info(f"  Failed: {failed_count}")
    logging.info("=" * 60)
    
    # Exit with error code if any files failed
    if failed_count > 0:
        sys.exit(1)
