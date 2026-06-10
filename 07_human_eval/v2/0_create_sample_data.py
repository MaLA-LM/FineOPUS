import argparse
import json
import multiprocessing
from datasets import load_dataset

RANDOM_SEED= 42 # Set a fixed random seed for reproducible data shuffling/sampling
import random
random.seed(RANDOM_SEED)


def stream_save_random_samples(hf_path, src_code, tgt_code, n_samples, buffer_size):
    
    # load data using streaming, shuffle with a buffer, and get n_samples samples
    relevant_hf_files = f"hf://datasets/{hf_path}/{src_code}-{tgt_code}/*.parquet"

    dataset = load_dataset(
        "parquet", 
        data_files={"train": relevant_hf_files}, 
        split="train",
        streaming=True
    ).shuffle(
        seed=RANDOM_SEED,
        buffer_size=buffer_size
    ).take(n_samples)
    
    # write the samples to a JSONL file    
    hf_path_converted = hf_path.replace("/", "_")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            processed_item = {
                "src_text": item.get("source_text", ""),
                "tgt_text": item.get("target_text", ""),
                "src_code": src_code,
                "tgt_code": tgt_code,
                "corpus": item.get("corpus", ""),
                "version": item.get("version", ""),
            }
            json.dump(processed_item, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"Successfully streamed {n_samples} random samples to {args.output_file}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Create sample data for human evaluation based on src and tgt codes.")
    
    # Use the target repo ID
    arg_parser.add_argument("--hf_path", type=str, default="MaLA-LM/FineOPUS-Deduplicated")
    arg_parser.add_argument("--n_samples", type=int, default=100)
    arg_parser.add_argument("--src_code", type=str, default="eng_Latn")
    arg_parser.add_argument("--tgt_code", type=str, default="zho_Hant")
    arg_parser.add_argument("--buffer_size", type=int, default=10000)
    arg_parser.add_argument("--output_folder", type=str, default="./data/samples")
    
    args = arg_parser.parse_args()
    args.output_file = f"{args.output_folder}/sample_{args.src_code}_{args.tgt_code}_{args.n_samples}.jsonl"
    
    process = multiprocessing.Process(
            target=stream_save_random_samples, 
            args=(args.hf_path, args.src_code, args.tgt_code, args.n_samples, args.buffer_size)
        )
        
    process.start()
    process.join() 