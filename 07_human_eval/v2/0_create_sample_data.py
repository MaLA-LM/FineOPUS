
import os
import json
import tqdm
import argparse
import multiprocessing

from itertools import chain
from datasets import load_dataset

import random 
random.seed(42) # Set a fixed random seed for reproducible data shuffling/sampling


# concat the src-tgt and tgt-src datasets into a single iterable for streaming and sampling
def chain_forward_backward_datasets(hf_path, lang1_code, lang2_code):
    files_forward = f"hf://datasets/{hf_path}/{lang1_code}-{lang2_code}/*.parquet"
    files_backward = f"hf://datasets/{hf_path}/{lang2_code}-{lang1_code}/*.parquet"

    ds_forward = load_dataset(
        "parquet",
        data_files={"train": files_forward},
        split="train",
        streaming=True,
    )

    ds_backward = load_dataset(
        "parquet",
        data_files={"train": files_backward},
        split="train",
        streaming=True,
    )

    return chain(ds_forward, ds_backward)


# proper reservoir sampling so every instance is equally likely to be sampled.
def reservoir_sampling(streamed_data, sample_size):
    reservoir = []
    for i, example in enumerate(streamed_data):
        if i < sample_size:
            reservoir.append(example)
        else:
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = example
    
    return reservoir


def stream_save_random_samples(args):
    
    streamed_data = chain_forward_backward_datasets(args.hf_path, args.lang1_code, args.lang2_code)
    
    sampled_data = reservoir_sampling(streamed_data, args.n_samples)
    
    # since sampled data contains both lang1-lang2 and lang2-lang1, we need to ensure that the lang1_code and lang2_code are consistent in the output
    # treat sampled_data[0] as the reference for lang1_code and lang2_code
    print(sampled_data)
    first_item_lang1_code = sampled_data[0]["src_lang"]
    first_item_lang2_code = sampled_data[0]["tgt_lang"]
    
    for item in sampled_data:
        if item["src_lang"] == first_item_lang1_code:
            continue # this item has the same src/tgt order as the first item
        else:
            assert item["src_lang"] == first_item_lang2_code
            assert item["tgt_lang"] == first_item_lang1_code
            # swap the src and tgt code and text fields
            item["src_lang"], item["tgt_lang"] = item["tgt_lang"], item["src_lang"]
            item["source_text"], item["target_text"] = item["target_text"], item["source_text"]
    
    # write the samples to a JSONL file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(sampled_data):
            processed_item = {
                "id": i,
                "lang1_text": item.get("source_text", ""),
                "lang2_text": item.get("target_text", ""),
                "lang1_code": item.get("src_lang", ""),
                "lang2_code": item.get("tgt_lang", ""),
                "corpus": args.hf_path,
                "version": item.get("corpus", "") + "__" + item.get("version", ""),
            }
            f.write(json.dumps(processed_item, ensure_ascii=False) + "\n")
    
    print(f"Successfully streamed {args.n_samples} random samples to {args.output_file}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Create sample data for human evaluation based on src and tgt codes. Sample from both src-tgt and tgt-src datasets.")
    
    # Use the target repo ID
    arg_parser.add_argument("--hf_path", type=str, default="MaLA-LM/FineOPUS-Deduplicated")
    arg_parser.add_argument("--n_samples", type=int, default=100)
    arg_parser.add_argument("--lang1_code", type=str, default="abk_Cyrl")
    arg_parser.add_argument("--lang2_code", type=str, default="bos_Latn")
    arg_parser.add_argument("--output_folder", type=str, default="./data/samples")
    
    args = arg_parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    args.output_file = f"{args.output_folder}/sample_{args.lang1_code}_{args.lang2_code}_{args.n_samples}.jsonl"
    
    stream_save_random_samples(args)
    
    # somehow I needed multiprocessing to avoid a memory error due to data loading...
    # process = multiprocessing.Process(
    #         target=stream_save_random_samples, 
    #         args=(args,)
    #     )
    # process.start()
    # process.join() 
    