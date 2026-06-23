"""
This script is an auxiliary script to  cross-check the FineOPUS-Original dataset (original)
against the FineOPUS-Filtered-Stage4 dataset (final) to find unique items in FineOPUS-Original.

We use a Bloom Filter:
- with a false positive rate of 0.0001
- assuming a maximum of 10 billion rows in the language we want

This script is heavily **hard-coded**
 for the specific datasets mentioned above as they use different key/fields

"""

import os
import json
import argparse

import datasets
datasets.config.STREAMING_READ_MAX_RETRIES = 200
datasets.config.STREAMING_READ_RETRY_INTERVAL = 10
from datasets import load_dataset

from itertools import chain
from rbloom import Bloom

import random
random.seed(42)


SAMPLE_SIZE = 100
FALSE_POSITIVE_RATE = 0.0001
# MAX_ROWS = 36521928606 # (total rows in FineOPUS-Filtered-Stage4)
MAX_ROWS = 10000000000 # a safe estimate for deu-eng


def main(lang1_code, lang2_code="eng_Latn"):

    # as the lang1-lang2 data can be stored in either direction, we need to check both folders.
    try:
        final_forward = load_dataset(
            "parquet",
            data_files={"train": f"hf://datasets/MaLA-LM/FineOPUS-Filtered-Stage4/{lang1_code}-{lang2_code}/*.parquet"},
            split="train",
            streaming=True,
        )
    except ValueError: # if lang1-lang2 folder does not exist
        final_forward = iter([])

    try:
        final_backward = load_dataset(
            "parquet",
            data_files={"train": f"hf://datasets/MaLA-LM/FineOPUS-Filtered-Stage4/{lang2_code}-{lang1_code}/*.parquet"},
            split="train",
            streaming=True,
        )
    except ValueError: # if lang2-lang1 folder does not exist
        final_backward = iter([])

    # create a Bloom Filter to store all source texts from the final dataset (which is smaller)
    bf = Bloom(MAX_ROWS, FALSE_POSITIVE_RATE)

    # add src-tgt texts from the FineOPUS-Filtered-Stage4 dataset to the Bloom Filter
    for item in final_forward:
        bf.add(item['source_text'] + "|||" + item['target_text'])
    for item in final_backward:
        bf.add(item['target_text'] + "|||" + item['source_text'])

    print(f"Bloom Filter created from FineOPUS-Filtered-Stage4 for {lang1_code}-{lang2_code}.", flush=True)

    try:
        original_forward = load_dataset(
            "parquet",
            data_files={"train": f"hf://datasets/MaLA-LM/FineOPUS-Original/{lang1_code}-{lang2_code}/*.parquet"},
            split="train",
            streaming=True,
        )
    except ValueError: # if lang1-lang2 folder does not exist
        original_forward = iter([])

    try:
        original_backward = load_dataset(
            "parquet",
            data_files={"train": f"hf://datasets/MaLA-LM/FineOPUS-Original/{lang2_code}-{lang1_code}/*.parquet"},
            split="train",
            streaming=True,
        )
    except ValueError: # if lang2-lang1 folder does not exist
        original_backward = iter([]) 

    # reservoir sampling as we stream through the original dataset
    reservoir = []
    negative_count = 0  # Tracks the number of non-existing items

    for item in original_forward:
        src_text = item['source_text']
        tgt_text = item['target_text']
        if src_text + "|||" + tgt_text not in bf:
            negative_count += 1

            if len(reservoir) < SAMPLE_SIZE:
                reservoir.append(item)
            else:
                replace_index = random.randint(0, negative_count - 1)
                if replace_index < SAMPLE_SIZE:
                    reservoir[replace_index] = item

    for item in original_backward:
        src_text = item['source_text']
        tgt_text = item['target_text']
        if tgt_text + "|||" + src_text not in bf:
            negative_count += 1

            if len(reservoir) < SAMPLE_SIZE:
                # swap the lang1 and lang2 codes and text fields
                item["conv_src_lang"], item["conv_tgt_lang"] = item["conv_tgt_lang"], item["conv_src_lang"]
                item["source_text"], item["target_text"] = item["target_text"], item["source_text"]
                reservoir.append(item)
            else:
                replace_index = random.randint(0, negative_count - 1)
                if replace_index < SAMPLE_SIZE:
                    # swap the lang1 and lang2 codes and text fields
                    item["conv_src_lang"], item["conv_tgt_lang"] = item["conv_tgt_lang"], item["conv_src_lang"]
                    item["source_text"], item["target_text"] = item["target_text"], item["source_text"]
                    reservoir[replace_index] = item

    filename = f"annotation_samples/sample_FineOPUS-Original_unique_{lang1_code}_{lang2_code}_{SAMPLE_SIZE}.jsonl"
    with open(filename, "w", encoding="utf-8") as f:
        for i, item in enumerate(reservoir):
            processed_item = {
                    "id": i,
                    "lang1_text": item.get("source_text", ""),
                    "lang2_text": item.get("target_text", ""),
                    "lang1_code": item.get("conv_src_lang", ""),
                    "lang2_code": item.get("conv_tgt_lang", ""),
                    "corpus": "FineOPUS-Original_unique",
                    "version": item.get("corpus", "") + "__" + item.get("version", ""),
                }
            f.write(json.dumps(processed_item, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cross-check FineOPUS-Original against FineOPUS-Filtered-Stage4.")
    parser.add_argument("--lang1_code", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Starting cross-check for language: {args.lang1_code}", flush=True)
    
    main(args.lang1_code)
