#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from typing import List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def setup_environment() -> str:
    """Setup environment for LUMI supercomputer

    Returns:
        Device string ('cuda' or 'cpu')
    """
    # Set CUDA device if available
    if torch.cuda.is_available():
        device = "cuda"
        logging.info(f"Using CUDA device: {torch.cuda.current_device()}")
        logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logging.info("CUDA not available, using CPU")

    # Set environment variables for better performance on LUMI
    os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count()))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings

    return device


def load_model(model_name: str, device: str = "cuda") -> Optional[Any]:
    """Load sentence transformer model with error handling

    Args:
        model_name: Name or path of the model
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded model or None if failed
    """
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        logging.info(f"Successfully loaded model: {model_name} on device: {device}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}")
        return None


def get_available_languages(ds: Any) -> List[str]:
    """Get all available language codes for dataset"""
    try:
        language_codes = set(ds["lang"])
        logging.info(f"Found {len(language_codes)} languages in dataset")
        return list(language_codes)

    except Exception as e:
        logging.error(f"Failed to get available languages for dataset: {e}")
        return []


def load_datasets(
    ds: Any, source_lang: str, target_lang: str
) -> Tuple[Optional[Any], Optional[Any]]:
    """Load both source and target language datasets"""
    try:
        source_ds = ds.filter(lambda x: x["lang"] == source_lang)
        target_ds = ds.filter(lambda x: x["lang"] == target_lang)
        # logging.info(f"Successfully loaded datasets for {source_lang} -> {target_lang}")
        # logging.info(f"Source dataset size: {len(source_ds)}")
        # logging.info(f"Target dataset size: {len(target_ds)}")

        return source_ds, target_ds

    except Exception as e:
        logging.error(f"Failed to load datasets {source_lang} -> {target_lang}: {e}")
        return None, None


def parse_language_pairs_file(file_path: str) -> List[Tuple[str, str]]:
    """Parse language pairs from a file
    
    Each line should be in format: source_lang-target_lang (e.g., zho_Hans-eng_Latn)
    
    Args:
        file_path: Path to the language pairs file
        
    Returns:
        List of (source_lang, target_lang) tuples
    """
    language_pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                
                if '-' not in line:
                    logging.warning(f"Line {line_num}: Invalid format '{line}', expected 'source_lang-target_lang'. Skipping.")
                    continue
                
                parts = line.split('-', 1)  # Split only on first '-' in case lang codes contain '-'
                if len(parts) != 2:
                    logging.warning(f"Line {line_num}: Invalid format '{line}'. Skipping.")
                    continue
                
                source_lang, target_lang = parts[0].strip(), parts[1].strip()
                if source_lang and target_lang:
                    language_pairs.append((source_lang, target_lang))
                else:
                    logging.warning(f"Line {line_num}: Empty language code in '{line}'. Skipping.")
                    
        logging.info(f"Loaded {len(language_pairs)} language pairs from {file_path}")
        return language_pairs
    except FileNotFoundError:
        logging.error(f"Language pairs file not found: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Failed to read language pairs file: {e}")
        return []


def process_dataset(
    args: argparse.Namespace, model: Any, output_file: Path
) -> Tuple[int, int, int]:
    """Process dataset"""
    logging.info("Processing dataset...")

    ds = load_dataset(args.dataset_name, split=args.split)

    # Get available languages in the dataset
    available_languages = set(get_available_languages(ds))
    if not available_languages:
        logging.error("Could not retrieve available languages. Exiting.")
        return 0, 0, 0
    logging.info(f"Dataset has {len(available_languages)} available languages")

    # Load language pairs from file
    language_pairs = parse_language_pairs_file(args.language_pairs_file)
    if not language_pairs:
        logging.error("No valid language pairs found. Exiting.")
        return 0, 0, 0
    
    # Filter language pairs to only include those available in the dataset
    valid_language_pairs = []
    for source_lang, target_lang in language_pairs:
        if source_lang not in available_languages:
            logging.warning(f"Source language '{source_lang}' not in dataset. Skipping pair {source_lang}-{target_lang}.")
            continue
        if target_lang not in available_languages:
            logging.warning(f"Target language '{target_lang}' not in dataset. Skipping pair {source_lang}-{target_lang}.")
            continue
        valid_language_pairs.append((source_lang, target_lang))
    
    logging.info(f"Found {len(valid_language_pairs)} valid language pairs out of {len(language_pairs)} total pairs")
    
    if not valid_language_pairs:
        logging.error("No valid language pairs available in dataset. Exiting.")
        return 0, 0, 0

    # Results storage
    results = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    # Evaluate across all language pairs
    for source_lang, target_lang in valid_language_pairs:
        logging.info(
            f"Processing language pair: {source_lang} -> {target_lang} ({processed_count + skipped_count + failed_count + 1}/{len(valid_language_pairs)})"
        )

        # Check if already processed
        if args.skip_processed:
            if is_language_processed(
                output_file, args.model, (source_lang, target_lang)
            ):
                logging.info(
                    f"Language pair {source_lang} -> {target_lang} already processed. Skipping."
                )
                skipped_count += 1
                continue

        # Load datasets for this language pair
        source_ds, target_ds = load_datasets(ds, source_lang, target_lang)
        if source_ds is None or target_ds is None:
            logging.warning(
                f"Failed to load datasets for language pair {source_lang} -> {target_lang}. Skipping."
            )
            failed_count += 1
            continue

        # Get source and target texts
        try:
            source_texts = list(source_ds["text"])
            target_texts = list(target_ds["text"])
            logging.info(
                f"Successfully extracted texts. Source: {len(source_texts)}, Target: {len(target_texts)}"
            )
        except Exception as e:
            logging.error(f"Failed to extract texts from datasets: {e}")
            failed_count += 1
            continue

        # Regular evaluation
        mrr, avg_rank = evaluate_model(
            model, args.model, source_texts, target_texts, batch_size=args.batch_size, normalize_embeddings=args.normalize_embeddings
        )

        results.append(
            {
                "model": args.model,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "MRR": mrr,
                "avg_rank": avg_rank,
            }
        )

        processed_count += 1

        # Save results after each language to avoid losing progress
        if results:
            save_results(output_file, results)
            results = []  # Clear the buffer

        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save any remaining results
    if results:
        save_results(output_file, results)

    return processed_count, skipped_count, failed_count


def encode_texts(
    model: Any, model_name: str, texts: List[str], batch_size: int = 32, normalize_embeddings: bool = False
) -> Optional[np.ndarray]:
    """Encode texts with batch processing for memory efficiency"""
    JINA_MODELS = {"jinaai/jina-embeddings-v3", "jinaai/jina-embeddings-v5-text-nano", "jinaai/jina-embeddings-v5-text-small"}
    HARRIER_MODELS = {"microsoft/harrier-oss-v1-0.6b", "microsoft/harrier-oss-v1-270m"}
    try:
        if model_name == "google/embeddinggemma-300m" or model_name == "codefuse-ai/F2LLM-v2-0.6B":
            embeddings = model.encode_document(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize_embeddings)
        elif model_name in HARRIER_MODELS:
            embeddings = model.encode(texts, prompt_name="sts_query", batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize_embeddings)
        elif model_name in JINA_MODELS:
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, task="text-matching", normalize_embeddings=normalize_embeddings)
        else:
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize_embeddings)

        # logging.info(f"Successfully encoded {len(texts)} texts")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to encode texts: {e}")
        return None


def get_correct_translation_ranks(sims: np.ndarray) -> List[int]:
    """Calculate ranks of correct translations"""
    argsorts = sims.argsort(axis=1)
    correct_translation_ranks = []

    for i, argsort in enumerate(argsorts):
        # Find rank of correct translation (diagonal element)
        correct_translation_rank = np.where(argsort[::-1] == i)[0][0] + 1
        correct_translation_ranks.append(correct_translation_rank)

    return correct_translation_ranks


def calculate_metrics(ranks: List[int]) -> Tuple[float, float]:
    """Calculate MRR and average rank"""
    ranks_array = np.array(ranks)
    mrr = (1 / ranks_array).mean()
    avg_rank = ranks_array.mean()
    return mrr, avg_rank


def evaluate_model(
    model: Any,
    model_name: str,
    source_texts: List[str],
    target_texts: List[str],
    batch_size: int = 32,
    normalize_embeddings: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """Evaluate a single model"""
    logging.info(f"Evaluating model: {model_name}")

    # Encode texts
    source_emb = encode_texts(model, model_name, source_texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings)
    target_emb = encode_texts(model, model_name, target_texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings)

    if source_emb is None or target_emb is None:
        logging.error(f"Failed to encode texts for model: {model_name}")
        return None, None

    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(source_emb, target_emb)

    # Get ranks
    ranks = get_correct_translation_ranks(sims)

    # Calculate metrics
    mrr, avg_rank = calculate_metrics(ranks)

    logging.info(f"{model_name} - MRR: {mrr:.4f}, Avg Rank: {avg_rank:.2f}")

    return mrr, avg_rank


def is_language_processed(
    output_file: Path, model_name: str, identifier: Tuple[str, str]
) -> bool:
    """Check if a language pair has already been processed for the given model"""

    if not output_file.exists():
        return False

    try:
        df = pd.read_csv(output_file)

        # Check source_lang and target_lang
        source_lang, target_lang = identifier
        mask = (
            (df["model"] == model_name)
            & (df["source_lang"] == source_lang)
            & (df["target_lang"] == target_lang)
        )

        return mask.any()
    except Exception as e:
        logging.warning(f"Could not read existing results file: {e}")
        return False


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in filename"""
    # Replace problematic characters with underscores
    sanitized = model_name.replace("/", "_").replace(":", "_").replace("-", "_")
    # Remove any other problematic characters
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    # Remove consecutive underscores and strip leading/trailing underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def save_results(output_file: Path, new_results: List[dict]) -> None:
    """Save results to CSV, appending to existing file if it exists"""
    df_new = pd.DataFrame(new_results)

    if output_file.exists():
        try:
            df_existing = pd.read_csv(output_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as e:
            logging.warning(f"Could not read existing file, creating new one: {e}")
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_csv(output_file, index=False)
    logging.info(f"Results saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentence transformer model")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Pretrained sentence transformer model to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="devtest",
        help="Dataset split (default: 'devtest')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results (default: './results')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--language_pairs_file",
        type=str,
        required=True,
        help="Path to file containing language pairs (one per line, format: source_lang-target_lang)",
    )
    parser.add_argument(
        "--skip_processed",
        action="store_true",
        help="Skip language pairs that have already been processed",
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="Normalize embeddings (default: False)",
    )

    args = parser.parse_args()

    logging.info("Arguments:")
    logging.info(f"  Dataset Name: {args.dataset_name}")
    logging.info(f"  Model: {args.model}")
    logging.info(f"  Output Directory: {args.output_dir}")
    logging.info(f"  Batch Size: {args.batch_size}")
    logging.info(f"  Language Pairs File: {args.language_pairs_file}")
    logging.info(f"  Skip Processed: {args.skip_processed}")
    logging.info(f"  Normalize Embeddings: {args.normalize_embeddings}")

    # Setup environment
    device = setup_environment()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename with sanitized names
    dataset_basename = (
        args.dataset_name.split("/")[-1]
        if "/" in args.dataset_name
        else args.dataset_name
    )
    sanitized_model_name = sanitize_model_name(args.model)
    output_file = output_dir / f"{dataset_basename}_{sanitized_model_name}.csv"

    # Load model once
    logging.info(f"Loading model: {args.model}")
    model = load_model(args.model, device=device)

    if model is None:
        logging.error("Failed to load model. Exiting.")
        sys.exit(1)

    # Process dataset
    processed_count, skipped_count, failed_count = process_dataset(
        args, model, output_file
    )

    # Clean up model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print summary
    logging.info("\n" + "=" * 80)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset_name}")
    logging.info(f"Language Pairs File: {args.language_pairs_file}")
    logging.info(
        f"Total language pairs: {processed_count + skipped_count + failed_count}"
    )
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Skipped (already processed): {skipped_count}")
    logging.info(f"Failed: {failed_count}")
    logging.info(f"Results saved to: {output_file}")

    # Display final results
    if output_file.exists():
        try:
            df = pd.read_csv(output_file)
            model_results = df[df["model"] == args.model]

            if not model_results.empty:
                logging.info("\nFINAL RESULTS:")
                logging.info(model_results.to_string(index=False))
        except Exception as e:
            logging.error(f"Could not display final results: {e}")

    logging.info("=" * 80)


if __name__ == "__main__":
    main()
