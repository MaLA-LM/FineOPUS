#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cross_decomposition import CCA

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


def load_models(model_names: List[str], device: str = "cuda") -> Dict[str, Any]:
    """Load multiple sentence transformer models with error handling

    Args:
        model_names: List of model names or paths
        device: Device to load models on ('cuda' or 'cpu')

    Returns:
        Dictionary mapping model names to loaded models
    """
    models = {}
    for model_name in model_names:
        try:
            model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
            models[model_name] = model
            logging.info(f"Successfully loaded model: {model_name} on device: {device}")
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            logging.warning(f"Skipping model {model_name} due to loading failure")
    
    if not models:
        logging.error("Failed to load any models")
    else:
        logging.info(f"Successfully loaded {len(models)}/{len(model_names)} models")
    
    return models


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
        logging.info(f"Successfully loaded datasets for {source_lang} -> {target_lang}")
        logging.info(f"Source dataset size: {len(source_ds)}")
        logging.info(f"Target dataset size: {len(target_ds)}")

        return source_ds, target_ds

    except Exception as e:
        logging.error(f"Failed to load datasets {source_lang} -> {target_lang}: {e}")
        return None, None


def process_dataset(
    args: argparse.Namespace, models: Dict[str, Any], output_file: Path
) -> Tuple[int, int, int]:
    """Process dataset with ensemble of models
    
    Args:
        args: Command-line arguments
        models: Dictionary of models for ensemble
        output_file: Path to output CSV file
    """
    logging.info("Processing dataset...")
    logging.info(f"Running in ENSEMBLE mode with {len(models)} models")
    
    # Generate ensemble identifier
    model_identifier = args.ensemble_name if hasattr(args, 'ensemble_name') and args.ensemble_name else "ensemble_" + "_".join([sanitize_model_name(m) for m in models.keys()])

    ds = load_dataset(args.dataset_name, split=args.split)

    # Get available languages
    if args.target_languages:
        target_languages = args.target_languages
        logging.info(f"Using specified target languages: {target_languages}")
    else:
        all_languages = get_available_languages(ds)
        if not all_languages:
            logging.error("Could not retrieve available languages. Exiting.")
            return 0, 0, 0
        # Remove source language from target languages
        target_languages = sorted([lang for lang in all_languages if lang != args.source_lang])
    logging.info(
        f"Found {len(target_languages)} target languages (excluding source language {args.source_lang})"
    )

    # Generate language pairs
    language_pairs = [
        (args.source_lang, target_lang) for target_lang in target_languages
    ]

    # Results storage
    results = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    # Evaluate across all language pairs
    for source_lang, target_lang in language_pairs:
        logging.info(
            f"Processing language pair: {source_lang} -> {target_lang} ({processed_count + skipped_count + failed_count + 1}/{len(language_pairs)})"
        )

        # Check if already processed
        if args.skip_processed:
            if is_language_processed(
                output_file, model_identifier, (source_lang, target_lang)
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

        # Evaluate ensemble
        mrr, avg_rank = evaluate_ensemble(
            models, model_identifier, source_texts, target_texts, 
            batch_size=args.batch_size, normalize_embeddings=args.normalize_embeddings,
            use_cca=args.use_cca
        )

        results.append(
            {
                "model": model_identifier,
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


def align_embeddings_with_cca(
    embeddings_list: List[np.ndarray], n_components: Optional[int] = None
) -> List[np.ndarray]:
    """Align embeddings from different models using CCA
    
    Args:
        embeddings_list: List of embedding arrays from different models
        n_components: Number of CCA components (default: min dimension across all embeddings)
    
    Returns:
        List of aligned embeddings with same dimensions
    """
    if len(embeddings_list) <= 1:
        return embeddings_list
    
    # Determine the target dimension
    dims = [emb.shape[1] for emb in embeddings_list]
    min_dim = min(dims)
    n_samples = embeddings_list[0].shape[0]
    
    # Use conservative target dimension to improve numerical stability
    # CCA requires n_samples > n_components, and we add buffer for stability
    if n_components is None:
        # Use min of: smallest dimension, 80% of samples, or 512 (reasonable max)
        n_components = min(min_dim, int(n_samples * 0.8), 512)
    
    # Ensure n_components is less than n_samples for numerical stability
    if n_components >= n_samples:
        n_components = max(1, int(n_samples * 0.8))
        logging.warning(f"Reduced n_components to {n_components} (80% of {n_samples} samples) for numerical stability")
    
    logging.info(f"Aligning embeddings with CCA. Original dimensions: {dims}, Target: {n_components}, Samples: {n_samples}")
    
    # Normalize embeddings for better numerical stability
    normalized_embeddings = []
    for emb in embeddings_list:
        # Standardize each embedding (zero mean, unit variance)
        emb_mean = emb.mean(axis=0, keepdims=True)
        emb_std = emb.std(axis=0, keepdims=True) + 1e-8  # Add small epsilon to avoid division by zero
        normalized_emb = (emb - emb_mean) / emb_std
        normalized_embeddings.append(normalized_emb)
    
    # Use first embedding as reference
    reference_emb = normalized_embeddings[0]
    aligned_embeddings = []
    
    # Align each embedding to the reference
    for i, emb in enumerate(normalized_embeddings):
        if i == 0:
            # First embedding: just truncate or use as is
            if reference_emb.shape[1] > n_components:
                aligned_embeddings.append(reference_emb[:, :n_components])
            else:
                aligned_embeddings.append(reference_emb)
            continue
        
        try:
            # Fit CCA with improved parameters
            cca = CCA(
                n_components=n_components,
                max_iter=1000,  # Increase max iterations
                tol=1e-4,       # Slightly relaxed tolerance
                scale=True      # Scale inputs
            )
            cca.fit(reference_emb, emb)
            
            # Transform embeddings to aligned space
            _, emb_transformed = cca.transform(reference_emb, emb)
            
            aligned_embeddings.append(emb_transformed)
            logging.info(f"Aligned embedding {i+1}/{len(embeddings_list)} using CCA (converged)")
        except Exception as e:
            logging.warning(f"CCA failed for embedding {i+1} with {n_components} components: {e}")
            
            # Try with reduced dimensions
            reduced_components = min(n_components // 2, int(n_samples * 0.5))
            logging.info(f"Retrying with reduced dimensions: {reduced_components}")
            
            try:
                cca = CCA(
                    n_components=reduced_components,
                    max_iter=1000,
                    tol=1e-3,
                    scale=True
                )
                cca.fit(reference_emb, emb)
                _, emb_transformed = cca.transform(reference_emb, emb)
                
                # Pad to match target dimension
                if emb_transformed.shape[1] < n_components:
                    padded = np.zeros((emb_transformed.shape[0], n_components))
                    padded[:, :emb_transformed.shape[1]] = emb_transformed
                    emb_transformed = padded
                
                aligned_embeddings.append(emb_transformed)
                logging.info(f"Aligned embedding {i+1}/{len(embeddings_list)} using CCA with reduced dimensions (converged)")
            except Exception as e2:
                logging.error(f"CCA failed again for embedding {i+1}: {e2}. Using truncation fallback.")
                # Final fallback: simple truncation
                if emb.shape[1] > n_components:
                    aligned_embeddings.append(emb[:, :n_components])
                else:
                    padded = np.zeros((emb.shape[0], n_components))
                    padded[:, :emb.shape[1]] = emb
                    aligned_embeddings.append(padded)
                logging.info(f"Used truncation fallback for embedding {i+1}/{len(embeddings_list)}")
    
    return aligned_embeddings


def encode_texts(
    model: Any, model_name: str, texts: List[str], batch_size: int = 32, normalize_embeddings: bool = False
) -> Optional[np.ndarray]:
    """Encode texts with batch processing for memory efficiency"""
    try:
        if model_name == "google/embeddinggemma-300m":
            embeddings = model.encode_document(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize_embeddings)
        elif model_name == "jinaai/jina-embeddings-v3":
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, task="text-matching", normalize_embeddings=normalize_embeddings)
        elif model_name == "Qwen/Qwen3-Embedding-0.6B":
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize_embeddings)
        elif model_name == "intfloat/multilingual-e5-large":
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize_embeddings)            
        elif model_name == "Alibaba-NLP/gte-multilingual-base":
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize_embeddings)
        else:
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize_embeddings)

        logging.info(f"Successfully encoded {len(texts)} texts with {model_name}")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to encode texts with {model_name}: {e}")
        return None


def encode_texts_ensemble(
    models: Dict[str, Any], texts: List[str], batch_size: int = 32, 
    normalize_embeddings: bool = False, use_cca: bool = True
) -> Optional[np.ndarray]:
    """Encode texts using ensemble of models with mean aggregation
    
    Args:
        models: Dictionary mapping model names to model objects
        texts: List of texts to encode
        batch_size: Batch size for encoding
        normalize_embeddings: Whether to normalize embeddings
        use_cca: Whether to use CCA for dimension alignment
    
    Returns:
        Mean of aligned embeddings from all models
    """
    if not models:
        logging.error("No models provided for ensemble encoding")
        return None
    
    embeddings_list = []
    successful_models = []
    
    # Encode with each model
    for model_name, model in models.items():
        emb = encode_texts(model, model_name, texts, batch_size, normalize_embeddings)
        if emb is not None:
            embeddings_list.append(emb)
            successful_models.append(model_name)
            logging.info(f"Encoded with {model_name}, shape: {emb.shape}")
    
    if not embeddings_list:
        logging.error("Failed to encode with any model in ensemble")
        return None
    
    logging.info(f"Successfully encoded with {len(embeddings_list)}/{len(models)} models")
    
    # Check if dimensions are different
    dims = [emb.shape[1] for emb in embeddings_list]
    if len(set(dims)) > 1:
        logging.info(f"Different embedding dimensions detected: {dims}")
        if use_cca:
            logging.info("Applying CCA alignment")
            embeddings_list = align_embeddings_with_cca(embeddings_list)
        else:
            # Fallback: truncate to minimum dimension
            min_dim = min(dims)
            logging.info(f"Truncating all embeddings to dimension {min_dim}")
            embeddings_list = [emb[:, :min_dim] for emb in embeddings_list]
    
    # Mean aggregation
    ensemble_embeddings = np.mean(embeddings_list, axis=0)
    logging.info(f"Ensemble embeddings shape: {ensemble_embeddings.shape}")
    
    return ensemble_embeddings


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


def evaluate_ensemble(
    models: Dict[str, Any],
    ensemble_name: str,
    source_texts: List[str],
    target_texts: List[str],
    batch_size: int = 32,
    normalize_embeddings: bool = False,
    use_cca: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """Evaluate ensemble of models"""
    logging.info(f"Evaluating ensemble: {ensemble_name} with {len(models)} models")

    # Encode texts with ensemble
    source_emb = encode_texts_ensemble(models, source_texts, batch_size=batch_size, 
                                       normalize_embeddings=normalize_embeddings, use_cca=use_cca)
    target_emb = encode_texts_ensemble(models, target_texts, batch_size=batch_size, 
                                       normalize_embeddings=normalize_embeddings, use_cca=use_cca)

    if source_emb is None or target_emb is None:
        logging.error(f"Failed to encode texts for ensemble: {ensemble_name}")
        return None, None

    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(source_emb, target_emb)

    # Get ranks
    ranks = get_correct_translation_ranks(sims)

    # Calculate metrics
    mrr, avg_rank = calculate_metrics(ranks)

    logging.info(f"{ensemble_name} - MRR: {mrr:.4f}, Avg Rank: {avg_rank:.2f}")

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
    parser = argparse.ArgumentParser(description="Evaluate ensemble of sentence transformer models")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Multiple models for ensemble (space-separated, minimum 1 model)",
    )
    parser.add_argument(
        "--ensemble_name",
        type=str,
        help="Name for the ensemble (optional, auto-generated if not provided)",
    )
    parser.add_argument(
        "--use_cca",
        action="store_true",
        default=True,
        help="Use CCA for dimension alignment (default: True)",
    )
    parser.add_argument(
        "--no_cca",
        action="store_true",
        help="Disable CCA and use truncation for dimension alignment",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="devtest",
        help="Dataset split (default: 'devtest')",
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default="eng_Latn",
        help="Source language code",
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
        "--skip_processed",
        action="store_true",
        help="Skip language pairs that have already been processed",
    )
    parser.add_argument(
        "--target_languages",
        type=str,
        nargs="*",
        help="Specific target languages to evaluate (if not provided, evaluates all)",
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="Normalize embeddings (default: False)",
    )

    args = parser.parse_args()
    
    # Handle CCA flag
    if args.no_cca:
        args.use_cca = False
    
    logging.info("=" * 80)
    logging.info("ENSEMBLE MODEL EVALUATION")
    logging.info("=" * 80)
    logging.info(f"  Dataset Name: {args.dataset_name}")
    logging.info(f"  Models: {args.models}")
    logging.info(f"  Ensemble Name: {args.ensemble_name if args.ensemble_name else 'auto-generated'}")
    logging.info(f"  Use CCA: {args.use_cca}")
    logging.info(f"  Source Language: {args.source_lang}")
    logging.info(f"  Output Directory: {args.output_dir}")
    logging.info(f"  Batch Size: {args.batch_size}")
    logging.info(f"  Skip Processed: {args.skip_processed}")
    logging.info(f"  Target Languages: {args.target_languages}")
    logging.info(f"  Normalize Embeddings: {args.normalize_embeddings}")
    logging.info("=" * 80)

    # Setup environment
    device = setup_environment()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models for ensemble
    models = args.models[0].split(" ")
    logging.info(f"Loading {len(models)} models for ensemble...")
    models_dict = load_models(models, device=device)
    
    if not models_dict:
        logging.error("Failed to load any models for ensemble. Exiting.")
        sys.exit(1)
    
    # Create output filename for ensemble
    dataset_basename = (
        args.dataset_name.split("/")[-1]
        if "/" in args.dataset_name
        else args.dataset_name
    )
    if args.ensemble_name:
        sanitized_name = sanitize_model_name(args.ensemble_name)
    else:
        sanitized_name = "ensemble_" + "_".join([sanitize_model_name(m) for m in models])
    output_file = output_dir / f"{dataset_basename}_{sanitized_name}.csv"

    # Process dataset
    processed_count, skipped_count, failed_count = process_dataset(
        args, models_dict, output_file
    )

    # Clean up models
    for m in models_dict.values():
        del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print summary
    logging.info("\n" + "=" * 80)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Ensemble with {len(models)} models")
    logging.info(f"Models: {', '.join(models)}")
    logging.info(f"Dataset: {args.dataset_name}")
    logging.info(f"Source Language: {args.source_lang}")
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
            # Get the model identifier used
            if args.ensemble_name:
                model_identifier = args.ensemble_name
            else:
                model_identifier = df["model"].iloc[0] if not df.empty else None
            
            if model_identifier:
                model_results = df[df["model"] == model_identifier]
                
                if not model_results.empty:
                    logging.info("\nFINAL RESULTS:")
                    logging.info(model_results.to_string(index=False))
        except Exception as e:
            logging.error(f"Could not display final results: {e}")

    logging.info("=" * 80)


if __name__ == "__main__":
    main()
