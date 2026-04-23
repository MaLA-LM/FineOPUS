#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect gold-pair similarity statistics for every (best_model, source_lang,
target_lang) triple listed in best_model_per_lang_pair_*.csv.

For each triple, we:
  1. Pull parallel gold sentences from FLORES-200 and BOUQuET_Sentence.
  2. Encode them with the assigned embedding model (same prompts / options
     as compute_similarity.py, so the scale matches the scored parquet data).
  3. Compute per-row cosine similarity and summarise the distribution
     (mean, std, min, max, p01, p05, p10, p25, p50, p75, p90, p95, p99).

The script is model-centric: a single invocation loads one model and handles
all (src, tgt) pairs assigned to it. Submit one SLURM job per model.

Usage:
  python collect_gold_stats.py \
      --model microsoft/harrier-oss-v1-0.6b \
      --best_model_csv ../results/best_model_per_lang_pair_by_flores_bouquet_combined_selected_models.csv \
      --datasets Zihao-Li/FLORES-200 Zihao-Li/BOUQuET_Sentence \
      --output stats/gold_score_stats.csv \
      --batch_size 64
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


JINA_MODELS = {
    "jinaai/jina-embeddings-v3",
    "jinaai/jina-embeddings-v5-text-nano",
    "jinaai/jina-embeddings-v5-text-small",
}
HARRIER_MODELS = {
    "microsoft/harrier-oss-v1-0.6b",
    "microsoft/harrier-oss-v1-270m",
}

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
QUANTILE_COLS = [f"p{int(q * 100):02d}" for q in QUANTILES]
OUTPUT_COLUMNS = [
    "model", "source_lang", "target_lang",
    "n_pairs", "n_datasets",
    "mean", "std", "min", "max",
    *QUANTILE_COLS,
]


def setup_environment() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("CUDA not available, using CPU")
    os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return device


def load_model(model_name: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading model: {model_name} on {device}")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    logger.info("Model loaded.")
    return model


def encode_texts(
    model: Any, model_name: str, texts: List[str], batch_size: int = 64, normalize: bool = True,
) -> np.ndarray:
    """Mirror compute_similarity.py to ensure identical embedding conventions."""
    if model_name == "google/embeddinggemma-300m" or model_name == "codefuse-ai/F2LLM-v2-0.6B":
        emb = model.encode_document(
            texts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=normalize,
        )
    elif model_name in HARRIER_MODELS:
        emb = model.encode(
            texts, prompt_name="sts_query", batch_size=batch_size,
            show_progress_bar=False, normalize_embeddings=normalize,
        )
    elif model_name in JINA_MODELS:
        emb = model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            task="text-matching", normalize_embeddings=normalize,
        )
    else:
        emb = model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=normalize,
        )
    return np.asarray(emb, dtype=np.float32)


def cosine_similarity_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.clip((a_norm * b_norm).sum(axis=1), -1.0, 1.0).astype(np.float32)


def quantile_stats(sims: np.ndarray) -> Dict[str, float]:
    out = {
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "min": float(sims.min()),
        "max": float(sims.max()),
    }
    qs = np.quantile(sims, QUANTILES)
    for name, q in zip(QUANTILE_COLS, qs):
        out[name] = float(q)
    return out


def _load_one_dataset(name: str, split: str) -> Any:
    """Try the given split first, then fall back to common alternatives."""
    from datasets import load_dataset
    candidates = [split, "test", "devtest", "validation", "train"]
    seen = set()
    last_err = None
    for s in candidates:
        if s in seen:
            continue
        seen.add(s)
        try:
            return load_dataset(name, split=s)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load any split from {name}. Last error: {last_err}")


def load_lang_texts(datasets: List[str], split: str) -> Dict[Tuple[str, str], List[str]]:
    """Return {(dataset, lang): [text_in_canonical_row_order]} for every dataset/lang.

    For FLORES-200 and BOUQuET_Sentence, rows are index-aligned across languages:
    the i-th text in each language is a translation of the same source sentence.
    """
    texts: Dict[Tuple[str, str], List[str]] = {}
    for name in datasets:
        logger.info(f"Loading dataset: {name}")
        try:
            ds = _load_one_dataset(name, split)
        except Exception as e:
            logger.warning(f"  Skipping {name}: {e}")
            continue
        cols = set(ds.column_names)
        if "lang" not in cols or "text" not in cols:
            logger.warning(f"  Dataset {name} missing 'lang' / 'text' columns; skipping.")
            continue
        logger.info(f"  {name}: {len(ds)} rows, {len(set(ds['lang']))} languages")
        by_lang: Dict[str, List[str]] = {}
        for row in ds:
            by_lang.setdefault(row["lang"], []).append(row["text"])
        for lang, lst in by_lang.items():
            texts[(name, lang)] = lst
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--best_model_csv",
        default=str(Path(__file__).resolve().parent.parent
                    / "results/best_model_per_lang_pair_by_flores_bouquet_combined_selected_models.csv"),
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["Zihao-Li/FLORES-200", "Zihao-Li/BOUQuET_Sentence"],
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "stats/gold_score_stats.csv"),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split", default="test")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"Model          : {args.model}")
    logger.info(f"Best model CSV : {args.best_model_csv}")
    logger.info(f"Datasets       : {args.datasets}")
    logger.info(f"Output         : {args.output}")
    logger.info(f"Batch size     : {args.batch_size}")
    logger.info("=" * 70)

    # Filter the best-model table to just this model
    df = pd.read_csv(args.best_model_csv)
    df = df[df["model"] == args.model].copy()
    df = df[df["source_lang"] != df["target_lang"]].reset_index(drop=True)
    logger.info(f"Assigned pairs for this model: {len(df):,}")
    if df.empty:
        logger.info("Nothing to do for this model. Exiting.")
        return

    # Resume support
    out_path = Path(args.output)
    done: set = set()
    if args.skip_existing and out_path.exists():
        try:
            prev = pd.read_csv(out_path)
            done = set(zip(
                prev["model"].astype(str),
                prev["source_lang"].astype(str),
                prev["target_lang"].astype(str),
            ))
        except Exception:
            pass
    if done:
        logger.info(f"Resume: skipping {len(done):,} pre-existing rows.")

    # Load gold sentences (indexed by dataset & language code)
    lang_texts = load_lang_texts(args.datasets, args.split)

    # Collect every language code we need to encode
    needed_langs = set(df["source_lang"]).union(df["target_lang"])
    logger.info(f"Languages to encode: {len(needed_langs)}")

    # Cache embeddings per (dataset, lang)
    device = setup_environment()
    model = load_model(args.model, device)

    emb_cache: Dict[Tuple[str, str], np.ndarray] = {}

    def get_emb(dataset: str, lang: str) -> np.ndarray:
        key = (dataset, lang)
        if key in emb_cache:
            return emb_cache[key]
        texts = lang_texts.get(key)
        if not texts:
            emb_cache[key] = np.zeros((0, 0), dtype=np.float32)
            return emb_cache[key]
        emb = encode_texts(model, args.model, texts, batch_size=args.batch_size, normalize=True)
        emb_cache[key] = emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return emb

    # Process pairs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_path.exists()
    with open(out_path, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=OUTPUT_COLUMNS)
        if not file_exists:
            writer.writeheader()
            fp.flush()

        processed = 0
        skipped = 0
        for i, row in df.iterrows():
            src = row["source_lang"]
            tgt = row["target_lang"]
            key = (args.model, src, tgt)
            if key in done:
                skipped += 1
                continue

            sims_chunks: List[np.ndarray] = []
            n_datasets_used = 0
            for dataset in args.datasets:
                src_texts = lang_texts.get((dataset, src))
                tgt_texts = lang_texts.get((dataset, tgt))
                if not src_texts or not tgt_texts:
                    continue
                # Align on common length (should match for FLORES/BOUQuET)
                n = min(len(src_texts), len(tgt_texts))
                if n == 0:
                    continue
                src_emb = get_emb(dataset, src)[:n]
                tgt_emb = get_emb(dataset, tgt)[:n]
                if src_emb.shape[0] == 0 or tgt_emb.shape[0] == 0:
                    continue
                sims_chunks.append(cosine_similarity_rowwise(src_emb, tgt_emb))
                n_datasets_used += 1

            if not sims_chunks:
                logger.warning(f"  {src}->{tgt}: no gold data found in any dataset.")
                continue

            sims = np.concatenate(sims_chunks)
            stats = quantile_stats(sims)
            writer.writerow({
                "model": args.model,
                "source_lang": src,
                "target_lang": tgt,
                "n_pairs": int(sims.size),
                "n_datasets": n_datasets_used,
                **stats,
            })
            fp.flush()

            processed += 1
            if processed % 50 == 0 or processed == 1:
                logger.info(
                    f"[{processed}/{len(df) - skipped}] {src}->{tgt}: "
                    f"n={sims.size}, mean={stats['mean']:.3f}, "
                    f"p05={stats['p05']:.3f}, p50={stats['p50']:.3f}"
                )

    logger.info(f"Done. Processed={processed}, skipped={skipped}.")


if __name__ == "__main__":
    main()
