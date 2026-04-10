#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large-scale per-row embedding similarity scoring for FineOPUS-Filtered-Stage2.

For each language pair assigned to this job:
  - Read {input_dir}/{lang_pair}/{lang_pair}_shard_0.parquet
  - Encode source_text and target_text with the assigned model
  - Compute per-row cosine similarity
  - Write output parquet to {output_dir}/{lang_pair}/{lang_pair}_shard_0.parquet
    with an added 'similarity_score' column

Usage:
  python compute_similarity.py \
    --model microsoft/harrier-oss-v1-0.6b \
    --chunk_id 0 \
    --total_chunks 625 \
    --input_dir /scratch/.../FineOPUS-Filtered-Stage2 \
    --output_dir /scratch/.../FineOPUS-Filtered-Stage2-Scored \
    --model_pairs_json /path/to/model_to_language_pairs.json \
    --batch_size 64
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment / model helpers (adapted from benchmarking.py)
# ---------------------------------------------------------------------------

JINA_MODELS = {
    "jinaai/jina-embeddings-v3",
    "jinaai/jina-embeddings-v5-text-nano",
    "jinaai/jina-embeddings-v5-text-small",
}
HARRIER_MODELS = {
    "microsoft/harrier-oss-v1-0.6b",
    "microsoft/harrier-oss-v1-270m",
}


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
    model: Any,
    model_name: str,
    texts: List[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a list of texts to embeddings, returning a float32 numpy array."""
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
    """Element-wise cosine similarity, guaranteed in [-1, 1].

    float32 arithmetic can produce values marginally outside [-1, 1] (e.g.
    1.0000002) when source and target embeddings are identical. np.clip ensures
    the output is always a valid cosine similarity value.
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.clip((a_norm * b_norm).sum(axis=1), -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

def get_shard_paths(base_dir: Path, lang_pair: str) -> List[Path]:
    """Return all shard parquet files for a language pair, sorted by shard index."""
    pair_dir = base_dir / lang_pair
    if not pair_dir.exists():
        return []
    shards = sorted(pair_dir.glob(f"{lang_pair}_shard_*.parquet"))
    return shards


def is_already_processed(output_path: Path) -> bool:
    """Return True if the output parquet exists and has 'similarity_score' column.

    Uses pyarrow.parquet.read_schema() which only reads the file footer metadata
    (no row data loaded), so this check is fast even for large shards.
    A corrupt/partial file will raise an exception and return False, triggering reprocessing.
    """
    if not output_path.exists():
        return False
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(output_path)
        return "similarity_score" in schema.names
    except Exception:
        return False


def process_shard(
    shard_input: Path,
    shard_output: Path,
    model: Any,
    model_name: str,
    batch_size: int,
    lang_pair: str,
) -> bool:
    """Process a single shard using two-pass streaming to keep RAM usage low.

    Pass 1 – reads only source_text / target_text in batches, encodes them,
              and collects similarity scores (N float32 values, tiny).
    Pass 2 – streams all columns in batches, appends the similarity column,
              and writes the output parquet without ever loading the full shard.

    This allows arbitrarily large shards to be processed within a fixed RAM
    budget (controlled by READ_ROWS below).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if is_already_processed(shard_output):
        logger.info(f"  [{shard_input.name}] Already processed. Skipping.")
        return True

    # Validate schema upfront (fast – only reads file footer metadata)
    try:
        schema = pq.read_schema(shard_input)
    except Exception as e:
        logger.error(f"  [{shard_input.name}] Cannot read schema: {e}")
        return False

    if "source_text" not in schema.names or "target_text" not in schema.names:
        logger.error(f"  [{shard_input.name}] Missing source_text/target_text columns. Skipping.")
        return False

    # Read total row count from file footer metadata (no row data loaded)
    try:
        total_rows_expected = pq.read_metadata(shard_input).num_rows
    except Exception:
        total_rows_expected = None

    # Rows read per streaming batch.  Each batch holds at most
    # READ_ROWS × avg_text_size in RAM while encoding.
    READ_ROWS = batch_size * 64  # e.g. 64 × 64 = 4096 rows per batch

    # ------------------------------------------------------------------
    # Pass 1: stream text columns only → encode → collect similarities
    # ------------------------------------------------------------------
    total_str = f"{total_rows_expected:,}" if total_rows_expected else "?"
    logger.info(f"  [{shard_input.name}] Pass 1: encoding {total_str} rows ...")
    all_sims: List[np.ndarray] = []
    total_rows = 0
    try:
        for batch in pq.ParquetFile(shard_input).iter_batches(
            batch_size=READ_ROWS, columns=["source_text", "target_text"]
        ):
            src = [s or "" for s in batch.column("source_text").to_pylist()]
            tgt = [t or "" for t in batch.column("target_text").to_pylist()]
            src_emb = encode_texts(model, model_name, src, batch_size=batch_size, normalize=True)
            tgt_emb = encode_texts(model, model_name, tgt, batch_size=batch_size, normalize=True)
            sims = cosine_similarity_rowwise(src_emb, tgt_emb)
            all_sims.append(sims)
            total_rows += len(sims)
            del src_emb, tgt_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if total_rows_expected:
                pct = total_rows / total_rows_expected * 100
                logger.info(
                    f"  [{shard_input.name}] Pass 1: {total_rows:,}/{total_rows_expected:,} rows ({pct:.1f}%)"
                )
            else:
                logger.info(f"  [{shard_input.name}] Pass 1: {total_rows:,} rows encoded")
    except Exception as e:
        logger.error(f"  [{shard_input.name}] Encoding failed: {e}")
        return False

    similarities = np.concatenate(all_sims).astype(np.float32)
    logger.info(
        f"  [{shard_input.name}] Encoded {total_rows:,} rows. "
        f"mean={similarities.mean():.4f}, min={similarities.min():.4f}, max={similarities.max():.4f}"
    )

    # ------------------------------------------------------------------
    # Pass 2: stream all columns → append similarity column → write
    # ------------------------------------------------------------------
    logger.info(f"  [{shard_input.name}] Pass 2: writing output ...")
    shard_output.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    row_idx = 0
    try:
        for batch in pq.ParquetFile(shard_input).iter_batches(batch_size=50_000):
            n = len(batch)
            score_col = pa.array(
                similarities[row_idx : row_idx + n], type=pa.float32()
            )
            row_idx += n
            new_batch = batch.append_column(
                pa.field("similarity_score", pa.float32()), score_col
            )
            if writer is None:
                writer = pq.ParquetWriter(shard_output, new_batch.schema, compression="zstd")
            writer.write_batch(new_batch)
    except Exception as e:
        logger.error(f"  [{shard_input.name}] Write failed: {e}")
        if shard_output.exists():
            shard_output.unlink()  # remove incomplete file so next run retries
        return False
    finally:
        if writer:
            writer.close()

    del similarities
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def process_pair(
    lang_pair: str,
    model: Any,
    model_name: str,
    input_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> bool:
    """
    Process all shards for a single language pair.
    Returns True if all shards succeeded (or were already done), False otherwise.
    """
    input_shards = get_shard_paths(input_dir, lang_pair)
    if not input_shards:
        logger.warning(f"[{lang_pair}] No shards found in {input_dir / lang_pair}. Skipping.")
        return False

    logger.info(f"[{lang_pair}] Found {len(input_shards)} shard(s).")
    all_ok = True
    for shard_in in input_shards:
        shard_out = output_dir / lang_pair / shard_in.name
        ok = process_shard(shard_in, shard_out, model, model_name, batch_size, lang_pair)
        if not ok:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-row embedding similarity for FineOPUS-Filtered-Stage2"
    )
    parser.add_argument("--model", required=True, help="Model name/path")
    parser.add_argument(
        "--model_pairs_json",
        required=True,
        help="Path to model_to_language_pairs.json",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root directory of FineOPUS-Filtered-Stage2",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root directory for output parquets with similarity scores",
    )
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=0,
        help="0-indexed chunk assigned to this job (default: 0)",
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=1,
        help="Total number of chunks the pairs are split into (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for model encoding (default: 64)",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"Model          : {args.model}")
    logger.info(f"Chunk          : {args.chunk_id} / {args.total_chunks}")
    logger.info(f"Input dir      : {args.input_dir}")
    logger.info(f"Output dir     : {args.output_dir}")
    logger.info(f"Batch size     : {args.batch_size}")
    logger.info("=" * 70)

    # Load pairs assigned to this model
    with open(args.model_pairs_json) as f:
        model_pairs: dict = json.load(f)

    if args.model not in model_pairs:
        logger.error(f"Model '{args.model}' not found in {args.model_pairs_json}")
        sys.exit(1)

    raw_entries = model_pairs[args.model]
    # Support both old format (list of str) and new format (list of [pair, bytes])
    pair_sizes: List[tuple] = [
        (e[0], e[1]) if isinstance(e, list) else (e, 0)
        for e in raw_entries
    ]
    logger.info(f"Total pairs for model: {len(pair_sizes)}")
    total_bytes = sum(b for _, b in pair_sizes)
    logger.info(f"Total data size: {total_bytes / 1e9:.2f} GB")

    # Greedy bin-packing: assign pairs to chunks minimising max chunk size.
    # Sort by size descending, then repeatedly add each pair to the lightest
    # chunk.  Deterministic (no randomness) → stable across retries.
    import heapq
    sorted_pairs = sorted(pair_sizes, key=lambda x: x[1], reverse=True)
    heap: List[tuple] = [(0, i, []) for i in range(args.total_chunks)]
    heapq.heapify(heap)
    for pair, size in sorted_pairs:
        load, chunk_id, members = heapq.heappop(heap)
        heapq.heappush(heap, (load + size, chunk_id, members + [pair]))

    chunks: List[List[str]] = [None] * args.total_chunks  # type: ignore[list-item]
    for load, chunk_id, members in heap:
        chunks[chunk_id] = members

    assigned_pairs: List[str] = chunks[args.chunk_id] or []

    if not assigned_pairs:
        logger.info("No pairs assigned to this chunk. Exiting.")
        return

    chunk_bytes = sum(b for p, b in pair_sizes if p in set(assigned_pairs))
    logger.info(
        f"This chunk: {len(assigned_pairs)} pairs, "
        f"{chunk_bytes / 1e9:.2f} GB"
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_environment()
    model = load_model(args.model, device)

    success, failed = 0, 0
    for i, lang_pair in enumerate(assigned_pairs):
        logger.info(f"--- [{i+1}/{len(assigned_pairs)}] {lang_pair} ---")
        ok = process_pair(
            lang_pair, model, args.model, input_dir, output_dir,
            batch_size=args.batch_size,
        )
        if ok:
            success += 1
        else:
            failed += 1

    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info(f"  Pairs assigned  : {len(assigned_pairs)}")
    logger.info(f"  Pairs succeeded : {success}")
    logger.info(f"  Pairs failed    : {failed}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
