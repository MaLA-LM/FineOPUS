"""Shared immutable artifacts, streaming IO and deterministic selection helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_INPUT = "/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage5"
DEFAULT_TOKENS = REPO / "tools/token_stats/FineOPUS-Filtered-Stage5_Qwen3_5.csv"
DEFAULT_MRR = REPO / "04_parallelism_check/thresholds/stats/thresholds.csv"
DEFAULT_TOKENIZER = REPO / "slm_from_scratch/tokenizer/qwen3_5/tokenizer.json"
PAIR_RE = re.compile(r"^[a-z]{2,4}_[A-Za-z]{4}-[a-z]{2,4}_[A-Za-z]{4}$")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def engine_digest():
    # Both feature extraction and IO/selection behavior are part of provenance.
    return digest({p: file_digest(HERE / p) for p in ("common.py", "scoring.py")})


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def enumerate_pairs(root, pairs=None, include_same_lang=False):
    root = Path(root).resolve(strict=True)
    result = [p.name for p in sorted(root.iterdir()) if p.is_dir()
              and PAIR_RE.fullmatch(p.name) and any(p.glob("*.parquet"))
              and (include_same_lang or len(set(p.name.split("-"))) == 2)]
    if pairs:
        wanted = set(pairs.split(","))
        if wanted - set(result):
            raise ValueError(f"Missing/invalid language pairs: {sorted(wanted - set(result))}")
        result = [p for p in result if p in wanted]
    if not result:
        raise ValueError(f"No language pairs under {root}")
    return result


def inventory(pair_dir):
    result = []
    for p in sorted(Path(pair_dir).glob("*.parquet")):
        st = p.stat()
        pf = pq.ParquetFile(p)
        if not {"source_text", "target_text"} <= set(pf.schema_arrow.names):
            raise ValueError(f"Missing required text columns: {p}")
        result.append({"file": p.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                       "rows": pf.metadata.num_rows, "schema": str(pf.schema_arrow)})
    if not result:
        raise ValueError(f"No shards: {pair_dir}")
    return result


def load_token_totals(path):
    if not path:
        return {}
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pair = row["lang_pair"]
            if pair in result:
                raise ValueError(f"Duplicate token statistics: {pair}")
            values = [int(row[k]) for k in
                      ("n_lines", "n_src_tokens_Qwen3_5", "n_tgt_tokens_Qwen3_5")]
            if any(v < 0 for v in values):
                raise ValueError(f"Negative token statistics: {pair}")
            result[pair] = values
    return result


def load_mrr(path):
    if not path:
        return {}
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pair = row.get("lang_pair") or f"{row['source_lang']}-{row['target_lang']}"
            try:
                mrr = float(row["MRR"])
            except (ValueError, KeyError):
                mrr = None
            if mrr is not None and (not np.isfinite(mrr) or not 0 <= mrr <= 1):
                mrr = None
            result[pair] = {"MRR": mrr, "model": row.get("model", "")}
    return result


def sample_rows(pair_dir, manifest, count, seed, columns):
    """Uniform global row offsets without replacement; read only touched row groups.

    Selection is independent of file size, file order, batches and null scores.
    At most one row group plus the requested sample is resident in memory.
    """
    total = sum(m["rows"] for m in manifest)
    if total == 0:
        raise ValueError(f"Empty language pair: {pair_dir}")
    rng = np.random.default_rng(seed)
    wanted = np.sort(rng.choice(total, min(count, total), replace=False))
    chunks = []
    offset = 0
    for m in manifest:
        pf = pq.ParquetFile(Path(pair_dir) / m["file"])
        cols = [c for c in columns if c in pf.schema_arrow.names]
        for group in range(pf.num_row_groups):
            n = pf.metadata.row_group(group).num_rows
            left, right = np.searchsorted(wanted, [offset, offset + n])
            if right > left:
                table = pf.read_row_group(group, columns=cols)
                chunks.append(table.take(pa.array(wanted[left:right] - offset)))
            offset += n
    return pa.concat_tables(chunks, promote_options="permissive")


def score_bins(scores, bins):
    return np.minimum((np.clip(scores, 0, 1) * bins).astype(np.int64), bins - 1)


def hash_uniform(pair, shard, offsets, seed):
    """Stable row-identity hash; invariant to batch boundaries and task counts.

    Input inventory validation prevents reuse after resharding. Vectorized
    SplitMix64 avoids hashing every complete text in a large boundary bucket.
    """
    salt = int.from_bytes(hashlib.blake2b(f"{seed}:{pair}:{shard}".encode(), digest_size=8).digest(), "little")
    with np.errstate(over="ignore"):
        x = np.asarray(offsets, dtype=np.uint64) + np.uint64(salt)
        x += np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
        x ^= x >> 31
    return (x >> 11).astype(np.float64) / 2**53


def selection_mask(scores, eligible, cohort, thresholds, pair, shard, offset, seed, bins):
    codes = score_bins(scores, bins)
    keep = np.zeros(len(scores), dtype=bool)
    for group, spec in thresholds.items():
        selected = eligible & (cohort == int(group))
        boundary = int(spec["boundary_bin"])
        keep |= selected & (codes > boundary)
        tied = np.flatnonzero(selected & (codes == boundary))
        if tied.size:
            keep[tied] = hash_uniform(pair, shard, tied + offset, seed) < float(spec["boundary_fraction"])
    return keep


class TokenCounter:
    """Local fast-tokenizer JSON, no model download or GPU dependency."""
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def lengths(self, texts):
        result = []
        for start in range(0, len(texts), 1024):
            result.extend(len(e.ids) for e in self.tokenizer.encode_batch(
                [x or "" for x in texts[start:start + 1024]], add_special_tokens=False))
        return np.asarray(result, dtype=np.int64)


class RotatingWriter:
    def __init__(self, directory, pair, schema, max_rows, compression):
        self.directory, self.pair, self.schema = Path(directory), pair, schema
        self.max_rows, self.compression = max_rows, compression
        self.writer = None
        self.shards = self.in_shard = 0

    def write(self, table):
        table = table.cast(self.schema)
        offset = 0
        while offset < table.num_rows:
            if self.writer is None:
                path = self.directory / f"{self.pair}_shard_{self.shards}.parquet"
                self.writer = pq.ParquetWriter(path, self.schema, compression=self.compression)
                self.shards += 1
                self.in_shard = 0
            n = min(table.num_rows - offset, self.max_rows - self.in_shard)
            self.writer.write_table(table.slice(offset, n))
            self.in_shard += n
            offset += n
            if self.in_shard == self.max_rows:
                self.close()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def validate_chunks(args):
    if args.n_chunks < 1 or not 0 <= args.chunk_id < args.n_chunks:
        raise ValueError("Require n_chunks >= 1 and 0 <= chunk_id < n_chunks")
