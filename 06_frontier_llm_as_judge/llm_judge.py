#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-a-judge scorer for parallel-corpus parquet shards.

Reads every `{dataset_dir}/{src}-{tgt}/*.parquet`, batches rows of
(source_text, target_text), asks an Azure-hosted LLM (DeepSeek-V4-Flash by
default) for reference-free MT quality scores, parses the JSON response, and
writes new parquet shards with an extra `llm_judge_score` column
(`overall_0to100` normalized to a 0.0-1.0 float).

Rate limiting respects both a tokens-per-minute and a requests-per-minute
budget via an asyncio token bucket. Designed to be run as a single process
per chunk (e.g. one SLURM array task per chunk).

Example (single machine):
    python llm_judge.py \\
        --dataset_dir /scratch/.../FineOPUS-Filtered-Stage3 \\
        --out_dir     /scratch/.../FineOPUS-Filtered-Stage3-LLMScored \\
        --batch_size 10 --concurrency 32

Example (SLURM array, one task per chunk):
    python llm_judge.py ... --n_chunks $SLURM_ARRAY_TASK_COUNT \\
                            --chunk_id $SLURM_ARRAY_TASK_ID
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers:
#   - httpx prints every successful POST at INFO level
#   - openai's own httpcore/openai loggers can also be chatty
#   - numexpr prints three "detected N virtual cores" lines on first import
for _noisy in ("httpx", "httpcore", "openai", "numexpr", "numexpr.utils"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
# Cap numexpr early in case any imported lib still pulls it in.
os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")

# ---------------------------------------------------------------------------
# Constants / output schema
# ---------------------------------------------------------------------------

SCORE_COL = "llm_judge_score"

STATS_COLUMNS = [
    "lang_pair", "source_lang", "target_lang",
    "n_shards_in", "n_shards_out",
    "rows_total", "rows_scored", "rows_failed",
    "mean_score", "elapsed_sec",
]


# ---------------------------------------------------------------------------
# Rate limiter (TPM + RPM, sliding 60s window)
# ---------------------------------------------------------------------------

class TpmRpmLimiter:
    """Sliding-window token + request bucket.

    Callers `await acquire(est_tokens)` before sending a request, and
    `add_actual(extra_tokens)` after the response with the delta between the
    actual token count and the estimate (positive or negative)."""

    def __init__(self, tpm: int, rpm: int):
        self.tpm = tpm
        self.rpm = rpm
        # Each entry is (timestamp, tokens). Requests track timestamp only.
        self._tokens: List[Tuple[float, int]] = []
        self._reqs: List[float] = []
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        # Drop entries older than 60s.
        i = 0
        while i < len(self._tokens) and self._tokens[i][0] < cutoff:
            i += 1
        if i:
            self._tokens = self._tokens[i:]
        j = 0
        while j < len(self._reqs) and self._reqs[j] < cutoff:
            j += 1
        if j:
            self._reqs = self._reqs[j:]

    async def acquire(self, est_tokens: int) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                cur_tokens = sum(n for _, n in self._tokens)
                cur_reqs = len(self._reqs)
                if cur_tokens + est_tokens <= self.tpm and cur_reqs < self.rpm:
                    self._tokens.append((now, est_tokens))
                    self._reqs.append(now)
                    return
                # Compute the soonest moment some window slot frees up.
                wait = 0.25
                if cur_reqs >= self.rpm and self._reqs:
                    wait = max(wait, 60.0 - (now - self._reqs[0]) + 0.05)
                if cur_tokens + est_tokens > self.tpm and self._tokens:
                    wait = max(wait, 60.0 - (now - self._tokens[0][0]) + 0.05)
                wait = min(wait, 10.0)
            await asyncio.sleep(wait)

    async def add_actual(self, extra_tokens: int) -> None:
        """After we know the true token usage, add a correction so the
        bucket reflects reality. `extra_tokens` can be negative."""
        if extra_tokens == 0:
            return
        async with self._lock:
            self._tokens.append((time.monotonic(), int(extra_tokens)))


# ---------------------------------------------------------------------------
# Prompt construction & response parsing
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a professional translation quality evaluator.

Below are {batch_size} source/translation segment pairs to evaluate.

Source language: {source_lang}
Target language: {target_lang}

{items_block}

Task: Reference-free MT quality scoring for EVERY item above.

Give each item a single overall quality score as an integer 0..100 (higher = better).
When deciding the score, consider all of these aspects together:
- accuracy & completeness (meaning preserved, no additions/omissions)
- terminology consistency
- fluency & coherence
- style, tone & audience fit
- locale formatting (numbers, punctuation, dates, tags if any)
- technical integrity (entities/units/code/markup preserved)
- cultural appropriateness

Output ONLY valid JSON with exactly this shape (no extra keys, no text outside JSON, all values integers):

{{
  "results": [
    {{
      "id": <int>,
      "overall_0to100": 0-100
    }}
  ]
}}

Return exactly {batch_size} items in "results", one per input segment, ordered by id."""


def _truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "\u2026"


def build_prompt(
    pairs: List[Tuple[str, str]],
    source_lang: str,
    target_lang: str,
    max_chars_per_field: int,
) -> str:
    blocks = []
    for i, (src, tgt) in enumerate(pairs):
        src_t = _truncate(src, max_chars_per_field).replace("\n", " ")
        tgt_t = _truncate(tgt, max_chars_per_field).replace("\n", " ")
        blocks.append(f"[{i}] SRC: {src_t}\n[{i}] TGT: {tgt_t}")
    items_block = "\n\n".join(blocks)
    return PROMPT_TEMPLATE.format(
        batch_size=len(pairs),
        source_lang=source_lang,
        target_lang=target_lang,
        items_block=items_block,
    )


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(content: str, expected_n: int) -> List[Optional[float]]:
    """Parse the JSON response. Returns an `overalls` list of length
    expected_n, with None entries where parsing failed for that id."""
    if content is None:
        raise ValueError("empty response content")
    # Strip code fences if any.
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    # Find the outermost JSON object.
    try:
        data = json.loads(text)
    except Exception:
        m = _JSON_OBJ_RE.search(text)
        if not m:
            raise ValueError(f"no JSON object in response: {text[:200]!r}")
        data = json.loads(m.group(0))
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("response missing 'results' list")

    by_id: Dict[int, dict] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            rid = int(r.get("id"))
        except (TypeError, ValueError):
            continue
        by_id[rid] = r

    overalls: List[Optional[float]] = [None] * expected_n
    for i in range(expected_n):
        r = by_id.get(i)
        if r is None:
            continue
        v = r.get("overall_0to100")
        if isinstance(v, (int, float)):
            # Normalize the model's 0..100 score to 0..1 and clip just in case.
            overalls[i] = max(0.0, min(1.0, float(v) / 100.0))
    return overalls


# ---------------------------------------------------------------------------
# Token estimation + self-calibration
# ---------------------------------------------------------------------------

def estimate_input_tokens(text: str) -> int:
    # Conservative: ~3 chars per token for mixed-script text.
    return max(1, len(text) // 3)


def estimate_output_tokens(batch_size: int) -> int:
    # Each result is ~20 tokens of JSON (id + overall).
    return 60 + 20 * batch_size


class TokenCalibration:
    """Learns a multiplicative correction factor that maps our cheap
    pre-request estimate to the real `usage.total_tokens` returned by the
    API. Until we have a few samples it returns 1.0 (i.e. trust the raw
    estimate)."""

    def __init__(self, min_samples: int = 5, log_every: int = 50):
        self._lock = asyncio.Lock()
        self._actual_sum = 0.0
        self._est_sum = 0.0
        self._samples = 0
        self._min_samples = min_samples
        self._log_every = log_every
        # Running totals for visibility / final summary.
        self.prompt_total = 0
        self.completion_total = 0

    async def update(
        self, actual_total: int, prompt_tokens: int, completion_tokens: int, est_raw: int
    ) -> Optional[str]:
        """Record one (actual, estimate) pair. Returns a summary string to
        log when we just crossed a `log_every` boundary, else None."""
        async with self._lock:
            self._actual_sum += float(actual_total)
            self._est_sum += max(1.0, float(est_raw))
            self._samples += 1
            self.prompt_total += int(prompt_tokens or 0)
            self.completion_total += int(completion_tokens or 0)
            if self._log_every and self._samples % self._log_every == 0:
                return self._summary_locked()
        return None

    def _summary_locked(self) -> str:
        factor = self.factor()
        return (
            f"token calibration: samples={self._samples} factor={factor:.3f} "
            f"(actual {int(self._actual_sum):,} / est {int(self._est_sum):,})  "
            f"prompt={self.prompt_total:,} completion={self.completion_total:,}"
        )

    def factor(self) -> float:
        if self._samples < self._min_samples or self._est_sum <= 0:
            return 1.0
        return self._actual_sum / self._est_sum

    def summary(self) -> str:
        return self._summary_locked() if self._samples else "token calibration: no samples"


# ---------------------------------------------------------------------------
# Per-batch scorer
# ---------------------------------------------------------------------------

class ScoringClient:
    def __init__(
        self,
        client: AsyncOpenAI,
        deployment: str,
        limiter: TpmRpmLimiter,
        sem: asyncio.Semaphore,
        calibration: "TokenCalibration",
        max_retries: int,
        request_timeout: float,
        max_chars_per_field: int,
        max_completion_tokens: int,
    ):
        self.client = client
        self.deployment = deployment
        self.limiter = limiter
        self.sem = sem
        self.calibration = calibration
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.max_chars_per_field = max_chars_per_field
        self.max_completion_tokens = max_completion_tokens

    async def score_batch(
        self,
        pairs: List[Tuple[str, str]],
        source_lang: str,
        target_lang: str,
    ) -> List[Optional[float]]:
        prompt = build_prompt(pairs, source_lang, target_lang, self.max_chars_per_field)
        raw_est = estimate_input_tokens(prompt) + estimate_output_tokens(len(pairs))
        # Calibrated estimate used both for the limiter reservation and for
        # the post-call delta correction.
        est_tokens = max(1, int(raw_est * self.calibration.factor()))

        last_err: Optional[Exception] = None
        async with self.sem:
            for attempt in range(self.max_retries):
                await self.limiter.acquire(est_tokens)
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.deployment,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=self.max_completion_tokens,
                        timeout=self.request_timeout,
                    )
                    content = resp.choices[0].message.content
                    # Reconcile the limiter and feed the calibrator with the
                    # *actual* token counts returned by the API.
                    try:
                        usage = resp.usage
                        if usage is not None:
                            prompt_t = int(usage.prompt_tokens or 0)
                            compl_t = int(usage.completion_tokens or 0)
                            actual = int(usage.total_tokens or (prompt_t + compl_t))
                            await self.limiter.add_actual(actual - est_tokens)
                            msg = await self.calibration.update(
                                actual, prompt_t, compl_t, raw_est
                            )
                            if msg:
                                logger.info(msg)
                    except Exception:
                        pass
                    return parse_response(content, len(pairs))
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    # Exponential backoff with jitter; longer for 429.
                    base = 5.0 if "429" in msg or "rate" in msg.lower() else 2.0
                    wait = min(60.0, base * (2 ** attempt)) + random.uniform(0, 0.5)
                    logger.warning(
                        f"  batch error (attempt {attempt + 1}/{self.max_retries}): "
                        f"{type(e).__name__}: {msg[:160]} -- retry in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
        logger.error(f"  batch failed after {self.max_retries} retries: {last_err}")
        return [None] * len(pairs)


# ---------------------------------------------------------------------------
# Per-shard worker
# ---------------------------------------------------------------------------

async def score_shard(
    in_path: Path,
    out_path: Path,
    src_lang: str,
    tgt_lang: str,
    scorer: ScoringClient,
    batch_size: int,
    compression: str,
    max_rows: Optional[int] = None,
    progress_prefix: str = "",
    progress_every_rows: int = 500,
    progress_every_sec: float = 15.0,
) -> Tuple[int, int, float]:
    """Score one parquet shard end-to-end. Returns (n_rows, n_failed, mean_score).

    If `max_rows` is set, only the first `max_rows` rows of the shard are
    read, scored, and written (used by the --max_rows option)."""
    table = pq.read_table(in_path)
    if max_rows is not None and table.num_rows > max_rows:
        table = table.slice(0, max_rows)
    n = table.num_rows
    if n == 0:
        pq.write_table(table, out_path, compression=compression)
        return 0, 0, float("nan")

    if "source_text" not in table.column_names or "target_text" not in table.column_names:
        raise ValueError(
            f"{in_path.name}: expected 'source_text' and 'target_text' columns, "
            f"got {table.column_names}"
        )

    sources = table["source_text"].to_pylist()
    targets = table["target_text"].to_pylist()

    overalls: List[Optional[float]] = [None] * n

    # Shared progress counters between concurrent batches.
    progress = {
        "rows_done": 0,
        "rows_failed": 0,
        "next_row_log": progress_every_rows,
        "next_time_log": time.monotonic() + progress_every_sec,
        "t0": time.monotonic(),
    }

    def _maybe_log_progress(force: bool = False) -> None:
        now = time.monotonic()
        rows_done = progress["rows_done"]
        if (
            force
            or rows_done >= progress["next_row_log"]
            or now >= progress["next_time_log"]
        ):
            elapsed = max(0.001, now - progress["t0"])
            rate = rows_done / elapsed
            eta = (n - rows_done) / rate if rate > 0 else float("inf")
            pct = 100.0 * rows_done / n if n else 100.0
            logger.info(
                f"{progress_prefix}  progress: {rows_done:,}/{n:,} "
                f"({pct:5.1f}%) failed={progress['rows_failed']:,} "
                f"{rate:.1f} rows/s ETA={eta:5.0f}s"
            )
            # Schedule the next thresholds (step past the count we just logged).
            while progress["next_row_log"] <= rows_done:
                progress["next_row_log"] += progress_every_rows
            progress["next_time_log"] = now + progress_every_sec

    async def run_one(start: int, end: int):
        idxs = list(range(start, end))
        pairs = [(sources[i] or "", targets[i] or "") for i in idxs]
        ov = await scorer.score_batch(pairs, src_lang, tgt_lang)
        for k, i in enumerate(idxs):
            overalls[i] = ov[k]
        # Update shared progress (single-threaded event loop, no lock needed).
        progress["rows_done"] += len(idxs)
        progress["rows_failed"] += sum(1 for v in ov if v is None)
        _maybe_log_progress()

    tasks = [
        run_one(start, min(start + batch_size, n))
        for start in range(0, n, batch_size)
    ]
    await asyncio.gather(*tasks)
    _maybe_log_progress(force=True)

    score_arr = pa.array(overalls, type=pa.float64())
    new_table = table.append_column(SCORE_COL, score_arr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, out_path, compression=compression)

    n_failed = sum(1 for v in overalls if v is None)
    valid = [v for v in overalls if v is not None]
    mean_score = (sum(valid) / len(valid)) if valid else float("nan")
    return n, n_failed, mean_score


# ---------------------------------------------------------------------------
# Lang-pair enumeration
# ---------------------------------------------------------------------------

def enumerate_pairs(dataset_dir: Path, include_same_lang: bool) -> List[Tuple[str, str]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")
    pairs: List[Tuple[str, str]] = []
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if "-" not in name:
            continue
        src, tgt = name.split("-", 1)
        if not include_same_lang and src == tgt:
            continue
        if not any(child.glob("*.parquet")):
            continue
        pairs.append((src, tgt))
    return pairs


def list_shards(pair_dir: Path) -> List[Path]:
    return sorted(p for p in pair_dir.glob("*.parquet") if p.is_file())


# ---------------------------------------------------------------------------
# Resource-class filtering (via the precomputed pair->combo JSON)
# ---------------------------------------------------------------------------

def parse_class_combos(spec: str) -> set:
    """Parse a comma-separated list of directional combos like
    "0-0,0-1,5-5" into a set of "{src_class}-{tgt_class}" strings."""
    combos = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            combos.add(tok)
    return combos


def pairs_from_combos_json(
    json_path: Path,
    combos: set,
    dataset_dir: Path,
    include_same_lang: bool,
) -> Tuple[List[Tuple[str, str]], List[str], int]:
    """Read the precomputed `{ "src_class-tgt_class": ["src-tgt", ...] }` JSON
    and return the (src, tgt) pairs belonging to the requested `combos`.

    Returns (pairs, missing_combos, skipped) where `missing_combos` lists
    requested combo keys absent from the JSON and `skipped` counts listed pairs
    whose directory is missing or has no parquet shards on disk."""
    with open(json_path) as fp:
        data = json.load(fp)

    missing_combos = sorted(c for c in combos if c not in data)
    names: List[str] = []
    for c in combos:
        names.extend(data.get(c, []))

    pairs: List[Tuple[str, str]] = []
    skipped = 0
    for name in sorted(set(names)):
        if "-" not in name:
            continue
        src, tgt = name.split("-", 1)
        if not include_same_lang and src == tgt:
            continue
        d = dataset_dir / name
        if not d.is_dir() or not any(d.glob("*.parquet")):
            skipped += 1
            continue
        pairs.append((src, tgt))
    return pairs, missing_combos, skipped


# ---------------------------------------------------------------------------
# Stats CSV I/O
# ---------------------------------------------------------------------------

def read_existing_pairs(stats_csv: Path) -> set:
    if not stats_csv.exists():
        return set()
    try:
        with open(stats_csv, newline="") as fp:
            return {row["lang_pair"] for row in csv.DictReader(fp)}
    except Exception:
        return set()


def append_stats_row(stats_csv: Path, row: dict) -> None:
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = stats_csv.exists()
    with open(stats_csv, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=STATS_COLUMNS)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in STATS_COLUMNS})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain(args: argparse.Namespace) -> None:
    load_dotenv(dotenv_path=args.env_file)

    api_key = os.environ.get("AZURE_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"AZURE_API_KEY not found in environment or {args.env_file}"
        )

    # The new Azure /openai/v1/ endpoint accepts both Bearer and api-key
    # headers; we send api-key explicitly to be safe.
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.endpoint.rstrip("/") + "/",
        default_headers={"api-key": api_key},
    )

    limiter = TpmRpmLimiter(tpm=args.tpm_limit, rpm=args.rpm_limit)
    sem = asyncio.Semaphore(args.concurrency)
    calibration = TokenCalibration(
        min_samples=args.calibration_warmup,
        log_every=args.calibration_log_every,
    )
    scorer = ScoringClient(
        client=client,
        deployment=args.deployment,
        limiter=limiter,
        sem=sem,
        calibration=calibration,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        max_chars_per_field=args.max_chars_per_field,
        max_completion_tokens=args.max_completion_tokens,
    )

    dataset_dir = Path(args.dataset_dir)
    out_root = Path(args.out_dir)
    stats_output = Path(args.stats_output)
    if args.n_chunks > 1:
        stats_output = stats_output.with_suffix(f".chunk{args.chunk_id:04d}.csv")

    out_root.mkdir(parents=True, exist_ok=True)

    # Resource-class filtering: when --class_combos is given we read the exact
    # pair list straight from the precomputed pair->combo JSON (no full-dataset
    # enumeration). Otherwise we enumerate every pair on disk.
    combos = parse_class_combos(args.class_combos) if args.class_combos else set()
    missing_combos: List[str] = []
    n_skipped_missing = 0
    if combos:
        pairs, missing_combos, n_skipped_missing = pairs_from_combos_json(
            Path(args.pair_combos_json), combos, dataset_dir, args.include_same_lang
        )
    else:
        pairs = enumerate_pairs(dataset_dir, args.include_same_lang)

    assigned = [p for i, p in enumerate(pairs) if i % args.n_chunks == args.chunk_id]

    logger.info("=" * 72)
    logger.info(f"dataset_dir       : {dataset_dir}")
    logger.info(f"out_dir           : {out_root}")
    logger.info(f"endpoint          : {args.endpoint}")
    logger.info(f"deployment        : {args.deployment}")
    logger.info(f"batch_size        : {args.batch_size}")
    logger.info(f"concurrency       : {args.concurrency}")
    logger.info(f"tpm_limit         : {args.tpm_limit:,}")
    logger.info(f"rpm_limit         : {args.rpm_limit:,}")
    logger.info(f"chunk_id          : {args.chunk_id} / {args.n_chunks}")
    logger.info(f"skip_existing     : {args.skip_existing}")
    logger.info(f"max_rows          : {args.max_rows or 'all'}")
    if combos:
        logger.info(f"pair_combos_json  : {args.pair_combos_json}")
        logger.info(f"class_combos      : {','.join(sorted(combos))}")
        logger.info(
            f"total pairs       : {len(pairs):,} (from combos JSON; "
            f"{n_skipped_missing:,} listed but missing on disk)"
        )
        if missing_combos:
            logger.warning(f"combos not in JSON: {','.join(missing_combos)}")
    else:
        logger.info(f"class_combos      : (none; all pairs)")
        logger.info(f"total pairs       : {len(pairs):,}")
    logger.info(f"assigned pairs    : {len(assigned):,}")
    logger.info(f"stats_output      : {stats_output}")
    logger.info("=" * 72)

    done_in_stats = read_existing_pairs(stats_output) if args.skip_existing else set()

    # Global row budget (across all pairs in this task). 0/None disables it.
    rows_budget: Optional[int] = (
        args.max_rows if args.max_rows and args.max_rows > 0 else None
    )

    for i, (src, tgt) in enumerate(assigned, 1):
        if rows_budget is not None and rows_budget <= 0:
            logger.info(f"Reached --max_rows budget; stopping after {i - 1} pair(s).")
            break
        lang_pair = f"{src}-{tgt}"
        in_dir = dataset_dir / lang_pair
        out_dir = out_root / lang_pair

        if args.skip_existing and (
            lang_pair in done_in_stats or (out_dir / "_DONE").exists()
        ):
            logger.info(f"[{i}/{len(assigned)}] {lang_pair}: already done, skip.")
            continue

        shards = list_shards(in_dir)
        if not shards:
            logger.warning(f"[{i}/{len(assigned)}] {lang_pair}: no parquet shards, skip.")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[{i}/{len(assigned)}] {lang_pair}: {len(shards)} shard(s) to score"
        )
        t0 = time.monotonic()
        rows_total = 0
        rows_failed = 0
        score_sum = 0.0
        score_n = 0
        n_shards_out = 0

        try:
            for j, shard in enumerate(shards, 1):
                if rows_budget is not None and rows_budget <= 0:
                    logger.info("  reached --max_rows budget, stopping.")
                    break
                out_path = out_dir / shard.name
                if args.skip_existing and out_path.exists():
                    logger.info(f"  [{j}/{len(shards)}] {shard.name}: output exists, skip shard.")
                    n_shards_out += 1
                    continue
                tt0 = time.monotonic()
                shard_prefix = (
                    f"[{i}/{len(assigned)}] {lang_pair} [{j}/{len(shards)}] {shard.name}"
                )
                logger.info(f"{shard_prefix}: scoring...")
                n_rows, n_fail, mean = await score_shard(
                    in_path=shard,
                    out_path=out_path,
                    src_lang=src,
                    tgt_lang=tgt,
                    scorer=scorer,
                    batch_size=args.batch_size,
                    compression=args.compression,
                    max_rows=rows_budget,
                    progress_prefix=shard_prefix,
                    progress_every_rows=args.progress_every_rows,
                    progress_every_sec=args.progress_every_sec,
                )
                if rows_budget is not None:
                    rows_budget -= n_rows
                rows_total += n_rows
                rows_failed += n_fail
                if n_rows > n_fail:
                    score_sum += mean * (n_rows - n_fail)
                    score_n += n_rows - n_fail
                n_shards_out += 1
                logger.info(
                    f"{shard_prefix}: done rows={n_rows:,} failed={n_fail:,} "
                    f"mean={mean:.2f} ({time.monotonic() - tt0:.1f}s)"
                )
        except Exception as e:
            logger.error(f"  {lang_pair}: aborted: {type(e).__name__}: {e}", exc_info=True)
            continue

        elapsed = time.monotonic() - t0
        rows_scored = rows_total - rows_failed
        mean_score = (score_sum / score_n) if score_n else float("nan")

        (out_dir / "_DONE").write_text(
            f"rows_total={rows_total}\nrows_failed={rows_failed}\n"
            f"mean_score={mean_score}\nelapsed_sec={elapsed:.1f}\n"
        )
        append_stats_row(stats_output, {
            "lang_pair": lang_pair,
            "source_lang": src,
            "target_lang": tgt,
            "n_shards_in": len(shards),
            "n_shards_out": n_shards_out,
            "rows_total": rows_total,
            "rows_scored": rows_scored,
            "rows_failed": rows_failed,
            "mean_score": f"{mean_score:.4f}",
            "elapsed_sec": f"{elapsed:.1f}",
        })
        logger.info(
            f"  {lang_pair}: total={rows_total:,} scored={rows_scored:,} "
            f"failed={rows_failed:,} mean={mean_score:.2f} ({elapsed:.1f}s)"
        )

    logger.info(calibration.summary())
    await client.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_dir", required=True,
                   help="Root parquet directory (one subdir per '{src}-{tgt}').")
    p.add_argument("--out_dir", required=True,
                   help="Output root; mirrors the input layout, with one extra column.")
    p.add_argument("--stats_output",
                   default=str(Path(__file__).resolve().parent / "stats/llm_judge_stats.csv"),
                   help="Per-pair stats CSV (one row per processed lang pair).")
    p.add_argument("--env_file",
                   default=str(Path(__file__).resolve().parent / ".env"),
                   help="dotenv file containing AZURE_API_KEY=...")
    p.add_argument("--endpoint",
                   default="https://fineopus-step6.services.ai.azure.com/openai/v1/")
    p.add_argument("--deployment", default="DeepSeek-V4-Flash")

    p.add_argument("--batch_size", type=int, default=10,
                   help="Segment pairs sent per API call.")
    p.add_argument("--concurrency", type=int, default=32,
                   help="Maximum in-flight requests.")
    p.add_argument("--tpm_limit", type=int, default=900_000,
                   help="Tokens-per-minute budget (default leaves 10%% headroom under 1M).")
    p.add_argument("--rpm_limit", type=int, default=900,
                   help="Requests-per-minute budget (default leaves 10%% headroom under 1K).")
    p.add_argument("--max_chars_per_field", type=int, default=2000,
                   help="Truncate each source/target text to this many characters.")
    p.add_argument("--max_completion_tokens", type=int, default=2048)
    p.add_argument("--request_timeout", type=float, default=120.0)
    p.add_argument("--max_retries", type=int, default=5)

    p.add_argument("--compression", default="zstd")
    p.add_argument("--include_same_lang", action="store_true")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip pairs with a _DONE sentinel or already recorded in stats.")

    p.add_argument("--pair_combos_json",
                   default=str(Path(__file__).resolve().parent / "fineopus_pair_class_combinations.json"),
                   help="Precomputed JSON mapping 'src_class-tgt_class' -> ['src-tgt', ...], "
                        "used by --class_combos to select exactly which pairs to score.")
    p.add_argument("--class_combos", default="",
                   help="Comma-separated directional resource-class combos to score, "
                        "e.g. '0-0,0-1,5-5' (src_class-tgt_class). Empty = all pairs. "
                        "The pair list is read directly from --pair_combos_json.")

    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--n_chunks", type=int, default=1)

    p.add_argument("--max_rows", type=int, default=0,
                   help="Test mode: if >0, score at most this many rows TOTAL across all "
                        "language pairs (this task). Output parquet is truncated to the "
                        "rows actually scored. 0 = no limit.")

    p.add_argument("--calibration_warmup", type=int, default=5,
                   help="Number of API responses to observe before the token-cost "
                        "self-calibration kicks in. During warm-up we trust the raw estimate.")
    p.add_argument("--calibration_log_every", type=int, default=50,
                   help="Log a token-calibration summary every N successful requests "
                        "(0 disables the periodic log; a final summary is always printed).")

    p.add_argument("--progress_every_rows", type=int, default=5000,
                   help="Within a shard, print a progress line every N scored rows.")
    p.add_argument("--progress_every_sec", type=float, default=15.0,
                   help="Within a shard, also print a progress line at least every N seconds.")

    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
