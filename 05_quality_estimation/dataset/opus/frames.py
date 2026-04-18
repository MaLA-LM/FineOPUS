from __future__ import annotations

from typing import Any

# Keys we strip from the example dict before emitting detail rows.
# ``src`` / ``tgt`` are scorer-facing aliases; the original parquet columns
# ``source_text`` / ``target_text`` are preserved as passthrough fields.
_RESERVED_EXAMPLE_KEYS = {"src", "tgt"}


def build_frames(
    model_name: str,
    dataset: str,
    split: str | None,
    src_lang: str,
    tgt_lang: str,
    scores: list[float],
    examples: list[dict[str, Any]],
    *,
    src_lang_seen: str | bool,
    tgt_lang_seen: str | bool,
    mean: float | None = None,
    median: float | None = None,
):
    """OPUS frame builder.

    OPUS keeps the original dataset rows intact. We only strip the scorer-only
    ``src``/``tgt`` aliases and append the four QE fields requested by the
    downstream OPUS pipeline.

    Failed scores (NaN from the scorer) are preserved and surface as JSON
    null on write — OPUS is the production application and must distinguish
    "low quality" from "could not be scored". This differs from FLORES,
    which is benchmarking and treats failures as worst-case 0.

    Shared scorer code still passes dataset metadata and summary statistics for
    API compatibility, but OPUS intentionally does not emit them into the row
    payload. The writer adds ``direction_key`` and ``shard_id`` later.
    """
    import pandas as pd

    # kept for compatibility with shared scorer code, not used in OPUS frames
    _ = (dataset, split, src_lang, tgt_lang, mean, median)

    detail_records: list[dict[str, Any]] = []
    for example, score in zip(examples, scores):
        record: dict[str, Any] = {
            key: value
            for key, value in example.items()
            if key not in _RESERVED_EXAMPLE_KEYS
        }
        record["qe_model"] = model_name
        record["qe_score"] = score
        record["qe_src_seen"] = src_lang_seen
        record["qe_tgt_seen"] = tgt_lang_seen
        detail_records.append(record)

    # ``from_records`` unions keys across rows and fills missing cells with
    # NaN; stage_writer._normalize_value converts those NaNs to None on write.
    return pd.DataFrame.from_records(detail_records)
