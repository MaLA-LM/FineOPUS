"""
pivot.py
--------
Mean/median ensemble logic (Steps 1–3).

Step 1: Pivot long-format scores → wide format (one row per sentence, one column per model).
Step 2: Compute all-inclusive and seen-only mean/median ensemble columns.
Step 3: Write as partitioned parquet (bucketed by direction_key hash).
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

from .config import MODELS, NUM_BUCKETS
from .db import setup_connection


def build_pivot_query(
    src_pattern: str,
    num_buckets: int = NUM_BUCKETS,
    models: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Return a SQL query that:
      - Pivots long-format rows into one wide row per sentence
      - Adds per-model score and both_seen flag columns
      - Computes ens_mean_all, ens_median_all, ens_mean_seen, ens_median_seen
      - Assigns a bucket_id for output partitioning
    """
    if models is None:
        models = MODELS

    # For each model: one score column + one both_seen flag column (ALL models)
    pivot_columns = []
    for hive_name, safe_name in MODELS:
        pivot_columns.append(
            f"MAX(CASE WHEN model = '{hive_name}' THEN score END)"
            f" AS {safe_name}_score"
        )
        pivot_columns.append(
            f"MAX(CASE WHEN model = '{hive_name}' AND src_lang_seen AND tgt_lang_seen THEN 1"
            f"         WHEN model = '{hive_name}' THEN 0 END)"
            f" AS {safe_name}_both_seen"
        )
    pivot_sql = ",\n            ".join(pivot_columns)

    # Score lists for ensembles — only *included* models
    all_items = [f"{s}_score" for _, s in models]
    seen_items = [f"CASE WHEN {s}_both_seen = 1 THEN {s}_score END" for _, s in models]
    all_list = ", ".join(all_items)
    seen_list = ", ".join(seen_items)

    # Output columns carry ALL models' scores
    per_model_cols = ", ".join(f"{s}_score, {s}_both_seen" for _, s in MODELS)

    return f"""
    WITH wide AS (
        SELECT
            src_txt, tgt_txt, direction_key, dataset,
            src_lang, tgt_lang, src_lang_family, tgt_lang_family,
            {pivot_sql}
        FROM read_parquet('{src_pattern}', union_by_name=True, hive_partitioning=True)
        WHERE row_type = 'detail'
        GROUP BY src_txt, tgt_txt, direction_key, dataset,
                 src_lang, tgt_lang, src_lang_family, tgt_lang_family
    )
    SELECT
        src_txt, tgt_txt, direction_key, dataset,
        src_lang, tgt_lang, src_lang_family, tgt_lang_family,
        {per_model_cols},
        list_aggregate([{all_list}], 'avg')                                       AS ens_mean_all,
        list_aggregate([{all_list}], 'median')                                    AS ens_median_all,
        list_aggregate(list_filter([{seen_list}], x -> x IS NOT NULL), 'avg')    AS ens_mean_seen,
        list_aggregate(list_filter([{seen_list}], x -> x IS NOT NULL), 'median') AS ens_median_seen,
        len(list_filter([{seen_list}], x -> x IS NOT NULL))                      AS num_models_seen,
        CAST(hash(direction_key) % {num_buckets} AS INTEGER)                     AS bucket_id
    FROM wide
    """


def run_pivot(
    src_root: Path,
    dst_root: Path,
    scratch_tmp: Path,
    preview: bool = False,
    models: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Execute the mean/median ensemble pivot."""
    src_pattern = str(src_root / "**" / "*.parquet")
    con = setup_connection(scratch_tmp)
    query = build_pivot_query(src_pattern, models=models)

    if preview:
        df = con.execute(f"SELECT * FROM ({query}) sub LIMIT 5").df()
        print(df.to_string())
        print(f"\nColumns ({len(df.columns)}): {list(df.columns)}")
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {dst_root} ...")

    t0 = time.time()
    con.execute(
        f"""
        COPY ({query})
        TO '{dst_root}'
        (FORMAT PARQUET, PARTITION_BY (bucket_id), OVERWRITE_OR_IGNORE)
    """
    )
    print(f"Done in {time.time() - t0:.1f}s")

    # Quick verification
    dst_pattern = str(dst_root / "**" / "*.parquet")
    print(
        con.execute(
            f"""
        SELECT bucket_id, COUNT(*) AS rows, COUNT(DISTINCT direction_key) AS directions
        FROM read_parquet('{dst_pattern}', hive_partitioning=True)
        GROUP BY bucket_id ORDER BY bucket_id
    """
        )
        .df()
        .to_string()
    )
