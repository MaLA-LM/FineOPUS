"""
weighted.py — Weighted-average ensemble (Steps 6–7 from thesis §8.4).

Computes 5 weighted-average ensemble scores per sentence:
  ens_wavg_all          — all models, global weights
  ens_wavg_both_seen    — only models where BOTH src & tgt are seen
  ens_wavg_src_seen     — only models where src lang is seen
  ens_wavg_tgt_seen     — only models where tgt lang is seen
  ens_wavg_both_unseen  — only models where BOTH langs are unseen

Weights = macro-average of each model's direction-level means, normalized to sum to 1.
Coverage-aware variants renormalize weights over the eligible model subset.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

from .config import MODELS, NUM_BUCKETS
from .db import setup_connection


def run_weighted(
    norm_root: Path,
    ens_root: Path,
    dst_root: Path,
    scratch_tmp: Path,
    preview: bool = False,
    models: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Execute the weighted-average ensemble computation (Steps 6 + 7)."""
    if models is None:
        models = MODELS

    norm_pattern = str(norm_root / "**" / "*.parquet")
    ens_pattern = str(ens_root / "**" / "*.parquet")
    con = setup_connection(scratch_tmp)

    # ── Step 6: global per-model weights ─────────────────────────────────
    #   w_k = macro_mean_k / Σ(macro_means)
    print("Step 6: Computing global weights...")
    t0 = time.time()

    model_filter = ", ".join(f"'{h}'" for h, _ in models)
    weights_df = con.execute(
        f"""
        WITH direction_means AS (
            SELECT model, direction_key, mean AS direction_mean
            FROM read_parquet('{norm_pattern}', union_by_name=True, hive_partitioning=True)
            WHERE row_type = 'summary'
              AND model IN ({model_filter})
        ),
        macro_avg AS (
            SELECT model, AVG(direction_mean) AS macro_mean
            FROM direction_means
            GROUP BY model
        ),
        total AS (
            SELECT SUM(macro_mean) AS sum_macro FROM macro_avg
        )
        SELECT m.model, m.macro_mean, m.macro_mean / t.sum_macro AS weight
        FROM macro_avg m CROSS JOIN total t
        ORDER BY m.model
    """
    ).df()

    print("\n  Global weights:")
    print(weights_df.to_string(index=False))
    print(f"  Computed in {time.time() - t0:.1f}s")

    w = dict(zip(weights_df["model"], weights_df["weight"]))

    # ── Pivot direction-level seen flags into a temp table ───────────────
    #   One row per direction_key, with {model}_src_seen / {model}_tgt_seen columns.
    print("\nPivoting direction-level seen flags...")
    t1 = time.time()

    seen_pivot_cols = []
    for hive, safe in MODELS:
        seen_pivot_cols.append(
            f"MAX(CASE WHEN model = '{hive}' THEN CAST(src_lang_seen AS INTEGER) END) AS {safe}_src_seen"
        )
        seen_pivot_cols.append(
            f"MAX(CASE WHEN model = '{hive}' THEN CAST(tgt_lang_seen AS INTEGER) END) AS {safe}_tgt_seen"
        )

    con.execute(
        f"""
        CREATE TEMP TABLE dir_seen_flags AS
        SELECT
            direction_key,
            {','.join(seen_pivot_cols)}
        FROM read_parquet('{norm_pattern}', union_by_name=True, hive_partitioning=True)
        WHERE row_type = 'summary'
        GROUP BY direction_key
    """
    )

    n_dirs = con.execute("SELECT COUNT(*) FROM dir_seen_flags").fetchone()[0]
    print(f"  {n_dirs} directions, pivoted in {time.time() - t1:.1f}s")

    # ── Step 7: build the weighted-average query ─────────────────────────
    #
    # Each coverage variant is: Σ(w_k * score_k * flag_k) / Σ(w_k * flag_k)
    # NULL when no models are eligible (denominator = 0).
    #
    # Tables:
    #   e = mean_median_ensembles (has per-model scores + both_seen flags)
    #   d = dir_seen_flags        (has per-model src_seen + tgt_seen flags)

    # -- ens_wavg_all: all models, no filtering --
    wavg_all = (
        "(" + " + ".join(f"{w[h]} * e.{s}_score" for h, s in models) + ")"
        " AS ens_wavg_all"
    )

    # -- ens_wavg_both_seen: both src & tgt seen (sentence-level flag from pivot) --
    bs_num = " + ".join(f"({w[h]} * e.{s}_score * e.{s}_both_seen)" for h, s in models)
    bs_den = " + ".join(f"({w[h]} * e.{s}_both_seen)" for h, s in models)
    wavg_both_seen = (
        f"CASE WHEN ({bs_den}) > 0 THEN ({bs_num}) / ({bs_den}) ELSE NULL END"
        " AS ens_wavg_both_seen"
    )

    # -- ens_wavg_src_seen: src lang seen (direction-level flag) --
    ss_num = " + ".join(f"({w[h]} * e.{s}_score * d.{s}_src_seen)" for h, s in models)
    ss_den = " + ".join(f"({w[h]} * d.{s}_src_seen)" for h, s in models)
    wavg_src_seen = (
        f"CASE WHEN ({ss_den}) > 0 THEN ({ss_num}) / ({ss_den}) ELSE NULL END"
        " AS ens_wavg_src_seen"
    )

    # -- ens_wavg_tgt_seen: tgt lang seen (direction-level flag) --
    ts_num = " + ".join(f"({w[h]} * e.{s}_score * d.{s}_tgt_seen)" for h, s in models)
    ts_den = " + ".join(f"({w[h]} * d.{s}_tgt_seen)" for h, s in models)
    wavg_tgt_seen = (
        f"CASE WHEN ({ts_den}) > 0 THEN ({ts_num}) / ({ts_den}) ELSE NULL END"
        " AS ens_wavg_tgt_seen"
    )

    # -- ens_wavg_both_unseen: both src & tgt unseen --
    def _bu(s):
        """Both-unseen flag: 1 when neither language was seen by model."""
        return f"CASE WHEN d.{s}_src_seen = 0 AND d.{s}_tgt_seen = 0 THEN 1 ELSE 0 END"

    bu_num = " + ".join(f"({w[h]} * e.{s}_score * ({_bu(s)}))" for h, s in models)
    bu_den = " + ".join(f"({w[h]} * ({_bu(s)}))" for h, s in models)
    wavg_both_unseen = (
        f"CASE WHEN ({bu_den}) > 0 THEN ({bu_num}) / ({bu_den}) ELSE NULL END"
        " AS ens_wavg_both_unseen"
    )

    # -- Eligible model counts per variant --
    num_src_seen = (
        f"({' + '.join(f'd.{s}_src_seen' for _, s in models)}) AS num_models_src_seen"
    )
    num_tgt_seen = (
        f"({' + '.join(f'd.{s}_tgt_seen' for _, s in models)}) AS num_models_tgt_seen"
    )
    num_both_unseen = (
        f"({' + '.join(f'({_bu(s)})' for _, s in models)}) AS num_models_both_unseen"
    )

    # -- Columns carried forward from mean_median ensemble (ALL models) --
    passthrough = [
        "e.src_txt",
        "e.tgt_txt",
        "e.direction_key",
        "e.dataset",
        "e.src_lang",
        "e.tgt_lang",
        "e.src_lang_family",
        "e.tgt_lang_family",
    ]
    for _, s in MODELS:
        passthrough += [f"e.{s}_score", f"e.{s}_both_seen"]
    passthrough += [
        "e.ens_mean_all",
        "e.ens_median_all",
        "e.ens_mean_seen",
        "e.ens_median_seen",
        "e.num_models_seen",
    ]

    # -- Per-model direction-level seen flags (ALL models) --
    dir_flag_cols = []
    for _, s in MODELS:
        dir_flag_cols += [f"d.{s}_src_seen", f"d.{s}_tgt_seen"]

    # -- Assemble the full query --
    query = f"""
    SELECT
        {','.join(passthrough)},
        {','.join(dir_flag_cols)},
        {wavg_all},
        {wavg_both_seen},
        {wavg_src_seen},
        {wavg_tgt_seen},
        {wavg_both_unseen},
        {num_src_seen},
        {num_tgt_seen},
        {num_both_unseen},
        CAST(hash(e.direction_key) % {NUM_BUCKETS} AS INTEGER) AS bucket_id
    FROM read_parquet('{ens_pattern}', union_by_name=True, hive_partitioning=True) e
    JOIN dir_seen_flags d ON e.direction_key = d.direction_key
    """

    # ── Execute ──────────────────────────────────────────────────────────
    if preview:
        df = con.execute(f"SELECT * FROM ({query}) sub LIMIT 5").df()
        new_cols = [
            c
            for c in df.columns
            if "wavg" in c or "src_seen" in c or "tgt_seen" in c or "num_models" in c
        ]
        print("\n--- New columns preview ---")
        print(df[new_cols].to_string())
        print(f"\nAll columns ({len(df.columns)}): {list(df.columns)}")
        print("\n--- Global weights used ---")
        for h, s in models:
            print(f"  {s}: {w[h]:.6f}")
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"\nStep 7: Writing to {dst_root} ...")

    t2 = time.time()
    con.execute(
        f"""
        COPY ({query})
        TO '{dst_root}'
        (FORMAT PARQUET, PARTITION_BY (bucket_id), OVERWRITE_OR_IGNORE)
    """
    )
    print(f"Done in {time.time() - t2:.1f}s")

    # ── Verification ─────────────────────────────────────────────────────
    dst_pattern = str(dst_root / "**" / "*.parquet")
    print("\nVerification:")
    print(
        con.execute(
            f"""
        SELECT
            bucket_id,
            COUNT(*) AS rows,
            COUNT(DISTINCT direction_key) AS directions,
            ROUND(AVG(ens_wavg_all),          4) AS avg_wavg_all,
            ROUND(AVG(ens_wavg_both_seen),    4) AS avg_wavg_both_seen,
            ROUND(AVG(ens_wavg_src_seen),     4) AS avg_wavg_src_seen,
            ROUND(AVG(ens_wavg_tgt_seen),     4) AS avg_wavg_tgt_seen,
            ROUND(AVG(ens_wavg_both_unseen),  4) AS avg_wavg_both_unseen
        FROM read_parquet('{dst_pattern}', hive_partitioning=True)
        GROUP BY bucket_id ORDER BY bucket_id
    """
        )
        .df()
        .to_string()
    )

    print("\nNull coverage (directions where variant is undefined):")
    print(
        con.execute(
            f"""
        SELECT
            COUNT(DISTINCT direction_key)                                                   AS total_directions,
            COUNT(DISTINCT CASE WHEN ens_wavg_both_seen   IS NULL THEN direction_key END)  AS null_both_seen,
            COUNT(DISTINCT CASE WHEN ens_wavg_src_seen    IS NULL THEN direction_key END)  AS null_src_seen,
            COUNT(DISTINCT CASE WHEN ens_wavg_tgt_seen    IS NULL THEN direction_key END)  AS null_tgt_seen,
            COUNT(DISTINCT CASE WHEN ens_wavg_both_unseen IS NULL THEN direction_key END)  AS null_both_unseen
        FROM read_parquet('{dst_pattern}', hive_partitioning=True)
    """
        )
        .df()
        .to_string()
    )
