from __future__ import annotations

import math

__all__ = [
    "build_source_view_sql",
    "global_summary_sql",
    "grouped_by_coverage_sql",
    "grouped_by_english_centric_sql",
    "threshold_alias",
]


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _read_parquet_arg(patterns: list[str]) -> str:
    if not patterns:
        raise ValueError("at least one parquet pattern is required")
    if len(patterns) == 1:
        return _sql_string(patterns[0])
    return "[" + ", ".join(_sql_string(pattern) for pattern in patterns) + "]"


def _threshold_literal(value: float) -> str:
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be finite: {value!r}")
    return f"{threshold:.12g}"


def threshold_alias(value: float) -> str:
    literal = _threshold_literal(value)
    return "tau_" + literal.replace("-", "neg_").replace(".", "_")


def _retention_columns(
    thresholds: tuple[float, ...], *, denominator: str = "COUNT(*)"
) -> str:
    columns: list[str] = []
    for threshold in thresholds:
        literal = _threshold_literal(threshold)
        alias = threshold_alias(threshold)
        retained_expr = f"SUM(CASE WHEN qe_score >= {literal} THEN 1 ELSE 0 END)"
        columns.extend(
            [
                f"{retained_expr} AS retained_{alias}",
                (
                    "ROUND("
                    f"100.0 * {retained_expr} / NULLIF({denominator}, 0), "
                    "4"
                    f") AS retained_{alias}_pct"
                ),
            ]
        )
    return ",\n        ".join(columns)


def build_source_view_sql(model_patterns: list[str]) -> str:
    """Build the DuckDB view over merged OPUS parquet files.

    OPUS merged rows intentionally do not carry ``direction_key``. The view
    recovers it from the grouped merge layout:

        <merged-base>/<model>/<direction_key>/<direction_key>.part-0000.parquet
    """
    parquet_arg = _read_parquet_arg(model_patterns)
    return f"""
    CREATE OR REPLACE TEMP VIEW opus_scored AS
    WITH
        raw AS (
            SELECT
                replace(filename, chr(92), '/') AS file_path,
                qe_model,
                TRY_CAST(qe_score AS DOUBLE) AS qe_score,
                qe_src_seen,
                qe_tgt_seen
            FROM read_parquet(
                {parquet_arg},
                union_by_name=True,
                filename=True
            )
        ),

        paths AS (
            SELECT
                *,
                regexp_extract(file_path, '/([^/]+)/[^/]+[.]parquet$', 1)
                    AS parent_key,
                regexp_extract(file_path, '/([^/]+)/[^/]+/[^/]+[.]parquet$', 1)
                    AS grandparent_key,
                regexp_extract(
                    file_path,
                    '/([^/]+?)(?:[.]part-[0-9]+)?[.]parquet$',
                    1
                ) AS file_key
            FROM raw
        ),

        directions AS (
            SELECT
                *,
                CASE
                    WHEN parent_key = file_key THEN grandparent_key
                    ELSE parent_key
                END AS model,
                CASE
                    WHEN parent_key = file_key THEN parent_key
                    ELSE file_key
                END AS direction_key
            FROM paths
        ),

        langs AS (
            SELECT
                direction_key,
                split_part(direction_key, '-', 1) AS src_lang,
                split_part(direction_key, '-', 2) AS tgt_lang,
                model,
                qe_model,
                qe_score,
                CASE
                    WHEN lower(trim(CAST(qe_src_seen AS VARCHAR))) IN (
                        'true', 't', '1', 'yes', 'y', 'supported', 'covered'
                    )
                    THEN TRUE
                    ELSE FALSE
                END AS src_covered,
                CASE
                    WHEN lower(trim(CAST(qe_tgt_seen AS VARCHAR))) IN (
                        'true', 't', '1', 'yes', 'y', 'supported', 'covered'
                    )
                    THEN TRUE
                    ELSE FALSE
                END AS tgt_covered
            FROM directions
            WHERE direction_key IS NOT NULL
              AND direction_key != ''
              AND direction_key LIKE '%-%'
        )

    SELECT
        direction_key,
        src_lang,
        tgt_lang,
        model,
        qe_model,
        qe_score,
        src_covered,
        tgt_covered,
        CASE
            WHEN src_covered AND tgt_covered THEN 'both-covered'
            WHEN src_covered OR tgt_covered THEN 'one-covered'
            ELSE 'both-uncovered'
        END AS coverage_status,
        CASE
            WHEN lower(split_part(src_lang, '_', 1)) IN ('eng', 'en')
              OR lower(split_part(tgt_lang, '_', 1)) IN ('eng', 'en')
            THEN 'english-centric'
            ELSE 'non-english-centric'
        END AS english_centric_status
    FROM langs
    """


def _scale_columns(*, row_count: str = "COUNT(*)") -> str:
    return f"""
        COUNT(DISTINCT model) AS models_scored,
        COUNT(DISTINCT direction_key) AS directions_scored,
        COUNT(DISTINCT model || chr(31) || direction_key)
            AS model_direction_pairs_scored,
        {row_count} AS sentence_pairs_scored,
        COUNT(DISTINCT src_lang) AS source_langs,
        COUNT(DISTINCT tgt_lang) AS target_langs,
        CAST(COUNT(DISTINCT src_lang) AS VARCHAR)
            || ' x '
            || CAST(COUNT(DISTINCT tgt_lang) AS VARCHAR)
            AS source_x_target_langs,
        COUNT(DISTINCT qe_model) AS qe_models_seen,
        COUNT(qe_score) AS non_null_scores,
        {row_count} - COUNT(qe_score) AS null_scores
    """.strip()


def global_summary_sql(scope: str, thresholds: tuple[float, ...]) -> str:
    return f"""
    SELECT
        {_sql_string(scope)} AS scope,
        {_scale_columns()},
        {_retention_columns(thresholds)}
    FROM opus_scored
    """


def grouped_by_coverage_sql(thresholds: tuple[float, ...]) -> str:
    return f"""
    WITH
        categories(coverage_status, sort_order) AS (
            VALUES
                ('both-covered', 1),
                ('one-covered', 2),
                ('both-uncovered', 3)
        ),
        expanded AS (
            SELECT
                c.coverage_status,
                c.sort_order,
                s.direction_key,
                s.src_lang,
                s.tgt_lang,
                s.model,
                s.qe_model,
                s.qe_score
            FROM categories c
            LEFT JOIN opus_scored s
              ON s.coverage_status = c.coverage_status
        )
    SELECT
        coverage_status,
        {_scale_columns(row_count="COUNT(direction_key)")},
        {_retention_columns(thresholds, denominator="COUNT(direction_key)")}
    FROM expanded
    GROUP BY coverage_status, sort_order
    ORDER BY sort_order
    """


def grouped_by_english_centric_sql(thresholds: tuple[float, ...]) -> str:
    return f"""
    WITH
        categories(english_centric_status, sort_order) AS (
            VALUES
                ('english-centric', 1),
                ('non-english-centric', 2)
        ),
        expanded AS (
            SELECT
                c.english_centric_status,
                c.sort_order,
                s.direction_key,
                s.src_lang,
                s.tgt_lang,
                s.model,
                s.qe_model,
                s.qe_score
            FROM categories c
            LEFT JOIN opus_scored s
              ON s.english_centric_status = c.english_centric_status
        )
    SELECT
        english_centric_status,
        {_scale_columns(row_count="COUNT(direction_key)")},
        {_retention_columns(thresholds, denominator="COUNT(direction_key)")}
    FROM expanded
    GROUP BY english_centric_status, sort_order
    ORDER BY sort_order
    """
