def patch_query(model_pattern: str, model_out: str) -> str:
    return f"""
    COPY (
        SELECT
            row_type, model_name, dataset, src_lang, tgt_lang,
            src_lang_family, tgt_lang_family,
            (src_s.code IS NOT NULL) AS src_lang_seen,
            (tgt_s.code IS NOT NULL) AS tgt_lang_seen,
            mean, median, score, z_score, rank_percentile,
            src_txt, tgt_txt, direction_key, bucket
        FROM read_parquet(
            '{model_pattern}',
            union_by_name=True,
            hive_partitioning=True
        ) AS data
        LEFT JOIN remedy_supported src_s ON data.src_lang = src_s.code
        LEFT JOIN remedy_supported tgt_s ON data.tgt_lang = tgt_s.code
    )
    TO '{model_out}'
    (
        FORMAT PARQUET,
        PARTITION_BY (bucket),
        OVERWRITE_OR_IGNORE
    )
    """


def validation_query(src_pattern: str, dst_pattern: str) -> str:
    return f"""
    WITH
        old AS (
            SELECT direction_key, src_lang_seen, tgt_lang_seen
            FROM read_parquet('{src_pattern}', union_by_name=True, hive_partitioning=True)
            WHERE row_type = 'summary'
        ),
        new AS (
            SELECT direction_key, src_lang_seen, tgt_lang_seen
            FROM read_parquet('{dst_pattern}', union_by_name=True, hive_partitioning=True)
            WHERE row_type = 'summary'
        )
    SELECT
        (SELECT COUNT(*) FROM read_parquet('{src_pattern}', union_by_name=True, hive_partitioning=True)) AS old_rows,
        (SELECT COUNT(*) FROM read_parquet('{dst_pattern}', union_by_name=True, hive_partitioning=True)) AS new_rows,
        COUNT(*) FILTER (
            WHERE old.src_lang_seen IS DISTINCT FROM new.src_lang_seen
        ) AS src_seen_changed,
        COUNT(*) FILTER (
            WHERE old.tgt_lang_seen IS DISTINCT FROM new.tgt_lang_seen
        ) AS tgt_seen_changed
    FROM old
    JOIN new USING (direction_key)
    """
