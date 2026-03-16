def normalization_query(model_pattern: str, model_out: str) -> str:
    return f"""
    COPY (
        WITH

        raw AS (
            SELECT *
            FROM read_parquet(
                '{model_pattern}',
                union_by_name=True,
                hive_partitioning=True
            )
        ),

        seen_flags AS (
            SELECT
                direction_key,
                bool_or(src_lang_seen) AS src_lang_seen,
                bool_or(tgt_lang_seen) AS tgt_lang_seen
            FROM raw
            WHERE row_type = 'summary'
            GROUP BY direction_key
        ),

        detail AS (
            SELECT
                'detail'                                        AS row_type,
                model_name, dataset, src_lang, tgt_lang,
                src_f.family                                    AS src_lang_family,
                tgt_f.family                                    AS tgt_lang_family,
                sf.src_lang_seen,
                sf.tgt_lang_seen,
                CAST(NULL AS DOUBLE)                            AS mean,
                CAST(NULL AS DOUBLE)                            AS median,
                COALESCE(score, 0.0)                            AS score,
                (COALESCE(score, 0) - avg(COALESCE(score, 0)) OVER w) /
                NULLIF(stddev_samp(COALESCE(score, 0)) OVER w, 0)      AS z_score,
                percent_rank()      OVER w                              AS rank_percentile,
                src_txt, tgt_txt, direction_key,
                bucket
            FROM raw
            LEFT JOIN seen_flags sf USING (direction_key)
            LEFT JOIN lang_family src_f ON src_lang = src_f.code
            LEFT JOIN lang_family tgt_f ON tgt_lang = tgt_f.code
            WHERE row_type = 'detail'
            WINDOW w AS (PARTITION BY model ORDER BY COALESCE(score, 0)
                         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
        ),

        detail_stats AS (
            SELECT
                direction_key,
                avg(score)    AS mean,
                median(score) AS median
            FROM detail
            GROUP BY direction_key
        ),

        summary AS (
            SELECT
                'summary'                       AS row_type,
                any_value(r.model_name)         AS model_name,
                any_value(r.dataset)            AS dataset,
                r.src_lang,
                r.tgt_lang,
                src_f.family                    AS src_lang_family,
                tgt_f.family                    AS tgt_lang_family,
                bool_or(r.src_lang_seen)        AS src_lang_seen,
                bool_or(r.tgt_lang_seen)        AS tgt_lang_seen,
                ds.mean,
                ds.median,
                CAST(NULL AS DOUBLE)            AS score,
                CAST(NULL AS DOUBLE)            AS z_score,
                CAST(NULL AS DOUBLE)            AS rank_percentile,
                CAST(NULL AS VARCHAR)           AS src_txt,
                CAST(NULL AS VARCHAR)           AS tgt_txt,
                r.direction_key,
                min(r.bucket)                   AS bucket
            FROM raw r
            JOIN detail_stats ds USING (direction_key)
            LEFT JOIN lang_family src_f ON r.src_lang = src_f.code
            LEFT JOIN lang_family tgt_f ON r.tgt_lang = tgt_f.code
            WHERE r.row_type = 'summary'
            GROUP BY r.src_lang, r.tgt_lang, r.direction_key,
                     src_f.family, tgt_f.family, ds.mean, ds.median
        )

        SELECT * FROM detail
        UNION ALL
        SELECT * FROM summary
    )
    TO '{model_out}'
    (
        FORMAT PARQUET,
        PARTITION_BY (bucket),
        OVERWRITE_OR_IGNORE
    )
    """
