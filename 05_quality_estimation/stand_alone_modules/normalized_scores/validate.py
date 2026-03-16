def validate_output(con, model_out: str, model: str) -> None:
    pattern = f"{model_out}/**/*.parquet"

    def fail(msg):
        raise RuntimeError(f"[validate:{model}] {msg}")

    def warn(msg):
        print(f"[validate:{model}] WARNING: {msg}")

    print(f"  Validating output for model: {model}")

    row = con.execute(
        f"""
        WITH data AS (
            SELECT row_type, score, z_score, rank_percentile, mean, median
            FROM read_parquet('{pattern}', hive_partitioning=True)
        )
        SELECT
            COUNT(*) FILTER (WHERE row_type = 'detail')                              AS n_detail,
            COUNT(*) FILTER (WHERE row_type = 'summary')                             AS n_summary,
            COUNT(*) FILTER (WHERE row_type = 'detail' AND score IS NULL)            AS null_scores,
            COUNT(*) FILTER (WHERE row_type = 'detail'
                                AND (z_score IS NULL OR isnan(z_score)))             AS null_z,
            avg(z_score)         FILTER (WHERE row_type = 'detail')                  AS z_mean,
            stddev_samp(z_score) FILTER (WHERE row_type = 'detail')                  AS z_std,
            COUNT(*) FILTER (WHERE row_type = 'detail'
                                AND (rank_percentile < 0 OR rank_percentile > 1))    AS bad_pct,
            COUNT(*) FILTER (WHERE row_type = 'summary'
                                AND (mean IS NULL OR median IS NULL))                 AS null_summary,
            MIN(score)   FILTER (WHERE row_type = 'detail')                          AS score_min,
            MAX(score)   FILTER (WHERE row_type = 'detail')                          AS score_max,
            MIN(mean)    FILTER (WHERE row_type = 'summary')                         AS mean_min,
            MAX(mean)    FILTER (WHERE row_type = 'summary')                         AS mean_max,
            MIN(median)  FILTER (WHERE row_type = 'summary')                         AS median_min,
            MAX(median)  FILTER (WHERE row_type = 'summary')                         AS median_max
        FROM data
        """
    ).fetchone()

    (
        n_detail, n_summary, null_scores, null_z, z_mean, z_std, bad_pct,
        null_summary, score_min, score_max, mean_min, mean_max, median_min, median_max,
    ) = row

    if n_detail == 0:
        fail("row_type='detail' is missing from output")
    if n_summary == 0:
        fail("row_type='summary' is missing from output")
    if null_scores > 0:
        fail(f"{null_scores} detail rows have NULL score")
    if null_z > 0:
        fail(f"{null_z} detail rows have NULL/NaN z_score — check window frame / stddev=0")
    if abs(z_mean) > 0.05:
        fail(f"z_score mean={z_mean:.4f} is too far from 0 (expected |mean| ≤ 0.05)")
    if z_std is None or not (0.9 <= z_std <= 1.1):
        fail(f"z_score std={z_std:.4f} is outside [0.9, 1.1] — standardisation may be wrong")
    if bad_pct > 0:
        fail(f"{bad_pct} detail rows have rank_percentile outside [0, 1]")
    if null_summary > 0:
        fail(f"{null_summary} summary rows have NULL mean or median")
    if mean_min < score_min or mean_max > score_max:
        warn(f"summary mean [{mean_min:.4f}, {mean_max:.4f}] outside score range [{score_min:.4f}, {score_max:.4f}]")
    if median_min < score_min or median_max > score_max:
        warn(f"summary median [{median_min:.4f}, {median_max:.4f}] outside score range [{score_min:.4f}, {score_max:.4f}]")

    print(f"  All checks passed for model: {model} (z_score mean={z_mean:.4f}, std={z_std:.4f})")
