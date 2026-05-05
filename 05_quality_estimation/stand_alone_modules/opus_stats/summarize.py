from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stand_alone_modules.opus_stats.queries import (
    build_source_view_sql,
    global_summary_sql,
    grouped_by_coverage_sql,
    grouped_by_english_centric_sql,
)
from utils.logger import logger

DEFAULT_MERGED_BASE = Path("/scratch/project_462001050/opus_qe/merged")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "stats"
DEFAULT_TMP_DIR = Path("/scratch/project_462001050/opus_qe/duckdb_tmp")
DEFAULT_THRESHOLDS = (0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class OpusStatsConfig:
    merged_base: Path = DEFAULT_MERGED_BASE
    models: tuple[str, ...] | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    tmp_dir: Path = DEFAULT_TMP_DIR
    threads: int = 8
    memory_limit: str | None = None
    max_temp_size: str | None = None


@dataclass(frozen=True)
class TableResult:
    title: str
    csv_path: Path
    columns: list[str]
    rows: list[tuple[Any, ...]]


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _parquet_pattern(model_dir: Path) -> str:
    return (model_dir / "**" / "*.parquet").as_posix()


def _first_parquet(path: Path) -> Path | None:
    return next(path.glob("**/*.parquet"), None)


def _selected_model_names(models: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    selected: list[str] = []
    for model in models:
        normalized = model.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)
    return tuple(selected)


def _resolve_model_dirs(config: OpusStatsConfig) -> list[Path]:
    if config.threads < 1:
        raise ValueError("--threads must be >= 1")
    if not config.thresholds:
        raise ValueError("at least one threshold is required")

    merged_base = config.merged_base
    if not merged_base.is_dir():
        raise SystemExit(f"Merged base does not exist: {merged_base}")

    if config.models:
        model_dirs: list[Path] = []
        for model in _selected_model_names(config.models):
            model_dir = merged_base / model
            if not model_dir.is_dir():
                raise SystemExit(f"Model directory does not exist: {model_dir}")
            if _first_parquet(model_dir) is None:
                raise SystemExit(f"No parquet files found under: {model_dir}")
            model_dirs.append(model_dir)
        if not model_dirs:
            raise SystemExit("--model did not select any non-empty model names")
        return model_dirs

    model_dirs = [
        path
        for path in sorted(merged_base.iterdir())
        if path.is_dir()
        and not path.name.startswith(".")
        and not path.name.startswith("_")
        and _first_parquet(path) is not None
    ]
    if not model_dirs:
        raise SystemExit(f"No model parquet files found under: {merged_base}")
    return model_dirs


def _connect(config: OpusStatsConfig):
    import duckdb

    config.tmp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={config.threads};")
    con.execute(f"SET temp_directory={_sql_string(str(config.tmp_dir))};")
    if config.memory_limit:
        con.execute(f"SET memory_limit={_sql_string(config.memory_limit)};")
    if config.max_temp_size:
        con.execute(
            f"SET max_temp_directory_size={_sql_string(config.max_temp_size)};"
        )
    return con


def _write_csv(path: Path, columns: list[str], rows: list[tuple[Any, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _format_markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _markdown_table(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_format_markdown_cell(value) for value in row) + " |"
        )
    return "\n".join(lines)


def _run_table(con, title: str, query: str, csv_path: Path) -> TableResult:
    cursor = con.execute(query)
    columns = [item[0] for item in cursor.description]
    rows = cursor.fetchall()
    _write_csv(csv_path, columns, rows)
    logger.info("Wrote %s: %s", title, csv_path)
    return TableResult(title=title, csv_path=csv_path, columns=columns, rows=rows)


def _write_report(
    path: Path,
    *,
    config: OpusStatsConfig,
    model_dirs: list[Path],
    tables: list[TableResult],
) -> None:
    model_names = [path.name for path in model_dirs]
    scope = "all models" if config.models is None else "selected models"
    lines = [
        f"# OPUS QE statistics: {scope}",
        "",
        f"Input: `{config.merged_base}`",
        "",
        f"Models included ({len(model_names)}): {', '.join(model_names)}",
        "",
        (
            "Retention percentages are pooled over sentence-pair rows. "
            "Rows with null `qe_score` are included in the denominator and "
            "are not retained by any threshold."
        ),
        "",
    ]
    for table in tables:
        lines.extend(
            [
                f"## {table.title}",
                "",
                _markdown_table(table.columns, table.rows),
                "",
                f"CSV: `{table.csv_path}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote Markdown report: %s", path)


def run(config: OpusStatsConfig) -> list[TableResult]:
    model_dirs = _resolve_model_dirs(config)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = tuple(float(value) for value in config.thresholds)
    patterns = [_parquet_pattern(model_dir) for model_dir in model_dirs]
    scope = "all_models" if config.models is None else "selected_models"

    logger.info(
        "Summarizing OPUS merged parquet: models=%d output=%s",
        len(model_dirs),
        output_dir,
    )
    con = _connect(config)
    try:
        con.execute(build_source_view_sql(patterns))
        tables = [
            _run_table(
                con,
                "Global Summary",
                global_summary_sql(scope, thresholds),
                output_dir / "global_summary.csv",
            ),
            _run_table(
                con,
                "By Coverage Status",
                grouped_by_coverage_sql(thresholds),
                output_dir / "by_coverage_status.csv",
            ),
            _run_table(
                con,
                "English-Centric Contrast",
                grouped_by_english_centric_sql(thresholds),
                output_dir / "english_centric_contrast.csv",
            ),
        ]
        _write_report(
            output_dir / "opus_stats_report.md",
            config=config,
            model_dirs=model_dirs,
            tables=tables,
        )
        return tables
    finally:
        con.close()
