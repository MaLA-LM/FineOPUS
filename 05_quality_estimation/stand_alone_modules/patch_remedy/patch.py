from pathlib import Path

import duckdb

from dataset.flores200_scripts.langcode_mapping import build_model_language_mapping
from models.language_data.remedy import REMEDY_SUPPORTED_LANGUAGES

from .queries import patch_query, validation_query


SCRATCH_TMP = (
    "/scratch/project_462001050/QE_flores200_scores/dataset=flores200/duckdb_tmp"
)

MODEL = "shaomutan_remedy-9b-22"

src_root = Path(
    "/scratch/project_462001050/QE_flores200_scores/dataset=flores200/buckets=normalized_scores"
)
dst_root = Path(
    "/scratch/project_462001050/QE_flores200_scores/dataset=flores200/buckets=normalized_scores_patched"
)


def _build_supported_codes() -> set[str]:
    mapping = build_model_language_mapping(REMEDY_SUPPORTED_LANGUAGES)
    return {code for code, info in mapping.items() if info[0]}


def _validate(con, model_out: str):
    src_pattern = str(src_root / f"model={MODEL}" / "**" / "*.parquet")
    dst_pattern = f"{model_out}/**/*.parquet"

    row = con.execute(validation_query(src_pattern, dst_pattern)).fetchone()
    old_rows, new_rows, src_changed, tgt_changed = row

    print(f"  Rows  — old: {old_rows}  new: {new_rows}")
    if old_rows != new_rows:
        raise RuntimeError(
            f"[validate:{MODEL}] row count mismatch: {old_rows} vs {new_rows}"
        )
    print(f"  src_lang_seen changed: {src_changed}")
    print(f"  tgt_lang_seen changed: {tgt_changed}")
    print(f"  Validation passed for {MODEL}")


def run():
    dst_root.mkdir(parents=True, exist_ok=True)

    supported_codes = _build_supported_codes()
    print(f"Remedy supported FLORES codes: {len(supported_codes)}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute(f"PRAGMA temp_directory='{SCRATCH_TMP}';")

    con.execute("CREATE TEMP TABLE remedy_supported(code VARCHAR)")
    con.executemany(
        "INSERT INTO remedy_supported VALUES (?)",
        [(c,) for c in supported_codes],
    )

    print(f"Patching model: {MODEL}")
    model_pattern = str(src_root / f"model={MODEL}" / "**" / "*.parquet")
    model_out = (dst_root / f"model={MODEL}").as_posix()

    con.execute(patch_query(model_pattern, model_out))
    print(f"Finished writing: {model_out}")

    _validate(con, model_out)
    print("Done.")
