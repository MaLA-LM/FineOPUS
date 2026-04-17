from pathlib import Path

import duckdb

from dataset.flores200.langfamily import CODE_TO_FAMILY

from .queries import normalization_query
from .validate import validate_output

SCRATCH_TMP = (
    "/scratch/project_462001050/QE_flores200_scores/dataset=flores200/duckdb_tmp"
)

DEFAULT_MODELS = [
    "qwen3-4b-instruct-2507-detailed",
    "qwen3-4b-instruct-2507-simple",
]
DEFAULT_SRC_ROOT = Path(
    "/scratch/project_462001050/QE_flores200_scores/dataset=flores200/buckets=raw_scores_new"
)
DEFAULT_DST_ROOT = Path(
    "/scratch/project_462001050/QE_flores200_scores/dataset=flores200/buckets=normalized_scores"
)


def run(config=None):
    if config is None:
        src_root = DEFAULT_SRC_ROOT
        dst_root = DEFAULT_DST_ROOT
        models = DEFAULT_MODELS
    else:
        src_root = config.src_root
        dst_root = config.dst_root
        models = config.models

    dst_root.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute(f"PRAGMA temp_directory='{SCRATCH_TMP}';")

    con.execute("CREATE TEMP TABLE lang_family(code VARCHAR, family VARCHAR)")
    con.executemany(
        "INSERT INTO lang_family VALUES (?, ?)", list(CODE_TO_FAMILY.items())
    )

    for model in models:
        print(f"Processing model: {model}")

        model_pattern = str(src_root / f"model={model}" / "**" / "part-*.parquet")
        model_out = (dst_root / f"model={model}").as_posix()

        con.execute(normalization_query(model_pattern, model_out))
        print(f"Finished model: {model}")

        if model == models[0]:
            validate_output(con, model_out, model)

    print("Done.")
