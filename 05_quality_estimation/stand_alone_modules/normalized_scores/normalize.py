from pathlib import Path

import duckdb

from dataset.flores200_scripts.langfamily import CODE_TO_FAMILY

from .queries import normalization_query
from .validate import validate_output


SCRATCH_TMP = (
    "/scratch/project_462001050/QE_flores200_scores/dataset=flores200/duckdb_tmp"
)

models = [
    "bicleaner-ai",
    "m-prometheus-7b",
    "qwen3-4b-instruct-2507",
    "shaomutan_remedy-9b-22",
    "xcomet-xl",
    "metricx24",
    "qwen3-14b",
    "qwen3-8b",
    "wmt23-cometkiwi-da-xl",
]

src_root = Path(
    "/scratch/project_462001050/QE_flores200_scores/dataset=flores200/buckets=raw_scores"
)
dst_root = Path(
    "/scratch/project_462001050/QE_flores200_scores/dataset=flores200/buckets=normalized_scores"
)


def run():
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
