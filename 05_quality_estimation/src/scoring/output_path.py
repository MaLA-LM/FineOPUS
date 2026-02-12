from __future__ import annotations

import re
from pathlib import Path


def sanitize_model_tag(name: str) -> str:
    tag = name.strip().lower()
    tag = re.sub(r"[^a-z0-9._-]+", "-", tag)
    tag = tag.strip("-")
    return tag or "model"


def build_output_path(
    output_base: str | Path,
    dataset: str,
    model_tag: str,
    split: str,
    src_lang: str,
    tgt_lang: str,
) -> Path:
    return (
        Path(output_base)
        / f"dataset={dataset}"
        / f"model={model_tag}"
        / f"split={split}"
        / f"src_lang={src_lang}"
        / f"tgt_lang={tgt_lang}"
        / "part.parquet"
    )
