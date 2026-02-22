from __future__ import annotations

from dataset.flores200_scripts.discovery import (
    SPLIT_VALUES,
    discover_directions,
    expected_detail_rows,
)
from dataset.flores200_scripts.flores200_builder import (
    DEFAULT_FLORES_ROOT,
    limit_rows,
    load_flores200_parallel,
)
from dataset.flores200_scripts.langcode_mapping import build_model_language_mapping

from dataset.flores200_scripts.langcodes import flores200_langcodes
from dataset.mediator import DatasetAdapter


FLORES200_ADAPTER = DatasetAdapter(
    id="flores200",
    default_root=DEFAULT_FLORES_ROOT,
    split_values=tuple(SPLIT_VALUES),
    load_parallel=load_flores200_parallel,
    limit_rows=limit_rows,
    discover_directions=discover_directions,
    expected_detail_rows=expected_detail_rows,
    language_codes=flores200_langcodes,
    langcode_to_name=build_model_language_mapping,
)
