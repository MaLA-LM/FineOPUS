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
from dataset.flores200_scripts.langcode_mapping import (
    apply_alias,
    base_name_from_flores_display,
    map_model_names_to_flores_codes,
)
from dataset.flores200_scripts.langcodes import flores200_langcodes
from dataset.mediator import DatasetAdapter


def iso639_1_from_flores(code: str, iso_map: dict[str, str]) -> str | None:
    if not code:
        return None
    display = flores200_langcodes.get(code)
    if not display:
        return None
    base = apply_alias(base_name_from_flores_display(display))
    return iso_map.get(base)


FLORES200_ADAPTER = DatasetAdapter(
    id="flores200",
    default_root=DEFAULT_FLORES_ROOT,
    split_values=tuple(SPLIT_VALUES),
    load_parallel=load_flores200_parallel,
    limit_rows=limit_rows,
    discover_directions=discover_directions,
    expected_detail_rows=expected_detail_rows,
    language_codes=flores200_langcodes,
    name_to_code_mapper=map_model_names_to_flores_codes,
    iso639_1_from_code=iso639_1_from_flores,
)
