from __future__ import annotations

from dataset.mediator import DatasetAdapter
from dataset.opus.builder import (
    limit_rows,
    load_opus_parallel,
)
from dataset.opus.discovery import (
    DEFAULT_OPUS_ROOT,
    SPLIT_VALUES,
    discover_directions,
    expected_detail_rows,
)
from dataset.opus.frames import build_frames as build_opus_frames
from dataset.opus.langcode_mapping import build_model_language_mapping
from dataset.opus.langcodes import opus_langcodes


OPUS_ADAPTER = DatasetAdapter(
    id="opus",
    default_root=DEFAULT_OPUS_ROOT,
    split_values=tuple(SPLIT_VALUES),
    load_parallel=load_opus_parallel,
    limit_rows=limit_rows,
    discover_directions=discover_directions,
    expected_detail_rows=expected_detail_rows,
    language_codes=opus_langcodes,
    langcode_to_name=build_model_language_mapping,
    build_frames=build_opus_frames,
)
