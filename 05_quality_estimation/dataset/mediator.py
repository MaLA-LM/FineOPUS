from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

Example = dict[str, Any]
DirectionSpec = tuple[str, str, str, Path]

LoadParallel = Callable[[str | Path, str, str], list[Example]]
LimitRows = Callable[[list[Example], int | None], list[Example]]
DiscoverDirections = Callable[[str | Path, str | None], list[DirectionSpec]]
ExpectedDetailRows = Callable[[str | Path, str, str, int | None], int | None]
LanguageCodeMapper = Callable[[set[str]], dict[str, list[object]]]
FrameBuilder = Callable[..., Any]  # returns a pandas.DataFrame


@dataclass(frozen=True)
class DatasetAdapter:
    id: str
    default_root: str | Path
    split_values: tuple[str, ...]
    load_parallel: LoadParallel
    limit_rows: LimitRows
    discover_directions: DiscoverDirections
    expected_detail_rows: ExpectedDetailRows
    language_codes: dict[str, str]
    langcode_to_name: LanguageCodeMapper
    build_frames: FrameBuilder


def limit_rows(examples: list[Example], max_rows: int | None) -> list[Example]:
    """Shared default row limiter used by dataset adapters."""
    if max_rows is None:
        return examples
    return examples[: min(max_rows, len(examples))]


DEFAULT_DATASET_ID = "flores200"


def get_dataset(dataset_id: str | None) -> DatasetAdapter:
    if not dataset_id:
        dataset_id = DEFAULT_DATASET_ID
    key = dataset_id.strip().lower()
    try:
        return DATASETS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASETS.keys()))
        raise SystemExit(
            f"Unknown dataset '{dataset_id}'. Supported: {supported}."
        ) from exc


def list_datasets() -> list[str]:
    return sorted(DATASETS.keys())


from dataset.flores200 import FLORES200_ADAPTER  # noqa: E402
from dataset.opus import OPUS_ADAPTER  # noqa: E402

DATASETS = {
    FLORES200_ADAPTER.id: FLORES200_ADAPTER,
    OPUS_ADAPTER.id: OPUS_ADAPTER,
}
