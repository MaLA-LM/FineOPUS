from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

Example = dict[str, str]
DirectionSpec = tuple[str, str, str, Path]

LanguageCodeMapper = Callable[
    [set[str], dict[str, str]],
    tuple[dict[str, set[str]], set[str], list[str]],
]
CodeNormalizer = Callable[[str], str]
Iso639Mapper = Callable[[str, dict[str, str]], str | None]

LoadParallel = Callable[..., list[Example]]
LimitRows = Callable[[list[Example], int | None], list[Example]]
DiscoverDirections = Callable[[str | Path, str | None], list[DirectionSpec]]
ExpectedDetailRows = Callable[[str | Path, str, str, int | None], int | None]


@dataclass(frozen=True)
class DatasetAdapter:
    id: str
    default_root: str | Path
    split_values: tuple[str, ...]
    load_parallel: LoadParallel
    limit_rows: LimitRows
    discover_directions: DiscoverDirections
    expected_detail_rows: ExpectedDetailRows
    language_codes: dict[str, str] | None = None
    name_to_code_mapper: LanguageCodeMapper | None = None
    code_normalizer: CodeNormalizer | None = None
    iso639_1_from_code: Iso639Mapper | None = None


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


from dataset.adapters.flores200 import FLORES200_ADAPTER  # noqa: E402

DATASETS = {FLORES200_ADAPTER.id: FLORES200_ADAPTER}
