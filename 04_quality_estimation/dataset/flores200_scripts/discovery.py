from __future__ import annotations

from functools import lru_cache
from pathlib import Path

EXPECTED_ROWS = {"dev": 997, "devtest": 1012}
SPLIT_VALUES = ("dev", "devtest")

DirectionSpec = tuple[str, str, str, Path]


# return split or all splits
def _resolve_splits(split: str | None) -> list[str]:
    if split is None or split == "all":
        return list(SPLIT_VALUES)
    return [split]


# return all languages in a given split
def _discover_languages(root: Path, split: str) -> list[str]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    suffix = f".{split}"
    languages: list[str] = []
    for path in split_dir.glob(f"*{suffix}"):
        if not path.is_file() or not path.name.endswith(suffix):
            continue
        lang = path.name[: -len(suffix)]
        if lang:
            languages.append(lang)
    return sorted(set(languages))


def discover_directions(
    root: str | Path, split: str | None = None
) -> list[DirectionSpec]:
    root_path = Path(root)
    directions: list[DirectionSpec] = []
    for split_name in _resolve_splits(split):
        languages = _discover_languages(root_path, split_name)
        split_dir = root_path / split_name
        for src_lang in languages:
            src_path = split_dir / f"{src_lang}.{split_name}"
            for tgt_lang in languages:
                if src_lang == tgt_lang:
                    continue
                directions.append((src_lang, tgt_lang, split_name, src_path))
    return directions


@lru_cache(maxsize=None)
def _count_lines_cached(path_str: str) -> int:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def expected_detail_rows(
    root: str | Path, split: str, src_lang: str, max_rows: int | None
) -> int | None:
    total = EXPECTED_ROWS.get(split)
    src_path = Path(root) / split / f"{src_lang}.{split}"
    try:
        actual = _count_lines_cached(str(src_path))
    except FileNotFoundError:
        actual = None
    if actual is not None:
        if total is None or actual != total:
            total = actual
    if total is None:
        return max_rows
    if max_rows is None:
        return total
    return min(max_rows, total)
