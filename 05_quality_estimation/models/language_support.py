from __future__ import annotations

from typing import Callable, Literal, Optional

from dataset.mediator import DatasetAdapter, DEFAULT_DATASET_ID, get_dataset
from models.language_data.comet import COMET_SUPPORTED_LANGUAGES
from models.language_data.metricx24 import METRICX24_SUPPORTED_LANGUAGES
from models.language_data.qwen import QWEN_SUPPORTED_LANGUAGES

LanguageCodeMapper = Callable[
    [set[str], dict[str, str]],
    tuple[dict[str, set[str]], set[str], list[str]],
]
CodeNormalizer = Callable[[str], str]
SupportStatus = bool | Literal["unknown"]


def _resolve_dataset(dataset: DatasetAdapter | str | None) -> DatasetAdapter:
    if isinstance(dataset, DatasetAdapter):
        return dataset
    return get_dataset(dataset or DEFAULT_DATASET_ID)


def _resolve_language_codes(
    language_codes: Optional[dict[str, str]],
    dataset: DatasetAdapter,
) -> dict[str, str] | None:
    if language_codes is not None:
        return language_codes
    return dataset.language_codes


def _resolve_name_to_code_mapper(
    name_to_code_mapper: Optional[LanguageCodeMapper],
    dataset: DatasetAdapter,
) -> LanguageCodeMapper | None:
    if name_to_code_mapper is not None:
        return name_to_code_mapper
    return dataset.name_to_code_mapper


def _default_code_normalizer(code: str) -> str:
    return code.strip().replace("-", "_")


class _BaseLanguageSupport:
    def __init__(
        self,
        supported_languages: set[str],
        language_codes: Optional[dict[str, str]],
        label: str,
        *,
        dataset: DatasetAdapter | str | None = None,
        name_to_code_mapper: Optional[LanguageCodeMapper] = None,
        code_normalizer: Optional[CodeNormalizer] = None,
    ) -> None:
        dataset = _resolve_dataset(dataset)
        self._supported_languages = supported_languages
        langcode = _resolve_language_codes(language_codes, dataset)
        mapper = _resolve_name_to_code_mapper(name_to_code_mapper, dataset)
        self._code_normalizer = (
            code_normalizer or dataset.code_normalizer or _default_code_normalizer
        )
        self._support_known = langcode is not None and mapper is not None
        if not self._support_known:
            self._name_to_codes = {}
            self._supported_codes = set()
            self._unmatched_supported_names = []
            return
        (
            self._name_to_codes,
            self._supported_codes,
            unmatched,
        ) = mapper(self._supported_languages, langcode)
        self._unmatched_supported_names = sorted(unmatched)

        if self._unmatched_supported_names:
            print(
                f"Unmatched language names for {label}: "
                f"{self._unmatched_supported_names}"
            )

    def get_all_languages(self) -> list[str]:
        return sorted(self._supported_languages)

    def get_supported_codes(self) -> set[str]:
        return set(self._supported_codes)

    def codes_for_name(self, name: str) -> set[str]:
        return set(self._name_to_codes.get(name, set()))

    def is_code_supported(self, code: str) -> bool:
        if not self._support_known:
            return False
        if not code:
            return False
        normalized = self._code_normalizer(code)
        return normalized in self._supported_codes

    def support_status(self, code: str) -> SupportStatus:
        if not self._support_known:
            return "unknown"
        return self.is_code_supported(code)

    def debug_unmatched(self) -> list[str]:
        if not self._support_known:
            return []
        return list(self._unmatched_supported_names)


class CometLanguages(_BaseLanguageSupport):
    """Class representing languages supported by COMET model."""

    def __init__(
        self,
        language_codes: Optional[dict[str, str]] = None,
        *,
        dataset: DatasetAdapter | str | None = None,
        name_to_code_mapper: Optional[LanguageCodeMapper] = None,
        code_normalizer: Optional[CodeNormalizer] = None,
    ) -> None:
        super().__init__(
            set(COMET_SUPPORTED_LANGUAGES),
            language_codes,
            "COMET",
            dataset=dataset,
            name_to_code_mapper=name_to_code_mapper,
            code_normalizer=code_normalizer,
        )


class QwenLanguages(_BaseLanguageSupport):
    """Class representing languages supported by Qwen model."""

    def __init__(
        self,
        language_codes: Optional[dict[str, str]] = None,
        *,
        dataset: DatasetAdapter | str | None = None,
        name_to_code_mapper: Optional[LanguageCodeMapper] = None,
        code_normalizer: Optional[CodeNormalizer] = None,
    ) -> None:
        super().__init__(
            set(QWEN_SUPPORTED_LANGUAGES),
            language_codes,
            "qwen3-14b",
            dataset=dataset,
            name_to_code_mapper=name_to_code_mapper,
            code_normalizer=code_normalizer,
        )


class MetricX24Languages(_BaseLanguageSupport):
    """Class representing languages supported by MetricX-24 model."""

    def __init__(
        self,
        language_codes: Optional[dict[str, str]] = None,
        *,
        dataset: DatasetAdapter | str | None = None,
        name_to_code_mapper: Optional[LanguageCodeMapper] = None,
        code_normalizer: Optional[CodeNormalizer] = None,
    ) -> None:
        super().__init__(
            set(METRICX24_SUPPORTED_LANGUAGES),
            language_codes,
            "MetricX-24",
            dataset=dataset,
            name_to_code_mapper=name_to_code_mapper,
            code_normalizer=code_normalizer,
        )
