from __future__ import annotations

from typing import Callable, Optional

from dataset.mediator import DatasetAdapter, get_dataset

LanguageCodeMapper = Callable[[set[str]], dict[str, list[object]]]

__all__ = ["BaseLanguageSupport", "LanguageCodeMapper"]


def _resolve_dataset(dataset: DatasetAdapter | str | None) -> DatasetAdapter:
    if isinstance(dataset, DatasetAdapter):
        return dataset
    return get_dataset(dataset)


def _resolve_language_codes(
    language_codes: Optional[dict[str, str]], dataset: DatasetAdapter
) -> Optional[dict[str, str]]:
    if language_codes is not None:
        return language_codes
    return dataset.language_codes


def _resolve_langcode_mapper(
    build_model_language_mapping: Optional[LanguageCodeMapper], dataset: DatasetAdapter
) -> Optional[LanguageCodeMapper]:
    if build_model_language_mapping is not None:
        return build_model_language_mapping
    return dataset.langcode_to_name


class BaseLanguageSupport:
    def __init__(
        self,
        supported_languages: set[str],
        language_codes: Optional[dict[str, str]],
        label: str,
        *,
        dataset: DatasetAdapter | str | None = None,
        langcode_to_name: Optional[LanguageCodeMapper] = None,
    ) -> None:
        dataset = _resolve_dataset(dataset)
        self._supported_languages = supported_languages
        self._label = label
        langcode = _resolve_language_codes(language_codes, dataset)
        mapper = _resolve_langcode_mapper(langcode_to_name, dataset)

        self._language_codes: dict[str, str] = langcode or {}
        self._support_known = langcode is not None and mapper is not None
        if not self._support_known:
            return

        self._name_to_codes = mapper(self._supported_languages)

    def get_all_languages(self) -> list[str]:
        return sorted(self._name_to_codes.keys())

    def get_supported_codes(self) -> set[str]:
        codes = set()
        for code_list in self._name_to_codes.values():
            codes.update(code_list[1])
        return codes

    def is_code_supported(self, code: str) -> bool:
        if not self._support_known:
            return False
        if not code:
            return False
        code_list = self._name_to_codes.get(code)
        if not code_list:
            return False
        return bool(code_list[0])

    def support_status(self, code: str) -> str | bool:
        if not self._support_known:
            return "unknown"
        return self.is_code_supported(code)

    def get_full_language_name(self, lang_code: str) -> str:
        if self._support_known:
            code_list = self._name_to_codes.get(lang_code)
            if code_list and len(code_list) > 1 and code_list[1]:
                return code_list[1]
        display_name = self._language_codes.get(lang_code)
        if display_name:
            return display_name
        return lang_code
