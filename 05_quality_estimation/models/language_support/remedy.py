from __future__ import annotations

from typing import Optional

from dataset.mediator import DatasetAdapter
from models.language_data.remedy import REMEDY_ISO_MAP, REMEDY_SUPPORTED_LANGUAGES
from models.language_support.base import (
    BaseLanguageSupport,
    LanguageCodeMapper,
    _resolve_dataset,
    _resolve_langcode_mapper,
)

__all__ = ["RemedyLanguages"]


class RemedyLanguages(BaseLanguageSupport):
    _LANGUAGE_TO_ISO = {name: code for code, name in REMEDY_ISO_MAP.items()}

    def __init__(
        self,
        language_codes: Optional[dict[str, str]] = None,
        *,
        dataset: DatasetAdapter | str | None = None,
        langcode_to_name: Optional[LanguageCodeMapper] = None,
    ) -> None:
        super().__init__(
            set(REMEDY_SUPPORTED_LANGUAGES),
            language_codes,
            "ReMedy",
            dataset=dataset,
            langcode_to_name=langcode_to_name,
        )
        self._iso_codes_by_dataset_code = self._build_iso_lookup(
            dataset, langcode_to_name
        )

    @classmethod
    def _build_iso_lookup(
        cls,
        dataset: DatasetAdapter | str | None,
        langcode_to_name: Optional[LanguageCodeMapper],
    ) -> dict[str, str]:
        dataset_obj = _resolve_dataset(dataset)
        mapper = _resolve_langcode_mapper(langcode_to_name, dataset_obj)
        if mapper is None:
            return {}
        mapping = mapper(set(REMEDY_ISO_MAP.values()))
        result: dict[str, str] = {}
        for lang_code, info in mapping.items():
            if not info or len(info) < 2:
                continue
            matched = info[1]
            if not matched:
                continue
            iso_code = cls._LANGUAGE_TO_ISO.get(matched)
            if iso_code:
                result[lang_code] = iso_code
        return result

    def iso639_1_from_language(self, language: str) -> Optional[str]:
        if not language:
            return None
        return self._LANGUAGE_TO_ISO.get(language)

    def iso639_1_from_code(self, lang_code: str) -> Optional[str]:
        if not lang_code:
            return None
        return self._iso_codes_by_dataset_code.get(lang_code)
