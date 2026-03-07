from __future__ import annotations

from typing import Callable, Literal, Optional

from dataset.mediator import DatasetAdapter, DEFAULT_DATASET_ID, get_dataset
from models.language_data.comet import COMET_SUPPORTED_LANGUAGES
from models.language_data.metricx24 import METRICX24_SUPPORTED_LANGUAGES
from models.language_data.prometheus import PROMETHEUS_SUPPORTED_LANGUAGES
from models.language_data.qwen import QWEN_SUPPORTED_LANGUAGES
from models.language_data.remedy import REMEDY_SUPPORTED_LANGUAGES, REMEDY_ISO_MAP


LanguageCodeMapper = Callable[[set[str]], dict[str, list[object]]]


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


class _BaseLanguageSupport:
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
        langcode = _resolve_language_codes(language_codes, dataset)
        mapper = _resolve_langcode_mapper(langcode_to_name, dataset)

        self._support_known = langcode is not None and mapper is not None
        if not self._support_known:
            self._code_to_name = {}
            return

        self._name_to_codes = mapper(self._supported_languages)

    # mapper -> mapping[language_code] = [is_supported, matched_model_language]

    # dataset langcodes
    def get_all_languages(self) -> list[str]:
        return sorted(self._name_to_codes.keys())

    def get_supported_codes(self) -> set[str]:
        codes = set()
        for code_list in self._name_to_codes.values():
            codes.update(code_list[1])  #  language
        return codes

    def is_code_supported(self, code: str) -> bool:
        if not self._support_known:
            return False
        if not code:
            return False
        code_list = self._name_to_codes.get(code)
        if not code_list:
            return False
        return bool(code_list[0])  # is_supported

    def support_status(self, code: str) -> str | bool:
        if not self._support_known:
            return "unknown"
        return self.is_code_supported(code)

    # get the full language name for a given code, if supported; otherwise return the code itself
    def get_full_language_name(self, lang_code: str) -> str:
        if not self._support_known:
            return lang_code
        name_to_codes = self._name_to_codes
        code_list = self._name_to_codes.get(lang_code)
        if code_list:
            return code_list[1] if len(code_list) > 1 else lang_code
        return lang_code


class CometLanguages(_BaseLanguageSupport):
    """Class representing languages supported by COMET model."""

    def __init__(
        self,
        language_codes: Optional[dict[str, str]] = None,
        *,
        dataset: DatasetAdapter | str | None = None,
        langcode_to_name: Optional[LanguageCodeMapper] = None,
    ) -> None:
        super().__init__(
            set(COMET_SUPPORTED_LANGUAGES),
            language_codes,
            "COMET",
            dataset=dataset,
            langcode_to_name=langcode_to_name,
        )


class QwenLanguages(_BaseLanguageSupport):
    """Class representing languages supported by Qwen model."""

    def __init__(
        self,
        language_codes: Optional[dict[str, str]] = None,
        *,
        dataset: DatasetAdapter | str | None = None,
        langcode_to_name: Optional[LanguageCodeMapper] = None,
    ) -> None:
        super().__init__(
            set(QWEN_SUPPORTED_LANGUAGES),
            language_codes,
            "qwen3",
            dataset=dataset,
            langcode_to_name=langcode_to_name,
        )


class PrometheusLanguages(_BaseLanguageSupport):
    """Class representing languages supported by Prometheus model."""

    def __init__(
        self,
        language_codes: Optional[dict[str, str]] = None,
        *,
        dataset: DatasetAdapter | str | None = None,
        langcode_to_name: Optional[LanguageCodeMapper] = None,
    ) -> None:
        super().__init__(
            set(PROMETHEUS_SUPPORTED_LANGUAGES),
            language_codes,
            "Prometheus",
            dataset=dataset,
            langcode_to_name=langcode_to_name,
        )


class MetricX24Languages(_BaseLanguageSupport):
    """Class representing languages supported by MetricX-24 model."""

    def __init__(
        self,
        language_codes: Optional[dict[str, str]] = None,
        *,
        dataset: DatasetAdapter | str | None = None,
        langcode_to_name: Optional[LanguageCodeMapper] = None,
    ) -> None:
        super().__init__(
            set(METRICX24_SUPPORTED_LANGUAGES),
            language_codes,
            "MetricX-24",
            dataset=dataset,
            langcode_to_name=langcode_to_name,
        )


class RemedyLanguages(_BaseLanguageSupport):
    """Language helper for ReMedy"""

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

    # REMEDY_ISO_MAP = { code: language_name, ... }
    def _iso639_1_from_language(self, language: str) -> Optional[str]:
        """Get ISO 639-1 code from a language code"""
        if not language:
            return None
        for code, name in REMEDY_ISO_MAP.items():
            if name == language:
                return code
        return None
