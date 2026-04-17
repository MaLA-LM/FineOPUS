from __future__ import annotations

from typing import Optional

from dataset.mediator import DatasetAdapter
from models.language_data.comet import COMET_SUPPORTED_LANGUAGES
from models.language_data.metricx24 import METRICX24_SUPPORTED_LANGUAGES
from models.language_data.prometheus import PROMETHEUS_SUPPORTED_LANGUAGES
from models.language_data.qwen import QWEN_SUPPORTED_LANGUAGES
from models.language_support.base import BaseLanguageSupport, LanguageCodeMapper

__all__ = [
    "CometLanguages",
    "QwenLanguages",
    "PrometheusLanguages",
    "MetricX24Languages",
]


class CometLanguages(BaseLanguageSupport):
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


class QwenLanguages(BaseLanguageSupport):
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


class PrometheusLanguages(BaseLanguageSupport):
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


class MetricX24Languages(BaseLanguageSupport):
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
