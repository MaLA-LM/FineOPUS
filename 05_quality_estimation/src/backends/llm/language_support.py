from __future__ import annotations

from dataset.mediator import DatasetAdapter
from models.language_support import (
    BaseLanguageSupport,
    PrometheusLanguages,
    QwenLanguages,
)
from src.backends.llm.backend.constants import RESPONSE_FORMAT_JSON_SCHEMA
from utils.logger import logger

__all__ = ["auto_response_format", "select_language_support"]


def select_language_support(
    model_key: str,
    dataset: DatasetAdapter,
) -> BaseLanguageSupport:
    if model_key.startswith("qwen3-"):
        language_support: BaseLanguageSupport = QwenLanguages(dataset=dataset)
        logger.info("Using QwenLanguages for model_key=%s", model_key)
        return language_support
    if model_key.startswith("m-prometheus-"):
        language_support = PrometheusLanguages(dataset=dataset)
        logger.info("Using PrometheusLanguages for model_key=%s", model_key)
        return language_support

    logger.info("Using default QwenLanguages for model_key=%s", model_key)
    return QwenLanguages(dataset=dataset)


def auto_response_format(model_key: str) -> str:
    _ = model_key
    return RESPONSE_FORMAT_JSON_SCHEMA
