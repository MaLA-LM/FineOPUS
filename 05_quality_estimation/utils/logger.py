from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from threading import Lock

__all__ = ["logger", "setup_logger"]

_LOGGER_NAME = "quality_estimation"
_DEFAULT_LEVEL = logging.INFO
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_BACKUP_COUNT = 5  # Keep up to 5 old log files
_FORMAT = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIG_LOCK = Lock()


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def _resolve_level(value: str | None) -> int:
    if not value:
        return _DEFAULT_LEVEL
    normalized = value.strip().upper()
    if normalized.isdigit():
        return int(normalized)
    level = logging.getLevelNamesMapping().get(normalized)
    if isinstance(level, int):
        return level
    return _DEFAULT_LEVEL


def _resolve_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)


def _build_stdout_handler(formatter: logging.Formatter) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.addFilter(_MaxLevelFilter(logging.WARNING))
    handler.setFormatter(formatter)
    return handler


def _build_stderr_handler(formatter: logging.Formatter) -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.ERROR)
    handler.setFormatter(formatter)
    return handler


def _build_rotating_handler(
    path: str,
    formatter: logging.Formatter,
    level: int,
    max_bytes: int,
    backup_count: int,
) -> logging.Handler:
    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def setup_logger() -> logging.Logger:
    with _CONFIG_LOCK:
        configured_logger = logging.getLogger(_LOGGER_NAME)
        if getattr(configured_logger, "_qe_configured", False):
            return configured_logger

        configured_logger.setLevel(_resolve_level(os.getenv("LOG_LEVEL")))
        configured_logger.propagate = False

        formatter = _build_formatter()
        configured_logger.addHandler(_build_stdout_handler(formatter))
        configured_logger.addHandler(_build_stderr_handler(formatter))

        log_file = os.getenv("LOG_FILE")
        err_file = os.getenv("LOG_ERR_FILE")
        max_bytes = _resolve_int_env("LOG_MAX_BYTES", _DEFAULT_MAX_BYTES)
        backup_count = _resolve_int_env("LOG_BACKUP_COUNT", _DEFAULT_BACKUP_COUNT)

        if log_file:
            file_handler = _build_rotating_handler(
                log_file,
                formatter,
                logging.DEBUG,
                max_bytes,
                backup_count,
            )
            if err_file:
                file_handler.addFilter(_MaxLevelFilter(logging.WARNING))
            configured_logger.addHandler(file_handler)

        if err_file:
            configured_logger.addHandler(
                _build_rotating_handler(
                    err_file,
                    formatter,
                    logging.ERROR,
                    max_bytes,
                    backup_count,
                )
            )

        for noisy_logger_name in ("urllib3", "asyncio"):
            logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

        configured_logger._qe_configured = True  # type: ignore[attr-defined]
        return configured_logger


logger = setup_logger()
