"""Regression tests for shared vLLM structured-output backend selection.

Run with:
    python -m execution.opus_queue.tests.test_llm_engine_backend
"""

from __future__ import annotations

import os
from unittest.mock import patch

from src.backends.llm.backend.constants import RESPONSE_FORMAT_JSON_SCHEMA, RESPONSE_FORMAT_NONE
from src.backends.llm.backend.engine import _resolve_structured_outputs_backend


def test_defaults_to_outlines_for_structured_outputs() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VLLM_STRUCTURED_OUTPUTS_BACKEND", None)
        os.environ.pop("STRUCTURED_OUTPUTS_BACKEND", None)
        assert (
            _resolve_structured_outputs_backend(RESPONSE_FORMAT_JSON_SCHEMA)
            == "outlines"
        )


def test_prefers_vllm_env_over_legacy_env() -> None:
    with patch.dict(
        os.environ,
        {
            "VLLM_STRUCTURED_OUTPUTS_BACKEND": " guidance ",
            "STRUCTURED_OUTPUTS_BACKEND": "outlines",
        },
        clear=False,
    ):
        assert (
            _resolve_structured_outputs_backend(RESPONSE_FORMAT_JSON_SCHEMA)
            == "guidance"
        )


def test_uses_legacy_env_when_vllm_env_is_missing() -> None:
    with patch.dict(
        os.environ,
        {
            "STRUCTURED_OUTPUTS_BACKEND": "lm-format-enforcer",
        },
        clear=False,
    ):
        os.environ.pop("VLLM_STRUCTURED_OUTPUTS_BACKEND", None)
        assert (
            _resolve_structured_outputs_backend(RESPONSE_FORMAT_JSON_SCHEMA)
            == "lm-format-enforcer"
        )


def test_disables_backend_when_response_format_is_none() -> None:
    with patch.dict(
        os.environ,
        {
            "VLLM_STRUCTURED_OUTPUTS_BACKEND": "outlines",
            "STRUCTURED_OUTPUTS_BACKEND": "xgrammar",
        },
        clear=False,
    ):
        assert _resolve_structured_outputs_backend(RESPONSE_FORMAT_NONE) is None


def run_test() -> None:
    test_defaults_to_outlines_for_structured_outputs()
    test_prefers_vllm_env_over_legacy_env()
    test_uses_legacy_env_when_vllm_env_is_missing()
    test_disables_backend_when_response_format_is_none()


if __name__ == "__main__":
    run_test()
