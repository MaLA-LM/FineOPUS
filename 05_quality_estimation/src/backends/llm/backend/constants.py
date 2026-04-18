from __future__ import annotations

from typing import Literal

PromptMode = Literal["detailed", "simple", "batch"]
PROMPT_MODE_DETAILED: PromptMode = "detailed"
PROMPT_MODE_SIMPLE: PromptMode = "simple"
PROMPT_MODE_BATCH: PromptMode = "batch"
PROMPT_MODES: tuple[PromptMode, ...] = (
    PROMPT_MODE_DETAILED,
    PROMPT_MODE_SIMPLE,
    PROMPT_MODE_BATCH,
)

ResponseFormat = Literal["none", "json_object", "json_schema"]
RESPONSE_FORMAT_NONE: ResponseFormat = "none"
RESPONSE_FORMAT_JSON_OBJECT: ResponseFormat = "json_object"
RESPONSE_FORMAT_JSON_SCHEMA: ResponseFormat = "json_schema"
RESPONSE_FORMATS: tuple[ResponseFormat, ...] = (
    RESPONSE_FORMAT_NONE,
    RESPONSE_FORMAT_JSON_OBJECT,
    RESPONSE_FORMAT_JSON_SCHEMA,
)

_SIMPLE_MAX_TOKENS = 32
_BATCH_TOKENS_PER_ITEM = 150
_RETRY_TEMPERATURE = 0.1
