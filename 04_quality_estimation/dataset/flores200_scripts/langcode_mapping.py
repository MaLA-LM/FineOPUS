
from __future__ import annotations

import logging
import re
from typing import Optional

LOGGER = logging.getLogger(__name__)

NAME_ALIASES: dict[str, str] = {
    "oriya": "odia",
    "persian": "western persian",
    "western frisian": "frisian",
    "burmese": "myanmar",
    "filipino": "tagalog",
    "scottish gaelic": "gaelic",
    "uyghur": "uighur",
}

_PUNCT_RE = re.compile(r"[-_/.,:;]+")
_APOSTROPHE_RE = re.compile(r"[']+")
_WHITESPACE_RE = re.compile(r"\s+")
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
_ROMANIZED_RE = re.compile(r"\s+romanized\s*$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    cleaned = text.strip().casefold()
    cleaned = cleaned.replace("&", " and ")
    cleaned = _APOSTROPHE_RE.sub("", cleaned)
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    
    # Extract last word if multiple words present
    words = cleaned.split()
    if len(words) > 1:
        cleaned = words[-1]
    
    return cleaned


def _strip_trailing_parentheticals(text: str) -> str:
    cleaned = text.strip()
    while True:
        updated = _PAREN_RE.sub("", cleaned).strip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def base_name_from_flores_display(display: str) -> str:
    return normalize_text(_strip_trailing_parentheticals(display))


def base_name_from_model_name(name: str) -> str:
    cleaned = _ROMANIZED_RE.sub("", name.strip()).strip()
    cleaned = _strip_trailing_parentheticals(cleaned)
    return normalize_text(cleaned)


def script_preference_from_model_name(name: str) -> Optional[str]:
    normalized = normalize_text(name)
    if "romanized" in normalized:
        return "Latn"
    if "chinese" in normalized and "simplified" in normalized:
        return "Hans"
    if "chinese" in normalized and "traditional" in normalized:
        return "Hant"
    return None


def script_from_flores_code(code: str) -> Optional[str]:
    cleaned = code.strip()
    if "_" not in cleaned:
        return None
    parts = cleaned.split("_")
    if len(parts) != 2:
        return None
    script = parts[1]
    if len(script) != 4 or not script.isalpha():
        return None
    return script


def build_flores_index(flores200_langcodes: dict[str, str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for code, display_name in flores200_langcodes.items():
        base = base_name_from_flores_display(display_name)
        index.setdefault(base, []).append(code)
    return index


def apply_alias(base: str) -> str:
    return NAME_ALIASES.get(base, base)


def map_model_names_to_flores_codes(
    supported_names: set[str],
    flores200_langcodes: dict[str, str],
) -> tuple[dict[str, set[str]], set[str], list[str]]:
    flores_index = build_flores_index(flores200_langcodes)
    name_to_codes: dict[str, set[str]] = {}
    supported_codes: set[str] = set()
    unmatched: list[str] = []

    for model_name in sorted(supported_names):
        base = base_name_from_model_name(model_name)
        aliased = apply_alias(base)
        candidates = flores_index.get(aliased, [])
        if not candidates and aliased != base:
            candidates = flores_index.get(base, [])
        if not candidates:
            unmatched.append(model_name)
            name_to_codes[model_name] = set()
            continue

        preferred_script = script_preference_from_model_name(model_name)
        if preferred_script is not None:
            chosen = {
                code
                for code in candidates
                if script_from_flores_code(code) == preferred_script
            }
            if not chosen:
                chosen = set(candidates)
        else:
            chosen = set(candidates)
            scripts = {script_from_flores_code(code) for code in candidates}
            scripts.discard(None)
            if len(scripts) > 1:
                LOGGER.info(
                    "No script preference for %s; mapping to %s",
                    model_name,
                    sorted(chosen),
                )

        name_to_codes[model_name] = chosen
        supported_codes.update(chosen)

    return name_to_codes, supported_codes, unmatched
