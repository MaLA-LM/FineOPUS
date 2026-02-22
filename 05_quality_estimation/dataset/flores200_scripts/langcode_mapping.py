from __future__ import annotations

import re
from typing import Optional
from dataset.flores200_scripts.langcodes import flores200_langcodes

# Hard-coded aliases for FLORES-200 names that cannot be resolved
# by the generic algorithm (different root word, not just a qualifier).
SPECIAL_ALIASES: dict[str, list[str]] = {
    "Odia": ["Oriya"],
    "Northern Kurdish": ["Kurdish (Kurmanji)"],
    "Yue Chinese": ["Cantonese"],
    "Eastern Panjabi": ["Punjabi"],
}


def _find_in_set(candidate: str, model_languages: set[str]) -> str | None:
    """Case-insensitive lookup. Returns the actual model-language string."""
    candidate_lower = candidate.lower()
    for lang in model_languages:
        if lang.lower() == candidate_lower:
            return lang
    return None


def _match_exact(name: str, model_languages: set[str]) -> str | None:
    """Step 1: exact match (case-insensitive)."""
    return _find_in_set(name, model_languages)


def _match_alias(name: str, model_languages: set[str]) -> str | None:
    """Step 2: try special-case aliases (Odia→Oriya, etc.)."""
    for alias in SPECIAL_ALIASES.get(name, []):
        match = _find_in_set(alias, model_languages)
        if match:
            return match
    return None


def _match_left_trim(name: str, model_languages: set[str]) -> str | None:
    """Step 3: progressively remove one word from the left
    ("South Azerbaijani" → "Azerbaijani")."""
    words = name.split()
    for i in range(1, len(words)):
        match = _find_in_set(" ".join(words[i:]), model_languages)
        if match:
            return match
    return None


def _match_strip_qualifiers(name: str, model_languages: set[str]) -> str | None:
    """Step 4: strip parenthesised qualifiers, then retry exact + left-trim
    ("Chinese (Simplified)" → "Chinese")."""
    stripped = re.sub(r"\s*\([^)]*\)", "", name).strip()
    if stripped == name:
        return None
    match = _match_exact(stripped, model_languages) or _match_left_trim(
        stripped, model_languages
    )
    return match


def _match_language(
    flores_display_name: str, model_languages: set[str]
) -> tuple[bool, str | None]:
    """Match a FLORES-200 display name to a model language.

    Matching strategy (tried in order):
      1. Exact match (case-insensitive)
      2. Special-case aliases
         (Odia→Oriya, Northern Kurdish→Kurdish (Kurmanji), Yue Chinese→Cantonese)
      3. Progressive left-word removal
         ("South Azerbaijani" → "Azerbaijani")
      4. Strip parenthesised qualifiers
         ("Chinese (Simplified)" → "Chinese"),
         then retry exact + left-word removal on the stripped form
    """
    match = (
        _match_exact(flores_display_name, model_languages)
        or _match_alias(flores_display_name, model_languages)
        or _match_left_trim(flores_display_name, model_languages)
        or _match_strip_qualifiers(flores_display_name, model_languages)
    )
    return (True, match) if match else (False, None)


def build_model_language_mapping(model_languages: set[str]) -> dict[str, list[object]]:
    mapping: dict[str, list[object]] = {}
    for language_code, flores_display_name in flores200_langcodes.items():
        is_supported, matched_model_language = _match_language(
            flores_display_name, model_languages
        )
        mapping[language_code] = [is_supported, matched_model_language]
    return mapping
