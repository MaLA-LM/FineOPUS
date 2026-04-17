from __future__ import annotations

from dataset.flores200.langcode_mapping import build_mapping_from_codes
from dataset.opus.langcodes import opus_langcodes


def build_model_language_mapping(model_languages: set[str]) -> dict[str, list[object]]:
    """OPUS wrapper around the shared 4-stage fuzzy matcher.

    Reuses the exact same matching pipeline (exact -> alias -> left-trim ->
    strip qualifiers) as FLORES-200 so that language-support classes behave
    identically across datasets; only the source langcodes dict changes.
    """
    return build_mapping_from_codes(opus_langcodes, model_languages)
