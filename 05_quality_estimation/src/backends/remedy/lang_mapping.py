from __future__ import annotations

from models.language_support import RemedyLanguages
from utils.logger import logger

__all__ = ["map_lang_codes_to_iso"]


def map_lang_codes_to_iso(
    src_lang: str,
    tgt_lang: str,
    language_support: RemedyLanguages,
) -> tuple[str, str]:
    remedy_src_lang = language_support.iso639_1_from_code(src_lang)
    remedy_tgt_lang = language_support.iso639_1_from_code(tgt_lang)

    if not remedy_src_lang:
        logger.warning(
            "Source language '%s' not recognized by ReMedy. Defaulting to 'en' for scoring. Scores may be inaccurate.",
            src_lang,
        )
        remedy_src_lang = "en"
    if not remedy_tgt_lang:
        logger.warning(
            "Target language '%s' not recognized by ReMedy. Defaulting to 'en' for scoring. Scores may be inaccurate.",
            tgt_lang,
        )
        remedy_tgt_lang = "en"

    return remedy_src_lang, remedy_tgt_lang
