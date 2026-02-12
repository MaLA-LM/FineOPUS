from __future__ import annotations

from typing import Callable, Literal, Optional

from dataset.mediator import DatasetAdapter, DEFAULT_DATASET_ID, get_dataset

LanguageCodeMapper = Callable[
    [set[str], dict[str, str]],
    tuple[dict[str, set[str]], set[str], list[str]],
]
CodeNormalizer = Callable[[str], str]
SupportStatus = bool | Literal["unknown"]


def _resolve_dataset(dataset: DatasetAdapter | str | None) -> DatasetAdapter:
    if isinstance(dataset, DatasetAdapter):
        return dataset
    return get_dataset(dataset or DEFAULT_DATASET_ID)


def _resolve_language_codes(
    language_codes: Optional[dict[str, str]],
    dataset: DatasetAdapter,
) -> dict[str, str] | None:
    if language_codes is not None:
        return language_codes
    return dataset.language_codes


def _resolve_name_to_code_mapper(
    name_to_code_mapper: Optional[LanguageCodeMapper],
    dataset: DatasetAdapter,
) -> LanguageCodeMapper | None:
    if name_to_code_mapper is not None:
        return name_to_code_mapper
    return dataset.name_to_code_mapper


def _default_code_normalizer(code: str) -> str:
    return code.strip().replace("-", "_")


class _BaseLanguageSupport:
    def __init__(
        self,
        supported_languages: set[str],
        language_codes: Optional[dict[str, str]],
        label: str,
        *,
        dataset: DatasetAdapter | str | None = None,
        name_to_code_mapper: Optional[LanguageCodeMapper] = None,
        code_normalizer: Optional[CodeNormalizer] = None,
    ) -> None:
        dataset = _resolve_dataset(dataset)
        self._supported_languages = supported_languages
        langcode = _resolve_language_codes(language_codes, dataset)
        mapper = _resolve_name_to_code_mapper(name_to_code_mapper, dataset)
        self._code_normalizer = (
            code_normalizer or dataset.code_normalizer or _default_code_normalizer
        )
        self._support_known = langcode is not None and mapper is not None
        if not self._support_known:
            self._name_to_codes = {}
            self._supported_codes = set()
            self._unmatched_supported_names = []
            return
        (
            self._name_to_codes,
            self._supported_codes,
            unmatched,
        ) = mapper(self._supported_languages, langcode)
        self._unmatched_supported_names = sorted(unmatched)

        if self._unmatched_supported_names:
            print(
                f"Unmatched language names for {label}: "
                f"{self._unmatched_supported_names}"
            )

    def get_all_languages(self) -> list[str]:
        return sorted(self._supported_languages)

    # model supported codes of a dataset
    def get_supported_codes(self) -> set[str]:
        return set(self._supported_codes)

    def codes_for_name(self, name: str) -> set[str]:
        return set(self._name_to_codes.get(name, set()))

    def is_code_supported(self, code: str) -> bool:
        if not self._support_known:
            return False
        if not code:
            return False
        normalized = self._code_normalizer(code)
        return normalized in self._supported_codes

    def support_status(self, code: str) -> SupportStatus:
        if not self._support_known:
            return "unknown"
        return self.is_code_supported(code)

    def debug_unmatched(self) -> list[str]:
        if not self._support_known:
            return []
        return list(self._unmatched_supported_names)


class CometLanguages(_BaseLanguageSupport):
    """Class representing languages supported by COMET model."""

    def __init__(
        self,
        language_codes: Optional[dict[str, str]] = None,
        *,
        dataset: DatasetAdapter | str | None = None,
        name_to_code_mapper: Optional[LanguageCodeMapper] = None,
        code_normalizer: Optional[CodeNormalizer] = None,
    ) -> None:
        supported_languages = {
            "Afrikaans",
            "Albanian",
            "Amharic",
            "Arabic",
            "Armenian",
            "Assamese",
            "Azerbaijani",
            "Basque",
            "Belarusian",
            "Bengali",
            "Bengali Romanized",
            "Bosnian",
            "Breton",
            "Bulgarian",
            "Burmese",
            "Catalan",
            "Chinese (Simplified)",
            "Chinese (Traditional)",
            "Croatian",
            "Czech",
            "Danish",
            "Dutch",
            "English",
            "Esperanto",
            "Estonian",
            "Filipino",
            "Finnish",
            "French",
            "Galician",
            "Georgian",
            "German",
            "Greek",
            "Gujarati",
            "Hausa",
            "Hebrew",
            "Hindi",
            "Hindi Romanized",
            "Hungarian",
            "Icelandic",
            "Indonesian",
            "Irish",
            "Italian",
            "Japanese",
            "Javanese",
            "Kannada",
            "Kazakh",
            "Khmer",
            "Korean",
            "Kurdish (Kurmanji)",
            "Kyrgyz",
            "Lao",
            "Latin",
            "Latvian",
            "Lithuanian",
            "Macedonian",
            "Malagasy",
            "Malay",
            "Malayalam",
            "Marathi",
            "Mongolian",
            "Nepali",
            "Norwegian",
            "Oriya",
            "Oromo",
            "Pashto",
            "Persian",
            "Polish",
            "Portuguese",
            "Punjabi",
            "Romanian",
            "Russian",
            "Sanskrit",
            "Scottish Gaelic",
            "Serbian",
            "Sindhi",
            "Sinhala",
            "Slovak",
            "Slovenian",
            "Somali",
            "Spanish",
            "Sundanese",
            "Swahili",
            "Swedish",
            "Tamil",
            "Tamil Romanized",
            "Telugu",
            "Telugu Romanized",
            "Thai",
            "Turkish",
            "Ukrainian",
            "Urdu",
            "Urdu Romanized",
            "Uyghur",
            "Uzbek",
            "Vietnamese",
            "Welsh",
            "Western Frisian",
            "Xhosa",
            "Yiddish",
        }
        super().__init__(
            supported_languages,
            language_codes,
            "COMET",
            dataset=dataset,
            name_to_code_mapper=name_to_code_mapper,
            code_normalizer=code_normalizer,
        )


class QwenLanguages(_BaseLanguageSupport):
    """Class representing languages supported by Qwen model."""

    def __init__(
        self,
        language_codes: Optional[dict[str, str]] = None,
        *,
        dataset: DatasetAdapter | str | None = None,
        name_to_code_mapper: Optional[LanguageCodeMapper] = None,
        code_normalizer: Optional[CodeNormalizer] = None,
    ) -> None:
        supported_languages = {
            "Afrikaans",
            "Arabic (Standard, Najdi, Levantine, Egyptian, Moroccan, Mesopotamian, Ta'izzi-Adeni, Tunisian)",
            "Armenian",
            "Assamese",
            "Asturian",
            "Awadhi",
            "Balinese",
            "Banjar",
            "Bashkir",
            "Basque",
            "Belarusian",
            "Bengali",
            "Bhojpuri",
            "Bosnian",
            "Bulgarian",
            "Burmese",
            "Catalan",
            "Cebuano",
            "Chhattisgarhi",
            "Chinese (Simplified Chinese, Traditional Chinese, Cantonese)",
            "Croatian",
            "Czech",
            "Danish",
            "Dari",
            "Dutch",
            "Eastern Yiddish",
            "English",
            "Estonian",
            "Faroese",
            "Finnish",
            "French",
            "Friulian",
            "Galician",
            "Georgian",
            "German",
            "Greek",
            "Gujarati",
            "Haitian",
            "Hebrew",
            "Hindi",
            "Hungarian",
            "Icelandic",
            "Iloko",
            "Indonesian",
            "Irish",
            "Italian",
            "Japanese",
            "Javanese",
            "Kabuverdianu",
            "Kannada",
            "Kazakh",
            "Khmer",
            "Korean",
            "Lao",
            "Latvian",
            "Ligurian",
            "Limburgish",
            "Lithuanian",
            "Lombard",
            "Luxembourgish",
            "Macedonian",
            "Magahi",
            "Maithili",
            "Malay",
            "Malayalam",
            "Maltese",
            "Marathi",
            "Minangkabau",
            "Nepali",
            "North Azerbaijani",
            "Northern Uzbek",
            "Norwegian (Bokmål)",
            "Norwegian (Nynorsk)",
            "Occitan",
            "Oriya",
            "Pangasinan",
            "Papiamento",
            "Persian",
            "Polish",
            "Portuguese",
            "Punjabi",
            "Romanian",
            "Russian",
            "Sardinian",
            "Serbian",
            "Sicilian",
            "Silesian",
            "Sindhi",
            "Sinhala",
            "Slovak",
            "Slovenian",
            "Spanish",
            "Sundanese",
            "Swahili",
            "Swedish",
            "Tagalog",
            "Tajik",
            "Tamil",
            "Tatar",
            "Telugu",
            "Thai",
            "Tok Pisin",
            "Tosk Albanian",
            "Turkish",
            "Ukrainian",
            "Urdu",
            "Venetian",
            "Vietnamese",
            "Waray (Philippines)",
            "Welsh",
        }
        super().__init__(
            supported_languages,
            language_codes,
            "qwen3-14b",
            dataset=dataset,
            name_to_code_mapper=name_to_code_mapper,
            code_normalizer=code_normalizer,
        )


class MetricX24Languages(_BaseLanguageSupport):
    """Class representing languages supported by MetricX-24 model."""

    def __init__(
        self,
        language_codes: Optional[dict[str, str]] = None,
        *,
        dataset: DatasetAdapter | str | None = None,
        name_to_code_mapper: Optional[LanguageCodeMapper] = None,
        code_normalizer: Optional[CodeNormalizer] = None,
    ) -> None:
        supported_languages = {
            "Afrikaans",
            "Albanian",
            "Amharic",
            "Arabic",
            "Armenian",
            "Azerbaijani",
            "Basque",
            "Belarusian",
            "Bengali",
            "Bulgarian",
            "Burmese",
            "Catalan",
            "Cebuano",
            "Chichewa",
            "Chinese",
            "Corsican",
            "Czech",
            "Danish",
            "Dutch",
            "English",
            "Esperanto",
            "Estonian",
            "Filipino",
            "Finnish",
            "French",
            "Galician",
            "Georgian",
            "German",
            "Greek",
            "Gujarati",
            "Haitian Creole",
            "Hausa",
            "Hawaiian",
            "Hebrew",
            "Hindi",
            "Hmong",
            "Hungarian",
            "Icelandic",
            "Igbo",
            "Indonesian",
            "Irish",
            "Italian",
            "Japanese",
            "Javanese",
            "Kannada",
            "Kazakh",
            "Khmer",
            "Korean",
            "Kurdish",
            "Kyrgyz",
            "Lao",
            "Latin",
            "Latvian",
            "Lithuanian",
            "Luxembourgish",
            "Macedonian",
            "Malagasy",
            "Malay",
            "Malayalam",
            "Maltese",
            "Maori",
            "Marathi",
            "Mongolian",
            "Nepali",
            "Norwegian",
            "Pashto",
            "Persian",
            "Polish",
            "Portuguese",
            "Punjabi",
            "Romanian",
            "Russian",
            "Samoan",
            "Scottish Gaelic",
            "Serbian",
            "Shona",
            "Sindhi",
            "Sinhala",
            "Slovak",
            "Slovenian",
            "Somali",
            "Sotho",
            "Spanish",
            "Sundanese",
            "Swahili",
            "Swedish",
            "Tajik",
            "Tamil",
            "Telugu",
            "Thai",
            "Turkish",
            "Ukrainian",
            "Urdu",
            "Uzbek",
            "Vietnamese",
            "Welsh",
            "West Frisian",
            "Xhosa",
            "Yiddish",
            "Yoruba",
            "Zulu",
        }
        super().__init__(
            supported_languages,
            language_codes,
            "MetricX-24",
            dataset=dataset,
            name_to_code_mapper=name_to_code_mapper,
            code_normalizer=code_normalizer,
        )
