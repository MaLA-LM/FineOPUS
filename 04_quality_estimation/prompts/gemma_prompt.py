from __future__ import annotations

PROMPT_TEMPLATE = """{source_lang} Source:
```{source_seg}```

{target_lang} Machine Translation:
```{target_seg}```

Evaluate the quality of a machine translation for a given segment, using the provided source text,
machine-translated text, source language, and target language.

You must analyze the translation without access to any human reference, considering the following:
- Meaning preservation (accuracy) and completeness (no omissions/additions).
- Terminology correctness and consistency.
- Fluency, grammar, and coherence in the target language.
- Appropriate style/tone for the target language and implied audience.
- Locale conventions and formatting (numbers, punctuation, dates, tags if any).
- Technical integrity (entities, units, code/markup, constraints preserved).
- Cultural appropriateness (no unintended offensiveness or pragmatic shifts).

Return ONLY valid JSON with integer scores, higher is  better, in the following format:
{{
  "dims_0to10": {{
    "accuracy_completeness": 0-10,
    "terminology_consistency": 0-10,
    "fluency_coherence": 0-10,
    "style_tone_audience": 0-10,
    "locale_formatting": 0-10,
    "technical_integrity": 0-10,
    "cultural_appropriateness": 0-10
  }},
  "overall_0to100": 0-100
}}

Important:
- Only use the source and MT segment for evaluation (no references).
- Output MUST be valid and must follow the structure.
- overall_0to100 must reflect the dimension scores, but you may adjust for severity of critical errors.
- Output JSON only (no extra text).
"""


def render_prompt(
    source_lang: str,
    source_seg: str,
    target_lang: str,
    target_seg: str,
) -> str:
    return PROMPT_TEMPLATE.format(
        source_lang=source_lang,
        source_seg=source_seg,
        target_lang=target_lang,
        target_seg=target_seg,
    )
