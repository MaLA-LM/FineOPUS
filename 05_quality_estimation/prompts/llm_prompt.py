from __future__ import annotations

PROMPT_TEMPLATE = """Source language:{source_lang} , Source text:
```{source_seg}```

Target language: {target_lang} , Machine Translation text:
```{target_seg}```

Task: Reference-free MT quality scoring for this single segment.
Score each dimension as an integer 0..10 (higher=better), then overall 0..100.

Dimensions:
1) accuracy_completeness (meaning preserved, no add/omit)
2) terminology_consistency
3) fluency_coherence
4) style_tone_audience
5) locale_formatting
6) technical_integrity (entities/units/code/markup preserved)
7) cultural_appropriateness

Output: ONLY valid JSON, exactly this shape:
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
