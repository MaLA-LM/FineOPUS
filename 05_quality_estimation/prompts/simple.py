from __future__ import annotations

SIMPLE_PROMPT_TEMPLATE = """\
You are a professional translation quality evaluator.

Source language: {source_lang}
Target language: {target_lang}
Source text:
```
{source_seg}
```
Machine Translation text:
```
{target_seg}
```

Task: Reference-free MT quality scoring for this single segment.
Score the overall translation quality as an integer from 0 to 100 (higher = better).

Output ONLY valid JSON with exactly this shape (no extra keys, no text outside JSON, value is an integer):
{{"overall_0to100": 0-100}}
"""


def render_simple_prompt(
    source_lang: str,
    source_seg: str,
    target_lang: str,
    target_seg: str,
) -> str:
    return SIMPLE_PROMPT_TEMPLATE.format(
        source_lang=source_lang,
        source_seg=source_seg,
        target_lang=target_lang,
        target_seg=target_seg,
    )
