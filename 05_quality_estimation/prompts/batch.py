from __future__ import annotations

from dataset.mediator import Example

BATCH_PROMPT_TEMPLATE = """\
You are a professional translation quality evaluator.
Below are {batch_size} source/translation segment pairs to evaluate.

{items_block}
Task: Reference-free MT quality scoring for EVERY item above.
Score each dimension as an integer 0..10 (higher = better), then overall 0..100.

Dimensions:
1) accuracy_completeness (meaning preserved, no additions/omissions)
2) terminology_consistency
3) fluency_coherence
4) style_tone_audience
5) locale_formatting (numbers, punctuation, dates, tags if any)
6) technical_integrity (entities/units/code/markup preserved)
7) cultural_appropriateness

Output ONLY valid JSON with exactly this shape (no extra keys, no text outside JSON, all values integers):
{{
  "results": [
    {{
      "id": <int>,
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
  ]
}}
Return exactly {batch_size} items in "results", one per input segment, ordered by id."""

_ITEM_TEMPLATE = """\
--- Item {item_id} ---
Source language: {source_lang}
Target language: {target_lang}
Source text:
```
{source_seg}
```
Machine Translation text:
```
{target_seg}
```"""


def _render_item(
    item_id: int,
    source_lang: str,
    target_lang: str,
    source_seg: str,
    target_seg: str,
) -> str:
    return _ITEM_TEMPLATE.format(
        item_id=item_id,
        source_lang=source_lang,
        target_lang=target_lang,
        source_seg=source_seg,
        target_seg=target_seg,
    )


def render_batch_prompt(
    examples: list[Example],
    src_lang: str,
    tgt_lang: str,
    *,
    start_id: int = 0,
) -> str:
    items = [
        _render_item(
            item_id=start_id + i,
            source_lang=src_lang,
            target_lang=tgt_lang,
            source_seg=ex["src"],
            target_seg=ex["tgt"],
        )
        for i, ex in enumerate(examples)
    ]
    return BATCH_PROMPT_TEMPLATE.format(
        items_block="\n\n".join(items),
        batch_size=len(examples),
    )
