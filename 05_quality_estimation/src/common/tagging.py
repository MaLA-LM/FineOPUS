from __future__ import annotations

import re


def sanitize_model_tag(name: str) -> str:
    tag = name.strip().lower()
    tag = re.sub(r"[^a-z0-9._-]+", "-", tag)
    tag = tag.strip("-")
    return tag or "model"
