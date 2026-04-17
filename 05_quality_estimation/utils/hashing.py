from __future__ import annotations

import hashlib


def stable_hash_int(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def direction_key(src_lang: str, tgt_lang: str) -> str:
    return f"{src_lang}->{tgt_lang}"
