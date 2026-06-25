from __future__ import annotations

import re
from collections import Counter

try:
    from .....core.config.constants import CATEGORICAL_LLM_MIN_REPEAT_COUNT
except ImportError:  # pragma: no cover
    from core.config.constants import CATEGORICAL_LLM_MIN_REPEAT_COUNT
from .text import is_numeric_like_value, normalized_text

SHORT_KOREAN_PREFIX_LEN = 2
ENTITY_COMPLETION_SUFFIXES = {
    "교",
    "원",
    "관",
    "소",
    "당",
    "집",
    "학교",
    "유치원",
    "어린이집",
    "병원",
    "의원",
    "약국",
    "학원",
    "센터",
    "회관",
    "복지관",
    "도서관",
    "보건소",
    "경로당",
    "관리소",
}


def is_normal_qualifier_suffix(suffix: str) -> bool:
    text = str(suffix or "").strip()
    if not text:
        return False

    normal_suffix_patterns = (
        r"^\d+호점$",
        r"^[A-Z]$",
        r"^[가-힣A-Z0-9]+점$",
        r"^도서관$",
        r"^분관$",
        r"^별관$",
        r"^본관$",
        r"^기관$",
        r"^시설$",
    )
    return any(re.fullmatch(pattern, text) for pattern in normal_suffix_patterns)


def is_short_korean_entity_prefix(short_norm: str, long_norm: str) -> bool:
    if len(short_norm) != SHORT_KOREAN_PREFIX_LEN:
        return False
    if not re.fullmatch(r"[가-힣]+", short_norm):
        return False
    if not long_norm.startswith(short_norm):
        return False

    suffix = long_norm[len(short_norm) :]
    if not suffix:
        return False
    return suffix in ENTITY_COMPLETION_SUFFIXES


def find_truncated_value_pairs(counter: Counter[str]) -> list[tuple[str, str]]:
    values = [value.strip() for value in counter if value and value.strip()]
    pairs: list[tuple[str, str]] = []

    for short_value in values:
        short_norm = normalized_text(short_value)
        if len(short_norm) < SHORT_KOREAN_PREFIX_LEN:
            continue
        if is_numeric_like_value(short_norm):
            continue
        for long_value in values:
            if short_value == long_value:
                continue
            long_norm = normalized_text(long_value)
            if is_numeric_like_value(long_norm):
                continue
            if len(long_norm) < len(short_norm) + 1:
                continue
            if short_norm == long_norm:
                continue
            if len(short_norm) < 3 and not is_short_korean_entity_prefix(short_norm, long_norm):
                continue
            if long_norm.startswith(short_norm) and len(long_norm) - len(short_norm) <= 3:
                if len(short_norm) / max(1, len(long_norm)) >= 0.45:
                    suffix = long_norm[len(short_norm) :]
                    if is_normal_qualifier_suffix(suffix):
                        continue
                    if (
                        counter[short_value] > CATEGORICAL_LLM_MIN_REPEAT_COUNT
                        or counter[long_value] < counter[short_value]
                    ):
                        continue
                    pairs.append((short_value, long_value))
    unique_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in sorted(pairs, key=lambda item: (len(normalized_text(item[0])), item[0], item[1])):
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    return unique_pairs
