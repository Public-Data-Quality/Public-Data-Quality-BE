from __future__ import annotations

import re
from collections import Counter

try:
    from .....core.config.constants import CATEGORICAL_LLM_MIN_REPEAT_COUNT
except ImportError:  # pragma: no cover
    from core.config.constants import CATEGORICAL_LLM_MIN_REPEAT_COUNT
from .column import (
    is_public_private_category_value,
    looks_institution_category_column,
    looks_route_name_column,
)
from .text import is_numeric_like_value, is_specific_normalization_reason


def canonical_normalization_key(value: str) -> str:
    return re.sub(r"[\s\-\.,()/·]+", "", value or "").strip().lower()


def visible_text_key(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_safe_normalization(source: str, target: str) -> bool:
    source_clean = str(source or "").strip()
    target_clean = str(target or "").strip()
    if not source_clean or not target_clean or source_clean == target_clean:
        return False
    if visible_text_key(source_clean) == visible_text_key(target_clean):
        return False

    return canonical_normalization_key(source_clean) == canonical_normalization_key(target_clean)


def is_llm_normalization_actionable(column, source: str, target: str, reason: str) -> bool:
    if not source or not target or source == target:
        return False
    if visible_text_key(source) == visible_text_key(target):
        return False
    if looks_route_name_column(column):
        return False
    if (
        looks_institution_category_column(column)
        and is_public_private_category_value(source)
        and is_public_private_category_value(target)
    ):
        return False
    if canonical_normalization_key(source) == canonical_normalization_key(target):
        return is_specific_normalization_reason(reason)
    return is_specific_normalization_reason(reason)


def has_mixed_surface_forms(counter: Counter[str], source: str, target: str) -> bool:
    source_clean = str(source or "").strip()
    target_clean = str(target or "").strip()
    if not source_clean or not target_clean:
        return False

    canonical = canonical_normalization_key(source_clean)
    variants = {
        value.strip()
        for value in counter
        if canonical_normalization_key(value) == canonical and value.strip()
    }
    return len(variants) >= 2


def find_surface_normalization_pairs(counter: Counter[str]) -> list[tuple[str, str]]:
    groups: dict[str, list[str]] = {}
    for value in counter:
        cleaned = value.strip()
        if not cleaned or is_numeric_like_value(cleaned):
            continue
        canonical = canonical_normalization_key(cleaned)
        if len(canonical) < 2:
            continue
        groups.setdefault(canonical, []).append(cleaned)

    pairs: list[tuple[str, str]] = []
    for variants in groups.values():
        unique_variants = sorted(set(variants))
        if len(unique_variants) < 2:
            continue
        target = max(unique_variants, key=lambda value: (counter[value], len(value), value))
        for source in unique_variants:
            if source == target:
                continue
            if visible_text_key(source) == visible_text_key(target):
                continue
            if counter[source] > CATEGORICAL_LLM_MIN_REPEAT_COUNT:
                continue
            if counter[target] < max(3, counter[source] * 3):
                continue
            if is_safe_normalization(source, target):
                pairs.append((source, target))

    unique_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in sorted(pairs, key=lambda item: (item[0], item[1])):
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    return unique_pairs
