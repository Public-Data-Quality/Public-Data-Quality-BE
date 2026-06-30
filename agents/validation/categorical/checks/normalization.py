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


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if left == right:
        return True

    mismatches = 0
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        mismatches += 1
        if mismatches > 1:
            return False
        if len(left) == len(right):
            i += 1
            j += 1
        elif len(left) < len(right):
            j += 1
        else:
            i += 1

    if i < len(left) or j < len(right):
        mismatches += 1
    return mismatches <= 1


def is_near_compact_domain_variant(source: str, target: str) -> bool:
    source_key = canonical_normalization_key(source)
    target_key = canonical_normalization_key(target)
    if not source_key or not target_key or source_key == target_key:
        return False
    if is_numeric_like_value(source_key) or is_numeric_like_value(target_key):
        return False
    if len(target_key) < 2 or len(target_key) > 10:
        return False
    if len(source_key) > len(target_key) + 1:
        return False
    if target_key.startswith(source_key) and len(source_key) >= 1:
        return True
    if len(source_key) == len(target_key) and sorted(source_key) == sorted(target_key):
        return True
    if len(source_key) == len(target_key) and source_key[0] != target_key[0]:
        return False
    return _edit_distance_at_most_one(source_key, target_key)


def find_compact_domain_variant_pairs(counter: Counter[str]) -> list[tuple[str, str]]:
    total = sum(counter.values())
    if total <= 0:
        return []

    dominant_values = [
        value
        for value, count in counter.most_common(5)
        if count / total >= 0.2 and 2 <= len(canonical_normalization_key(value)) <= 10
    ]
    if not dominant_values:
        return []

    pairs: list[tuple[str, str]] = []
    for source, source_count in counter.items():
        source_ratio = source_count / total
        if source_ratio >= 0.2:
            continue
        for target in dominant_values:
            if source == target or source_count >= counter[target]:
                continue
            if is_near_compact_domain_variant(source, target):
                pairs.append((source, target))
                break

    unique_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in sorted(pairs, key=lambda item: (item[1], item[0])):
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    return unique_pairs
