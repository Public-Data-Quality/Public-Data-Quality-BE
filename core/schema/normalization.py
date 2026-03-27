from __future__ import annotations

import re

from ..config.constants import (
    DEFAULT_COLUMN_ROUTING_CONFIDENCE,
    MAX_ROUTING_CONFIDENCE,
    SEMANTIC_TAG_PATTERNS,
    TAG_CONFIDENCE_CAP,
    TAG_CONFIDENCE_STEP,
    TAG_RULE_MAP,
)
from .models import ColumnProfile

PARENS_RE = re.compile(r"\(([^()]*)\)")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_column_name(raw_name: str, synonym_index: dict[str, str]) -> tuple[str, str | None]:
    del synonym_index
    unit = None
    name = raw_name.strip()
    paren_match = PARENS_RE.search(name)
    if paren_match:
        maybe_unit = paren_match.group(1).strip()
        if maybe_unit:
            unit = maybe_unit
        name = PARENS_RE.sub("", name)

    name = MULTISPACE_RE.sub(" ", name).strip()
    return name, unit


def tokenize_korean_label(label: str) -> list[str]:
    parts = re.split(r"[\s/,_-]+", label)
    return [part for part in parts if part]


def infer_semantic_tags(name: str) -> list[str]:
    tags: set[str] = set()
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["date"]):
        tags.add("date")
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["phone"]):
        tags.add("phone")
    if "위도" in name:
        tags.update({"geo_lat", "coordinate_pair"})
    if "경도" in name:
        tags.update({"geo_lon", "coordinate_pair"})
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["address"]):
        tags.add("address")
    if "여부" in name:
        tags.add("boolean")
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["numeric"]):
        tags.add("numeric")
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["width"]):
        tags.add("width")
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["count"]):
        tags.add("count")
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["amount"]):
        tags.add("amount")
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["quantity"]):
        tags.add("quantity")
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["rate"]):
        tags.add("rate")
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["enum"]):
        tags.add("enum")
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["code"]):
        tags.add("code")
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["identifier"]):
        tags.add("identifier")
    if any(token in name for token in SEMANTIC_TAG_PATTERNS["name"]):
        tags.add("name")
    return sorted(tags)


def assign_rules(tags: list[str]) -> list[str]:
    rules: list[str] = []
    for tag in tags:
        for rule in TAG_RULE_MAP.get(tag, []):
            if rule not in rules:
                rules.append(rule)
    return rules


def build_column_profile(
    raw_name: str,
    source: str,
    synonym_index: dict[str, str],
) -> ColumnProfile:
    normalized_name, unit = normalize_column_name(raw_name, synonym_index)
    tags = infer_semantic_tags(normalized_name)
    rules = assign_rules(tags)
    confidence = DEFAULT_COLUMN_ROUTING_CONFIDENCE
    if tags:
        confidence += min(TAG_CONFIDENCE_CAP, TAG_CONFIDENCE_STEP * len(tags))
    return ColumnProfile(
        raw_name=raw_name,
        normalized_name=normalized_name,
        source=source,
        unit=unit,
        tokens=tokenize_korean_label(normalized_name),
        semantic_tags=tags,
        assigned_rules=rules,
        routing_confidence=min(confidence, MAX_ROUTING_CONFIDENCE),
        rag_required=not rules,
    )
