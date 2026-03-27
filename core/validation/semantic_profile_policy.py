from __future__ import annotations

from ..config.constants import (
    LLM_SEMANTIC_PROFILE_ALWAYS_TRIGGER_NAME_TOKENS,
    LLM_SEMANTIC_PROFILE_ALWAYS_TRIGGER_TAGS,
    LLM_SEMANTIC_PROFILE_AMBIGUOUS_TERMS,
    LLM_SEMANTIC_PROFILE_SKIP_TAGS,
    LLM_SEMANTIC_PROFILE_TRIGGER_MATCH_TYPES,
)
from ..schema.models import ColumnProfile


def _is_free_text_column(column: ColumnProfile) -> bool:
    name = column.raw_name.strip()
    if any(token in name for token in LLM_SEMANTIC_PROFILE_ALWAYS_TRIGGER_NAME_TOKENS):
        return True

    if column.inferred_primitive_type != "string":
        return False

    if not column.sample_values:
        return False

    long_samples = [value.strip() for value in column.sample_values if len(value.strip()) >= 12]
    return len(long_samples) >= 2


def _is_structured_column(column: ColumnProfile) -> bool:
    if not column.semantic_tags:
        return False
    return set(column.semantic_tags).issubset(LLM_SEMANTIC_PROFILE_SKIP_TAGS)


def semantic_profile_llm_reasons(column: ColumnProfile) -> list[str]:
    reasons: list[str] = []
    name = column.raw_name.strip()

    if set(column.semantic_tags).intersection(LLM_SEMANTIC_PROFILE_ALWAYS_TRIGGER_TAGS):
        reasons.append("주소/위치 계열 컬럼")
    if _is_free_text_column(column):
        reasons.append("자유서술형 문자열 컬럼")
    if not column.semantic_tags:
        reasons.append("semantic tag 없음")
    if (column.standard_match_type or "unmatched") in LLM_SEMANTIC_PROFILE_TRIGGER_MATCH_TYPES:
        reasons.append(f"매핑 유형={column.standard_match_type or 'unmatched'}")
    if len(column.semantic_tags) >= 3:
        reasons.append("semantic tag 다중 충돌")
    if not column.standard_candidates:
        reasons.append("표준용어 후보 없음")
    if name in LLM_SEMANTIC_PROFILE_AMBIGUOUS_TERMS:
        reasons.append("다의적 일반명사 컬럼")
    if len(name) <= 2:
        reasons.append("컬럼명이 너무 짧음")
    if column.sample_values:
        top_sample = str(column.sample_values[0]).strip()
        if top_sample and len(top_sample) > 20 and column.inferred_primitive_type == "string":
            reasons.append("문자열 샘플 의미 해석 필요")

    if reasons:
        return list(dict.fromkeys(reasons))

    if _is_structured_column(column):
        return []

    if column.inferred_primitive_type == "string":
        reasons.append("문자열 샘플 의미 해석 필요")
    return list(dict.fromkeys(reasons))
