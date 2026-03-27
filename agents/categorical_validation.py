from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from ..core.config.constants import (
    CATEGORICAL_LLM_CONFIDENCE_THRESHOLD,
    CATEGORICAL_LLM_MAX_DISTINCT,
    CATEGORICAL_LLM_MAX_VALUES,
    CATEGORICAL_LLM_MIN_DISTINCT,
    CATEGORICAL_LLM_MIN_REPEAT_COUNT,
    LLM_DEFAULT_MODEL,
)
from ..core.validation.helpers import build_finding
from .base import BaseAgent

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None


def _parse_json_content(content: str) -> dict[str, Any] | None:
    try:
        return json.loads(content)
    except Exception:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


class LLMCategoricalValueValidator:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or LLM_DEFAULT_MODEL
        self._llm = None

    @property
    def enabled(self) -> bool:
        return bool(ChatOpenAI and os.getenv("OPENAI_API_KEY"))

    def _client(self):
        if not self.enabled:
            return None
        if self._llm is None:
            self._llm = ChatOpenAI(model=self.model_name, temperature=0)
        return self._llm

    def validate(
        self,
        *,
        dataset_name: str,
        provider_name: str,
        column_name: str,
        standard_candidate: str | None,
        semantic_tags: list[str],
        values: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        llm = self._client()
        if llm is None:
            return None

        prompt = f"""
You are validating a categorical string column in a Korean public dataset.
Decide whether the distinct values belong to one coherent semantic domain.

Return strict JSON only with keys:
- domain_label: string
- canonical_values: list[string]
- normalizations: list[{{"source": string, "target": string, "reason": string, "confidence": float}}]
- out_of_domain_values: list[{{"value": string, "reason": string, "confidence": float}}]
- needs_manual_review: list[{{"value": string, "reason": string, "confidence": float}}]
- overall_confidence: float

Rules:
- Focus on categorical domain consistency, not spelling or whitespace only.
- Use Korean in reasons.
- Mark abbreviations or shorthand forms as normalizations.
- Mark values from a different semantic taxonomy as out_of_domain_values.
- Only include items you are reasonably confident about.

Dataset:
- name: {dataset_name}
- provider: {provider_name}

Column:
- name: {column_name}
- standard_candidate: {standard_candidate or ""}
- semantic_tags: {semantic_tags}
- distinct_values_with_counts: {json.dumps(values, ensure_ascii=False)}
"""
        response = llm.invoke(prompt)
        payload = _parse_json_content(response.content)
        if payload is None:
            return None
        payload.setdefault("domain_label", "")
        payload.setdefault("canonical_values", [])
        payload.setdefault("normalizations", [])
        payload.setdefault("out_of_domain_values", [])
        payload.setdefault("needs_manual_review", [])
        payload.setdefault("overall_confidence", 0.0)
        return payload


class CategoricalSemanticValidationAgent(BaseAgent):
    name = "categorical_semantic_validator"

    def __init__(self, validator: LLMCategoricalValueValidator | None = None):
        self.validator = validator

    @staticmethod
    def _is_candidate_column(column) -> bool:
        if column.inferred_primitive_type != "string":
            return False
        if column.distinct_count is None:
            return False
        if not (CATEGORICAL_LLM_MIN_DISTINCT <= column.distinct_count <= CATEGORICAL_LLM_MAX_DISTINCT):
            return False
        if not column.top_values:
            return False
        if max(count for _, count in column.top_values) < CATEGORICAL_LLM_MIN_REPEAT_COUNT:
            return False

        categorical_tokens = ("구분", "유형", "종류", "상태", "여부", "급", "분류", "코드", "명칭")
        categorical_tags = {"enum", "code", "boolean", "name"}
        return bool(categorical_tags.intersection(set(column.semantic_tags))) or any(
            token in column.raw_name for token in categorical_tokens
        )

    @staticmethod
    def _value_rows(rows: list[dict[str, str]], column_name: str, target_value: str) -> list[int]:
        indexes: list[int] = []
        for row_index, row in enumerate(rows, start=1):
            value = (row.get(column_name) or "").strip()
            if value == target_value:
                indexes.append(row_index)
        return indexes

    def run(self, state):
        traces = list(state.get("agent_traces", []))
        findings = list(state.get("findings", []))
        rows = state.get("preview_rows", [])
        use_llm = bool(state.get("use_llm_agents")) and self.validator is not None

        if not use_llm:
            traces.append(self.trace(action="categorical_semantic_validate", detail="skipped:llm_disabled"))
            return {"findings": findings, "agent_traces": traces}

        dataset_meta = state["dataset_meta"]
        for column in state["columns"]:
            if not self._is_candidate_column(column):
                continue

            counter = Counter()
            for row in rows:
                value = (row.get(column.raw_name) or "").strip()
                if value:
                    counter[value] += 1
            if not (CATEGORICAL_LLM_MIN_DISTINCT <= len(counter) <= CATEGORICAL_LLM_MAX_DISTINCT):
                continue

            values = [
                {"value": value, "count": count}
                for value, count in counter.most_common(CATEGORICAL_LLM_MAX_VALUES)
            ]
            result = self.validator.validate(
                dataset_name=dataset_meta.dataset_name,
                provider_name=dataset_meta.provider_name,
                column_name=column.raw_name,
                standard_candidate=column.standard_candidates[0] if column.standard_candidates else None,
                semantic_tags=column.semantic_tags,
                values=values,
            )
            if not result:
                traces.append(
                    self.trace(
                        action="categorical_semantic_validate",
                        target=column.raw_name,
                        detail="llm_no_result",
                    )
                )
                continue

            overall_confidence = float(result.get("overall_confidence") or 0.0)
            generated = 0

            for item in result.get("normalizations", []):
                confidence = float(item.get("confidence") or 0.0)
                if confidence < CATEGORICAL_LLM_CONFIDENCE_THRESHOLD:
                    continue
                source = str(item.get("source") or "").strip()
                target = str(item.get("target") or "").strip()
                if not source or not target or source == target:
                    continue
                findings.append(
                    build_finding(
                        column_name=column.raw_name,
                        severity="warning",
                        category_group="domain_validity",
                        criterion_name="categorical_semantic_domain",
                        rule_id="categorical_value_normalization",
                        message=f"'{source}' 값은 '{target}'로 표준화하는 것이 적절합니다. {item.get('reason', '').strip()}".strip(),
                        row_indexes=self._value_rows(rows, column.raw_name, source),
                        related_columns=[column.raw_name],
                        evidence=[f"domain:{result.get('domain_label', '')}", f"confidence:{confidence:.2f}"],
                    )
                )
                generated += 1

            for item in result.get("out_of_domain_values", []):
                confidence = float(item.get("confidence") or 0.0)
                if confidence < CATEGORICAL_LLM_CONFIDENCE_THRESHOLD:
                    continue
                value = str(item.get("value") or "").strip()
                if not value:
                    continue
                findings.append(
                    build_finding(
                        column_name=column.raw_name,
                        severity="warning",
                        category_group="domain_validity",
                        criterion_name="categorical_semantic_domain",
                        rule_id="categorical_value_out_of_domain",
                        message=f"'{value}' 값은 해당 컬럼의 의미 도메인과 맞지 않을 수 있습니다. {item.get('reason', '').strip()}".strip(),
                        row_indexes=self._value_rows(rows, column.raw_name, value),
                        related_columns=[column.raw_name],
                        evidence=[f"domain:{result.get('domain_label', '')}", f"confidence:{confidence:.2f}"],
                    )
                )
                generated += 1

            for item in result.get("needs_manual_review", []):
                confidence = float(item.get("confidence") or 0.0)
                value = str(item.get("value") or "").strip()
                if not value or confidence >= CATEGORICAL_LLM_CONFIDENCE_THRESHOLD:
                    continue
                findings.append(
                    build_finding(
                        column_name=column.raw_name,
                        severity="info",
                        category_group="domain_validity",
                        criterion_name="categorical_semantic_domain",
                        rule_id="categorical_value_manual_review",
                        message=f"'{value}' 값은 의미 판정이 애매해 수동 검토가 필요합니다. {item.get('reason', '').strip()}".strip(),
                        row_indexes=self._value_rows(rows, column.raw_name, value),
                        related_columns=[column.raw_name],
                        evidence=[f"domain:{result.get('domain_label', '')}", f"confidence:{confidence:.2f}"],
                    )
                )
                generated += 1

            traces.append(
                self.trace(
                    action="categorical_semantic_validate",
                    target=column.raw_name,
                    detail=(
                        f"values={len(values)}, findings={generated}, "
                        f"domain={result.get('domain_label', '')}, overall_confidence={overall_confidence:.2f}"
                    ),
                )
            )

        return {"findings": findings, "agent_traces": traces}
