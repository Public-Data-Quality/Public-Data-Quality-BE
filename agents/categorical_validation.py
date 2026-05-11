from __future__ import annotations

import json
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
from ..core.llm import ChatLLMClient
from ..core.validation.helpers import build_finding
from .base import BaseAgent


def _parse_json_content(content: str) -> dict[str, Any] | None:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _clean_reason_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""

    # Drop obviously low-quality freeform reasons to avoid leaking broken Korean into user-facing findings.
    lowered = text.lower()
    if any(token in lowered for token in ("lorem", "asdf", "n/a", "unknown")):
        return ""
    if re.search(r"[A-Za-z]{8,}", text):
        return ""

    words = [word for word in re.split(r"\s+", text) if word]
    if len(words) >= 3:
        suspicious_endings = ("하세요", "해세요", "습니다", "입니다", "적절", "일치", "통일", "권장", "권합니다")
        valid_endings = sum(1 for word in words if word.endswith(suspicious_endings))
        if valid_endings == 0:
            return ""

    return text


def _is_specific_out_of_domain_reason(reason: str) -> bool:
    cleaned = _clean_reason_text(reason)
    if not cleaned:
        return False

    # Suppress generic explanations that merely restate that values are different.
    generic_reasons = {
        "다른 기관명",
        "다른 값",
        "상이한 값",
        "형태가 다름",
        "표현이 다름",
        "이름이 다름",
        "기관명이 다름",
        "다른 명칭",
    }
    if cleaned in generic_reasons:
        return False

    specific_markers = (
        "도메인",
        "범주",
        "체계",
        "분류",
        "tax",
        "taxonomy",
        "기관명이 아님",
        "부서명이 아님",
        "경찰서명이 아님",
        "학교명이 아님",
        "주소가 아님",
        "날짜가 아님",
        "숫자가 아님",
        "코드가 아님",
        "서로 다른 유형",
    )
    return any(marker in cleaned for marker in specific_markers)


def _canonical_normalization_key(value: str) -> str:
    return re.sub(r"[\s\-\.,()/·]+", "", value or "").strip().lower()


def _is_safe_normalization(source: str, target: str) -> bool:
    source_clean = str(source or "").strip()
    target_clean = str(target or "").strip()
    if not source_clean or not target_clean or source_clean == target_clean:
        return False

    # Only allow surface-form normalization such as spacing or punctuation changes.
    # Reject suggestions that add/remove substantive tokens like administrative areas or department names.
    return _canonical_normalization_key(source_clean) == _canonical_normalization_key(target_clean)


def _has_mixed_surface_forms(counter: Counter[str], source: str, target: str) -> bool:
    source_clean = str(source or "").strip()
    target_clean = str(target or "").strip()
    if not source_clean or not target_clean:
        return False

    canonical = _canonical_normalization_key(source_clean)
    variants = {
        value.strip()
        for value in counter
        if _canonical_normalization_key(value) == canonical and value.strip()
    }
    return len(variants) >= 2


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _find_truncated_value_pairs(counter: Counter[str]) -> list[tuple[str, str]]:
    values = [value.strip() for value in counter if value and value.strip()]
    pairs: list[tuple[str, str]] = []

    for short_value in values:
        short_norm = _normalized_text(short_value)
        if len(short_norm) < 3:
            continue
        for long_value in values:
            if short_value == long_value:
                continue
            long_norm = _normalized_text(long_value)
            if len(long_norm) < len(short_norm) + 1:
                continue
            if short_norm == long_norm:
                continue
            # Catch damaged/truncated categorical values like "초등학교" -> "초등".
            if long_norm.startswith(short_norm) and len(long_norm) - len(short_norm) <= 3:
                if len(short_norm) / max(1, len(long_norm)) >= 0.45:
                    pairs.append((short_value, long_value))
    unique_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in sorted(pairs, key=lambda item: (len(_normalized_text(item[0])), item[0], item[1])):
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    return unique_pairs


def _finding_key(finding) -> tuple[str, str, str, tuple[int, ...]]:
    return (
        finding.column_name,
        finding.rule_id,
        finding.message,
        tuple(finding.row_indexes),
    )


def _looks_malformed_text_value(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if "�" in text:
        return True
    if re.search(r"[ㄱ-ㅎㅏ-ㅣ]{2,}", text):
        return True
    if re.search(r"[?!]{2,}", text):
        return True
    if re.search(r"[가-힣A-Za-z0-9][?！!]{1,}$", text):
        return True
    return False


def _looks_date_column(column) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    return "date" in column.semantic_tags or any(token in name for token in ("일자", "일시", "날짜", "년월", "등록일", "기준일"))


def _looks_boolean_column(column) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    return "boolean" in column.semantic_tags or any(token in name for token in ("여부", "유무", "YN", "Yn", "yn", "Y/N"))


def _looks_date_value(value: str) -> bool:
    return bool(re.match(r"^\d{4}[-./]?\d{1,2}[-./]?\d{1,2}$", value.strip()))


def _is_yn_value(value: str) -> bool:
    return value.strip().upper() in {"Y", "N"}


class LLMCategoricalValueValidator:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or LLM_DEFAULT_MODEL
        self._llm = ChatLLMClient(model_name=self.model_name)

    @property
    def enabled(self) -> bool:
        return self._llm.enabled

    def _client(self):
        return self._llm if self.enabled else None

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
You are validating values in one column of a Korean public dataset.
Find data-quality issues that require correction or normalization.

Return strict JSON only with keys:
- domain_label: string
- canonical_values: list[string]
- normalizations: list[{{"source": string, "target": string, "reason": string, "confidence": float}}]
- out_of_domain_values: list[{{"value": string, "reason": string, "confidence": float}}]
- invalid_format_values: list[{{"value": string, "issue_type": string, "reason": string, "confidence": float}}]
- inconsistent_format_groups: list[{{"values": list[string], "target_format": string, "reason": string, "confidence": float}}]
- needs_manual_review: list[{{"value": string, "reason": string, "confidence": float}}]
- overall_confidence: float

Rules:
- Detect categorical abbreviation/full-form inconsistencies, e.g. "초등" vs "초등학교".
- Detect incomplete or truncated Korean text values in semi-structured/free-text columns, e.g. "야간 조" when other values show "야간 조명 부족".
- Detect malformed free-text/category values with stray punctuation or broken suffixes, e.g. "불법주정차빈??".
- Detect date format inconsistency, e.g. "2024-01-01" mixed with "20240102".
- Detect invalid Y/N values in boolean columns, e.g. "I" in a column whose values should be Y or N.
- Do not put categorical values like school levels in inconsistent_format_groups.
- For invalid Y/N values, use invalid_format_values with issue_type "boolean_invalid"; do not guess a replacement.
- For incomplete/truncated values, use invalid_format_values with issue_type "truncated_text" or needs_manual_review.
- For malformed values with stray punctuation or corrupted text, use invalid_format_values with issue_type "malformed_text".
- Do not invent issues. Only include items with concrete evidence from the provided values.
- Use Korean in reasons.
- Mark abbreviations or shorthand forms as normalizations.
- Mark values from a different semantic taxonomy as out_of_domain_values.
- Mark format/type violations as invalid_format_values.
- Mark mixed but individually valid formats as inconsistent_format_groups.
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
        response = llm.invoke_json(
            prompt,
            system_prompt=(
                "You validate categorical values in Korean public datasets. "
                "Return a single JSON object only. No markdown, no explanation, no code fences."
            ),
        )
        if response is None:
            return None
        payload = _parse_json_content(response.content)
        if payload is None:
            llm.last_error = f"llm_parse_error:{response.content[:200]}"
            return None
        payload.setdefault("domain_label", "")
        payload.setdefault("canonical_values", [])
        payload.setdefault("normalizations", [])
        payload.setdefault("out_of_domain_values", [])
        payload.setdefault("invalid_format_values", [])
        payload.setdefault("inconsistent_format_groups", [])
        payload.setdefault("needs_manual_review", [])
        payload.setdefault("overall_confidence", 0.0)
        return payload


class CategoricalSemanticValidationAgent(BaseAgent):
    name = "categorical_semantic_validator"

    def __init__(self, validator: LLMCategoricalValueValidator | None = None):
        self.validator = validator

    def _llm_debug_detail(self, use_llm: bool) -> tuple[str, str]:
        if not use_llm or self.validator is None:
            return "", ""
        client = self.validator._llm
        return client.last_error, client.last_response_preview

    @staticmethod
    def _is_candidate_column(column) -> bool:
        if column.distinct_count is None:
            return False
        if not (CATEGORICAL_LLM_MIN_DISTINCT <= column.distinct_count <= CATEGORICAL_LLM_MAX_DISTINCT):
            return False
        if not column.top_values:
            return False

        categorical_tokens = (
            "구분",
            "유형",
            "종류",
            "상태",
            "여부",
            "유무",
            "급",
            "분류",
            "코드",
            "명칭",
            "일자",
            "일시",
            "날짜",
            "년월",
            "내용",
            "설명",
            "사유",
            "비고",
            "메모",
            "특이사항",
            "조치",
            "민원",
            "안내",
        )
        categorical_tags = {"enum", "code", "boolean", "name", "date"}
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
            counter = Counter()
            for row in rows:
                value = (row.get(column.raw_name) or "").strip()
                if value:
                    counter[value] += 1

            existing_finding_keys = {_finding_key(finding) for finding in findings}
            malformed_values = [value for value in counter if _looks_malformed_text_value(value)]
            for value in malformed_values:
                row_indexes = self._value_rows(rows, column.raw_name, value)
                finding = build_finding(
                    column_name=column.raw_name,
                    severity="warning",
                    category_group="completeness",
                    criterion_name="garbled_text",
                    rule_id="garbled_text",
                    message=f"'{value}' 값은 불필요한 기호 또는 깨진 텍스트가 포함된 것으로 보입니다.",
                    row_indexes=row_indexes,
                    related_columns=[column.raw_name],
                    evidence=["detector:malformed_text"],
                )
                key = _finding_key(finding)
                if key not in existing_finding_keys:
                    findings.append(finding)
                    existing_finding_keys.add(key)

            truncated_pairs = _find_truncated_value_pairs(counter)
            for source, target in truncated_pairs:
                row_indexes = self._value_rows(rows, column.raw_name, source)
                finding = build_finding(
                    column_name=column.raw_name,
                    severity="warning",
                    category_group="domain_validity",
                    criterion_name="categorical_semantic_domain",
                    rule_id="categorical_value_truncated",
                    message=f"'{source}' 값은 '{target}' 값의 앞부분과 일치해 입력 중 잘림 가능성이 있습니다.",
                    row_indexes=row_indexes,
                    related_columns=[column.raw_name],
                    evidence=[f"matched_full_value:{target}", "detector:prefix_truncation"],
                )
                key = _finding_key(finding)
                if key not in existing_finding_keys:
                    findings.append(finding)
                    existing_finding_keys.add(key)

            if not self._is_candidate_column(column):
                if truncated_pairs or malformed_values:
                    traces.append(
                        self.trace(
                            action="categorical_semantic_validate",
                            target=column.raw_name,
                            detail=(
                                f"local_truncated_findings={len(truncated_pairs)}, "
                                f"local_malformed_findings={len(malformed_values)}, skipped:llm_candidate_filter"
                            ),
                        )
                    )
                continue

            if not (CATEGORICAL_LLM_MIN_DISTINCT <= len(counter) <= CATEGORICAL_LLM_MAX_DISTINCT):
                if truncated_pairs or malformed_values:
                    traces.append(
                        self.trace(
                            action="categorical_semantic_validate",
                            target=column.raw_name,
                            detail=(
                                f"local_truncated_findings={len(truncated_pairs)}, "
                                f"local_malformed_findings={len(malformed_values)}, skipped:distinct_count={len(counter)}"
                            ),
                        )
                    )
                continue

            if not counter:
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
                llm_error, llm_preview = self._llm_debug_detail(use_llm)
                traces.append(
                    self.trace(
                        action="categorical_semantic_validate",
                        target=column.raw_name,
                        detail=(f"llm_no_result,error={llm_error},preview={llm_preview}"),
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
                reason = _clean_reason_text(item.get("reason"))
                if not source or not target or source == target:
                    continue
                if _looks_boolean_column(column) and not _is_yn_value(source) and _is_yn_value(target):
                    evidence = [f"domain:{result.get('domain_label', '')}", f"confidence:{confidence:.2f}"]
                    if reason:
                        evidence.append(f"reason:{reason}")
                    findings.append(
                        build_finding(
                            column_name=column.raw_name,
                            severity="warning",
                            category_group="domain_validity",
                            criterion_name="boolean_domain",
                            rule_id="boolean_domain",
                            message=f"'{source}' 값은 Y/N 여부 컬럼의 허용값과 맞지 않을 수 있습니다.",
                            row_indexes=self._value_rows(rows, column.raw_name, source),
                            related_columns=[column.raw_name],
                            evidence=evidence,
                        )
                    )
                    generated += 1
                    continue
                evidence = [f"domain:{result.get('domain_label', '')}", f"confidence:{confidence:.2f}"]
                if reason:
                    evidence.append(f"reason:{reason}")
                findings.append(
                    build_finding(
                        column_name=column.raw_name,
                        severity="warning",
                        category_group="domain_validity",
                        criterion_name="categorical_semantic_domain",
                        rule_id="categorical_value_normalization",
                        message=f"'{source}' 값은 '{target}'로 표준화하는 것이 적절합니다.",
                        row_indexes=self._value_rows(rows, column.raw_name, source),
                        related_columns=[column.raw_name],
                        evidence=evidence,
                    )
                )
                generated += 1

            for item in result.get("invalid_format_values", []):
                confidence = float(item.get("confidence") or 0.0)
                if confidence < CATEGORICAL_LLM_CONFIDENCE_THRESHOLD:
                    continue
                value = str(item.get("value") or "").strip()
                issue_type = str(item.get("issue_type") or "").strip()
                reason = _clean_reason_text(item.get("reason"))
                if not value:
                    continue
                rule_id = (
                    "boolean_domain"
                    if issue_type == "boolean_invalid"
                    else "date_domain"
                    if issue_type == "date_invalid"
                    else "garbled_text"
                    if issue_type == "malformed_text"
                    else "categorical_value_truncated"
                    if issue_type == "truncated_text"
                    else "categorical_value_out_of_domain"
                )
                criterion_name = (
                    "boolean_domain"
                    if issue_type == "boolean_invalid"
                    else "date_domain"
                    if issue_type == "date_invalid"
                    else "garbled_text"
                    if issue_type == "malformed_text"
                    else "categorical_semantic_domain"
                )
                category_group = "completeness" if issue_type == "malformed_text" else "domain_validity"
                evidence = [f"domain:{result.get('domain_label', '')}", f"confidence:{confidence:.2f}", f"issue_type:{issue_type}"]
                if reason:
                    evidence.append(f"reason:{reason}")
                message = (
                    f"'{value}' 값은 불필요한 기호 또는 깨진 텍스트가 포함된 것으로 보입니다."
                    if issue_type == "malformed_text"
                    else
                    f"'{value}' 값은 문맥상 입력 중 잘렸거나 불완전한 텍스트일 수 있습니다."
                    if issue_type == "truncated_text"
                    else f"'{value}' 값은 컬럼의 형식 또는 허용값과 맞지 않을 수 있습니다."
                )
                findings.append(
                    build_finding(
                        column_name=column.raw_name,
                        severity="warning",
                        category_group=category_group,
                        criterion_name=criterion_name,
                        rule_id=rule_id,
                        message=message,
                        row_indexes=self._value_rows(rows, column.raw_name, value),
                        related_columns=[column.raw_name],
                        evidence=evidence,
                    )
                )
                generated += 1

            for item in result.get("inconsistent_format_groups", []):
                confidence = float(item.get("confidence") or 0.0)
                if confidence < CATEGORICAL_LLM_CONFIDENCE_THRESHOLD:
                    continue
                values_in_group = [str(value).strip() for value in item.get("values", []) if str(value).strip()]
                target_format = str(item.get("target_format") or "").strip()
                reason = _clean_reason_text(item.get("reason"))
                if not values_in_group:
                    continue
                if not _looks_date_column(column) and not all(_looks_date_value(value) for value in values_in_group):
                    continue
                evidence = [f"domain:{result.get('domain_label', '')}", f"confidence:{confidence:.2f}"]
                if target_format:
                    evidence.append(f"target_format:{target_format}")
                if reason:
                    evidence.append(f"reason:{reason}")
                findings.append(
                    build_finding(
                        column_name=column.raw_name,
                        severity="warning",
                        category_group="domain_validity",
                        criterion_name="date_domain",
                        rule_id="date_format_inconsistent",
                        message=f"날짜 또는 형식 컬럼에서 표기 형식이 혼용됩니다: {', '.join(values_in_group)}",
                        row_indexes=[
                            row_index
                            for value in values_in_group
                            for row_index in self._value_rows(rows, column.raw_name, value)
                        ],
                        related_columns=[column.raw_name],
                        evidence=evidence,
                    )
                )
                generated += 1

            for item in result.get("out_of_domain_values", []):
                confidence = float(item.get("confidence") or 0.0)
                if confidence < CATEGORICAL_LLM_CONFIDENCE_THRESHOLD:
                    continue
                value = str(item.get("value") or "").strip()
                reason = _clean_reason_text(item.get("reason"))
                if not value or not _is_specific_out_of_domain_reason(reason):
                    continue
                evidence = [f"domain:{result.get('domain_label', '')}", f"confidence:{confidence:.2f}"]
                if reason:
                    evidence.append(f"reason:{reason}")
                findings.append(
                    build_finding(
                        column_name=column.raw_name,
                        severity="warning",
                        category_group="domain_validity",
                        criterion_name="categorical_semantic_domain",
                        rule_id="categorical_value_out_of_domain",
                        message=f"'{value}' 값은 해당 컬럼의 의미 도메인과 맞지 않을 수 있습니다.",
                        row_indexes=self._value_rows(rows, column.raw_name, value),
                        related_columns=[column.raw_name],
                        evidence=evidence,
                    )
                )
                generated += 1

            for item in result.get("needs_manual_review", []):
                confidence = float(item.get("confidence") or 0.0)
                value = str(item.get("value") or "").strip()
                reason = _clean_reason_text(item.get("reason"))
                if not value:
                    continue
                evidence = [f"domain:{result.get('domain_label', '')}", f"confidence:{confidence:.2f}"]
                if reason:
                    evidence.append(f"reason:{reason}")
                findings.append(
                    build_finding(
                        column_name=column.raw_name,
                        severity="info",
                        category_group="domain_validity",
                        criterion_name="categorical_semantic_domain",
                        rule_id="categorical_value_manual_review",
                        message=f"'{value}' 값은 의미 판정이 애매해 수동 검토가 필요합니다.",
                        row_indexes=self._value_rows(rows, column.raw_name, value),
                        related_columns=[column.raw_name],
                        evidence=evidence,
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
