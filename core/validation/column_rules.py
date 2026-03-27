from __future__ import annotations

from ..schema.models import ColumnProfile, DatasetMeta, StandardTerm, ValidationFinding
from .helpers import (
    BOOLEAN_ALLOWED_VALUES,
    PHONE_DIGIT_RE,
    allowed_values,
    build_finding,
    contains_broken_text,
    parse_datetime,
    parse_number,
    has_whitespace_or_special_issue,
)


def _matching_row_indexes(
    rows: list[dict[str, str]],
    column_name: str,
    predicate,
) -> list[int]:
    indexes: list[int] = []
    for row_index, row in enumerate(rows, start=1):
        value = (row.get(column_name) or "").strip()
        if predicate(value):
            indexes.append(row_index)
    return indexes


def _duplicate_value_row_indexes(rows: list[dict[str, str]], column_name: str) -> list[int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = (row.get(column_name) or "").strip()
        if value:
            counts[value] = counts.get(value, 0) + 1

    indexes: list[int] = []
    for row_index, row in enumerate(rows, start=1):
        value = (row.get(column_name) or "").strip()
        if value and counts.get(value, 0) > 1:
            indexes.append(row_index)
    return indexes


def is_likely_required(column: ColumnProfile) -> bool:
    required_tags = {"identifier", "name", "date", "address", "phone"}
    return bool(required_tags.intersection(column.semantic_tags)) or any(
        token in column.normalized_name for token in ("명", "이름", "번호", "일자", "주소", "전화")
    )


def looks_numeric_column(column: ColumnProfile) -> bool:
    numeric_tags = {"numeric", "count", "quantity", "amount", "rate", "width"}
    return bool(numeric_tags.intersection(column.semantic_tags)) or column.inferred_primitive_type == "numeric"


def validate_column(
    column: ColumnProfile,
    dataset_meta: DatasetMeta,
    standard_terms: dict[str, StandardTerm],
    rows: list[dict[str, str]] | None = None,
) -> list[ValidationFinding]:
    del dataset_meta
    rows = rows or []
    findings: list[ValidationFinding] = []
    best_term = standard_terms.get(column.standard_candidates[0]) if column.standard_candidates else None

    if not column.assigned_rules:
        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group="completeness",
                criterion_name="required_value",
                message="검증 규칙이 할당되지 않아 수동 검토가 필요합니다.",
                rule_id="manual_review_required",
                evidence=column.rag_evidence,
            )
        )
        return findings

    if contains_broken_text(column.raw_name) or any(contains_broken_text(value) for value in column.sample_values):
        row_indexes = _matching_row_indexes(rows, column.raw_name, contains_broken_text)
        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="error",
                category_group="completeness",
                criterion_name="garbled_text",
                message="컬럼명 또는 샘플 데이터에 글자 깨짐이 의심됩니다.",
                row_indexes=row_indexes,
                evidence=column.sample_values[:3],
            )
        )

    if has_whitespace_or_special_issue(column.raw_name) or any(
        has_whitespace_or_special_issue(value) for value in column.sample_values[:5]
    ):
        row_indexes = _matching_row_indexes(rows, column.raw_name, has_whitespace_or_special_issue)
        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group="completeness",
                criterion_name="whitespace_special_characters",
                message="컬럼명 또는 값에 불필요한 공백/특수문자가 포함된 것으로 보입니다.",
                row_indexes=row_indexes,
                evidence=column.sample_values[:3],
            )
        )

    if is_likely_required(column) and (column.null_count or 0) > 0:
        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group="completeness",
                criterion_name="required_value",
                message=f"필수성이 높은 컬럼으로 추정되나 결측값 {column.null_count}건이 존재합니다.",
                row_indexes=_matching_row_indexes(rows, column.raw_name, lambda value: not value.strip()),
                evidence=[f"null_ratio:{column.null_ratio}"],
            )
        )

    if "identifier" in column.semantic_tags and column.distinct_count is not None and column.non_empty_count > column.distinct_count:
        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group="completeness",
                criterion_name="duplicate_data",
                message="식별자 성격의 컬럼에서 중복 데이터가 탐지되었습니다.",
                row_indexes=_duplicate_value_row_indexes(rows, column.raw_name),
                evidence=[f"non_empty:{column.non_empty_count}", f"distinct:{column.distinct_count}"],
            )
        )

    if "date" in column.semantic_tags and column.date_parse_ratio is not None and column.date_parse_ratio < 1.0:
        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group="domain_validity",
                criterion_name="date_domain",
                message="날짜 도메인 컬럼에서 유효하지 않은 날짜 형식 또는 범위 이탈 값이 존재합니다.",
                row_indexes=_matching_row_indexes(
                    rows,
                    column.raw_name,
                    lambda value: bool(value) and parse_datetime(value) is None,
                ),
                evidence=[f"date_parse_ratio:{column.date_parse_ratio:.2f}"],
            )
        )

    if "phone" in column.semantic_tags:
        invalid_phone = [value for value in column.sample_values if value and not PHONE_DIGIT_RE.match(value)]
        if invalid_phone:
            findings.append(
                build_finding(
                    column_name=column.raw_name,
                    severity="warning",
                    category_group="domain_validity",
                    criterion_name="number_domain",
                    message="번호 도메인 컬럼에 규칙을 벗어난 값이 포함된 것으로 보입니다.",
                    row_indexes=_matching_row_indexes(
                        rows,
                        column.raw_name,
                        lambda value: bool(value) and not PHONE_DIGIT_RE.match(value),
                    ),
                    evidence=invalid_phone[:3],
                )
            )

    if "boolean" in column.semantic_tags:
        invalid_boolean = [value for value, _ in column.top_values if value.strip().lower() not in BOOLEAN_ALLOWED_VALUES]
        if invalid_boolean:
            findings.append(
                build_finding(
                    column_name=column.raw_name,
                    severity="warning",
                    category_group="domain_validity",
                    criterion_name="boolean_domain",
                    message="여부 도메인 컬럼에 2값 범위를 벗어난 값이 존재합니다.",
                    row_indexes=_matching_row_indexes(
                        rows,
                        column.raw_name,
                        lambda value: bool(value.strip()) and value.strip().lower() not in BOOLEAN_ALLOWED_VALUES,
                    ),
                    evidence=invalid_boolean[:5],
                )
            )

    term_allowed_values = allowed_values(best_term)
    if ("enum" in column.semantic_tags or "code" in column.semantic_tags) and term_allowed_values:
        invalid_codes = [value for value, _ in column.top_values if value and value not in term_allowed_values]
        if invalid_codes:
            findings.append(
                build_finding(
                    column_name=column.raw_name,
                    severity="warning",
                    category_group="domain_validity",
                    criterion_name="code_domain",
                    message="코드 도메인 컬럼이 표준 허용값과 일치하지 않습니다.",
                    row_indexes=_matching_row_indexes(
                        rows,
                        column.raw_name,
                        lambda value: bool(value) and value not in term_allowed_values,
                    ),
                    evidence=[f"allowed:{', '.join(term_allowed_values[:10])}", *invalid_codes[:3]],
                )
            )

    if "amount" in column.semantic_tags:
        if column.numeric_parse_ratio is not None and column.numeric_parse_ratio < 1.0:
            findings.append(
                build_finding(
                    column_name=column.raw_name,
                    severity="warning",
                    category_group="domain_validity",
                    criterion_name="amount_domain",
                    message="금액 도메인 컬럼에 숫자 파싱이 되지 않는 값이 존재합니다.",
                    row_indexes=_matching_row_indexes(
                        rows,
                        column.raw_name,
                        lambda value: bool(value) and parse_number(value) is None,
                    ),
                    evidence=[f"numeric_parse_ratio:{column.numeric_parse_ratio:.2f}"],
                )
            )
        elif column.numeric_min is not None and column.numeric_min < 0:
            findings.append(
                build_finding(
                    column_name=column.raw_name,
                    severity="warning",
                    category_group="domain_validity",
                    criterion_name="amount_domain",
                    message="금액 도메인 컬럼에 음수 값이 포함되어 있습니다.",
                    row_indexes=_matching_row_indexes(
                        rows,
                        column.raw_name,
                        lambda value: (parsed := parse_number(value)) is not None and parsed < 0,
                    ),
                    evidence=[f"min:{column.numeric_min}"],
                )
            )

    if "quantity" in column.semantic_tags or "count" in column.semantic_tags:
        if column.numeric_parse_ratio is not None and column.numeric_parse_ratio < 1.0:
            findings.append(
                build_finding(
                    column_name=column.raw_name,
                    severity="warning",
                    category_group="domain_validity",
                    criterion_name="quantity_domain",
                    message="수량 도메인 컬럼에 숫자 파싱이 되지 않는 값이 존재합니다.",
                    row_indexes=_matching_row_indexes(
                        rows,
                        column.raw_name,
                        lambda value: bool(value) and parse_number(value) is None,
                    ),
                    evidence=[f"numeric_parse_ratio:{column.numeric_parse_ratio:.2f}"],
                )
            )
        elif column.numeric_min is not None and column.numeric_min < 0:
            findings.append(
                build_finding(
                    column_name=column.raw_name,
                    severity="warning",
                    category_group="domain_validity",
                    criterion_name="quantity_domain",
                    message="수량 도메인 컬럼에 음수 값이 포함되어 있습니다.",
                    row_indexes=_matching_row_indexes(
                        rows,
                        column.raw_name,
                        lambda value: (parsed := parse_number(value)) is not None and parsed < 0,
                    ),
                    evidence=[f"min:{column.numeric_min}"],
                )
            )

    if "rate" in column.semantic_tags and column.numeric_min is not None and column.numeric_max is not None:
        if column.numeric_min < 0 or column.numeric_max > 100:
            findings.append(
                build_finding(
                    column_name=column.raw_name,
                    severity="warning",
                    category_group="domain_validity",
                    criterion_name="rate_domain",
                    message="율 도메인 컬럼 값이 일반적인 0~100 범위를 벗어났습니다.",
                    row_indexes=_matching_row_indexes(
                        rows,
                        column.raw_name,
                        lambda value: (parsed := parse_number(value)) is not None and (parsed < 0 or parsed > 100),
                    ),
                    evidence=[f"min:{column.numeric_min}", f"max:{column.numeric_max}"],
                )
            )

    if "geo_lat" in column.semantic_tags and column.numeric_min is not None and column.numeric_max is not None:
        if column.numeric_min < -90 or column.numeric_max > 90:
            findings.append(
                build_finding(
                    column_name=column.raw_name,
                    severity="error",
                    category_group="domain_validity",
                    criterion_name="number_domain",
                    message="위도 값이 허용 범위를 벗어났습니다.",
                    row_indexes=_matching_row_indexes(
                        rows,
                        column.raw_name,
                        lambda value: (parsed := parse_number(value)) is not None and (parsed < -90 or parsed > 90),
                    ),
                    evidence=[f"min:{column.numeric_min}", f"max:{column.numeric_max}"],
                )
            )

    if "geo_lon" in column.semantic_tags and column.numeric_min is not None and column.numeric_max is not None:
        if column.numeric_min < -180 or column.numeric_max > 180:
            findings.append(
                build_finding(
                    column_name=column.raw_name,
                    severity="error",
                    category_group="domain_validity",
                    criterion_name="number_domain",
                    message="경도 값이 허용 범위를 벗어났습니다.",
                    row_indexes=_matching_row_indexes(
                        rows,
                        column.raw_name,
                        lambda value: (parsed := parse_number(value)) is not None and (parsed < -180 or parsed > 180),
                    ),
                    evidence=[f"min:{column.numeric_min}", f"max:{column.numeric_max}"],
                )
            )

    if not column.standard_candidates:
        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group="completeness",
                criterion_name="required_value",
                rule_id="standard_term_missing",
                message="표준용어 후보를 찾지 못해 정밀 검증 범위가 제한됩니다.",
                evidence=[f"normalized:{column.normalized_name}"],
            )
        )

    return findings


def build_repair_suggestion(column: ColumnProfile) -> str | None:
    if column.raw_name == column.normalized_name and not column.standard_candidates:
        return None

    parts: list[str] = []
    if column.raw_name != column.normalized_name:
        parts.append(f"컬럼명을 '{column.normalized_name}'로 정규화")
    if column.unit:
        parts.append(f"단위 '{column.unit}'를 별도 메타데이터로 분리")
    if column.standard_candidates:
        parts.append(f"표준용어 후보 '{column.standard_candidates[0]}'에 매핑")
    return ", ".join(parts) if parts else None
