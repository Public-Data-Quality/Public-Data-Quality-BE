from __future__ import annotations

from ...schema.models import ValidationFinding
from .context import ColumnRuleContext
from .helpers import matching_row_indexes
from ..helpers import (
    BOOLEAN_ALLOWED_VALUES,
    PHONE_DIGIT_RE,
    allowed_values,
    build_finding,
    parse_datetime,
)

DATE_DOMAIN_NAME_TOKENS = (
    "일자",
    "일시",
    "날짜",
    "년월",
    "연도",
    "년도",
    "등록일",
    "기준일",
    "개방일",
    "운영일",
    "시행일",
    "공개일",
    "마감일",
    "시작일",
    "종료일",
    "접수일",
    "처리일",
    "발생일",
    "생성일",
    "수정일",
    "갱신일",
)

TIME_ONLY_NAME_TOKENS = ("시간", "시각")
CODE_DOMAIN_INVALID_MAJORITY_RATIO = 0.5


def _looks_time_only_column(column) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    return any(token in name for token in TIME_ONLY_NAME_TOKENS) and not any(
        token in name for token in DATE_DOMAIN_NAME_TOKENS
    )


def _looks_date_domain_column(column) -> bool:
    if _looks_time_only_column(column):
        return False

    name = f"{column.raw_name} {column.normalized_name}"
    if any(token in name for token in DATE_DOMAIN_NAME_TOKENS):
        return True

    return column.inferred_primitive_type == "date" or (column.date_parse_ratio or 0) > 0


def find_invalid_dates(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if not (
        "date" in column.semantic_tags
        and _looks_date_domain_column(column)
        and column.date_parse_ratio is not None
        and column.date_parse_ratio < 1.0
    ):
        return []

    row_indexes = matching_row_indexes(
        context.rows,
        column.raw_name,
        lambda value: bool(value) and parse_datetime(value) is None,
    )
    if not row_indexes:
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="domain_validity",
            criterion_name="date_domain",
            message=(
                "날짜 도메인 컬럼에서 유효하지 않은 날짜 형식 또는 "
                "범위 이탈 값이 존재합니다."
            ),
            row_indexes=row_indexes,
            evidence=[f"date_parse_ratio:{column.date_parse_ratio:.2f}"],
        )
    ]


def find_invalid_phone_numbers(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if "phone" not in column.semantic_tags:
        return []

    invalid_phone = [value for value in column.sample_values if value and not PHONE_DIGIT_RE.match(value)]
    if not invalid_phone:
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="domain_validity",
            criterion_name="number_domain",
            message="번호 도메인 컬럼에 규칙을 벗어난 값이 포함된 것으로 보입니다.",
            row_indexes=matching_row_indexes(
                context.rows,
                column.raw_name,
                lambda value: bool(value) and not PHONE_DIGIT_RE.match(value),
            ),
            evidence=invalid_phone[:3],
        )
    ]


def find_invalid_booleans(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if "boolean" not in column.semantic_tags:
        return []

    invalid_boolean = [
        value
        for value, _ in column.top_values
        if value.strip().lower() not in BOOLEAN_ALLOWED_VALUES
    ]
    if not invalid_boolean:
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="domain_validity",
            criterion_name="boolean_domain",
            message="여부 도메인 컬럼에 2값 범위를 벗어난 값이 존재합니다.",
            row_indexes=matching_row_indexes(
                context.rows,
                column.raw_name,
                lambda value: bool(value.strip()) and value.strip().lower() not in BOOLEAN_ALLOWED_VALUES,
            ),
            evidence=invalid_boolean[:5],
        )
    ]


def find_invalid_codes(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    term_allowed_values = allowed_values(context.best_term)
    if not (("enum" in column.semantic_tags or "code" in column.semantic_tags) and term_allowed_values):
        return []

    allowed_value_set = set(term_allowed_values)
    invalid_row_indexes: list[int] = []
    invalid_examples: list[str] = []
    non_empty_count = 0

    for row_index, row in enumerate(context.rows, start=1):
        value = (row.get(column.raw_name) or "").strip()
        if not value:
            continue
        non_empty_count += 1
        if value in allowed_value_set:
            continue
        invalid_row_indexes.append(row_index)
        if value not in invalid_examples:
            invalid_examples.append(value)

    if not invalid_row_indexes:
        return []

    invalid_ratio = len(invalid_row_indexes) / max(1, non_empty_count)
    if invalid_ratio > CODE_DOMAIN_INVALID_MAJORITY_RATIO:
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="domain_validity",
            criterion_name="code_domain",
            message="코드 도메인 컬럼이 표준 허용값과 일치하지 않습니다.",
            row_indexes=invalid_row_indexes,
            evidence=[
                f"allowed:{', '.join(term_allowed_values[:10])}",
                f"invalid_ratio:{invalid_ratio:.2f}",
                *invalid_examples[:3],
            ],
        )
    ]
