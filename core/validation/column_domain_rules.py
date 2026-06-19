from __future__ import annotations

from ..schema.models import ValidationFinding
from .column_rule_context import ColumnRuleContext
from .column_rule_helpers import matching_row_indexes
from .helpers import (
    BOOLEAN_ALLOWED_VALUES,
    PHONE_DIGIT_RE,
    allowed_values,
    build_finding,
    parse_datetime,
)


def find_invalid_dates(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if not (
        "date" in column.semantic_tags
        and column.date_parse_ratio is not None
        and column.date_parse_ratio < 1.0
    ):
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
            row_indexes=matching_row_indexes(
                context.rows,
                column.raw_name,
                lambda value: bool(value) and parse_datetime(value) is None,
            ),
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

    invalid_codes = [value for value, _ in column.top_values if value and value not in term_allowed_values]
    if not invalid_codes:
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="domain_validity",
            criterion_name="code_domain",
            message="코드 도메인 컬럼이 표준 허용값과 일치하지 않습니다.",
            row_indexes=matching_row_indexes(
                context.rows,
                column.raw_name,
                lambda value: bool(value) and value not in term_allowed_values,
            ),
            evidence=[f"allowed:{', '.join(term_allowed_values[:10])}", *invalid_codes[:3]],
        )
    ]
