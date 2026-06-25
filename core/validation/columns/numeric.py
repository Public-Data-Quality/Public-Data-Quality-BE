from __future__ import annotations

from ...schema.models import ValidationFinding
from .context import ColumnRuleContext
from .helpers import matching_row_indexes
from ..helpers import build_finding, parse_number


def _numeric_parse_or_negative_findings(
    context: ColumnRuleContext,
    *,
    criterion_name: str,
    parse_error_message: str,
    negative_value_message: str,
) -> list[ValidationFinding]:
    column = context.column
    if column.numeric_parse_ratio is not None and column.numeric_parse_ratio < 1.0:
        return [
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group="domain_validity",
                criterion_name=criterion_name,
                message=parse_error_message,
                row_indexes=matching_row_indexes(
                    context.rows,
                    column.raw_name,
                    lambda value: bool(value) and parse_number(value) is None,
                ),
                evidence=[f"numeric_parse_ratio:{column.numeric_parse_ratio:.2f}"],
            )
        ]

    if column.numeric_min is not None and column.numeric_min < 0:
        return [
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group="domain_validity",
                criterion_name=criterion_name,
                message=negative_value_message,
                row_indexes=matching_row_indexes(
                    context.rows,
                    column.raw_name,
                    lambda value: (parsed := parse_number(value)) is not None and parsed < 0,
                ),
                evidence=[f"min:{column.numeric_min}"],
            )
        ]

    return []


def find_amount_domain_issues(context: ColumnRuleContext) -> list[ValidationFinding]:
    if "amount" not in context.column.semantic_tags:
        return []

    return _numeric_parse_or_negative_findings(
        context,
        criterion_name="amount_domain",
        parse_error_message="금액 도메인 컬럼에 숫자 파싱이 되지 않는 값이 존재합니다.",
        negative_value_message="금액 도메인 컬럼에 음수 값이 포함되어 있습니다.",
    )


def find_quantity_domain_issues(context: ColumnRuleContext) -> list[ValidationFinding]:
    if "quantity" not in context.column.semantic_tags and "count" not in context.column.semantic_tags:
        return []

    return _numeric_parse_or_negative_findings(
        context,
        criterion_name="quantity_domain",
        parse_error_message="수량 도메인 컬럼에 숫자 파싱이 되지 않는 값이 존재합니다.",
        negative_value_message="수량 도메인 컬럼에 음수 값이 포함되어 있습니다.",
    )


def find_rate_domain_issues(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if not (
        "rate" in column.semantic_tags
        and column.numeric_min is not None
        and column.numeric_max is not None
    ):
        return []
    if column.numeric_min >= 0 and column.numeric_max <= 100:
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="domain_validity",
            criterion_name="rate_domain",
            message="율 도메인 컬럼 값이 일반적인 0~100 범위를 벗어났습니다.",
            row_indexes=matching_row_indexes(
                context.rows,
                column.raw_name,
                lambda value: (parsed := parse_number(value)) is not None and (parsed < 0 or parsed > 100),
            ),
            evidence=[f"min:{column.numeric_min}", f"max:{column.numeric_max}"],
        )
    ]


def _numeric_range_findings(
    context: ColumnRuleContext,
    *,
    tag: str,
    lower_bound: float,
    upper_bound: float,
    message: str,
) -> list[ValidationFinding]:
    column = context.column
    if not (
        tag in column.semantic_tags
        and column.numeric_min is not None
        and column.numeric_max is not None
    ):
        return []
    if column.numeric_min >= lower_bound and column.numeric_max <= upper_bound:
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="error",
            category_group="domain_validity",
            criterion_name="number_domain",
            message=message,
            row_indexes=matching_row_indexes(
                context.rows,
                column.raw_name,
                lambda value: (parsed := parse_number(value)) is not None
                and (parsed < lower_bound or parsed > upper_bound),
            ),
            evidence=[f"min:{column.numeric_min}", f"max:{column.numeric_max}"],
        )
    ]


def find_latitude_domain_issues(context: ColumnRuleContext) -> list[ValidationFinding]:
    return _numeric_range_findings(
        context,
        tag="geo_lat",
        lower_bound=-90,
        upper_bound=90,
        message="위도 값이 허용 범위를 벗어났습니다.",
    )


def find_longitude_domain_issues(context: ColumnRuleContext) -> list[ValidationFinding]:
    return _numeric_range_findings(
        context,
        tag="geo_lon",
        lower_bound=-180,
        upper_bound=180,
        message="경도 값이 허용 범위를 벗어났습니다.",
    )
