from __future__ import annotations

from ..schema.models import ValidationFinding
from .column_rule_context import ColumnRuleContext
from .column_rule_helpers import (
    address_context_columns,
    duplicate_value_row_indexes,
    incomplete_detail_address_row_indexes,
    is_likely_required,
    looks_address_column,
    looks_detail_address_column,
    matching_row_indexes,
    truncated_address_row_indexes,
)
from .helpers import (
    build_finding,
    contains_broken_text,
    has_special_char_issue,
    has_whitespace_issue,
)

UNIQUE_IDENTIFIER_NAME_TOKENS = (
    "고유번호",
    "고유아이디",
    "식별번호",
    "식별자",
    "일련번호",
    "관리번호",
    "등록번호",
    "허가번호",
    "면허번호",
    "접수번호",
    "문서번호",
    "데이터번호",
    "센터ID",
    "센터아이디",
    "차량ID",
    "차량아이디",
    "ID",
    "아이디",
)
NON_UNIQUE_NAME_TOKENS = (
    "명",
    "명칭",
    "이름",
    "기관",
    "부서",
    "담당",
    "경찰서",
    "시설",
    "업소",
    "주소",
    "소재지",
    "전화",
    "연락처",
    "종류",
    "유형",
    "구분",
    "상태",
    "여부",
    "유무",
    "일자",
    "날짜",
)


def _normalize_name_for_identifier_check(value: str) -> str:
    return value.replace(" ", "").replace("_", "").replace("-", "").upper()


def _looks_unique_identifier_column(context: ColumnRuleContext) -> bool:
    column = context.column
    name = _normalize_name_for_identifier_check(f"{column.raw_name}{column.normalized_name}")
    if not any(token.upper() in name for token in UNIQUE_IDENTIFIER_NAME_TOKENS):
        return False
    if any(token.upper() in name for token in NON_UNIQUE_NAME_TOKENS):
        return False
    if column.non_empty_count <= 1 or column.distinct_count is None:
        return False
    distinct_ratio = column.distinct_count / column.non_empty_count
    return distinct_ratio >= 0.8


def find_garbled_text(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if not (
        contains_broken_text(column.raw_name)
        or any(contains_broken_text(value) for value in column.sample_values)
    ):
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="error",
            category_group="completeness",
            criterion_name="garbled_text",
            message="컬럼명 또는 샘플 데이터에 글자 깨짐이 의심됩니다.",
            row_indexes=matching_row_indexes(context.rows, column.raw_name, contains_broken_text),
            evidence=column.sample_values[:3],
        )
    ]


def find_whitespace_issues(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    has_value_issue = any(has_whitespace_issue(value) for value in context.sample_values)
    if not (has_whitespace_issue(column.raw_name) or has_value_issue):
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="completeness",
            criterion_name="whitespace_special_characters",
            rule_id="whitespace_issue",
            message="컬럼명 또는 값에 앞뒤 공백이나 연속 공백이 포함된 것으로 보입니다.",
            row_indexes=matching_row_indexes(
                context.rows,
                column.raw_name,
                has_whitespace_issue,
                strip_value=False,
            ),
            evidence=[value for value in context.sample_values if has_whitespace_issue(value)][:3],
        )
    ]


def find_special_character_issues(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    has_value_issue = any(has_special_char_issue(value) for value in context.sample_values)
    if not (has_special_char_issue(column.raw_name) or has_value_issue):
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="completeness",
            criterion_name="whitespace_special_characters",
            rule_id="special_character_issue",
            message=(
                "컬럼명 또는 값에 허용 범위를 벗어난 특수문자가 포함된 것으로 보입니다."
            ),
            row_indexes=matching_row_indexes(
                context.rows,
                column.raw_name,
                has_special_char_issue,
                strip_value=False,
            ),
            evidence=[value for value in context.sample_values if has_special_char_issue(value)][:3],
        )
    ]


def find_incomplete_detail_address(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if not looks_detail_address_column(column):
        return []

    related_columns = address_context_columns(context.rows, column.raw_name)
    row_indexes = incomplete_detail_address_row_indexes(
        context.rows,
        column.raw_name,
        related_columns,
    )
    if not row_indexes:
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="domain_validity",
            criterion_name="categorical_semantic_domain",
            rule_id="categorical_value_truncated",
            message=(
                "상세주소 값이 한두 글자 조각으로만 입력되어 문맥상 잘렸거나 "
                "불완전한 주소일 수 있습니다."
            ),
            row_indexes=row_indexes,
            related_columns=[column.raw_name, *related_columns],
            evidence=[f"incomplete_detail_address_rows:{len(row_indexes)}"],
        )
    ]


def find_truncated_address(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if not looks_address_column(column):
        return []

    row_indexes = truncated_address_row_indexes(context.rows, column.raw_name)
    if not row_indexes:
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="domain_validity",
            criterion_name="categorical_semantic_domain",
            rule_id="categorical_value_truncated",
            message=(
                "주소 값의 괄호 또는 시설명 부분이 닫히지 않아 입력 중 잘렸거나 "
                "불완전한 주소일 수 있습니다."
            ),
            row_indexes=row_indexes,
            related_columns=[column.raw_name],
            evidence=[f"truncated_address_rows:{len(row_indexes)}"],
        )
    ]


def find_missing_assigned_rules(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if column.assigned_rules:
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="completeness",
            criterion_name="required_value",
            message="검증 규칙이 할당되지 않아 수동 검토가 필요합니다.",
            rule_id="manual_review_required",
            evidence=column.rag_evidence,
        )
    ]


def find_required_nulls(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if not (is_likely_required(column) and (column.null_count or 0) > 0):
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="completeness",
            criterion_name="required_value",
            message=(
                f"필수성이 높은 컬럼으로 추정되나 결측값 {column.null_count}건이 존재합니다."
            ),
            row_indexes=matching_row_indexes(
                context.rows,
                column.raw_name,
                lambda value: not value.strip(),
            ),
            evidence=[f"null_ratio:{column.null_ratio}"],
        )
    ]


def find_duplicate_identifiers(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if not (
        ("identifier" in column.semantic_tags or "duplicate_data" in column.assigned_rules)
        and _looks_unique_identifier_column(context)
        and column.distinct_count is not None
        and column.non_empty_count > column.distinct_count
    ):
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="completeness",
            criterion_name="duplicate_data",
            message="식별자 성격의 컬럼에서 중복 데이터가 탐지되었습니다.",
            row_indexes=duplicate_value_row_indexes(context.rows, column.raw_name),
            evidence=[f"non_empty:{column.non_empty_count}", f"distinct:{column.distinct_count}"],
        )
    ]


def find_missing_standard_term(context: ColumnRuleContext) -> list[ValidationFinding]:
    column = context.column
    if column.standard_candidates:
        return []

    return [
        build_finding(
            column_name=column.raw_name,
            severity="warning",
            category_group="completeness",
            criterion_name="required_value",
            rule_id="standard_term_missing",
            message="표준용어 후보를 찾지 못해 정밀 검증 범위가 제한됩니다.",
            evidence=[f"normalized:{column.normalized_name}"],
        )
    ]
