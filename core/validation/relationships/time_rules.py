from __future__ import annotations

from typing import Any

from ...schema.models import ColumnProfile, ValidationFinding
from ..helpers import TIME_ORDER_TOKENS, build_finding, parse_datetime
from .common import candidate_pairs, find_matching_columns


def validate_time_relationships(
    columns: list[ColumnProfile],
    rows: list[dict[str, str]],
    relationship_candidates: list[dict[str, Any]] | None = None,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    selected_pairs = candidate_pairs(
        relationship_candidates,
        {"time_sequence_consistency", "precedence_accuracy"},
        columns,
    )
    if relationship_candidates is not None:
        pairs = selected_pairs
    else:
        pairs = [
            pair
            for left_token, right_token in TIME_ORDER_TOKENS
            for pair in find_matching_columns(columns, left_token, right_token)
        ]
    for left, right in pairs:
        reversed_row_indexes: list[int] = []
        for row_index, row in enumerate(rows, start=1):
            left_value = parse_datetime(row.get(left.raw_name, ""))
            right_value = parse_datetime(row.get(right.raw_name, ""))
            if left_value and right_value and left_value > right_value:
                reversed_row_indexes.append(row_index)
        reversed_count = len(reversed_row_indexes)
        if reversed_count:
            findings.append(
                build_finding(
                    column_name=left.raw_name,
                    severity="error",
                    category_group="relation_consistency",
                    criterion_name="time_sequence_consistency",
                    message=f"'{left.raw_name}'와 '{right.raw_name}' 간 시간순서가 뒤바뀐 행이 {reversed_count}건 존재합니다.",
                    row_indexes=reversed_row_indexes,
                    related_columns=[left.raw_name, right.raw_name],
                    evidence=[f"reversed_rows:{reversed_count}"],
                )
            )
            findings.append(
                build_finding(
                    column_name=left.raw_name,
                    severity="warning",
                    category_group="relation_consistency",
                    criterion_name="precedence_accuracy",
                    message=f"선후관계를 가져야 하는 '{left.raw_name}' -> '{right.raw_name}' 규칙이 지켜지지 않았습니다.",
                    row_indexes=reversed_row_indexes,
                    related_columns=[left.raw_name, right.raw_name],
                    evidence=[f"reversed_rows:{reversed_count}"],
                )
            )
    return findings
