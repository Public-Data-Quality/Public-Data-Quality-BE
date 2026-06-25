from __future__ import annotations

from typing import Any

from ...schema.models import ColumnProfile, ValidationFinding
from ..columns import looks_numeric_column
from ..helpers import build_finding, parse_number
from .common import candidate_groups, is_related_numeric_pair


def validate_calculation_relationships(
    columns: list[ColumnProfile],
    rows: list[dict[str, str]],
    relationship_candidates: list[dict[str, Any]] | None = None,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    selected_groups = candidate_groups(relationship_candidates, {"calculation_formula"}, columns)
    if relationship_candidates is not None:
        groups = [(group[0], group[1:]) for group in selected_groups if len(group) >= 3]
    else:
        total_columns = [column for column in columns if "총" in column.normalized_name and looks_numeric_column(column)]
        part_columns = [column for column in columns if looks_numeric_column(column)]
        groups = [
            (
                total_col,
                [
                    column
                    for column in part_columns
                    if column.raw_name != total_col.raw_name and is_related_numeric_pair(total_col, column)
                ],
            )
            for total_col in total_columns
        ]

    for total_col, siblings in groups:
        if len(siblings) < 2:
            continue
        for left_index in range(len(siblings)):
            for right_index in range(left_index + 1, len(siblings)):
                left = siblings[left_index]
                right = siblings[right_index]
                mismatch = 0
                comparable = 0
                mismatch_row_indexes: list[int] = []
                for row_index, row in enumerate(rows, start=1):
                    total = parse_number(row.get(total_col.raw_name, ""))
                    left_value = parse_number(row.get(left.raw_name, ""))
                    right_value = parse_number(row.get(right.raw_name, ""))
                    if total is None or left_value is None or right_value is None:
                        continue
                    comparable += 1
                    if abs(total - (left_value + right_value)) > 1e-6:
                        mismatch += 1
                        mismatch_row_indexes.append(row_index)
                if comparable and mismatch and mismatch / comparable >= 0.3:
                    findings.append(
                        build_finding(
                            column_name=total_col.raw_name,
                            severity="warning",
                            category_group="relation_consistency",
                            criterion_name="calculation_formula",
                            message=(
                                f"'{total_col.raw_name}'가 '{left.raw_name} + {right.raw_name}'와 일치하지 않는 행이 "
                                f"{mismatch}건 존재합니다."
                            ),
                            row_indexes=mismatch_row_indexes,
                            related_columns=[total_col.raw_name, left.raw_name, right.raw_name],
                            evidence=[f"checked_rows:{comparable}", f"mismatch_rows:{mismatch}"],
                        )
                    )
                    break
            else:
                continue
            break
    return findings
