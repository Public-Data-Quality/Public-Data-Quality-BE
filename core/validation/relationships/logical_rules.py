from __future__ import annotations

from typing import Any

from ...schema.models import ColumnProfile, ValidationFinding
from ..helpers import build_finding, parse_number
from .common import candidate_pairs


def validate_logical_relationships(
    columns: list[ColumnProfile],
    rows: list[dict[str, str]],
    relationship_candidates: list[dict[str, Any]] | None = None,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    selected_pairs = candidate_pairs(relationship_candidates, {"logical_consistency"}, columns)
    if relationship_candidates is not None:
        pairs = [
            (left, right)
            for left, right in selected_pairs
            if "boolean" in left.semantic_tags and {"count", "quantity"}.intersection(right.semantic_tags)
        ] + [
            (right, left)
            for left, right in selected_pairs
            if "boolean" in right.semantic_tags and {"count", "quantity"}.intersection(left.semantic_tags)
        ]
    else:
        boolean_columns = [column for column in columns if "boolean" in column.semantic_tags]
        quantity_columns = [column for column in columns if {"count", "quantity"}.intersection(column.semantic_tags)]
        pairs = [(flag_col, qty_col) for flag_col in boolean_columns for qty_col in quantity_columns]

    for flag_col, qty_col in pairs:
        stem = flag_col.normalized_name.replace("여부", "").replace("유무", "")
        if stem and stem in qty_col.normalized_name:
            inconsistent_row_indexes: list[int] = []
            for row_index, row in enumerate(rows, start=1):
                flag_value = row.get(flag_col.raw_name, "").strip().lower()
                qty_value = parse_number(row.get(qty_col.raw_name, ""))
                if flag_value in {"n", "no", "false", "0", "아니오", "무"} and qty_value and qty_value > 0:
                    inconsistent_row_indexes.append(row_index)
            inconsistency_count = len(inconsistent_row_indexes)
            if inconsistency_count:
                findings.append(
                    build_finding(
                        column_name=flag_col.raw_name,
                        severity="warning",
                        category_group="relation_consistency",
                        criterion_name="logical_consistency",
                        message=(
                            f"'{flag_col.raw_name}'가 부정값인데 '{qty_col.raw_name}'가 양수인 행이 "
                            f"{inconsistency_count}건 존재합니다."
                        ),
                        row_indexes=inconsistent_row_indexes,
                        related_columns=[flag_col.raw_name, qty_col.raw_name],
                        evidence=[f"inconsistent_rows:{inconsistency_count}"],
                    )
                )
    return findings
