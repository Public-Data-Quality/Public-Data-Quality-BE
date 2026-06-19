from __future__ import annotations

import re
from typing import Any

from ...schema.models import ColumnProfile, ValidationFinding
from ..helpers import build_finding
from .common import candidate_pairs
from .region import REGION_VALUE_RE, address_region_prefix, looks_address_column, looks_region_column


def validate_region_address_relationships(
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
            if looks_region_column(left, rows) and looks_address_column(right)
        ] + [
            (right, left)
            for left, right in selected_pairs
            if looks_region_column(right, rows) and looks_address_column(left)
        ]
    else:
        region_columns = [column for column in columns if looks_region_column(column, rows)]
        address_columns = [column for column in columns if looks_address_column(column)]
        pairs = [(region_col, address_col) for region_col in region_columns for address_col in address_columns]

    for region_col, address_col in pairs:
        if region_col.raw_name == address_col.raw_name:
            continue
        conflict_rows: list[int] = []
        conflict_examples: list[str] = []

        for row_index, row in enumerate(rows, start=1):
            region_value = (row.get(region_col.raw_name) or "").strip()
            address_value = re.sub(r"\s+", " ", row.get(address_col.raw_name) or "").strip()
            if not region_value or not address_value:
                continue
            if not REGION_VALUE_RE.fullmatch(region_value):
                continue
            address_region = address_region_prefix(address_value)
            if not address_region or address_region == region_value:
                continue
            conflict_rows.append(row_index)
            if len(conflict_examples) < 3:
                conflict_examples.append(f"{region_value} != {address_region}: {address_value}")

        if not conflict_rows:
            continue

        findings.append(
            build_finding(
                column_name=address_col.raw_name,
                severity="warning",
                category_group="relation_consistency",
                criterion_name="logical_consistency",
                rule_id="address_region_prefix_mismatch",
                message=(
                    f"'{address_col.raw_name}'에 명시된 광역지역명이 같은 행의 '{region_col.raw_name}' 값과 "
                    f"다른 행이 {len(conflict_rows)}건 존재합니다."
                ),
                row_indexes=conflict_rows,
                related_columns=[region_col.raw_name, address_col.raw_name],
                evidence=conflict_examples,
            )
        )
    return findings
