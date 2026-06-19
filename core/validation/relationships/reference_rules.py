from __future__ import annotations

from typing import Any

from ...schema.models import ColumnProfile, ValidationFinding
from ..helpers import REFERENCE_PAIR_TOKENS, build_finding
from .common import candidate_pairs, find_matching_columns


def validate_reference_relationships(
    columns: list[ColumnProfile],
    rows: list[dict[str, str]],
    relationship_candidates: list[dict[str, Any]] | None = None,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    selected_pairs = candidate_pairs(relationship_candidates, {"reference_relation"}, columns)
    if relationship_candidates is not None:
        pairs = selected_pairs
    else:
        pairs = [
            pair
            for code_token, name_token in REFERENCE_PAIR_TOKENS
            for pair in find_matching_columns(columns, code_token, name_token)
        ]
    for code_col, name_col in pairs:
        mapping: dict[str, set[str]] = {}
        ambiguous_row_indexes: list[int] = []
        for row_index, row in enumerate(rows, start=1):
            code_value = row.get(code_col.raw_name, "").strip()
            name_value = row.get(name_col.raw_name, "").strip()
            if not code_value or not name_value:
                continue
            mapping.setdefault(code_value, set()).add(name_value)
        ambiguous = {code: names for code, names in mapping.items() if len(names) > 1}
        if ambiguous:
            sample_code, sample_names = next(iter(ambiguous.items()))
            for row_index, row in enumerate(rows, start=1):
                if row.get(code_col.raw_name, "").strip() == sample_code:
                    ambiguous_row_indexes.append(row_index)
            findings.append(
                build_finding(
                    column_name=code_col.raw_name,
                    severity="warning",
                    category_group="relation_consistency",
                    criterion_name="reference_relation",
                    message=(
                        f"참조 관계가 불안정합니다. 동일한 '{code_col.raw_name}' 값이 "
                        f"여러 '{name_col.raw_name}' 값과 연결됩니다."
                    ),
                    row_indexes=ambiguous_row_indexes,
                    related_columns=[code_col.raw_name, name_col.raw_name],
                    evidence=[f"{sample_code}:{', '.join(sorted(sample_names)[:3])}"],
                )
            )
    return findings
