from __future__ import annotations


def finding_key(finding) -> tuple[str, str, str, tuple[int, ...]]:
    return (
        finding.column_name,
        finding.rule_id,
        finding.message,
        tuple(finding.row_indexes),
    )


def value_rows(rows: list[dict[str, str]], column_name: str, target_value: str) -> list[int]:
    indexes: list[int] = []
    for row_index, row in enumerate(rows, start=1):
        value = (row.get(column_name) or "").strip()
        if value == target_value:
            indexes.append(row_index)
    return indexes
