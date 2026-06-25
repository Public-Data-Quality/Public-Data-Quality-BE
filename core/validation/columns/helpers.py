from __future__ import annotations

import re
from collections.abc import Callable

from ...schema.models import ColumnProfile
from .context import Rows

RowPredicate = Callable[[str], bool]


def matching_row_indexes(
    rows: Rows,
    column_name: str,
    predicate: RowPredicate,
    *,
    strip_value: bool = True,
) -> list[int]:
    indexes: list[int] = []
    for row_index, row in enumerate(rows, start=1):
        value = row.get(column_name) or ""
        if strip_value:
            value = value.strip()
        if predicate(value):
            indexes.append(row_index)
    return indexes


def duplicate_value_row_indexes(rows: Rows, column_name: str) -> list[int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = (row.get(column_name) or "").strip()
        if value:
            counts[value] = counts.get(value, 0) + 1

    indexes: list[int] = []
    for row_index, row in enumerate(rows, start=1):
        value = (row.get(column_name) or "").strip()
        if value and counts.get(value, 0) > 1:
            indexes.append(row_index)
    return indexes


def is_likely_required(column: ColumnProfile) -> bool:
    required_tags = {"identifier", "name", "date", "address", "phone"}
    return bool(required_tags.intersection(column.semantic_tags)) or any(
        token in column.normalized_name for token in ("명", "이름", "번호", "일자", "주소", "전화")
    )


def looks_numeric_column(column: ColumnProfile) -> bool:
    numeric_tags = {"numeric", "count", "quantity", "amount", "rate", "width"}
    return bool(numeric_tags.intersection(column.semantic_tags)) or column.inferred_primitive_type == "numeric"


def looks_detail_address_column(column: ColumnProfile) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    return "상세주소" in name or ("상세" in name and "주소" in name)


def looks_address_column(column: ColumnProfile) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    return "address" in column.semantic_tags or "주소" in name or "소재지" in name


def address_context_columns(rows: Rows, detail_column_name: str) -> list[str]:
    if not rows:
        return []
    columns = list(rows[0].keys())
    return [
        name
        for name in columns
        if name != detail_column_name and "주소" in name and "상세" not in name
    ]


def looks_incomplete_detail_address(value: str) -> bool:
    text = re.sub(r"\s+", "", value or "")
    if not text:
        return False
    if len(text) <= 1 and re.search(r"[가-힣]", text):
        return True
    if len(text) <= 2 and re.fullmatch(r"[가-힣]+", text):
        return True
    return False


def incomplete_detail_address_row_indexes(
    rows: Rows,
    detail_column_name: str,
    address_columns: list[str],
) -> list[int]:
    indexes: list[int] = []
    for row_index, row in enumerate(rows, start=1):
        value = (row.get(detail_column_name) or "").strip()
        if not looks_incomplete_detail_address(value):
            continue
        if address_columns and not any((row.get(name) or "").strip() for name in address_columns):
            continue
        indexes.append(row_index)
    return indexes


def looks_truncated_address_value(value: str) -> bool:
    text = re.sub(r"\s+", " ", value or "").strip()
    if not text:
        return False
    if text.count("(") > text.count(")"):
        return True
    if text.count("[") > text.count("]"):
        return True
    if text.count("{") > text.count("}"):
        return True
    if re.search(r"\([^)]+$", text):
        return True
    if (
        re.search(r"(요양|노인요|어린이|초등|중학|고등|복지|센터|기관|시설)$", text)
        and "(" in text
    ):
        return True
    return False


def truncated_address_row_indexes(rows: Rows, column_name: str) -> list[int]:
    return matching_row_indexes(rows, column_name, looks_truncated_address_value)


def build_repair_suggestion(column: ColumnProfile) -> str | None:
    if column.raw_name == column.normalized_name and not column.standard_candidates:
        return None

    parts: list[str] = []
    if column.raw_name != column.normalized_name:
        parts.append(f"컬럼명을 '{column.normalized_name}'로 정규화")
    if column.unit:
        parts.append(f"단위 '{column.unit}'를 별도 메타데이터로 분리")
    if column.standard_candidates:
        parts.append(f"표준용어 후보 '{column.standard_candidates[0]}'에 매핑")
    return ", ".join(parts) if parts else None
