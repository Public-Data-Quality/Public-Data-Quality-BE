from __future__ import annotations

import re

from ...schema.models import ColumnProfile


REGION_VALUE_RE = re.compile(
    r"^(?:"
    r"서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|"
    r"세종특별자치시|제주특별자치도|"
    r"[가-힣]+도|[가-힣]+특별시|[가-힣]+광역시|[가-힣]+특별자치도|[가-힣]+특별자치시"
    r")$"
)

REGION_PREFIX_RE = re.compile(
    r"^(서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|"
    r"세종특별자치시|제주특별자치도|"
    r"[가-힣]+도|[가-힣]+특별시|[가-힣]+광역시|[가-힣]+특별자치도|[가-힣]+특별자치시)(?:\s|$)"
)


def looks_region_column(column: ColumnProfile, rows: list[dict[str, str]]) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    if any(token in name for token in ("시도", "광역", "지역", "도명", "소속센터")):
        return True

    non_empty = 0
    region_like = 0
    for row in rows[:200]:
        value = (row.get(column.raw_name) or "").strip()
        if not value:
            continue
        non_empty += 1
        if REGION_VALUE_RE.fullmatch(value):
            region_like += 1
    return non_empty > 0 and region_like / non_empty >= 0.6


def looks_address_column(column: ColumnProfile) -> bool:
    name = f"{column.raw_name} {column.normalized_name}"
    return ("address" in column.semantic_tags or "주소" in name or "소재지" in name) and "상세" not in name


def address_region_prefix(address_value: str) -> str:
    match = REGION_PREFIX_RE.match(address_value)
    return match.group(1) if match else ""
