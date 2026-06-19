from __future__ import annotations

from collections import Counter
from typing import Any


def context_columns(columns) -> list[dict[str, Any]]:
    useful_tokens = (
        "지역",
        "시도",
        "주소",
        "우편",
        "시설",
        "기관",
        "센터",
        "소속",
        "명",
        "구분",
        "분류",
        "인원",
        "정원",
        "수용",
    )
    selected = []
    for column in columns:
        name = f"{column.raw_name} {column.normalized_name}"
        if any(token in name for token in useful_tokens) or {
            "address",
            "name",
            "enum",
            "quantity",
            "count",
        }.intersection(column.semantic_tags):
            selected.append(
                {
                    "raw_name": column.raw_name,
                    "normalized_name": column.normalized_name,
                    "semantic_tags": column.semantic_tags,
                    "semantic_profile_label": column.semantic_profile_label,
                }
            )
    return selected[:20]


def looks_row_context_signal_column(header: str) -> bool:
    return any(
        token in header
        for token in ("지역", "시도", "광역", "센터", "소속", "구분", "분류", "유형", "관리청")
    )


def row_context_signal_score(header: str, count: int) -> int:
    if count > 2:
        return 0
    if any(token in header for token in ("지역", "시도", "광역")):
        return 100 if count == 1 else 80
    if any(token in header for token in ("센터", "소속")):
        return 90 if count == 1 else 70
    if any(token in header for token in ("구분", "분류", "유형")):
        return 60 if count == 1 else 40
    return 30 if count == 1 else 20


def context_rows(rows: list[dict[str, str]], headers: list[str], limit: int = 80) -> list[dict[str, Any]]:
    value_counts: dict[str, Counter[str]] = {}
    for header in headers:
        if not looks_row_context_signal_column(header):
            continue
        counter = Counter()
        for row in rows:
            value = (row.get(header) or "").strip()
            if value:
                counter[value] += 1
        value_counts[header] = counter

    candidates: dict[int, dict[str, Any]] = {}
    for row_index, row in enumerate(rows, start=1):
        reasons: list[str] = []
        score = 0
        for header, counter in value_counts.items():
            value = (row.get(header) or "").strip()
            if not value:
                continue
            count = counter.get(value, 0)
            score += row_context_signal_score(header, count)
            if count == 1:
                reasons.append(f"{header} has unique value '{value}'")
            elif count == 2:
                reasons.append(f"{header} has rare value '{value}'")
        if reasons:
            candidates[row_index] = {
                "row_index": row_index,
                "candidate_score": score,
                "candidate_reasons": reasons[:4],
                "values": {header: row.get(header, "") for header in headers},
            }

    prioritized_candidates = sorted(
        candidates.values(),
        key=lambda item: (-int(item.get("candidate_score") or 0), int(item.get("row_index") or 0)),
    )
    selected: list[dict[str, Any]] = prioritized_candidates[: max(0, limit - 30)]
    selected_indexes = {item["row_index"] for item in selected}
    for row_index, row in enumerate(rows[:30], start=1):
        if row_index in selected_indexes:
            continue
        selected.append(
            {
                "row_index": row_index,
                "candidate_reasons": ["early sample row"],
                "values": {header: row.get(header, "") for header in headers},
            }
        )
        if len(selected) >= limit:
            break
    return selected[:limit]
