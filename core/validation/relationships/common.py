from __future__ import annotations

from typing import Any

from ...schema.models import ColumnProfile


def base_stem(name: str) -> str:
    stem = name
    for token in ("총", "합계", "전체", "수", "개수", "건수", "금액", "비율", "율"):
        stem = stem.replace(token, "")
    return stem.strip()


def is_related_numeric_pair(total_col: ColumnProfile, candidate: ColumnProfile) -> bool:
    total_stem = base_stem(total_col.normalized_name)
    candidate_stem = base_stem(candidate.normalized_name)
    if total_stem and candidate_stem and total_stem == candidate_stem:
        return True
    if total_stem and candidate_stem and (total_stem in candidate_stem or candidate_stem in total_stem):
        return True
    if total_col.unit and candidate.unit and total_col.unit == candidate.unit:
        return True
    total_candidate = total_col.standard_candidates[0] if total_col.standard_candidates else ""
    candidate_name = candidate.standard_candidates[0] if candidate.standard_candidates else ""
    if total_candidate and candidate_name and total_candidate == candidate_name:
        return True
    return False


def find_matching_columns(
    columns: list[ColumnProfile],
    left_token: str,
    right_token: str,
) -> list[tuple[ColumnProfile, ColumnProfile]]:
    matches: list[tuple[ColumnProfile, ColumnProfile]] = []
    left_candidates = [column for column in columns if left_token in column.normalized_name]
    right_candidates = [column for column in columns if right_token in column.normalized_name]
    for left in left_candidates:
        stem = left.normalized_name.replace(left_token, "")
        for right in right_candidates:
            other_stem = right.normalized_name.replace(right_token, "")
            if stem and stem == other_stem:
                matches.append((left, right))
    return matches


def columns_by_name(columns: list[ColumnProfile]) -> dict[str, ColumnProfile]:
    return {column.raw_name: column for column in columns}


def candidate_groups(
    relationship_candidates: list[dict[str, Any]] | None,
    rule_ids: set[str],
    columns: list[ColumnProfile],
) -> list[list[ColumnProfile]]:
    if not relationship_candidates:
        return []

    by_name = columns_by_name(columns)
    groups: list[list[ColumnProfile]] = []
    for candidate in relationship_candidates:
        if candidate.get("rule_id") not in rule_ids:
            continue
        names = candidate.get("columns") or []
        if not isinstance(names, list):
            continue
        group = [by_name[name] for name in names if isinstance(name, str) and name in by_name]
        if len(group) >= 2:
            groups.append(group)
    return groups


def candidate_pairs(
    relationship_candidates: list[dict[str, Any]] | None,
    rule_ids: set[str],
    columns: list[ColumnProfile],
) -> list[tuple[ColumnProfile, ColumnProfile]]:
    pairs: list[tuple[ColumnProfile, ColumnProfile]] = []
    for group in candidate_groups(relationship_candidates, rule_ids, columns):
        pairs.append((group[0], group[1]))
    return pairs
