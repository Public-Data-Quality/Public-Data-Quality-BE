from __future__ import annotations

from ..schema.models import ColumnProfile, ValidationFinding
from .column_rules import looks_numeric_column
from .helpers import REFERENCE_PAIR_TOKENS, TIME_ORDER_TOKENS, build_finding, parse_datetime, parse_number


def _base_stem(name: str) -> str:
    stem = name
    for token in ("총", "합계", "전체", "수", "개수", "건수", "금액", "비율", "율"):
        stem = stem.replace(token, "")
    return stem.strip()


def _is_related_numeric_pair(total_col: ColumnProfile, candidate: ColumnProfile) -> bool:
    total_stem = _base_stem(total_col.normalized_name)
    candidate_stem = _base_stem(candidate.normalized_name)
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


def validate_dataset_relationships(
    columns: list[ColumnProfile],
    rows: list[dict[str, str]],
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    if not rows:
        return findings

    findings.extend(_validate_time_relationships(columns, rows))
    findings.extend(_validate_logical_relationships(columns, rows))
    findings.extend(_validate_calculation_relationships(columns, rows))
    findings.extend(_validate_reference_relationships(columns, rows))
    return findings


def _validate_time_relationships(
    columns: list[ColumnProfile],
    rows: list[dict[str, str]],
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for left_token, right_token in TIME_ORDER_TOKENS:
        for left, right in find_matching_columns(columns, left_token, right_token):
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


def _validate_logical_relationships(
    columns: list[ColumnProfile],
    rows: list[dict[str, str]],
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    boolean_columns = [column for column in columns if "boolean" in column.semantic_tags]
    quantity_columns = [column for column in columns if {"count", "quantity"}.intersection(column.semantic_tags)]
    for flag_col in boolean_columns:
        for qty_col in quantity_columns:
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


def _validate_calculation_relationships(
    columns: list[ColumnProfile],
    rows: list[dict[str, str]],
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    total_columns = [column for column in columns if "총" in column.normalized_name and looks_numeric_column(column)]
    part_columns = [column for column in columns if looks_numeric_column(column)]
    for total_col in total_columns:
        siblings = [
            column
            for column in part_columns
            if column.raw_name != total_col.raw_name and _is_related_numeric_pair(total_col, column)
        ]
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


def _validate_reference_relationships(
    columns: list[ColumnProfile],
    rows: list[dict[str, str]],
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for code_token, name_token in REFERENCE_PAIR_TOKENS:
        for code_col, name_col in find_matching_columns(columns, code_token, name_token):
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
