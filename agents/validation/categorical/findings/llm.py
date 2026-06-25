from __future__ import annotations

try:
    from .....core.config.constants import CATEGORICAL_LLM_CONFIDENCE_THRESHOLD
    from .....core.validation.helpers import build_finding
except ImportError:  # pragma: no cover
    from core.config.constants import CATEGORICAL_LLM_CONFIDENCE_THRESHOLD
    from core.validation.helpers import build_finding
from ..checks.column import (
    is_public_private_category_value,
    is_yn_value,
    looks_boolean_column,
    looks_date_column,
    looks_date_value,
    looks_institution_category_column,
)
from ..checks.normalization import is_llm_normalization_actionable
from ..checks.text import clean_reason_text, is_specific_out_of_domain_reason
from .utils import value_rows


def apply_llm_categorical_findings(
    *,
    column,
    rows: list[dict[str, str]],
    result: dict,
    findings: list,
) -> int:
    generated = 0
    generated += _append_normalization_findings(column=column, rows=rows, result=result, findings=findings)
    generated += _append_invalid_format_findings(column=column, rows=rows, result=result, findings=findings)
    generated += _append_inconsistent_format_findings(column=column, rows=rows, result=result, findings=findings)
    generated += _append_out_of_domain_findings(column=column, rows=rows, result=result, findings=findings)
    generated += _append_manual_review_findings(column=column, rows=rows, result=result, findings=findings)
    return generated


def _append_normalization_findings(*, column, rows: list[dict[str, str]], result: dict, findings: list) -> int:
    generated = 0
    for item in result.get("normalizations", []):
        confidence = float(item.get("confidence") or 0.0)
        if confidence < CATEGORICAL_LLM_CONFIDENCE_THRESHOLD:
            continue
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        reason = clean_reason_text(item.get("reason"))
        if not is_llm_normalization_actionable(column, source, target, reason):
            continue

        evidence = _llm_evidence(result, confidence, reason)
        if looks_boolean_column(column) and not is_yn_value(source) and is_yn_value(target):
            findings.append(
                build_finding(
                    column_name=column.raw_name,
                    severity="warning",
                    category_group="domain_validity",
                    criterion_name="boolean_domain",
                    rule_id="boolean_domain",
                    message=f"'{source}' 값은 Y/N 여부 컬럼의 허용값과 맞지 않을 수 있습니다.",
                    row_indexes=value_rows(rows, column.raw_name, source),
                    related_columns=[column.raw_name],
                    evidence=evidence,
                )
            )
            generated += 1
            continue

        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group="domain_validity",
                criterion_name="categorical_semantic_domain",
                rule_id="categorical_value_normalization",
                message=f"'{source}' 값은 '{target}'로 표준화하는 것이 적절합니다.",
                row_indexes=value_rows(rows, column.raw_name, source),
                related_columns=[column.raw_name],
                evidence=evidence,
            )
        )
        generated += 1
    return generated


def _append_invalid_format_findings(*, column, rows: list[dict[str, str]], result: dict, findings: list) -> int:
    generated = 0
    for item in result.get("invalid_format_values", []):
        confidence = float(item.get("confidence") or 0.0)
        if confidence < CATEGORICAL_LLM_CONFIDENCE_THRESHOLD:
            continue
        value = str(item.get("value") or "").strip()
        issue_type = str(item.get("issue_type") or "").strip()
        reason = clean_reason_text(item.get("reason"))
        if not value:
            continue
        rule_id = _invalid_format_rule_id(issue_type)
        criterion_name = _invalid_format_criterion_name(issue_type)
        category_group = "completeness" if issue_type == "malformed_text" else "domain_validity"
        evidence = _llm_evidence(result, confidence, reason, f"issue_type:{issue_type}")
        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group=category_group,
                criterion_name=criterion_name,
                rule_id=rule_id,
                message=_invalid_format_message(value, issue_type),
                row_indexes=value_rows(rows, column.raw_name, value),
                related_columns=[column.raw_name],
                evidence=evidence,
            )
        )
        generated += 1
    return generated


def _append_inconsistent_format_findings(
    *,
    column,
    rows: list[dict[str, str]],
    result: dict,
    findings: list,
) -> int:
    generated = 0
    for item in result.get("inconsistent_format_groups", []):
        confidence = float(item.get("confidence") or 0.0)
        if confidence < CATEGORICAL_LLM_CONFIDENCE_THRESHOLD:
            continue
        values_in_group = [str(value).strip() for value in item.get("values", []) if str(value).strip()]
        target_format = str(item.get("target_format") or "").strip()
        reason = clean_reason_text(item.get("reason"))
        if not values_in_group:
            continue
        if not looks_date_column(column) and not all(looks_date_value(value) for value in values_in_group):
            continue

        evidence = _llm_evidence(result, confidence, "")
        if target_format:
            evidence.append(f"target_format:{target_format}")
        if reason:
            evidence.append(f"reason:{reason}")
        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group="domain_validity",
                criterion_name="date_domain",
                rule_id="date_format_inconsistent",
                message=(
                    f"날짜 또는 형식 컬럼에서 표기 형식이 혼용됩니다: "
                    f"{', '.join(values_in_group)}"
                ),
                row_indexes=[
                    row_index
                    for value in values_in_group
                    for row_index in value_rows(rows, column.raw_name, value)
                ],
                related_columns=[column.raw_name],
                evidence=evidence,
            )
        )
        generated += 1
    return generated


def _append_out_of_domain_findings(*, column, rows: list[dict[str, str]], result: dict, findings: list) -> int:
    generated = 0
    for item in result.get("out_of_domain_values", []):
        confidence = float(item.get("confidence") or 0.0)
        if confidence < CATEGORICAL_LLM_CONFIDENCE_THRESHOLD:
            continue
        value = str(item.get("value") or "").strip()
        reason = clean_reason_text(item.get("reason"))
        if looks_institution_category_column(column) and is_public_private_category_value(value):
            continue
        if not value or not is_specific_out_of_domain_reason(reason):
            continue
        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="warning",
                category_group="domain_validity",
                criterion_name="categorical_semantic_domain",
                rule_id="categorical_value_out_of_domain",
                message=f"'{value}' 값은 해당 컬럼의 의미 도메인과 맞지 않을 수 있습니다.",
                row_indexes=value_rows(rows, column.raw_name, value),
                related_columns=[column.raw_name],
                evidence=_llm_evidence(result, confidence, reason),
            )
        )
        generated += 1
    return generated


def _append_manual_review_findings(*, column, rows: list[dict[str, str]], result: dict, findings: list) -> int:
    generated = 0
    for item in result.get("needs_manual_review", []):
        confidence = float(item.get("confidence") or 0.0)
        value = str(item.get("value") or "").strip()
        reason = clean_reason_text(item.get("reason"))
        if not value:
            continue
        row_indexes = value_rows(rows, column.raw_name, value)
        if _has_existing_value_finding(
            findings,
            column_name=column.raw_name,
            row_indexes=row_indexes,
        ):
            continue
        findings.append(
            build_finding(
                column_name=column.raw_name,
                severity="info",
                category_group="domain_validity",
                criterion_name="categorical_semantic_domain",
                rule_id="categorical_value_manual_review",
                message=f"'{value}' 값은 의미 판정이 애매해 수동 검토가 필요합니다.",
                row_indexes=row_indexes,
                related_columns=[column.raw_name],
                evidence=_llm_evidence(result, confidence, reason),
            )
        )
        generated += 1
    return generated


def _has_existing_value_finding(findings: list, *, column_name: str, row_indexes: list[int]) -> bool:
    row_index_set = set(row_indexes)
    for finding in findings:
        if finding.column_name != column_name:
            continue
        if set(finding.row_indexes) != row_index_set:
            continue
        if finding.finding_type == "issue" or finding.rule_id == "categorical_value_manual_review":
            return True
    return False


def _llm_evidence(result: dict, confidence: float, reason: str, *extra: str) -> list[str]:
    evidence = [f"domain:{result.get('domain_label', '')}", f"confidence:{confidence:.2f}", *extra]
    if reason:
        evidence.append(f"reason:{reason}")
    return evidence


def _invalid_format_rule_id(issue_type: str) -> str:
    if issue_type == "boolean_invalid":
        return "boolean_domain"
    if issue_type == "date_invalid":
        return "date_domain"
    if issue_type == "malformed_text":
        return "garbled_text"
    if issue_type == "truncated_text":
        return "categorical_value_truncated"
    return "categorical_value_out_of_domain"


def _invalid_format_criterion_name(issue_type: str) -> str:
    if issue_type == "boolean_invalid":
        return "boolean_domain"
    if issue_type == "date_invalid":
        return "date_domain"
    if issue_type == "malformed_text":
        return "garbled_text"
    return "categorical_semantic_domain"


def _invalid_format_message(value: str, issue_type: str) -> str:
    if issue_type == "malformed_text":
        return f"'{value}' 값은 불필요한 기호 또는 깨진 텍스트가 포함된 것으로 보입니다."
    if issue_type == "truncated_text":
        return f"'{value}' 값은 문맥상 입력 중 잘렸거나 불완전한 텍스트일 수 있습니다."
    return f"'{value}' 값은 컬럼의 형식 또는 허용값과 맞지 않을 수 있습니다."
