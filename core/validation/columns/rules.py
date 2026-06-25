from __future__ import annotations

from ...schema.models import ColumnProfile, DatasetMeta, StandardTerm, ValidationFinding
from .completeness import (
    find_duplicate_identifiers,
    find_garbled_text,
    find_incomplete_detail_address,
    find_missing_assigned_rules,
    find_missing_standard_term,
    find_required_nulls,
    find_special_character_issues,
    find_truncated_address,
    find_whitespace_issues,
)
from .domain import (
    find_invalid_booleans,
    find_invalid_codes,
    find_invalid_dates,
    find_invalid_phone_numbers,
)
from .numeric import (
    find_amount_domain_issues,
    find_latitude_domain_issues,
    find_longitude_domain_issues,
    find_quantity_domain_issues,
    find_rate_domain_issues,
)
from .context import ColumnRuleCheck, build_column_rule_context, collect_findings
from .helpers import build_repair_suggestion, is_likely_required, looks_numeric_column

BASE_RULE_CHECKS: tuple[ColumnRuleCheck, ...] = (
    find_garbled_text,
    find_whitespace_issues,
    find_special_character_issues,
    find_incomplete_detail_address,
    find_truncated_address,
)

ASSIGNED_RULE_CHECKS: tuple[ColumnRuleCheck, ...] = (
    find_required_nulls,
    find_duplicate_identifiers,
    find_invalid_dates,
    find_invalid_phone_numbers,
    find_invalid_booleans,
    find_invalid_codes,
    find_amount_domain_issues,
    find_quantity_domain_issues,
    find_rate_domain_issues,
    find_latitude_domain_issues,
    find_longitude_domain_issues,
    find_missing_standard_term,
)

__all__ = [
    "build_repair_suggestion",
    "is_likely_required",
    "looks_numeric_column",
    "validate_column",
]


def validate_column(
    column: ColumnProfile,
    dataset_meta: DatasetMeta,
    standard_terms: dict[str, StandardTerm],
    rows: list[dict[str, str]] | None = None,
) -> list[ValidationFinding]:
    del dataset_meta
    context = build_column_rule_context(column, standard_terms, rows or [])
    findings = collect_findings(BASE_RULE_CHECKS, context)

    missing_rule_findings = find_missing_assigned_rules(context)
    if missing_rule_findings:
        findings.extend(missing_rule_findings)
        return findings

    findings.extend(collect_findings(ASSIGNED_RULE_CHECKS, context))
    return findings
