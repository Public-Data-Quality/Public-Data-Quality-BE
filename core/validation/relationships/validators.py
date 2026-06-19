from __future__ import annotations

from typing import Any, Callable

from ...schema.models import ColumnProfile, ValidationFinding
from .calculation_rules import validate_calculation_relationships
from .logical_rules import validate_logical_relationships
from .reference_rules import validate_reference_relationships
from .region_address import validate_region_address_relationships
from .time_rules import validate_time_relationships

RelationshipValidator = Callable[
    [list[ColumnProfile], list[dict[str, str]], list[dict[str, Any]] | None],
    list[ValidationFinding],
]

RELATIONSHIP_VALIDATORS: tuple[RelationshipValidator, ...] = (
    validate_time_relationships,
    validate_logical_relationships,
    validate_calculation_relationships,
    validate_reference_relationships,
    validate_region_address_relationships,
)


def validate_dataset_relationships(
    columns: list[ColumnProfile],
    rows: list[dict[str, str]],
    relationship_candidates: list[dict[str, Any]] | None = None,
) -> list[ValidationFinding]:
    if not rows:
        return []

    findings: list[ValidationFinding] = []
    for validator in RELATIONSHIP_VALIDATORS:
        findings.extend(validator(columns, rows, relationship_candidates))
    return findings
