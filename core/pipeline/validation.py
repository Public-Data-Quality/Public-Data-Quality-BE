from __future__ import annotations

from ..schema.models import PipelineState, ValidationFinding
from ..validation.rules import validate_column, validate_dataset_relationships
from .tracing import pipeline_trace

VALIDATION_STEP_NAME = "validator"


def validate_quality(state: PipelineState) -> PipelineState:
    findings: list[ValidationFinding] = []
    traces = list(state.get("agent_traces", []))
    preview_rows = state.get("preview_rows", [])

    for column in state["columns"]:
        column_findings = validate_column(column, state["dataset_meta"], state["standard_terms"], preview_rows)
        findings.extend(column_findings)
        traces.append(
            pipeline_trace(
                VALIDATION_STEP_NAME,
                action="validate_column",
                target=column.raw_name,
                detail=f"findings={len(column_findings)}",
            )
        )

    relationship_findings = validate_dataset_relationships(
        state["columns"],
        state.get("preview_rows", []),
        state.get("relationship_candidates"),
    )
    findings.extend(relationship_findings)
    traces.append(
        pipeline_trace(
            VALIDATION_STEP_NAME,
            action="validate_relationships",
            target=state["dataset_meta"].dataset_id,
            detail=(
                f"findings={len(relationship_findings)}, "
                f"candidates={len(state.get('relationship_candidates') or [])}"
            ),
        )
    )

    return {"columns": state["columns"], "findings": findings, "agent_traces": traces}
