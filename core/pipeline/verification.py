from __future__ import annotations

import json
from collections import Counter

from ..schema.models import PipelineState
from .tracing import pipeline_trace

VERIFICATION_STEP_NAME = "verifier"


def verify_results(state: PipelineState) -> PipelineState:
    coverage = 0
    repairs = 0
    manual_review = 0
    traces = list(state.get("agent_traces", []))
    match_breakdown = Counter()

    for column in state["columns"]:
        if column.standard_candidates:
            coverage += 1
            column.verification_notes.append("표준용어 후보 존재")
        match_breakdown[column.standard_match_type or "unmatched"] += 1
        if column.repair_suggestion:
            repairs += 1
            column.verification_notes.append("수정 제안 생성")
        if not column.assigned_rules:
            manual_review += 1
            column.verification_notes.append("규칙 미할당")

    summary = _build_quality_summary(
        state,
        coverage=coverage,
        repairs=repairs,
        manual_review=manual_review,
        match_breakdown=match_breakdown,
    )
    traces.append(
        pipeline_trace(
            VERIFICATION_STEP_NAME,
            action="verify_results",
            target=state["dataset_meta"].dataset_id,
            detail=json.dumps(summary, ensure_ascii=False),
        )
    )
    return {"summary": summary, "columns": state["columns"], "agent_traces": traces}


def _build_quality_summary(
    state: PipelineState,
    *,
    coverage: int,
    repairs: int,
    manual_review: int,
    match_breakdown: Counter,
) -> dict:
    findings = state["findings"]
    return {
        "dataset_id": state["dataset_meta"].dataset_id,
        "dataset_name": state["dataset_meta"].dataset_name,
        "provider_name": state["dataset_meta"].provider_name,
        "column_count": len(state["columns"]),
        "row_count": state["dataset_meta"].total_rows,
        "standard_term_coverage": round(coverage / max(1, len(state["columns"])), 4),
        "standard_term_coverage_breakdown": dict(match_breakdown),
        "repair_suggestion_count": repairs,
        "manual_review_count": manual_review,
        "finding_count": len(findings),
        "manual_review_finding_count": sum(
            1 for finding in findings if finding.finding_type == "manual_review"
        ),
        "issue_finding_count": sum(1 for finding in findings if finding.finding_type == "issue"),
        "finding_breakdown": dict(Counter(finding.category_label for finding in findings)),
        "finding_type_breakdown": dict(Counter(finding.display_label for finding in findings)),
        "llm_agents_enabled": bool(state.get("use_llm_agents")),
    }
