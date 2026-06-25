from __future__ import annotations

from typing import Callable

try:
    from .....core.schema.models import AgentTrace, PipelineState, ValidationFinding
except ImportError:  # pragma: no cover
    from core.schema.models import AgentTrace, PipelineState, ValidationFinding
from ..value_validator import LLMCategoricalValueValidator
from .row_context_results import append_row_context_findings
from .row_selection import (
    context_columns,
    context_rows,
    looks_row_context_signal_column,
    row_context_signal_score,
)

TraceFactory = Callable[[str, str | None, str], AgentTrace]
DebugDetail = Callable[[], tuple[str, str]]

__all__ = [
    "context_columns",
    "context_rows",
    "looks_row_context_signal_column",
    "row_context_signal_score",
    "run_llm_row_context_validation",
]


def run_llm_row_context_validation(
    *,
    state: PipelineState,
    findings: list[ValidationFinding],
    traces: list[AgentTrace],
    validator: LLMCategoricalValueValidator | None,
    trace: TraceFactory,
    debug_detail: DebugDetail,
) -> tuple[list[ValidationFinding], list[AgentTrace]]:
    if validator is None or not validator.enabled:
        return findings, traces

    rows = state.get("preview_rows", [])
    if not rows:
        return findings, traces

    selected_context_columns = context_columns(state["columns"])
    if not selected_context_columns:
        return findings, traces

    context_headers = [column["raw_name"] for column in selected_context_columns]
    selected_context_rows = context_rows(rows, context_headers)
    dataset_meta = state["dataset_meta"]
    result = validator.validate_row_context(
        dataset_name=dataset_meta.dataset_name,
        provider_name=dataset_meta.provider_name,
        columns=selected_context_columns,
        rows=selected_context_rows,
    )
    if not result:
        llm_error, llm_preview = debug_detail()
        traces.append(
            trace(
                "row_context_validate",
                dataset_meta.dataset_id,
                f"llm_no_result,error={llm_error},preview={llm_preview}",
            )
        )
        return findings, traces

    generated, manual_generated = append_row_context_findings(
        result=result,
        rows=rows,
        columns=selected_context_columns,
        findings=findings,
    )

    traces.append(
        trace(
            "row_context_validate",
            dataset_meta.dataset_id,
            (
                f"rows={len(selected_context_rows)}, findings={generated}, "
                f"manual_reviews={manual_generated}, "
                f"overall_confidence={float(result.get('overall_confidence') or 0.0):.2f}, "
                f"model={result.get('_llm_model', '')}, "
                f"stage={result.get('_llm_stage', '')}, "
                f"escalated={bool(result.get('_llm_escalated'))}"
            ),
        )
    )
    return findings, traces
