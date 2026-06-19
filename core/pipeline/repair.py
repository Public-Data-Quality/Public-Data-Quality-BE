from __future__ import annotations

from ..schema.models import PipelineState
from ..validation.rules import build_repair_suggestion
from .tracing import pipeline_trace

REPAIR_STEP_NAME = "repair_planner"


def propose_repairs(state: PipelineState) -> PipelineState:
    traces = list(state.get("agent_traces", []))
    updated = []
    for column in state["columns"]:
        column.repair_suggestion = build_repair_suggestion(column)
        traces.append(
            pipeline_trace(
                REPAIR_STEP_NAME,
                action="propose_repair",
                target=column.raw_name,
                detail=column.repair_suggestion or "none",
            )
        )
        updated.append(column)
    return {"columns": updated, "agent_traces": traces}
