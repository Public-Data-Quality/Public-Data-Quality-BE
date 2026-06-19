from __future__ import annotations

from ..schema.models import AgentTrace


def pipeline_trace(
    step_name: str,
    action: str,
    target: str | None = None,
    detail: str = "",
) -> AgentTrace:
    return AgentTrace(agent_name=step_name, action=action, target=target, detail=detail)
