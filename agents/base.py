from __future__ import annotations

try:
    from ..core.schema.models import AgentTrace
except ImportError:  # pragma: no cover
    from core.schema.models import AgentTrace


class BaseAgent:
    name = "base_agent"

    def trace(self, action: str, target: str | None = None, detail: str = "") -> AgentTrace:
        return AgentTrace(agent_name=self.name, action=action, target=target, detail=detail)
