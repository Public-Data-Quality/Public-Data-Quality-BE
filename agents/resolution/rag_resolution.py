from __future__ import annotations

try:
    from ...core.schema.models import PipelineState
    from ...core.schema.retrieval import resolve_with_rag
except ImportError:  # pragma: no cover
    from core.schema.models import PipelineState
    from core.schema.retrieval import resolve_with_rag
from ..base import BaseAgent


class RAGResolverAgent(BaseAgent):
    name = "rag_resolver"

    def run(self, state: PipelineState) -> PipelineState:
        traces = list(state.get("agent_traces", []))
        resolved = []

        for column in state["columns"]:
            if not column.rag_required:
                resolved.append(column)
                continue

            column = resolve_with_rag(
                column=column,
                dataset_meta=state["dataset_meta"],
                standard_terms=state["standard_terms"],
                synonym_index=state["synonym_index"],
                example_index=state["example_index"],
            )
            if column.standard_candidates and column.standard_match_type in {"unmatched", "rule_only"}:
                column.standard_match_type = "rag_resolved"

            traces.append(
                self.trace(
                    action="resolve_column",
                    target=column.raw_name,
                    detail=(
                        f"candidates={column.standard_candidates}, "
                        f"evidence={len(column.rag_evidence)}"
                    ),
                )
            )
            resolved.append(column)

        return {"columns": resolved, "agent_traces": traces}


def should_run_rag(state: PipelineState) -> str:
    if any(column.rag_required for column in state["columns"]):
        return "rag_resolve"
    return "semantic_profile"
