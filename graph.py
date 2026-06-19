from __future__ import annotations

from langgraph.graph import END, START, StateGraph

try:
    from .agents import build_agents, should_run_rag
    from .core.pipeline import profile_values, propose_repairs, validate_quality, verify_results
    from .core.schema.models import PipelineState
except ImportError:  # pragma: no cover
    from agents import build_agents, should_run_rag
    from core.pipeline import profile_values, propose_repairs, validate_quality, verify_results
    from core.schema.models import PipelineState


def build_graph(
    llm_model: str | None = None,
    llm_fast_model: str | None = None,
    llm_strong_model: str | None = None,
):
    agents = build_agents(
        llm_model=llm_model,
        llm_fast_model=llm_fast_model,
        llm_strong_model=llm_strong_model,
    )
    graph = StateGraph(PipelineState)
    graph.add_node("load_reference_data", agents["reference_loader"].run)
    graph.add_node("normalize_columns", agents["schema_parser"].run)
    graph.add_node("profile_values", profile_values)
    graph.add_node("route_rules", agents["rule_router"].run)
    graph.add_node("rag_resolve", agents["rag_resolver"].run)
    graph.add_node("semantic_profile", agents["semantic_profiler"].run)
    graph.add_node("validate", validate_quality)
    graph.add_node("categorical_semantic_validate", agents["categorical_semantic_validator"].run)
    graph.add_node("propose_repairs", propose_repairs)
    graph.add_node("verify_results", verify_results)

    graph.add_edge(START, "load_reference_data")
    graph.add_edge("load_reference_data", "normalize_columns")
    graph.add_edge("normalize_columns", "profile_values")
    graph.add_edge("profile_values", "route_rules")
    graph.add_conditional_edges(
        "route_rules",
        should_run_rag,
        {"rag_resolve": "rag_resolve", "semantic_profile": "semantic_profile"},
    )
    graph.add_edge("rag_resolve", "semantic_profile")
    graph.add_edge("semantic_profile", "validate")
    graph.add_edge("validate", "categorical_semantic_validate")
    graph.add_edge("categorical_semantic_validate", "propose_repairs")
    graph.add_edge("propose_repairs", "verify_results")
    graph.add_edge("verify_results", END)
    return graph.compile()
