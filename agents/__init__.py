from .base import BaseAgent
from .categorical_validation import CategoricalSemanticValidationAgent, LLMCategoricalValueValidator
from .ingestion import ReferenceLoaderAgent, SchemaParsingAgent
from .quality import DataProfilingAgent, RepairAgent, ValidationAgent, VerificationAgent
from .resolution import (
    LLMRoutingAgent,
    LLMSemanticProfiler,
    RAGResolverAgent,
    SemanticProfilingAgent,
    should_run_rag,
)


def build_agents(llm_model: str | None = None) -> dict[str, BaseAgent]:
    semantic_profiler = LLMSemanticProfiler(model_name=llm_model)
    categorical_validator = LLMCategoricalValueValidator(model_name=llm_model)
    return {
        "reference_loader": ReferenceLoaderAgent(),
        "schema_parser": SchemaParsingAgent(),
        "profiler": DataProfilingAgent(),
        "rule_router": LLMRoutingAgent(),
        "rag_resolver": RAGResolverAgent(),
        "semantic_profiler": SemanticProfilingAgent(semantic_profiler=semantic_profiler),
        "validator": ValidationAgent(),
        "categorical_semantic_validator": CategoricalSemanticValidationAgent(validator=categorical_validator),
        "repair_planner": RepairAgent(),
        "verifier": VerificationAgent(),
    }


__all__ = ["build_agents", "should_run_rag"]
