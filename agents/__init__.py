from .base import BaseAgent
from .categorical_validation import CategoricalSemanticValidationAgent, LLMCategoricalValueValidator
from .ingestion import ReferenceLoaderAgent, SchemaParsingAgent
from .quality import DataProfilingAgent, RepairAgent, ValidationAgent, VerificationAgent
from .resolution import (
    LLMColumnResolver,
    LLMSemanticProfiler,
    RAGResolverAgent,
    RuleRoutingAgent,
    SemanticProfilingAgent,
    should_run_rag,
)


def build_agents(llm_model: str | None = None) -> dict[str, BaseAgent]:
    llm_resolver = LLMColumnResolver(model_name=llm_model)
    semantic_profiler = LLMSemanticProfiler(model_name=llm_model)
    categorical_validator = LLMCategoricalValueValidator(model_name=llm_model)
    return {
        "reference_loader": ReferenceLoaderAgent(),
        "schema_parser": SchemaParsingAgent(),
        "profiler": DataProfilingAgent(),
        "rule_router": RuleRoutingAgent(),
        "rag_resolver": RAGResolverAgent(llm_resolver=llm_resolver),
        "semantic_profiler": SemanticProfilingAgent(semantic_profiler=semantic_profiler),
        "validator": ValidationAgent(),
        "categorical_semantic_validator": CategoricalSemanticValidationAgent(validator=categorical_validator),
        "repair_planner": RepairAgent(),
        "verifier": VerificationAgent(),
    }


__all__ = ["build_agents", "should_run_rag"]
