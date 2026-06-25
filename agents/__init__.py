from .base import BaseAgent
from .ingestion import ReferenceLoaderAgent, SchemaParsingAgent
from .resolution import (
    LLMColumnResolver,
    LLMRoutingAgent,
    LLMSemanticProfiler,
    RAGResolverAgent,
    SemanticProfilingAgent,
    should_run_rag,
)
from .validation.categorical import CategoricalSemanticValidationAgent, LLMCategoricalValueValidator


def build_agents(
    llm_model: str | None = None,
    llm_fast_model: str | None = None,
    llm_strong_model: str | None = None,
) -> dict[str, BaseAgent]:
    column_resolver = LLMColumnResolver(
        model_name=llm_model,
        fast_model_name=llm_fast_model,
        strong_model_name=llm_strong_model,
    )
    semantic_profiler = LLMSemanticProfiler(
        model_name=llm_model,
        fast_model_name=llm_fast_model,
        strong_model_name=llm_strong_model,
    )
    categorical_validator = LLMCategoricalValueValidator(
        model_name=llm_model,
        fast_model_name=llm_fast_model,
        strong_model_name=llm_strong_model,
    )
    return {
        "reference_loader": ReferenceLoaderAgent(),
        "schema_parser": SchemaParsingAgent(),
        "rule_router": LLMRoutingAgent(column_resolver=column_resolver),
        "rag_resolver": RAGResolverAgent(),
        "semantic_profiler": SemanticProfilingAgent(semantic_profiler=semantic_profiler),
        "categorical_semantic_validator": CategoricalSemanticValidationAgent(validator=categorical_validator),
    }


__all__ = ["build_agents", "should_run_rag"]
