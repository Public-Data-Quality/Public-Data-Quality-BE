from __future__ import annotations

import json
import os
import re
from typing import Any

from ..core.config.constants import (
    LLM_DEFAULT_MODEL,
    LLM_LOW_CONFIDENCE_THRESHOLD,
    LLM_RESOLUTION_CONFIDENCE,
    LLM_SEMANTIC_PROFILE_CONFIDENCE_DEFAULT,
    LLM_STANDARD_TERM_SAMPLE_SIZE,
    ROUTING_EXACT_MATCH_CONFIDENCE,
    ROUTING_PARTIAL_MATCH_CONFIDENCE,
    ROUTING_PARTIAL_MAX_CANDIDATES,
    ROUTING_PARTIAL_MIN_LENGTH,
    ROUTING_RULE_ONLY_THRESHOLD,
    ROUTING_SYNONYM_MATCH_CONFIDENCE,
)
from ..core.schema.models import ColumnProfile, PipelineState
from ..core.schema.retrieval import resolve_with_rag
from ..core.validation import semantic_profile_llm_reasons
from .base import BaseAgent

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None


class RuleRoutingAgent(BaseAgent):
    name = "rule_router"

    @staticmethod
    def _partial_standard_candidates(name: str, standard_terms: dict[str, object], synonym_index: dict[str, str]) -> list[str]:
        candidates: list[str] = []
        lookup_values = list(standard_terms.keys()) + list(synonym_index.keys())
        for candidate in lookup_values:
            if len(candidate) < ROUTING_PARTIAL_MIN_LENGTH:
                continue
            if name == candidate or candidate in name or name in candidate:
                canonical = synonym_index.get(candidate, candidate)
                if canonical not in candidates:
                    candidates.append(canonical)
            if len(candidates) >= ROUTING_PARTIAL_MAX_CANDIDATES:
                break
        return candidates

    def run(self, state: PipelineState) -> PipelineState:
        standard_terms = state["standard_terms"]
        synonym_index = state["synonym_index"]
        traces = list(state.get("agent_traces", []))
        updated = []
        rag_count = 0

        for column in state["columns"]:
            if column.normalized_name in standard_terms:
                column.standard_candidates = [column.normalized_name]
                column.standard_match_type = "exact"
                column.routing_confidence = max(column.routing_confidence, ROUTING_EXACT_MATCH_CONFIDENCE)
                column.rag_required = False
            elif column.normalized_name in synonym_index:
                column.standard_candidates = [synonym_index[column.normalized_name]]
                column.standard_match_type = "synonym"
                column.routing_confidence = max(column.routing_confidence, ROUTING_SYNONYM_MATCH_CONFIDENCE)
                column.rag_required = False
            else:
                partial_candidates = self._partial_standard_candidates(
                    column.normalized_name,
                    standard_terms=standard_terms,
                    synonym_index=synonym_index,
                )
                if partial_candidates:
                    column.standard_candidates = partial_candidates
                    column.standard_match_type = "partial"
                    column.routing_confidence = max(column.routing_confidence, ROUTING_PARTIAL_MATCH_CONFIDENCE)
                    column.rag_required = False
                elif column.assigned_rules and column.routing_confidence >= ROUTING_RULE_ONLY_THRESHOLD:
                    column.standard_match_type = "rule_only"
                    column.rag_required = False
                else:
                    column.standard_match_type = "unmatched"
                    column.rag_required = True
                    rag_count += 1

            traces.append(
                self.trace(
                    action="route_rules",
                    target=column.raw_name,
                    detail=(
                        f"rules={column.assigned_rules}, confidence={column.routing_confidence:.2f}, "
                        f"rag={column.rag_required}, match_type={column.standard_match_type}, candidates={column.standard_candidates}"
                    ),
                )
            )
            updated.append(column)

        traces.append(self.trace(action="routing_summary", detail=f"rag_required={rag_count}"))
        return {"columns": updated, "agent_traces": traces}


class LLMColumnResolver:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or LLM_DEFAULT_MODEL
        self._llm = None

    @property
    def enabled(self) -> bool:
        return bool(ChatOpenAI and os.getenv("OPENAI_API_KEY"))

    def _client(self):
        if not self.enabled:
            return None
        if self._llm is None:
            self._llm = ChatOpenAI(model=self.model_name, temperature=0)
        return self._llm

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any] | None:
        try:
            return json.loads(content)
        except Exception:
            pass

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def resolve(self, state: PipelineState, column: ColumnProfile) -> dict[str, Any] | None:
        llm = self._client()
        if llm is None:
            return None

        dataset_meta = state["dataset_meta"]
        standard_terms = list(state["standard_terms"].keys())[:LLM_STANDARD_TERM_SAMPLE_SIZE]
        prompt = f"""
You are a public-data schema routing agent.
Return strict JSON with keys:
- normalized_name: string
- semantic_tags: list[string]
- assigned_rules: list[string]
- standard_candidates: list[string]
- reason: string

Dataset:
- name: {dataset_meta.dataset_name}
- provider: {dataset_meta.provider_name}
- keywords: {", ".join(dataset_meta.keywords)}

Column:
- raw_name: {column.raw_name}
- source: {column.source}

Known standard terms sample:
{json.dumps(standard_terms, ensure_ascii=False)}
"""
        response = llm.invoke(prompt)
        return self._parse_json_content(response.content)


class LLMSemanticProfiler:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or LLM_DEFAULT_MODEL
        self._llm = None

    @property
    def enabled(self) -> bool:
        return bool(ChatOpenAI and os.getenv("OPENAI_API_KEY"))

    def _client(self):
        if not self.enabled:
            return None
        if self._llm is None:
            self._llm = ChatOpenAI(model=self.model_name, temperature=0)
        return self._llm

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any] | None:
        try:
            return json.loads(content)
        except Exception:
            pass

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def profile(self, state: PipelineState, column: ColumnProfile) -> dict[str, Any] | None:
        llm = self._client()
        if llm is None:
            return None

        dataset_meta = state["dataset_meta"]
        prompt = f"""
You are a semantic profiling agent for Korean public datasets.
Return strict JSON with keys:
- label: short semantic role name in Korean
- description: one sentence in Korean about the business meaning of this column
- confidence: float between 0 and 1

Rules:
- label must be written in Korean only
- description must be written in Korean only
- do not use English unless the original column itself is an English acronym that must remain unchanged
- prefer concise public-data terminology
- output JSON only

Dataset:
- name: {dataset_meta.dataset_name}
- provider: {dataset_meta.provider_name}
- format: {dataset_meta.data_format}

Column:
- raw_name: {column.raw_name}
- normalized_name: {column.normalized_name}
- semantic_tags: {column.semantic_tags}
- standard_candidates: {column.standard_candidates}
- inferred_type: {column.inferred_primitive_type}
- sample_values: {column.sample_values}
- top_values: {column.top_values[:3]}
"""
        response = llm.invoke(prompt)
        payload = self._parse_json_content(response.content)
        if payload is None:
            return None
        confidence = payload.get("confidence")
        if confidence is None:
            payload["confidence"] = LLM_SEMANTIC_PROFILE_CONFIDENCE_DEFAULT
        if payload.get("label"):
            payload["label"] = str(payload["label"]).strip()
        if payload.get("description"):
            payload["description"] = str(payload["description"]).strip()
        return payload


class RAGResolverAgent(BaseAgent):
    name = "rag_resolver"

    def __init__(self, llm_resolver: LLMColumnResolver | None = None):
        self.llm_resolver = llm_resolver

    def run(self, state: PipelineState) -> PipelineState:
        traces = list(state.get("agent_traces", []))
        resolved = []
        use_llm = bool(state.get("use_llm_agents")) and self.llm_resolver is not None

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
            if column.standard_candidates and column.standard_match_type == "unmatched":
                column.standard_match_type = "rag_resolved"

            llm_used = False
            if use_llm and (not column.standard_candidates or column.routing_confidence < LLM_LOW_CONFIDENCE_THRESHOLD):
                decision = self.llm_resolver.resolve(state, column)
                if decision:
                    column.normalized_name = decision.get("normalized_name") or column.normalized_name
                    column.semantic_tags = decision.get("semantic_tags") or column.semantic_tags
                    column.assigned_rules = decision.get("assigned_rules") or column.assigned_rules
                    column.standard_candidates = decision.get("standard_candidates") or column.standard_candidates
                    if column.standard_candidates:
                        column.standard_match_type = "llm_resolved"
                    column.routing_confidence = max(column.routing_confidence, LLM_RESOLUTION_CONFIDENCE)
                    column.rag_required = False
                    column.rag_evidence.append(f"llm_reason:{decision.get('reason', '')}")
                    llm_used = True

            traces.append(
                self.trace(
                    action="resolve_column",
                    target=column.raw_name,
                    detail=f"llm_used={llm_used}, candidates={column.standard_candidates}, evidence={len(column.rag_evidence)}",
                )
            )
            resolved.append(column)

        return {"columns": resolved, "agent_traces": traces}


class SemanticProfilingAgent(BaseAgent):
    name = "semantic_profiler"

    def __init__(self, semantic_profiler: LLMSemanticProfiler | None = None):
        self.semantic_profiler = semantic_profiler

    def run(self, state: PipelineState) -> PipelineState:
        traces = list(state.get("agent_traces", []))
        updated = []
        use_llm = bool(state.get("use_llm_agents")) and self.semantic_profiler is not None

        for column in state["columns"]:
            llm_reasons = semantic_profile_llm_reasons(column)
            needs_llm = bool(llm_reasons)
            column.semantic_profile_llm_needed = needs_llm
            column.semantic_profile_llm_reasons = llm_reasons
            if use_llm and needs_llm:
                profile = self.semantic_profiler.profile(state, column)
                if profile:
                    column.semantic_profile_label = profile.get("label")
                    column.semantic_profile_description = profile.get("description")
                    column.semantic_profile_confidence = profile.get("confidence")
            traces.append(
                self.trace(
                    action="semantic_profile",
                    target=column.raw_name,
                    detail=(
                        f"label={column.semantic_profile_label}, "
                        f"confidence={column.semantic_profile_confidence}, "
                        f"llm_needed={needs_llm}, reasons={llm_reasons}"
                    ),
                )
            )
            updated.append(column)

        return {"columns": updated, "agent_traces": traces}


def should_run_rag(state: PipelineState) -> str:
    if any(column.rag_required for column in state["columns"]):
        return "rag_resolve"
    return "validate"
