from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from ..core.config.constants import (
        LLM_DEFAULT_MODEL,
        LLM_FAST_MODEL,
        LLM_RESOLUTION_CONFIDENCE,
        LLM_SEMANTIC_PROFILE_CONFIDENCE_DEFAULT,
        LLM_STRONG_FALLBACK_CONFIDENCE,
        LLM_STRONG_MODEL,
        LLM_STANDARD_TERM_SAMPLE_SIZE,
        TAG_RULE_MAP,
        VALIDATION_CRITERIA,
    )
    from ..core.llm import ChatLLMClient
    from ..core.schema.models import ColumnProfile, PipelineState
    from ..core.schema.retrieval import resolve_with_rag
    from ..core.validation import semantic_profile_llm_reasons
except ImportError:  # pragma: no cover
    from core.config.constants import (
        LLM_DEFAULT_MODEL,
        LLM_FAST_MODEL,
        LLM_RESOLUTION_CONFIDENCE,
        LLM_SEMANTIC_PROFILE_CONFIDENCE_DEFAULT,
        LLM_STRONG_FALLBACK_CONFIDENCE,
        LLM_STRONG_MODEL,
        LLM_STANDARD_TERM_SAMPLE_SIZE,
        TAG_RULE_MAP,
        VALIDATION_CRITERIA,
    )
    from core.llm import ChatLLMClient
    from core.schema.models import ColumnProfile, PipelineState
    from core.schema.retrieval import resolve_with_rag
    from core.validation import semantic_profile_llm_reasons
from .base import BaseAgent


class LLMRoutingAgent(BaseAgent):
    name = "rule_router"
    NON_UNIQUE_NAME_TOKENS = (
        "명",
        "명칭",
        "이름",
        "기관",
        "부서",
        "담당",
        "경찰서",
        "시설",
        "업소",
        "주소",
        "소재지",
    )

    def __init__(self, column_resolver: "LLMColumnResolver | None" = None):
        self.column_resolver = column_resolver

    @staticmethod
    def _rule_tags(column: ColumnProfile) -> list[str]:
        tags = set(column.semantic_tags)
        name = f"{column.raw_name} {column.normalized_name}"
        if any(token in name for token in ("일자", "일시", "날짜", "년월", "등록일", "기준일")):
            tags.add("date")
        if any(token in name for token in ("주소", "소재지")):
            tags.add("address")
        if "위도" in name:
            tags.add("geo_lat")
        if "경도" in name:
            tags.add("geo_lon")
        if any(token in name for token in ("여부", "유무", "YN", "Yn", "yn", "Y/N")):
            tags.add("boolean")
        if any(token in name for token in ("구분", "유형", "종류", "상태", "분류")):
            tags.add("enum")
        if any(token in name for token in ("코드",)):
            tags.add("code")
        if any(token in name for token in ("명", "명칭", "이름", "기관명", "시설명", "경찰서명")):
            tags.add("name")
        if any(token in name for token in ("대수", "개수", "건수", "수량", "좌석수", "정원수")):
            tags.add("quantity")
        if any(token in name for token in ("폭", "너비")):
            tags.add("width")
        if any(token in name for token in ("전화", "연락처", "휴대전화")):
            tags.add("phone")
        return sorted(tags)

    @staticmethod
    def _allowed_rule_ids() -> set[str]:
        return {
            rule_id
            for category in VALIDATION_CRITERIA.values()
            for rule_id in category["criteria"].keys()
        }

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @classmethod
    def _looks_non_unique_name_column(cls, column: ColumnProfile) -> bool:
        name = f"{column.raw_name} {column.normalized_name}"
        return any(token in name for token in cls.NON_UNIQUE_NAME_TOKENS)

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return LLM_RESOLUTION_CONFIDENCE

    def _apply_rule_fallback(self, column: ColumnProfile) -> ColumnProfile:
        rule_ids: list[str] = []
        rule_tags = self._rule_tags(column)
        if rule_tags:
            column.semantic_tags = list(dict.fromkeys(rule_tags))
        for tag in rule_tags:
            rule_ids.extend(TAG_RULE_MAP.get(tag, []))
        column.assigned_rules = list(dict.fromkeys(rule_ids))
        column.standard_match_type = "unmatched"
        column.rag_required = True
        return column

    def _apply_rule_route(self, state: PipelineState, column: ColumnProfile) -> tuple[ColumnProfile, str]:
        column = self._apply_rule_fallback(column)
        if column.normalized_name in state["standard_terms"]:
            column.standard_candidates = [column.normalized_name]
            column.standard_match_type = "exact"
            column.routing_confidence = max(column.routing_confidence, 0.96)
            column.rag_required = False
            column.rag_evidence = [f"rule_exact:{column.normalized_name}"]
            return column, "rule_exact"
        return column, "rule_fallback"

    def _apply_llm_route(self, state: PipelineState, column: ColumnProfile, payload: dict[str, Any]) -> ColumnProfile:
        allowed_tags = set(TAG_RULE_MAP)
        allowed_rules = self._allowed_rule_ids()
        standard_terms = state["standard_terms"]

        normalized_name = payload.get("normalized_name")
        if isinstance(normalized_name, str) and normalized_name.strip():
            column.normalized_name = normalized_name.strip()

        semantic_tags = [
            tag for tag in self._string_list(payload.get("semantic_tags"))
            if tag in allowed_tags
        ]
        if self._looks_non_unique_name_column(column):
            semantic_tags = [tag for tag in semantic_tags if tag != "identifier"]
            if "name" not in semantic_tags:
                semantic_tags.append("name")
        if semantic_tags:
            column.semantic_tags = list(dict.fromkeys(semantic_tags))

        assigned_rules = [
            rule_id for rule_id in self._string_list(payload.get("assigned_rules"))
            if rule_id in allowed_rules
        ]
        if self._looks_non_unique_name_column(column):
            assigned_rules = [
                rule_id
                for rule_id in assigned_rules
                if rule_id not in {"duplicate_data", "number_domain"}
            ]
        if not assigned_rules and column.semantic_tags:
            for tag in column.semantic_tags:
                assigned_rules.extend(TAG_RULE_MAP.get(tag, []))
        column.assigned_rules = list(dict.fromkeys(assigned_rules))

        standard_candidates = [
            name for name in self._string_list(payload.get("standard_candidates"))
            if name in standard_terms
        ]
        column.standard_candidates = list(dict.fromkeys(standard_candidates))
        column.routing_confidence = max(column.routing_confidence, self._confidence(payload.get("confidence")))
        column.standard_match_type = "llm_resolved" if column.standard_candidates else "rule_only"
        column.rag_required = not bool(column.standard_candidates)
        if payload.get("reason"):
            column.rag_evidence = [f"llm_route:{str(payload['reason']).strip()}"]
        return column

    def run(self, state: PipelineState) -> PipelineState:
        traces = list(state.get("agent_traces", []))
        updated: list[ColumnProfile] = []
        rag_count = 0
        use_llm = (
            bool(state.get("use_llm_agents"))
            and self.column_resolver is not None
            and self.column_resolver.enabled
        )
        relationship_candidates: list[dict[str, Any]] | None = None

        for column in state["columns"]:
            column, route_source = self._apply_rule_route(state, column)
            llm_error = ""
            llm_model = ""
            llm_stage = ""
            llm_escalated = ""
            if use_llm and column.rag_required:
                payload = self.column_resolver.resolve(state, column)
                if payload:
                    column = self._apply_llm_route(state, column, payload)
                    route_source = f"llm:{payload.get('_llm_stage', 'fast')}"
                    llm_model = str(payload.get("_llm_model", ""))
                    llm_stage = str(payload.get("_llm_stage", ""))
                    llm_escalated = str(payload.get("_llm_escalated", ""))
                else:
                    column = self._apply_rule_fallback(column)
                    llm_error = self.column_resolver.last_error
                    llm_model = getattr(self.column_resolver, "last_model_name", "")
                    llm_stage = getattr(self.column_resolver, "last_stage", "")

            if column.rag_required:
                rag_count += 1

            traces.append(
                self.trace(
                    action="route_rules",
                    target=column.raw_name,
                    detail=(
                        f"source={route_source}, "
                        f"rules={column.assigned_rules}, "
                        f"confidence={column.routing_confidence:.2f}, rag={column.rag_required}, "
                        f"match_type={column.standard_match_type}, candidates={column.standard_candidates}, "
                        f"model={llm_model}, stage={llm_stage}, escalated={llm_escalated}, "
                        f"llm_error={llm_error}"
                    ),
                )
            )
            updated.append(column)

        if use_llm:
            relationship_candidates = self.column_resolver.resolve_relationships(state, updated)
            traces.append(
                self.trace(
                    action="route_relationships",
                    detail=(
                        f"candidates={len(relationship_candidates)}, "
                        f"model={self.column_resolver.last_model_name}, "
                        f"stage={self.column_resolver.last_stage}, "
                        f"llm_error={self.column_resolver.last_error}"
                    ),
                )
            )

        traces.append(self.trace(action="routing_summary", detail=f"rag_required={rag_count}"))
        result: PipelineState = {
            "columns": updated,
            "agent_traces": traces,
        }
        if relationship_candidates is not None:
            result["relationship_candidates"] = relationship_candidates
        return result


class LLMColumnResolver:
    def __init__(
        self,
        model_name: str | None = None,
        fast_model_name: str | None = None,
        strong_model_name: str | None = None,
    ):
        self.fast_model_name = fast_model_name or os.getenv("OLLAMA_FAST_MODEL") or os.getenv("OLLAMA_MODEL") or LLM_FAST_MODEL
        self.strong_model_name = strong_model_name or os.getenv("OLLAMA_STRONG_MODEL") or model_name or LLM_STRONG_MODEL
        self.model_name = self.fast_model_name
        self._llm = ChatLLMClient(model_name=self.fast_model_name)
        self._strong_llm = ChatLLMClient(model_name=self.strong_model_name)
        self.last_error = ""
        self.last_response_preview = ""
        self.last_model_name = self.fast_model_name
        self.last_stage = "fast"

    @property
    def enabled(self) -> bool:
        return self._llm.enabled

    def _client(self):
        return self._llm if self.enabled else None

    @property
    def _strong_enabled(self) -> bool:
        return (
            bool(self.strong_model_name)
            and self.strong_model_name != self.fast_model_name
            and self._strong_llm.enabled
        )

    def _record_attempt(self, llm: ChatLLMClient, stage: str) -> None:
        self.last_error = llm.last_error
        self.last_response_preview = llm.last_response_preview
        self.last_model_name = llm.model_name
        self.last_stage = stage

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any] | None:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _invoke_json_payload(
        self,
        prompt: str,
        *,
        system_prompt: str,
        difficult: Any,
    ) -> dict[str, Any] | None:
        fast_payload = self._invoke_json_payload_once(
            self._llm,
            "fast",
            prompt,
            system_prompt=system_prompt,
        )
        if fast_payload is not None and not difficult(fast_payload):
            return fast_payload

        if self._strong_enabled:
            strong_payload = self._invoke_json_payload_once(
                self._strong_llm,
                "strong",
                prompt,
                system_prompt=system_prompt,
            )
            if strong_payload is not None:
                strong_payload["_llm_escalated"] = True
                if fast_payload is not None:
                    strong_payload["_llm_fast_confidence"] = fast_payload.get("confidence")
                return strong_payload

        return fast_payload

    def _invoke_json_payload_once(
        self,
        llm: ChatLLMClient,
        stage: str,
        prompt: str,
        *,
        system_prompt: str,
    ) -> dict[str, Any] | None:
        response = llm.invoke_json(prompt, system_prompt=system_prompt)
        self._record_attempt(llm, stage)
        if response is None:
            return None

        payload = self._parse_json_content(response.content)
        if payload is None:
            llm.last_error = f"llm_parse_error:{response.content[:200]}"
            self._record_attempt(llm, stage)
            return None

        payload["_llm_model"] = llm.model_name
        payload["_llm_stage"] = stage
        payload["_llm_escalated"] = False
        self._record_attempt(llm, stage)
        return payload

    @staticmethod
    def _list_payload(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    @classmethod
    def _routing_needs_strong(cls, payload: dict[str, Any]) -> bool:
        confidence = LLMRoutingAgent._confidence(payload.get("confidence"))
        if confidence < LLM_STRONG_FALLBACK_CONFIDENCE:
            return True
        if not cls._list_payload(payload.get("assigned_rules")) and not cls._list_payload(payload.get("semantic_tags")):
            return True
        return False

    @staticmethod
    def _relationship_needs_strong(payload: dict[str, Any]) -> bool:
        candidates = payload.get("relationship_candidates")
        if not isinstance(candidates, list):
            return True
        if not candidates:
            return False
        confidences = [
            LLMRoutingAgent._confidence(candidate.get("confidence"))
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
        return bool(confidences) and max(confidences) < LLM_STRONG_FALLBACK_CONFIDENCE

    def resolve(self, state: PipelineState, column: ColumnProfile) -> dict[str, Any] | None:
        llm = self._client()
        if llm is None:
            return None

        dataset_meta = state["dataset_meta"]
        standard_terms = list(state["standard_terms"].keys())[:LLM_STANDARD_TERM_SAMPLE_SIZE]
        allowed_tags = sorted(TAG_RULE_MAP.keys())
        allowed_rules = sorted(
            {
                rule_id
                for category in VALIDATION_CRITERIA.values()
                for rule_id in category["criteria"].keys()
            }
        )
        all_columns = [candidate.raw_name for candidate in state.get("columns", [])]
        prompt = f"""
You are a public-data schema routing agent.
Return strict JSON with keys:
- normalized_name: string
- semantic_tags: list[string]
- assigned_rules: list[string]
- standard_candidates: list[string]
- confidence: float between 0 and 1
- reason: string

Dataset:
- name: {dataset_meta.dataset_name}
- provider: {dataset_meta.provider_name}
- keywords: {", ".join(dataset_meta.keywords)}
- format: {dataset_meta.data_format}
- all_columns: {all_columns}

Column:
- raw_name: {column.raw_name}
- normalized_name: {column.normalized_name}
- source: {column.source}
- inferred_type: {column.inferred_primitive_type}
- sample_values: {column.sample_values}
- top_values: {column.top_values}

Instructions:
- Infer semantic_tags and assigned_rules from the column meaning and dataset context.
- semantic_tags must use only these values: {allowed_tags}
- assigned_rules must use only these values: {allowed_rules}
- Columns whose names end with 명, 명칭, 기관명, 시설명, 경찰서명, 부서명, or 담당자명 are descriptive names, not row identifiers.
- Do not assign identifier semantic_tags or duplicate_data rules to descriptive name columns unless the column name explicitly contains 고유번호, 식별번호, 일련번호, 관리번호, ID, or 아이디.
- standard_candidates should contain the best matching canonical standard terms only.
- If there is no confident standard term match, return an empty list for standard_candidates.
- assigned_rules may be non-empty even when standard_candidates is empty if the column meaning is still clear.
- confidence should reflect routing confidence for this column.
- reason should be a short Korean sentence.

Known standard terms sample:
{json.dumps(standard_terms, ensure_ascii=False)}
"""
        return self._invoke_json_payload(
            prompt,
            system_prompt="You are a careful public-data schema routing assistant.",
            difficult=self._routing_needs_strong,
        )

    def resolve_relationships(self, state: PipelineState, columns: list[ColumnProfile]) -> list[dict[str, Any]]:
        llm = self._client()
        if llm is None:
            return []

        dataset_meta = state["dataset_meta"]
        allowed_rules = [
            "time_sequence_consistency",
            "precedence_accuracy",
            "logical_consistency",
            "calculation_formula",
            "reference_relation",
        ]
        column_payload = [
            {
                "raw_name": column.raw_name,
                "normalized_name": column.normalized_name,
                "semantic_tags": column.semantic_tags,
                "assigned_rules": column.assigned_rules,
                "inferred_type": column.inferred_primitive_type,
                "sample_values": column.sample_values,
                "top_values": column.top_values,
            }
            for column in columns
        ]
        prompt = f"""
You are a public-data relationship routing agent.
Return strict JSON with one key:
- relationship_candidates: list of objects

Each object must have:
- rule_id: one of {allowed_rules}
- columns: list of existing raw column names involved in the relationship
- confidence: float between 0 and 1
- reason: short Korean sentence

Dataset:
- name: {dataset_meta.dataset_name}
- provider: {dataset_meta.provider_name}
- keywords: {", ".join(dataset_meta.keywords)}
- format: {dataset_meta.data_format}

Columns:
{json.dumps(column_payload, ensure_ascii=False)}

Instructions:
- Propose only relationships that are strongly implied by column meanings and samples.
- Do not propose a relationship just because names share a generic token.
- For time_sequence_consistency or precedence_accuracy, use exactly two date/time columns.
- For logical_consistency, use two columns with a clear business dependency, such as a yes/no flag and its count, or a region column and an address column.
- For calculation_formula, use one result/total column and two or more numeric component columns.
- For reference_relation, use exactly two columns where one is a code/id/number and the other is its name/label.
- Return an empty list if no relationship is clear.
- Use only raw column names that appear in Columns.
- Output JSON only.
"""
        payload = self._invoke_json_payload(
            prompt,
            system_prompt=(
                "You are a careful public-data relationship routing assistant. "
                "Respond with a single JSON object only."
            ),
            difficult=self._relationship_needs_strong,
        )
        if payload is None:
            return []

        raw_names = {column.raw_name for column in columns}
        candidates = payload.get("relationship_candidates")
        if not isinstance(candidates, list):
            return []

        sanitized: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            rule_id = str(candidate.get("rule_id") or "").strip()
            if rule_id not in allowed_rules:
                continue
            candidate_columns = [
                str(name).strip()
                for name in candidate.get("columns", [])
                if str(name).strip() in raw_names
            ]
            if len(candidate_columns) < 2:
                continue
            sanitized.append(
                {
                    "rule_id": rule_id,
                    "columns": list(dict.fromkeys(candidate_columns)),
                    "confidence": LLMRoutingAgent._confidence(candidate.get("confidence")),
                    "reason": str(candidate.get("reason") or "").strip(),
                }
            )
        return sanitized


class LLMSemanticProfiler:
    def __init__(
        self,
        model_name: str | None = None,
        fast_model_name: str | None = None,
        strong_model_name: str | None = None,
    ):
        self.fast_model_name = fast_model_name or os.getenv("OLLAMA_FAST_MODEL") or os.getenv("OLLAMA_MODEL") or LLM_FAST_MODEL
        self.strong_model_name = strong_model_name or os.getenv("OLLAMA_STRONG_MODEL") or model_name or LLM_STRONG_MODEL
        self.model_name = self.fast_model_name
        self._llm = ChatLLMClient(model_name=self.fast_model_name)
        self._strong_llm = ChatLLMClient(model_name=self.strong_model_name)
        self.last_error = ""
        self.last_response_preview = ""
        self.last_model_name = self.fast_model_name
        self.last_stage = "fast"

    @property
    def enabled(self) -> bool:
        return self._llm.enabled

    def _client(self):
        return self._llm if self.enabled else None

    @property
    def _strong_enabled(self) -> bool:
        return (
            bool(self.strong_model_name)
            and self.strong_model_name != self.fast_model_name
            and self._strong_llm.enabled
        )

    def _record_attempt(self, llm: ChatLLMClient, stage: str) -> None:
        self.last_error = llm.last_error
        self.last_response_preview = llm.last_response_preview
        self.last_model_name = llm.model_name
        self.last_stage = stage

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any] | None:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _invoke_json_payload(
        self,
        prompt: str,
        *,
        system_prompt: str,
    ) -> dict[str, Any] | None:
        fast_payload = self._invoke_json_payload_once(
            self._llm,
            "fast",
            prompt,
            system_prompt=system_prompt,
        )
        if fast_payload is not None and not self._profile_needs_strong(fast_payload):
            return fast_payload

        if self._strong_enabled:
            strong_payload = self._invoke_json_payload_once(
                self._strong_llm,
                "strong",
                prompt,
                system_prompt=system_prompt,
            )
            if strong_payload is not None:
                strong_payload["_llm_escalated"] = True
                if fast_payload is not None:
                    strong_payload["_llm_fast_confidence"] = fast_payload.get("confidence")
                return strong_payload

        return fast_payload

    def _invoke_json_payload_once(
        self,
        llm: ChatLLMClient,
        stage: str,
        prompt: str,
        *,
        system_prompt: str,
    ) -> dict[str, Any] | None:
        response = llm.invoke_json(prompt, system_prompt=system_prompt)
        self._record_attempt(llm, stage)
        if response is None:
            return None

        payload = self._parse_json_content(response.content)
        if payload is None:
            llm.last_error = f"llm_parse_error:{response.content[:200]}"
            self._record_attempt(llm, stage)
            return None

        payload["_llm_model"] = llm.model_name
        payload["_llm_stage"] = stage
        payload["_llm_escalated"] = False
        self._record_attempt(llm, stage)
        return payload

    @staticmethod
    def _profile_needs_strong(payload: dict[str, Any]) -> bool:
        if payload.get("confidence") is None:
            return True
        return LLMRoutingAgent._confidence(payload.get("confidence")) < LLM_STRONG_FALLBACK_CONFIDENCE

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
- only explain what this column represents or means in the dataset
- do not claim that a problem exists unless it is visible in the provided samples
- do not mention standard term mapping
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
- top_values: {column.top_values}
"""
        payload = self._invoke_json_payload(
            prompt,
            system_prompt=(
                "You are a semantic profiling assistant for Korean public datasets. "
                "Respond with a single JSON object only. No markdown, no explanation, no code fences."
            ),
        )
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
            if column.standard_candidates and column.standard_match_type == "unmatched":
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


class SemanticProfilingAgent(BaseAgent):
    name = "semantic_profiler"

    def __init__(self, semantic_profiler: LLMSemanticProfiler | None = None):
        self.semantic_profiler = semantic_profiler

    def _llm_debug_detail(self, use_llm: bool, llm_attempted: bool) -> tuple[str, str]:
        if not use_llm or not llm_attempted or self.semantic_profiler is None:
            return "", ""
        return self.semantic_profiler.last_error, self.semantic_profiler.last_response_preview

    def run(self, state: PipelineState) -> PipelineState:
        traces = list(state.get("agent_traces", []))
        updated = []
        use_llm = bool(state.get("use_llm_agents")) and self.semantic_profiler is not None

        for column in state["columns"]:
            llm_reasons = semantic_profile_llm_reasons(column)
            needs_llm = use_llm and bool(llm_reasons)
            llm_attempted = False
            column.semantic_profile_llm_needed = needs_llm
            column.semantic_profile_llm_reasons = llm_reasons
            if needs_llm:
                llm_attempted = True
                profile = self.semantic_profiler.profile(state, column)
                if profile:
                    column.semantic_profile_label = profile.get("label")
                    column.semantic_profile_description = profile.get("description")
                    column.semantic_profile_confidence = profile.get("confidence")
            llm_error, llm_preview = self._llm_debug_detail(use_llm, llm_attempted)
            traces.append(
                self.trace(
                    action="semantic_profile",
                    target=column.raw_name,
                    detail=(
                        f"label={column.semantic_profile_label}, "
                        f"confidence={column.semantic_profile_confidence}, "
                        f"llm_needed={needs_llm}, reasons={llm_reasons}, "
                        f"model={getattr(self.semantic_profiler, 'last_model_name', '')}, "
                        f"stage={getattr(self.semantic_profiler, 'last_stage', '')}, "
                        f"llm_error={llm_error}, "
                        f"llm_preview={llm_preview}"
                    ),
                )
            )
            updated.append(column)

        return {"columns": updated, "agent_traces": traces}


def should_run_rag(state: PipelineState) -> str:
    if any(column.rag_required for column in state["columns"]):
        return "rag_resolve"
    return "semantic_profile"
