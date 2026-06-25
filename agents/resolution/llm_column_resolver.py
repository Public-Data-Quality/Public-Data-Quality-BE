from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from ...core.config.constants import (
        LLM_FAST_MODEL,
        LLM_STRONG_FALLBACK_CONFIDENCE,
        LLM_STRONG_MODEL,
        LLM_STANDARD_TERM_SAMPLE_SIZE,
        TAG_RULE_MAP,
        VALIDATION_CRITERIA,
    )
    from ...core.llm import ChatLLMClient
    from ...core.llm.resolution import (
        RELATIONSHIP_ROUTING_SYSTEM_PROMPT,
        SCHEMA_ROUTING_SYSTEM_PROMPT,
        relationship_routing_prompt,
        schema_routing_prompt,
    )
    from ...core.schema.models import ColumnProfile, PipelineState
except ImportError:  # pragma: no cover
    from core.config.constants import (
        LLM_FAST_MODEL,
        LLM_STRONG_FALLBACK_CONFIDENCE,
        LLM_STRONG_MODEL,
        LLM_STANDARD_TERM_SAMPLE_SIZE,
        TAG_RULE_MAP,
        VALIDATION_CRITERIA,
    )
    from core.llm import ChatLLMClient
    from core.llm.resolution import (
        RELATIONSHIP_ROUTING_SYSTEM_PROMPT,
        SCHEMA_ROUTING_SYSTEM_PROMPT,
        relationship_routing_prompt,
        schema_routing_prompt,
    )
    from core.schema.models import ColumnProfile, PipelineState
from .routing import LLMRoutingAgent


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
        return self._invoke_json_payload(
            schema_routing_prompt(
                dataset_name=dataset_meta.dataset_name,
                provider_name=dataset_meta.provider_name,
                keywords=dataset_meta.keywords,
                data_format=dataset_meta.data_format,
                all_columns=all_columns,
                column_raw_name=column.raw_name,
                column_normalized_name=column.normalized_name,
                column_source=column.source,
                column_inferred_type=column.inferred_primitive_type,
                sample_values=column.sample_values,
                top_values=column.top_values,
                allowed_tags=allowed_tags,
                allowed_rules=allowed_rules,
                standard_terms=standard_terms,
            ),
            system_prompt=SCHEMA_ROUTING_SYSTEM_PROMPT,
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
        payload = self._invoke_json_payload(
            relationship_routing_prompt(
                dataset_name=dataset_meta.dataset_name,
                provider_name=dataset_meta.provider_name,
                keywords=dataset_meta.keywords,
                data_format=dataset_meta.data_format,
                columns=column_payload,
                allowed_rules=allowed_rules,
            ),
            system_prompt=RELATIONSHIP_ROUTING_SYSTEM_PROMPT,
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
