from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from ...core.config.constants import (
        LLM_FAST_MODEL,
        LLM_SEMANTIC_PROFILE_CONFIDENCE_DEFAULT,
        LLM_STRONG_FALLBACK_CONFIDENCE,
        LLM_STRONG_MODEL,
    )
    from ...core.llm import ChatLLMClient
    from ...core.llm.resolution import SEMANTIC_PROFILE_SYSTEM_PROMPT, semantic_profile_prompt
    from ...core.schema.models import ColumnProfile, PipelineState
    from ...core.validation import semantic_profile_llm_reasons
except ImportError:  # pragma: no cover
    from core.config.constants import (
        LLM_FAST_MODEL,
        LLM_SEMANTIC_PROFILE_CONFIDENCE_DEFAULT,
        LLM_STRONG_FALLBACK_CONFIDENCE,
        LLM_STRONG_MODEL,
    )
    from core.llm import ChatLLMClient
    from core.llm.resolution import SEMANTIC_PROFILE_SYSTEM_PROMPT, semantic_profile_prompt
    from core.schema.models import ColumnProfile, PipelineState
    from core.validation import semantic_profile_llm_reasons
from ..base import BaseAgent
from .routing import LLMRoutingAgent


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
        payload = self._invoke_json_payload(
            semantic_profile_prompt(
                dataset_name=dataset_meta.dataset_name,
                provider_name=dataset_meta.provider_name,
                data_format=dataset_meta.data_format,
                column_raw_name=column.raw_name,
                column_normalized_name=column.normalized_name,
                semantic_tags=column.semantic_tags,
                standard_candidates=column.standard_candidates,
                column_inferred_type=column.inferred_primitive_type,
                sample_values=column.sample_values,
                top_values=column.top_values,
            ),
            system_prompt=SEMANTIC_PROFILE_SYSTEM_PROMPT,
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
