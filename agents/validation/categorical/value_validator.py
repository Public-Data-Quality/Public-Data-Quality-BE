from __future__ import annotations

import os
from typing import Any

try:
    from ....core.config.constants import LLM_FAST_MODEL, LLM_STRONG_MODEL
    from ....core.llm import ChatLLMClient
    from ....core.llm.categorical import (
        CATEGORICAL_VALUE_SYSTEM_PROMPT,
        ROW_CONTEXT_SYSTEM_PROMPT,
        categorical_value_validation_prompt,
        row_context_validation_prompt,
    )
except ImportError:  # pragma: no cover
    from core.config.constants import LLM_FAST_MODEL, LLM_STRONG_MODEL
    from core.llm import ChatLLMClient
    from core.llm.categorical import (
        CATEGORICAL_VALUE_SYSTEM_PROMPT,
        ROW_CONTEXT_SYSTEM_PROMPT,
        categorical_value_validation_prompt,
        row_context_validation_prompt,
    )
from .utils import parse_json_content


class LLMCategoricalValueValidator:
    VALUE_REVIEW_KEYS = (
        "normalizations",
        "out_of_domain_values",
        "invalid_format_values",
        "inconsistent_format_groups",
        "needs_manual_review",
    )
    ROW_CONTEXT_REVIEW_KEYS = (
        "row_context_issues",
        "row_context_manual_reviews",
    )

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
        self._llm.last_error = llm.last_error
        self._llm.last_response_preview = llm.last_response_preview

    def _invoke_json_payload(
        self,
        prompt: str,
        *,
        system_prompt: str,
        review_keys: tuple[str, ...],
    ) -> dict[str, Any] | None:
        fast_payload = self._invoke_json_payload_once(
            self._llm,
            "fast",
            prompt,
            system_prompt=system_prompt,
        )
        if fast_payload is not None and not self._payload_needs_strong(fast_payload, review_keys):
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
                    strong_payload["_llm_fast_overall_confidence"] = fast_payload.get("overall_confidence")
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
        payload = parse_json_content(response.content)
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
    def _payload_needs_strong(payload: dict[str, Any], review_keys: tuple[str, ...]) -> bool:
        return any(isinstance(payload.get(key), list) and bool(payload.get(key)) for key in review_keys)

    def validate(
        self,
        *,
        dataset_name: str,
        provider_name: str,
        column_name: str,
        standard_candidate: str | None,
        semantic_tags: list[str],
        values: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        llm = self._client()
        if llm is None:
            return None

        payload = self._invoke_json_payload(
            categorical_value_validation_prompt(
                dataset_name=dataset_name,
                provider_name=provider_name,
                column_name=column_name,
                standard_candidate=standard_candidate,
                semantic_tags=semantic_tags,
                values=values,
            ),
            system_prompt=CATEGORICAL_VALUE_SYSTEM_PROMPT,
            review_keys=self.VALUE_REVIEW_KEYS,
        )
        if payload is None:
            return None
        payload.setdefault("domain_label", "")
        payload.setdefault("canonical_values", [])
        payload.setdefault("normalizations", [])
        payload.setdefault("out_of_domain_values", [])
        payload.setdefault("invalid_format_values", [])
        payload.setdefault("inconsistent_format_groups", [])
        payload.setdefault("needs_manual_review", [])
        payload.setdefault("overall_confidence", 0.0)
        return payload

    def validate_row_context(
        self,
        *,
        dataset_name: str,
        provider_name: str,
        columns: list[dict[str, Any]],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        llm = self._client()
        if llm is None:
            return None

        payload = self._invoke_json_payload(
            row_context_validation_prompt(
                dataset_name=dataset_name,
                provider_name=provider_name,
                columns=columns,
                rows=rows,
            ),
            system_prompt=ROW_CONTEXT_SYSTEM_PROMPT,
            review_keys=self.ROW_CONTEXT_REVIEW_KEYS,
        )
        if payload is None:
            return None
        payload.setdefault("row_context_issues", [])
        payload.setdefault("row_context_manual_reviews", [])
        payload.setdefault("overall_confidence", 0.0)
        return payload
