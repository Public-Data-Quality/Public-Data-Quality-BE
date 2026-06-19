from __future__ import annotations

from collections import Counter
from typing import Any

try:
    from ...core.config.constants import CATEGORICAL_LLM_MAX_DISTINCT, CATEGORICAL_LLM_MIN_DISTINCT
except ImportError:  # pragma: no cover
    from core.config.constants import CATEGORICAL_LLM_MAX_DISTINCT, CATEGORICAL_LLM_MIN_DISTINCT
from ..base import BaseAgent
from .findings import (
    LocalCategoricalFindingCounts,
    apply_llm_categorical_findings,
    apply_local_categorical_findings,
    context_columns,
    context_rows,
    value_rows,
    looks_row_context_signal_column,
    row_context_signal_score,
    run_llm_row_context_validation,
)
from .value_validator import LLMCategoricalValueValidator


class CategoricalSemanticValidationAgent(BaseAgent):
    name = "categorical_semantic_validator"

    def __init__(self, validator: LLMCategoricalValueValidator | None = None):
        self.validator = validator

    def _llm_debug_detail(self, use_llm: bool) -> tuple[str, str]:
        if not use_llm or self.validator is None:
            return "", ""
        return self.validator.last_error, self.validator.last_response_preview

    @staticmethod
    def _is_candidate_column(column) -> bool:
        if column.distinct_count is None:
            return False
        if not (CATEGORICAL_LLM_MIN_DISTINCT <= column.distinct_count <= CATEGORICAL_LLM_MAX_DISTINCT):
            return False
        if not column.top_values:
            return False

        categorical_tokens = (
            "구분",
            "유형",
            "종류",
            "상태",
            "여부",
            "유무",
            "급",
            "분류",
            "코드",
            "명칭",
            "일자",
            "일시",
            "날짜",
            "년월",
            "내용",
            "설명",
            "사유",
            "비고",
            "메모",
            "특이사항",
            "조치",
            "민원",
            "안내",
        )
        categorical_tags = {"enum", "code", "boolean", "name", "date"}
        return bool(categorical_tags.intersection(set(column.semantic_tags))) or any(
            token in column.raw_name for token in categorical_tokens
        )

    @staticmethod
    def _value_rows(rows: list[dict[str, str]], column_name: str, target_value: str) -> list[int]:
        return value_rows(rows, column_name, target_value)

    @staticmethod
    def _context_columns(columns) -> list[dict[str, Any]]:
        return context_columns(columns)

    @staticmethod
    def _looks_row_context_signal_column(header: str) -> bool:
        return looks_row_context_signal_column(header)

    @staticmethod
    def _row_context_signal_score(header: str, count: int) -> int:
        return row_context_signal_score(header, count)

    @staticmethod
    def _context_rows(rows: list[dict[str, str]], headers: list[str], limit: int = 80) -> list[dict[str, Any]]:
        return context_rows(rows, headers, limit)

    def _run_llm_row_context_validation(self, *, state, findings, traces):
        return run_llm_row_context_validation(
            state=state,
            findings=findings,
            traces=traces,
            validator=self.validator,
            trace=self.trace,
            debug_detail=lambda: self._llm_debug_detail(True),
        )

    def run(self, state):
        traces = list(state.get("agent_traces", []))
        findings = list(state.get("findings", []))
        rows = state.get("preview_rows", [])
        use_llm = bool(state.get("use_llm_agents")) and self.validator is not None

        if not use_llm:
            traces.append(
                self.trace(
                    action="categorical_semantic_validate",
                    detail="llm_disabled; running_local_text_detectors_only",
                )
            )

        dataset_meta = state["dataset_meta"]
        for column in state["columns"]:
            counter = _column_value_counter(rows, column.raw_name)
            local_counts = apply_local_categorical_findings(
                column=column,
                rows=rows,
                counter=counter,
                findings=findings,
            )

            if not use_llm:
                self._trace_local_skip(traces, column, local_counts, "llm_disabled")
                continue

            if not self._is_candidate_column(column):
                self._trace_local_skip(traces, column, local_counts, "llm_candidate_filter")
                continue

            if not (CATEGORICAL_LLM_MIN_DISTINCT <= len(counter) <= CATEGORICAL_LLM_MAX_DISTINCT):
                self._trace_local_skip(traces, column, local_counts, f"distinct_count={len(counter)}")
                continue

            if not counter:
                continue

            values = [{"value": value, "count": count} for value, count in counter.most_common()]
            result = self.validator.validate(
                dataset_name=dataset_meta.dataset_name,
                provider_name=dataset_meta.provider_name,
                column_name=column.raw_name,
                standard_candidate=column.standard_candidates[0] if column.standard_candidates else None,
                semantic_tags=column.semantic_tags,
                values=values,
            )
            if not result:
                llm_error, llm_preview = self._llm_debug_detail(use_llm)
                traces.append(
                    self.trace(
                        action="categorical_semantic_validate",
                        target=column.raw_name,
                        detail=(f"llm_no_result,error={llm_error},preview={llm_preview}"),
                    )
                )
                continue

            generated = apply_llm_categorical_findings(
                column=column,
                rows=rows,
                result=result,
                findings=findings,
            )
            traces.append(
                self.trace(
                    action="categorical_semantic_validate",
                    target=column.raw_name,
                    detail=(
                        f"values={len(values)}, findings={generated}, "
                        f"domain={result.get('domain_label', '')}, "
                        f"overall_confidence={float(result.get('overall_confidence') or 0.0):.2f}, "
                        f"model={result.get('_llm_model', '')}, "
                        f"stage={result.get('_llm_stage', '')}, "
                        f"escalated={bool(result.get('_llm_escalated'))}"
                    ),
                )
            )

        if use_llm:
            findings, traces = self._run_llm_row_context_validation(state=state, findings=findings, traces=traces)

        return {"findings": findings, "agent_traces": traces}

    def _trace_local_skip(
        self,
        traces: list,
        column,
        local_counts: LocalCategoricalFindingCounts,
        skipped_reason: str,
    ) -> None:
        if not local_counts.has_findings:
            return
        traces.append(
            self.trace(
                action="categorical_semantic_validate",
                target=column.raw_name,
                detail=local_counts.trace_detail(skipped_reason),
            )
        )


def _column_value_counter(rows: list[dict[str, str]], column_name: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = (row.get(column_name) or "").strip()
        if value:
            counter[value] += 1
    return counter
