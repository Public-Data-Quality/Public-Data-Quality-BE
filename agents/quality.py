from __future__ import annotations

import json
from collections import Counter
from datetime import datetime

from ..core.config.constants import (
    PROFILE_DISTINCT_TRACK_LIMIT,
    PROFILE_SAMPLE_VALUES_LIMIT,
    PROFILE_TOP_VALUE_LIMIT,
    PROFILE_TYPE_INFERENCE_THRESHOLD,
)
from ..core.io.loaders import iter_uploaded_rows
from ..core.schema.models import PipelineState, ValidationFinding
from ..core.validation.rules import build_repair_suggestion, validate_column, validate_dataset_relationships
from .base import BaseAgent


def _is_numeric(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def _is_date(value: str) -> bool:
    patterns = ("%Y-%m-%d", "%Y%m%d", "%Y.%m.%d", "%Y/%m/%d", "%Y-%m", "%Y%m")
    for pattern in patterns:
        try:
            datetime.strptime(value, pattern)
            return True
        except ValueError:
            continue
    return False


class DataProfilingAgent(BaseAgent):
    name = "profiler"

    def run(self, state: PipelineState) -> PipelineState:
        uploaded_path = state.get("uploaded_dataset_path")
        traces = list(state.get("agent_traces", []))
        if not uploaded_path:
            traces.append(self.trace(action="profile_values", detail="skipped:no_uploaded_dataset"))
            return {"columns": state["columns"], "agent_traces": traces}

        columns_by_name = {column.raw_name: column for column in state["columns"]}
        preview_rows: list[dict[str, str]] = []
        stats = {
            name: {
                "rows": 0,
                "null_count": 0,
                "non_empty_count": 0,
                "samples": [],
                "distinct": set(),
                "distinct_overflow": False,
                "value_counter": Counter(),
                "numeric_count": 0,
                "date_count": 0,
                "numeric_values": [],
            }
            for name in columns_by_name
        }

        for row in iter_uploaded_rows(uploaded_path):
            preview_rows.append({key: (value or "") for key, value in row.items()})
            for name, column in columns_by_name.items():
                value = (row.get(name) or "").strip()
                bucket = stats[name]
                bucket["rows"] += 1
                if not value:
                    bucket["null_count"] += 1
                    continue
                bucket["non_empty_count"] += 1
                if len(bucket["samples"]) < PROFILE_SAMPLE_VALUES_LIMIT and value not in bucket["samples"]:
                    bucket["samples"].append(value)
                if not bucket["distinct_overflow"]:
                    bucket["distinct"].add(value)
                    if len(bucket["distinct"]) > PROFILE_DISTINCT_TRACK_LIMIT:
                        bucket["distinct_overflow"] = True
                bucket["value_counter"][value] += 1
                if _is_numeric(value):
                    bucket["numeric_count"] += 1
                    bucket["numeric_values"].append(float(value.replace(",", "")))
                if _is_date(value):
                    bucket["date_count"] += 1

        updated = []
        for name, column in columns_by_name.items():
            bucket = stats[name]
            non_empty = bucket["non_empty_count"]
            rows = bucket["rows"]
            column.null_count = bucket["null_count"]
            column.non_empty_count = non_empty
            column.null_ratio = round(bucket["null_count"] / rows, 4) if rows else None
            column.distinct_count = None if bucket["distinct_overflow"] else len(bucket["distinct"])
            column.sample_values = bucket["samples"]
            column.top_values = bucket["value_counter"].most_common(PROFILE_TOP_VALUE_LIMIT)
            column.numeric_parse_ratio = round(bucket["numeric_count"] / non_empty, 4) if non_empty else None
            column.date_parse_ratio = round(bucket["date_count"] / non_empty, 4) if non_empty else None
            if bucket["numeric_values"]:
                column.numeric_min = min(bucket["numeric_values"])
                column.numeric_max = max(bucket["numeric_values"])
                column.numeric_mean = round(sum(bucket["numeric_values"]) / len(bucket["numeric_values"]), 4)
            if non_empty == 0:
                column.inferred_primitive_type = "empty"
            elif (column.numeric_parse_ratio or 0) >= PROFILE_TYPE_INFERENCE_THRESHOLD:
                column.inferred_primitive_type = "numeric"
            elif (column.date_parse_ratio or 0) >= PROFILE_TYPE_INFERENCE_THRESHOLD:
                column.inferred_primitive_type = "date"
            else:
                column.inferred_primitive_type = "string"
            updated.append(column)
            traces.append(
                self.trace(
                    action="profile_column",
                    target=column.raw_name,
                    detail=(
                        f"null_ratio={column.null_ratio}, distinct_count={column.distinct_count}, "
                        f"inferred={column.inferred_primitive_type}, top_values={column.top_values[:2]}"
                    ),
                )
            )

        return {
            "columns": updated,
            "preview_headers": list(columns_by_name.keys()),
            "preview_rows": preview_rows,
            "agent_traces": traces,
        }


class ValidationAgent(BaseAgent):
    name = "validator"

    def run(self, state: PipelineState) -> PipelineState:
        findings: list[ValidationFinding] = []
        traces = list(state.get("agent_traces", []))
        preview_rows = state.get("preview_rows", [])

        for column in state["columns"]:
            column_findings = validate_column(column, state["dataset_meta"], state["standard_terms"], preview_rows)
            findings.extend(column_findings)
            traces.append(
                self.trace(
                    action="validate_column",
                    target=column.raw_name,
                    detail=f"findings={len(column_findings)}",
                )
            )

        relationship_findings = validate_dataset_relationships(state["columns"], state.get("preview_rows", []))
        findings.extend(relationship_findings)
        traces.append(
            self.trace(
                action="validate_relationships",
                target=state["dataset_meta"].dataset_id,
                detail=f"findings={len(relationship_findings)}",
            )
        )

        return {"columns": state["columns"], "findings": findings, "agent_traces": traces}


class RepairAgent(BaseAgent):
    name = "repair_planner"

    def run(self, state: PipelineState) -> PipelineState:
        traces = list(state.get("agent_traces", []))
        updated = []
        for column in state["columns"]:
            column.repair_suggestion = build_repair_suggestion(column)
            traces.append(
                self.trace(
                    action="propose_repair",
                    target=column.raw_name,
                    detail=column.repair_suggestion or "none",
                )
            )
            updated.append(column)
        return {"columns": updated, "agent_traces": traces}


class VerificationAgent(BaseAgent):
    name = "verifier"

    def run(self, state: PipelineState) -> PipelineState:
        coverage = 0
        repairs = 0
        manual_review = 0
        traces = list(state.get("agent_traces", []))
        match_breakdown = Counter()

        for column in state["columns"]:
            if column.standard_candidates:
                coverage += 1
                column.verification_notes.append("표준용어 후보 존재")
            match_breakdown[column.standard_match_type or "unmatched"] += 1
            if column.repair_suggestion:
                repairs += 1
                column.verification_notes.append("수정 제안 생성")
            if not column.assigned_rules:
                manual_review += 1
                column.verification_notes.append("규칙 미할당")

        summary = {
            "dataset_id": state["dataset_meta"].dataset_id,
            "dataset_name": state["dataset_meta"].dataset_name,
            "provider_name": state["dataset_meta"].provider_name,
            "column_count": len(state["columns"]),
            "row_count": state["dataset_meta"].total_rows,
            "standard_term_coverage": round(coverage / max(1, len(state["columns"])), 4),
            "standard_term_coverage_breakdown": dict(match_breakdown),
            "repair_suggestion_count": repairs,
            "manual_review_count": manual_review,
            "finding_count": len(state["findings"]),
            "finding_breakdown": dict(Counter(finding.category_label for finding in state["findings"])),
            "llm_agents_enabled": bool(state.get("use_llm_agents")),
        }
        traces.append(
            self.trace(
                action="verify_results",
                target=state["dataset_meta"].dataset_id,
                detail=json.dumps(summary, ensure_ascii=False),
            )
        )
        return {"summary": summary, "columns": state["columns"], "agent_traces": traces}
