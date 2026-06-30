from __future__ import annotations

import math
from collections import Counter

from ..config.constants import (
    PROFILE_DISTINCT_TRACK_LIMIT,
    PROFILE_SAMPLE_VALUES_LIMIT,
    PROFILE_TOP_VALUE_LIMIT,
    PROFILE_TYPE_INFERENCE_THRESHOLD,
)
from ..io.loaders import iter_uploaded_rows
from ..schema.models import AgentTrace, ColumnProfile, PipelineState
from ..validation.helpers import parse_datetime
from .tracing import pipeline_trace

PROFILE_STEP_NAME = "profiler"
ProfileStats = dict[str, dict]


def profile_values(state: PipelineState) -> PipelineState:
    uploaded_path = state.get("uploaded_dataset_path")
    traces = list(state.get("agent_traces", []))
    if not uploaded_path:
        traces.append(
            pipeline_trace(
                PROFILE_STEP_NAME,
                action="profile_values",
                detail="skipped:no_uploaded_dataset",
            )
        )
        return {"columns": state["columns"], "agent_traces": traces}

    columns_by_name = {column.raw_name: column for column in state["columns"]}
    preview_rows: list[dict[str, str]] = []
    stats = _initial_profile_stats(columns_by_name)

    for row in iter_uploaded_rows(uploaded_path):
        preview_rows.append({key: (value or "") for key, value in row.items()})
        _update_profile_stats(row, columns_by_name, stats)

    updated = _apply_profile_stats(columns_by_name, stats, traces)
    dataset_meta = state["dataset_meta"]
    if updated:
        dataset_meta.total_rows = stats[updated[0].raw_name]["rows"]

    return {
        "columns": updated,
        "preview_headers": list(columns_by_name.keys()),
        "preview_rows": preview_rows,
        "dataset_meta": dataset_meta,
        "agent_traces": traces,
    }


def _initial_profile_stats(columns_by_name: dict[str, ColumnProfile]) -> ProfileStats:
    return {
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
            "numeric_sum": 0.0,
            "numeric_min": None,
            "numeric_max": None,
        }
        for name in columns_by_name
    }


def _update_profile_stats(
    row: dict[str, str],
    columns_by_name: dict[str, ColumnProfile],
    stats: ProfileStats,
) -> None:
    for name in columns_by_name:
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
        _append_numeric_value(bucket, value)
        if _is_date(value):
            bucket["date_count"] += 1


def _apply_profile_stats(
    columns_by_name: dict[str, ColumnProfile],
    stats: ProfileStats,
    traces: list[AgentTrace],
) -> list[ColumnProfile]:
    updated = []
    for name, column in columns_by_name.items():
        bucket = stats[name]
        non_empty = bucket["non_empty_count"]
        rows = bucket["rows"]
        column.total_count = rows
        column.null_count = bucket["null_count"]
        column.non_empty_count = non_empty
        column.null_ratio = round(bucket["null_count"] / rows, 4) if rows else None
        column.distinct_count = None if bucket["distinct_overflow"] else len(bucket["distinct"])
        column.sample_values = bucket["samples"]
        column.top_values = bucket["value_counter"].most_common(PROFILE_TOP_VALUE_LIMIT)
        column.numeric_parse_ratio = round(bucket["numeric_count"] / non_empty, 4) if non_empty else None
        column.date_parse_ratio = round(bucket["date_count"] / non_empty, 4) if non_empty else None
        if bucket["numeric_count"]:
            column.numeric_min = bucket["numeric_min"]
            column.numeric_max = bucket["numeric_max"]
            column.numeric_mean = round(bucket["numeric_sum"] / bucket["numeric_count"], 4)
        _set_inferred_primitive_type(column, non_empty)
        updated.append(column)
        traces.append(
            pipeline_trace(
                PROFILE_STEP_NAME,
                action="profile_column",
                target=column.raw_name,
                detail=(
                    f"null_ratio={column.null_ratio}, distinct_count={column.distinct_count}, "
                    f"inferred={column.inferred_primitive_type}, top_values={column.top_values}"
                ),
            )
        )
    return updated


def _set_inferred_primitive_type(column: ColumnProfile, non_empty: int) -> None:
    if non_empty == 0:
        column.inferred_primitive_type = "empty"
    elif (column.numeric_parse_ratio or 0) >= PROFILE_TYPE_INFERENCE_THRESHOLD:
        column.inferred_primitive_type = "numeric"
    elif (column.date_parse_ratio or 0) >= PROFILE_TYPE_INFERENCE_THRESHOLD:
        column.inferred_primitive_type = "date"
    else:
        column.inferred_primitive_type = "string"


def _parse_finite_number(value: str) -> float | None:
    try:
        parsed = float(value.replace(",", ""))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _append_numeric_value(bucket: dict, value: str) -> None:
    parsed = _parse_finite_number(value)
    if parsed is None:
        return
    bucket["numeric_count"] += 1
    bucket["numeric_sum"] += parsed
    bucket["numeric_min"] = parsed if bucket["numeric_min"] is None else min(bucket["numeric_min"], parsed)
    bucket["numeric_max"] = parsed if bucket["numeric_max"] is None else max(bucket["numeric_max"], parsed)


def _is_date(value: str) -> bool:
    return parse_datetime(value) is not None
