from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

try:
    from .core.config.constants import (
        DEFAULT_META_CSV_NAME,
        DEFAULT_STANDARD_TERMS_CSV_NAME,
        QUALITY_DETECTION_RESULTS_CSV_NAME,
        VALIDATION_OUTPUT_DIR_NAME,
    )
    from .graph import build_graph
except ImportError:  # pragma: no cover
    from core.config.constants import (
        DEFAULT_META_CSV_NAME,
        DEFAULT_STANDARD_TERMS_CSV_NAME,
        QUALITY_DETECTION_RESULTS_CSV_NAME,
        VALIDATION_OUTPUT_DIR_NAME,
    )
    from graph import build_graph


DETECTION_MATRIX_METADATA_FIELDS = [
    "dataset_name",
    "row_index",
]


def default_data_paths(base_dir: Path | None = None) -> tuple[Path, Path]:
    base = base_dir or Path(__file__).resolve().parent
    data_dir = base / "data"
    meta_path = data_dir / DEFAULT_META_CSV_NAME
    standard_path = data_dir / DEFAULT_STANDARD_TERMS_CSV_NAME
    return meta_path, standard_path


def validation_output_dir(base_dir: Path | None = None) -> Path:
    base = base_dir or Path(__file__).resolve().parent.parent
    return base / VALIDATION_OUTPUT_DIR_NAME


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _unique_column_headers(column_names: list[str]) -> list[tuple[str, str]]:
    seen: dict[str, int] = {}
    reserved = set(DETECTION_MATRIX_METADATA_FIELDS)
    used = set(reserved)
    headers = []

    for column_name in column_names:
        base = column_name.strip() or "column"
        if base in reserved:
            base = f"data_{base}"
        seen[base] = seen.get(base, 0) + 1
        header = base if seen[base] == 1 else f"{base}_{seen[base]}"
        while header in used:
            seen[base] += 1
            header = f"{base}_{seen[base]}"
        used.add(header)
        headers.append((column_name, header))

    return headers


def _detection_row_count(result: dict) -> int:
    summary = result["summary"]
    row_count = int(summary.get("row_count") or 0)
    preview_row_count = len(result.get("preview_rows") or [])
    max_finding_row_index = max(
        (
            int(row_index)
            for finding in result.get("findings", [])
            for row_index in (finding.get("row_indexes") or [])
            if str(row_index).isdigit()
        ),
        default=0,
    )
    return max(row_count, preview_row_count, max_finding_row_index)


def _detection_column_headers(result: dict) -> list[tuple[str, str]]:
    column_names = [column.get("raw_name", "") for column in result.get("columns", [])]
    if not column_names:
        column_names = list(result.get("preview_headers") or [])
    return _unique_column_headers([str(column_name) for column_name in column_names])


def _issue_cells(result: dict, column_headers: list[tuple[str, str]]) -> set[tuple[int, str]]:
    headers_by_raw_name: dict[str, list[str]] = {}
    for raw_name, header in column_headers:
        headers_by_raw_name.setdefault(raw_name, []).append(header)

    cells: set[tuple[int, str]] = set()
    for finding in result.get("findings", []):
        if finding.get("finding_type") != "issue":
            continue

        row_indexes = [
            int(row_index)
            for row_index in (finding.get("row_indexes") or [])
            if str(row_index).isdigit() and int(row_index) > 0
        ]
        if not row_indexes:
            continue

        column_name = str(finding.get("column_name") or "")
        for header in headers_by_raw_name.get(column_name, []):
            cells.update((row_index, header) for row_index in row_indexes)

    return cells


def _write_detection_result_csv(result: dict, output_dir: Path | None = None) -> str:
    summary = result["summary"]
    output_path = (output_dir or validation_output_dir()) / QUALITY_DETECTION_RESULTS_CSV_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    column_headers = _detection_column_headers(result)
    fieldnames = DETECTION_MATRIX_METADATA_FIELDS + [header for _, header in column_headers]
    issue_cells = _issue_cells(result, column_headers)
    row_count = _detection_row_count(result)

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row_index in range(1, row_count + 1):
            row = {
                "dataset_name": summary.get("dataset_name", ""),
                "row_index": row_index,
            }
            for _, header in column_headers:
                row[header] = 1 if (row_index, header) in issue_cells else 0
            writer.writerow(row)

    return str(output_path)


def run_pipeline(
    *,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    meta_csv: str | None = None,
    standard_terms_csv: str | None = None,
    uploaded_dataset_csv: str | None = None,
    uploaded_dataset_name: str | None = None,
    use_llm_agents: bool = False,
    llm_model: str | None = None,
    llm_fast_model: str | None = None,
    llm_strong_model: str | None = None,
) -> dict:
    if not uploaded_dataset_csv and not dataset_id and not dataset_name:
        raise ValueError("uploaded_dataset_csv, dataset_id, or dataset_name 중 하나는 필요합니다.")

    default_meta, default_standard = default_data_paths()
    graph = build_graph(
        llm_model=llm_model,
        llm_fast_model=llm_fast_model,
        llm_strong_model=llm_strong_model,
    )
    result = graph.invoke(
        {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "meta_csv_path": str(Path(meta_csv) if meta_csv else default_meta),
            "standard_terms_path": str(Path(standard_terms_csv) if standard_terms_csv else default_standard),
            "uploaded_dataset_path": str(Path(uploaded_dataset_csv)) if uploaded_dataset_csv else None,
            "uploaded_dataset_name": uploaded_dataset_name,
            "use_llm_agents": use_llm_agents,
            "llm_model": llm_model,
            "llm_fast_model": llm_fast_model,
            "llm_strong_model": llm_strong_model,
        }
    )

    response = _json_safe({
        "summary": result["summary"],
        "preview_headers": result.get("preview_headers", []),
        "preview_rows": result.get("preview_rows", []),
        "columns": [column.model_dump() for column in result["columns"]],
        "findings": [finding.model_dump() for finding in result["findings"]],
        "agent_traces": [trace.model_dump() for trace in result.get("agent_traces", [])],
    })
    response["summary"]["validation_result_csv"] = _write_detection_result_csv(response)
    return response
