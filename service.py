from __future__ import annotations

from pathlib import Path

from .core.config.constants import DEFAULT_META_CSV_NAME, DEFAULT_STANDARD_TERMS_CSV_NAME
from .graph import build_graph


def default_data_paths(base_dir: Path | None = None) -> tuple[Path, Path]:
    base = base_dir or Path(__file__).resolve().parent
    data_dir = base / "data"
    meta_path = data_dir / DEFAULT_META_CSV_NAME
    standard_path = data_dir / DEFAULT_STANDARD_TERMS_CSV_NAME
    return meta_path, standard_path


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
) -> dict:
    if not uploaded_dataset_csv and not dataset_id and not dataset_name:
        raise ValueError("uploaded_dataset_csv is required.")

    default_meta, default_standard = default_data_paths()
    graph = build_graph(llm_model=llm_model)
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
        }
    )

    return {
        "summary": result["summary"],
        "preview_headers": result.get("preview_headers", []),
        "preview_rows": result.get("preview_rows", []),
        "columns": [column.model_dump() for column in result["columns"]],
        "findings": [finding.model_dump() for finding in result["findings"]],
        "agent_traces": [trace.model_dump() for trace in result.get("agent_traces", [])],
    }
