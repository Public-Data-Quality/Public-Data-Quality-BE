from __future__ import annotations

try:
    from ..core.io.loaders import build_example_index, load_dataset_meta, load_standard_terms, load_uploaded_dataset_meta
    from ..core.schema.models import ColumnProfile, PipelineState
    from ..core.schema.normalization import build_column_profile
except ImportError:  # pragma: no cover
    from core.io.loaders import build_example_index, load_dataset_meta, load_standard_terms, load_uploaded_dataset_meta
    from core.schema.models import ColumnProfile, PipelineState
    from core.schema.normalization import build_column_profile
from .base import BaseAgent


class ReferenceLoaderAgent(BaseAgent):
    name = "reference_loader"

    def run(self, state: PipelineState) -> PipelineState:
        standard_terms, synonym_index = load_standard_terms(state["standard_terms_path"])
        if state.get("uploaded_dataset_path"):
            dataset_meta = load_uploaded_dataset_meta(
                state["uploaded_dataset_path"],
                dataset_name=state.get("uploaded_dataset_name") or state.get("dataset_name"),
            )
        else:
            dataset_meta = load_dataset_meta(
                state["meta_csv_path"],
                dataset_id=state.get("dataset_id"),
                dataset_name=state.get("dataset_name"),
            )
        example_index = build_example_index(state["meta_csv_path"])
        traces = list(state.get("agent_traces", []))
        traces.append(
            self.trace(
                action="load_reference_data",
                target=dataset_meta.dataset_id,
                detail=(
                    f"dataset={dataset_meta.dataset_name}, standard_terms={len(standard_terms)}, "
                    f"uploaded={bool(state.get('uploaded_dataset_path'))}"
                ),
            )
        )
        return {
            "dataset_id": dataset_meta.dataset_id,
            "dataset_name": dataset_meta.dataset_name,
            "dataset_meta": dataset_meta,
            "standard_terms": standard_terms,
            "synonym_index": synonym_index,
            "example_index": example_index,
            "findings": [],
            "agent_traces": traces,
        }


class SchemaParsingAgent(BaseAgent):
    name = "schema_parser"

    def run(self, state: PipelineState) -> PipelineState:
        dataset_meta = state["dataset_meta"]
        synonym_index = state["synonym_index"]
        columns: list[ColumnProfile] = []

        for raw_name in dataset_meta.request_fields:
            columns.append(build_column_profile(raw_name, "request", synonym_index))
        for raw_name in dataset_meta.response_fields:
            columns.append(build_column_profile(raw_name, "response", synonym_index))

        traces = list(state.get("agent_traces", []))
        traces.append(
            self.trace(
                action="parse_schema",
                target=dataset_meta.dataset_id,
                detail=f"columns={len(columns)}",
            )
        )
        return {"columns": columns, "agent_traces": traces}
