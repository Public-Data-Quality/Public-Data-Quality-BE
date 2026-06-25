from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class StandardTerm(BaseModel):
    name: str
    description: str = ""
    abbreviation: str = ""
    domain_name: str = ""
    allowed_values: str = ""
    storage_format: str = ""
    expression_format: str = ""
    code_name: str = ""
    owner_org: str = ""
    synonyms: list[str] = Field(default_factory=list)


class DatasetMeta(BaseModel):
    dataset_id: str
    dataset_name: str
    keywords: list[str] = Field(default_factory=list)
    provider_name: str = ""
    provider_code: str = ""
    dataset_type: str = ""
    service_type: str = ""
    data_format: str = ""
    request_fields: list[str] = Field(default_factory=list)
    response_fields: list[str] = Field(default_factory=list)
    update_cycle: str = ""
    total_rows: int | None = None


class ColumnProfile(BaseModel):
    raw_name: str
    normalized_name: str
    source: Literal["request", "response"]
    unit: str | None = None
    tokens: list[str] = Field(default_factory=list)
    semantic_tags: list[str] = Field(default_factory=list)
    standard_candidates: list[str] = Field(default_factory=list)
    standard_match_type: str | None = None
    routing_confidence: float = 0.0
    assigned_rules: list[str] = Field(default_factory=list)
    rag_required: bool = False
    rag_evidence: list[str] = Field(default_factory=list)
    total_count: int | None = None
    non_empty_count: int = 0
    null_count: int = 0
    null_ratio: float | None = None
    distinct_count: int | None = None
    sample_values: list[str] = Field(default_factory=list)
    top_values: list[tuple[str, int]] = Field(default_factory=list)
    inferred_primitive_type: str | None = None
    numeric_parse_ratio: float | None = None
    date_parse_ratio: float | None = None
    numeric_min: float | None = None
    numeric_max: float | None = None
    numeric_mean: float | None = None
    semantic_profile_label: str | None = None
    semantic_profile_description: str | None = None
    semantic_profile_confidence: float | None = None
    semantic_profile_llm_needed: bool | None = None
    semantic_profile_llm_reasons: list[str] = Field(default_factory=list)
    repair_suggestion: str | None = None
    verification_notes: list[str] = Field(default_factory=list)


class ValidationFinding(BaseModel):
    column_name: str
    severity: Literal["info", "warning", "error"]
    finding_type: Literal["manual_review", "issue"]
    display_label: str
    category_group: str
    category_label: str
    criterion_name: str
    criterion_description: str = ""
    rule_id: str
    message: str
    row_indexes: list[int] = Field(default_factory=list)
    related_columns: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class AgentTrace(BaseModel):
    agent_name: str
    action: str
    target: str | None = None
    detail: str = ""


class PipelineState(TypedDict, total=False):
    meta_csv_path: str
    standard_terms_path: str
    uploaded_dataset_path: str
    uploaded_dataset_name: str
    use_llm_agents: bool
    llm_model: str | None
    llm_fast_model: str | None
    llm_strong_model: str | None
    dataset_id: str
    dataset_name: str
    dataset_meta: DatasetMeta
    standard_terms: dict[str, StandardTerm]
    synonym_index: dict[str, str]
    example_index: dict[str, list[str]]
    preview_headers: list[str]
    preview_rows: list[dict[str, str]]
    relationship_candidates: list[dict[str, Any]]
    columns: list[ColumnProfile]
    findings: list[ValidationFinding]
    agent_traces: list[AgentTrace]
    summary: dict[str, Any]
