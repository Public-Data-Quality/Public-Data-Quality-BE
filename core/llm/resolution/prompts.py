from __future__ import annotations

import json
from typing import Any

SCHEMA_ROUTING_SYSTEM_PROMPT = "You are a careful public-data schema routing assistant."

RELATIONSHIP_ROUTING_SYSTEM_PROMPT = (
    "You are a careful public-data relationship routing assistant. "
    "Respond with a single JSON object only."
)

SEMANTIC_PROFILE_SYSTEM_PROMPT = (
    "You are a semantic profiling assistant for Korean public datasets. "
    "Respond with a single JSON object only. No markdown, no explanation, no code fences."
)


def schema_routing_prompt(
    *,
    dataset_name: str,
    provider_name: str,
    keywords: list[str],
    data_format: str,
    all_columns: list[str],
    column_raw_name: str,
    column_normalized_name: str,
    column_source: str,
    column_inferred_type: str,
    sample_values: list[Any],
    top_values: list[dict[str, Any]],
    allowed_tags: list[str],
    allowed_rules: list[str],
    standard_terms: list[str],
) -> str:
    return f"""
You are a public-data schema routing agent.
Return strict JSON with keys:
- normalized_name: string
- semantic_tags: list[string]
- assigned_rules: list[string]
- standard_candidates: list[string]
- confidence: float between 0 and 1
- reason: string

Dataset:
- name: {dataset_name}
- provider: {provider_name}
- keywords: {", ".join(keywords)}
- format: {data_format}
- all_columns: {all_columns}

Column:
- raw_name: {column_raw_name}
- normalized_name: {column_normalized_name}
- source: {column_source}
- inferred_type: {column_inferred_type}
- sample_values: {sample_values}
- top_values: {top_values}

Instructions:
- Infer semantic_tags and assigned_rules from the column meaning and dataset context.
- semantic_tags must use only these values: {allowed_tags}
- assigned_rules must use only these values: {allowed_rules}
- Columns whose names end with 명, 명칭, 기관명, 시설명, 경찰서명, 부서명, or 담당자명 are descriptive names, not row identifiers.
- Do not assign identifier semantic_tags or duplicate_data rules to descriptive name columns unless the column name explicitly contains 고유번호, 식별번호, 일련번호, 관리번호, ID, or 아이디.
- standard_candidates should contain the best matching canonical standard terms only.
- If there is no confident standard term match, return an empty list for standard_candidates.
- assigned_rules may be non-empty even when standard_candidates is empty if the column meaning is still clear.
- confidence should reflect routing confidence for this column.
- reason should be a short Korean sentence.

Known standard terms sample:
{json.dumps(standard_terms, ensure_ascii=False)}
"""


def relationship_routing_prompt(
    *,
    dataset_name: str,
    provider_name: str,
    keywords: list[str],
    data_format: str,
    columns: list[dict[str, Any]],
    allowed_rules: list[str],
) -> str:
    return f"""
You are a public-data relationship routing agent.
Return strict JSON with one key:
- relationship_candidates: list of objects

Each object must have:
- rule_id: one of {allowed_rules}
- columns: list of existing raw column names involved in the relationship
- confidence: float between 0 and 1
- reason: short Korean sentence

Dataset:
- name: {dataset_name}
- provider: {provider_name}
- keywords: {", ".join(keywords)}
- format: {data_format}

Columns:
{json.dumps(columns, ensure_ascii=False)}

Instructions:
- Propose only relationships that are strongly implied by column meanings and samples.
- Do not propose a relationship just because names share a generic token.
- For time_sequence_consistency or precedence_accuracy, use exactly two date/time columns.
- For logical_consistency, use two columns with a clear business dependency, such as a yes/no flag and its count, or a region column and an address column.
- For calculation_formula, use one result/total column and two or more numeric component columns.
- For reference_relation, use exactly two columns where one is a code/id/number and the other is its name/label.
- Return an empty list if no relationship is clear.
- Use only raw column names that appear in Columns.
- Output JSON only.
"""


def semantic_profile_prompt(
    *,
    dataset_name: str,
    provider_name: str,
    data_format: str,
    column_raw_name: str,
    column_normalized_name: str,
    semantic_tags: list[str],
    standard_candidates: list[str],
    column_inferred_type: str,
    sample_values: list[Any],
    top_values: list[dict[str, Any]],
) -> str:
    return f"""
You are a semantic profiling agent for Korean public datasets.
Return strict JSON with keys:
- label: short semantic role name in Korean
- description: one sentence in Korean about the business meaning of this column
- confidence: float between 0 and 1

Rules:
- label must be written in Korean only
- description must be written in Korean only
- do not use English unless the original column itself is an English acronym that must remain unchanged
- prefer concise public-data terminology
- only explain what this column represents or means in the dataset
- do not claim that a problem exists unless it is visible in the provided samples
- do not mention standard term mapping
- output JSON only

Dataset:
- name: {dataset_name}
- provider: {provider_name}
- format: {data_format}

Column:
- raw_name: {column_raw_name}
- normalized_name: {column_normalized_name}
- semantic_tags: {semantic_tags}
- standard_candidates: {standard_candidates}
- inferred_type: {column_inferred_type}
- sample_values: {sample_values}
- top_values: {top_values}
"""
