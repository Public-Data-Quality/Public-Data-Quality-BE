"""Compatibility helpers for older categorical imports.

New code should import from ``agents.validation.categorical.checks``,
``agents.validation.categorical.findings``, or
``agents.validation.categorical.utils``.
"""

from __future__ import annotations

from .checks import (
    allows_local_prefix_truncation,
    allows_local_surface_normalization,
    canonical_normalization_key,
    clean_reason_text,
    find_surface_normalization_pairs,
    find_truncated_value_pairs,
    has_mixed_surface_forms,
    is_llm_normalization_actionable,
    is_normal_qualifier_suffix,
    is_numeric_like_value,
    is_public_private_category_value,
    is_safe_normalization,
    is_specific_normalization_reason,
    is_specific_out_of_domain_reason,
    is_specific_row_context_reason,
    is_yn_value,
    looks_boolean_column,
    looks_date_column,
    looks_date_value,
    looks_institution_category_column,
    looks_malformed_text_value,
    looks_route_name_column,
    normalized_text,
    visible_text_key,
)
from .findings import finding_key, value_rows
from .utils import parse_json_content

__all__ = [
    "allows_local_prefix_truncation",
    "allows_local_surface_normalization",
    "canonical_normalization_key",
    "clean_reason_text",
    "find_surface_normalization_pairs",
    "find_truncated_value_pairs",
    "finding_key",
    "has_mixed_surface_forms",
    "is_llm_normalization_actionable",
    "is_normal_qualifier_suffix",
    "is_numeric_like_value",
    "is_public_private_category_value",
    "is_safe_normalization",
    "is_specific_normalization_reason",
    "is_specific_out_of_domain_reason",
    "is_specific_row_context_reason",
    "is_yn_value",
    "looks_boolean_column",
    "looks_date_column",
    "looks_date_value",
    "looks_institution_category_column",
    "looks_malformed_text_value",
    "looks_route_name_column",
    "normalized_text",
    "parse_json_content",
    "value_rows",
    "visible_text_key",
]
