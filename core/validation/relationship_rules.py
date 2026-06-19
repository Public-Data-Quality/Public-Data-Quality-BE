"""Compatibility import path for dataset relationship validation rules.

New code should import from ``core.validation.relationships``.
"""

from __future__ import annotations

from .relationships import find_matching_columns, validate_dataset_relationships

__all__ = [
    "find_matching_columns",
    "validate_dataset_relationships",
]
