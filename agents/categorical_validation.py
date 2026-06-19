"""Compatibility import path for the categorical validation agent.

New code should import from ``agents.categorical``.
"""

from __future__ import annotations

from .categorical import CategoricalSemanticValidationAgent, LLMCategoricalValueValidator

__all__ = [
    "CategoricalSemanticValidationAgent",
    "LLMCategoricalValueValidator",
]
