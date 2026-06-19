"""Legacy BaseAgent adapters around deterministic pipeline steps.

The pipeline calls ``core.pipeline`` functions directly; these classes remain
for older agent-registry imports.
"""

from __future__ import annotations

try:
    from ..core.pipeline import profile_values, propose_repairs, validate_quality, verify_results
    from ..core.schema.models import PipelineState
except ImportError:  # pragma: no cover
    from core.pipeline import profile_values, propose_repairs, validate_quality, verify_results
    from core.schema.models import PipelineState
from .base import BaseAgent


class DataProfilingAgent(BaseAgent):
    name = "profiler"

    def run(self, state: PipelineState) -> PipelineState:
        return profile_values(state)


class ValidationAgent(BaseAgent):
    name = "validator"

    def run(self, state: PipelineState) -> PipelineState:
        return validate_quality(state)


class RepairAgent(BaseAgent):
    name = "repair_planner"

    def run(self, state: PipelineState) -> PipelineState:
        return propose_repairs(state)


class VerificationAgent(BaseAgent):
    name = "verifier"

    def run(self, state: PipelineState) -> PipelineState:
        return verify_results(state)
