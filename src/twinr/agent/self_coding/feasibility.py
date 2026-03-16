"""Run deterministic feasibility checks for Adaptive Skill Engine requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .capability_registry import SelfCodingCapabilityRegistry
from .contracts import FeasibilityResult, SkillSpec
from .status import CapabilityStatus, CompileTarget, FeasibilityOutcome

_AUTOMATION_MAX_CAPABILITIES = 4
_AUTOMATION_MAX_TRIGGER_CONDITIONS = 4
_AUTOMATION_MAX_CONSTRAINTS = 4
_AUTOMATION_MAX_SCOPE_LEAVES = 8


class SelfCodingFeasibilityChecker:
    """Decide whether a skill request fits the current ASE build envelope."""

    def __init__(
        self,
        registry: SelfCodingCapabilityRegistry,
        *,
        automation_max_capabilities: int = _AUTOMATION_MAX_CAPABILITIES,
        automation_max_trigger_conditions: int = _AUTOMATION_MAX_TRIGGER_CONDITIONS,
        automation_max_constraints: int = _AUTOMATION_MAX_CONSTRAINTS,
        automation_max_scope_leaves: int = _AUTOMATION_MAX_SCOPE_LEAVES,
    ) -> None:
        self.registry = registry
        self.automation_max_capabilities = max(1, int(automation_max_capabilities))
        self.automation_max_trigger_conditions = max(1, int(automation_max_trigger_conditions))
        self.automation_max_constraints = max(1, int(automation_max_constraints))
        self.automation_max_scope_leaves = max(1, int(automation_max_scope_leaves))

    def check(self, spec: SkillSpec) -> FeasibilityResult:
        """Return the deterministic feasibility result for one skill spec."""

        readiness = {item.capability_id: item for item in self.registry.availability_snapshot()}
        missing_capabilities: list[str] = []
        reasons: list[str] = []

        for capability_id in spec.capabilities:
            availability = readiness.get(capability_id)
            if availability is None:
                missing_capabilities.append(capability_id)
                reasons.append(f"Capability `{capability_id}` is not part of the ASE registry.")
                continue
            if availability.status != CapabilityStatus.READY:
                missing_capabilities.append(capability_id)
                detail = availability.detail or f"Capability status is {availability.status.value}."
                reasons.append(f"Capability `{capability_id}` is not ready: {detail}")

        if missing_capabilities:
            return FeasibilityResult(
                outcome=FeasibilityOutcome.RED,
                summary="Twinr is missing one or more required capabilities for this skill.",
                reasons=tuple(reasons),
                missing_capabilities=tuple(missing_capabilities),
                suggested_target=None,
            )

        suggested_target, target_reasons = self._suggest_target(spec)
        reasons.extend(target_reasons)
        if suggested_target == CompileTarget.AUTOMATION_MANIFEST:
            return FeasibilityResult(
                outcome=FeasibilityOutcome.GREEN,
                summary="Required capabilities are ready and the request fits the automation-first build path.",
                reasons=tuple(reasons),
                missing_capabilities=(),
                suggested_target=suggested_target,
            )
        return FeasibilityResult(
            outcome=FeasibilityOutcome.YELLOW,
            summary="Required capabilities are ready, but this request needs the later skill-package path.",
            reasons=tuple(reasons),
            missing_capabilities=(),
            suggested_target=suggested_target,
        )

    def _suggest_target(self, spec: SkillSpec) -> tuple[CompileTarget, tuple[str, ...]]:
        reasons: list[str] = []
        if spec.trigger.mode != "push":
            reasons.append("Pull-style on-demand skills require the later skill-package runner instead of the automation-first path.")
            return CompileTarget.SKILL_PACKAGE, tuple(reasons)
        if len(spec.capabilities) > self.automation_max_capabilities:
            reasons.append(
                f"The current automation-first path is limited to {self.automation_max_capabilities} capabilities per skill."
            )
            return CompileTarget.SKILL_PACKAGE, tuple(reasons)
        if len(spec.trigger.conditions) > self.automation_max_trigger_conditions:
            reasons.append(
                f"The current automation-first path supports at most {self.automation_max_trigger_conditions} trigger conditions."
            )
            return CompileTarget.SKILL_PACKAGE, tuple(reasons)
        if len(spec.constraints) > self.automation_max_constraints:
            reasons.append(
                f"The current automation-first path supports at most {self.automation_max_constraints} constraints."
            )
            return CompileTarget.SKILL_PACKAGE, tuple(reasons)
        if _scope_leaf_count(spec.scope) > self.automation_max_scope_leaves:
            reasons.append(
                f"The current automation-first path supports at most {self.automation_max_scope_leaves} scope leaf values."
            )
            return CompileTarget.SKILL_PACKAGE, tuple(reasons)
        reasons.append("The request stays within the current automation-first compile envelope.")
        return CompileTarget.AUTOMATION_MANIFEST, tuple(reasons)


def _scope_leaf_count(payload: Mapping[str, Any]) -> int:
    total = 0
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            if not current:
                total += 1
                continue
            stack.extend(current.values())
            continue
        if isinstance(current, (list, tuple)):
            if not current:
                total += 1
                continue
            stack.extend(current)
            continue
        total += 1
    return total
