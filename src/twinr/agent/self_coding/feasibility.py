"""Run deterministic feasibility checks for Adaptive Skill Engine requests."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .capability_registry import SelfCodingCapabilityRegistry
from .contracts import FeasibilityResult, SkillSpec
from .status import CapabilityStatus, CompileTarget, FeasibilityOutcome

_AUTOMATION_MAX_CAPABILITIES = 4
_AUTOMATION_MAX_TRIGGER_CONDITIONS = 4
_AUTOMATION_MAX_CONSTRAINTS = 4
_AUTOMATION_MAX_SCOPE_LEAVES = 8

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ValidatedSkillSpec:
    capabilities: tuple[str, ...]
    trigger_mode: str
    trigger_conditions: tuple[Any, ...]
    constraints: tuple[Any, ...]
    scope: Mapping[str, Any]


class _ScopeTraversalError(ValueError):
    """Raised when scope traversal encounters malformed cyclic data."""


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
        # AUDIT-FIX(#2): Parse configuration values defensively so bad .env input does not crash construction.
        self.automation_max_capabilities = _coerce_positive_int(
            automation_max_capabilities,
            default=_AUTOMATION_MAX_CAPABILITIES,
        )
        # AUDIT-FIX(#2): Parse configuration values defensively so bad .env input does not crash construction.
        self.automation_max_trigger_conditions = _coerce_positive_int(
            automation_max_trigger_conditions,
            default=_AUTOMATION_MAX_TRIGGER_CONDITIONS,
        )
        # AUDIT-FIX(#2): Parse configuration values defensively so bad .env input does not crash construction.
        self.automation_max_constraints = _coerce_positive_int(
            automation_max_constraints,
            default=_AUTOMATION_MAX_CONSTRAINTS,
        )
        # AUDIT-FIX(#2): Parse configuration values defensively so bad .env input does not crash construction.
        self.automation_max_scope_leaves = _coerce_positive_int(
            automation_max_scope_leaves,
            default=_AUTOMATION_MAX_SCOPE_LEAVES,
        )

    def check(self, spec: SkillSpec) -> FeasibilityResult:
        """Return the deterministic feasibility result for one skill spec."""

        # AUDIT-FIX(#4): Validate external spec shape once so malformed input degrades to RED instead of raising.
        try:
            validated_spec, validation_reasons = _validate_spec(spec)
        except Exception:
            logger.exception("Skill specification validation failed during feasibility check.")
            return FeasibilityResult(
                outcome=FeasibilityOutcome.RED,
                summary="Twinr could not evaluate this skill request because the specification is malformed.",
                reasons=("Skill spec validation failed unexpectedly.",),
                missing_capabilities=(),
                suggested_target=None,
            )

        if validation_reasons:
            return FeasibilityResult(
                outcome=FeasibilityOutcome.RED,
                summary="Twinr could not evaluate this skill request because the specification is malformed.",
                reasons=validation_reasons,
                missing_capabilities=(),
                suggested_target=None,
            )

        # AUDIT-FIX(#3): Guard the registry boundary so transient registry faults do not escape as 500 errors.
        readiness, registry_reasons, registry_missing_capabilities = self._build_readiness_index()
        if registry_reasons:
            return FeasibilityResult(
                outcome=FeasibilityOutcome.RED,
                summary="Twinr could not safely read capability readiness for this skill request.",
                reasons=registry_reasons,
                missing_capabilities=registry_missing_capabilities,
                suggested_target=None,
            )

        missing_capabilities: list[str] = []
        reasons: list[str] = []

        for capability_id in validated_spec.capabilities:
            availability = readiness.get(capability_id)
            if availability is None:
                missing_capabilities.append(capability_id)
                reasons.append(f"Capability `{capability_id}` is not part of the ASE registry.")
                continue

            try:
                status = getattr(availability, "status")
            except Exception:
                logger.exception("Capability registry entry for %s exposed an unreadable status.", capability_id)
                missing_capabilities.append(capability_id)
                reasons.append(f"Capability `{capability_id}` could not be evaluated safely.")
                continue

            if status != CapabilityStatus.READY:
                missing_capabilities.append(capability_id)
                # AUDIT-FIX(#7): Return only user-safe status text instead of raw internal detail blobs.
                detail = _status_reason(status)
                reasons.append(f"Capability `{capability_id}` is not ready: {detail}")

        if missing_capabilities:
            return FeasibilityResult(
                outcome=FeasibilityOutcome.RED,
                # AUDIT-FIX(#9): Use accurate wording because capabilities may exist but still be unavailable.
                summary="Twinr cannot safely build this skill because one or more required capabilities are unavailable.",
                reasons=tuple(reasons),
                missing_capabilities=tuple(missing_capabilities),
                suggested_target=None,
            )

        try:
            suggested_target, target_reasons = self._suggest_target(validated_spec)
        except _ScopeTraversalError as exc:
            return FeasibilityResult(
                outcome=FeasibilityOutcome.RED,
                summary="Twinr could not evaluate this skill request because the specification is malformed.",
                reasons=(str(exc),),
                missing_capabilities=(),
                suggested_target=None,
            )

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

    def _build_readiness_index(self) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
        # AUDIT-FIX(#3): Catch unexpected registry failures and downgrade them to deterministic RED results.
        try:
            snapshot = tuple(self.registry.availability_snapshot())
        except Exception:
            logger.exception("Capability registry snapshot failed during feasibility check.")
            return {}, ("Capability registry snapshot failed unexpectedly.",), ()

        readiness: dict[str, Any] = {}
        duplicate_capability_ids: list[str] = []

        for item in snapshot:
            capability_id = getattr(item, "capability_id", None)
            if not isinstance(capability_id, str) or not capability_id.strip():
                logger.error("Capability registry snapshot contained an entry without a valid capability_id.")
                return {}, ("Capability registry snapshot contained a malformed entry.",), ()
            normalized_capability_id = capability_id.strip()
            # AUDIT-FIX(#5): Reject duplicate registry identifiers instead of silently letting later entries win.
            if normalized_capability_id in readiness:
                duplicate_capability_ids.append(normalized_capability_id)
                continue
            readiness[normalized_capability_id] = item

        if duplicate_capability_ids:
            unique_duplicates = tuple(_unique_strings(duplicate_capability_ids))
            return (
                {},
                tuple(
                    f"Capability `{capability_id}` appears multiple times in the ASE registry snapshot."
                    for capability_id in unique_duplicates
                ),
                (),
            )
        return readiness, (), ()

    def _suggest_target(self, spec: _ValidatedSkillSpec) -> tuple[CompileTarget, tuple[str, ...]]:
        reasons: list[str] = []
        requires_skill_package = False

        if spec.trigger_mode != "push":
            reasons.append("Pull-style on-demand skills require the later skill-package runner instead of the automation-first path.")
            requires_skill_package = True
        if len(spec.capabilities) > self.automation_max_capabilities:
            reasons.append(
                f"The current automation-first path is limited to {self.automation_max_capabilities} capabilities per skill."
            )
            requires_skill_package = True
        if len(spec.trigger_conditions) > self.automation_max_trigger_conditions:
            reasons.append(
                f"The current automation-first path supports at most {self.automation_max_trigger_conditions} trigger conditions."
            )
            requires_skill_package = True
        if len(spec.constraints) > self.automation_max_constraints:
            reasons.append(
                f"The current automation-first path supports at most {self.automation_max_constraints} constraints."
            )
            requires_skill_package = True
        # AUDIT-FIX(#1): Bound scope traversal work to the first failing threshold so large payloads cannot monopolize the process.
        scope_leaf_count = _scope_leaf_count(
            spec.scope,
            max_count=self.automation_max_scope_leaves + 1,
        )
        if scope_leaf_count > self.automation_max_scope_leaves:
            reasons.append(
                f"The current automation-first path supports at most {self.automation_max_scope_leaves} scope leaf values."
            )
            requires_skill_package = True

        # AUDIT-FIX(#8): Report every violated envelope rule in one pass so remediation is complete.
        if requires_skill_package:
            return CompileTarget.SKILL_PACKAGE, tuple(reasons)
        reasons.append("The request stays within the current automation-first compile envelope.")
        return CompileTarget.AUTOMATION_MANIFEST, tuple(reasons)


def _coerce_positive_int(value: Any, *, default: int) -> int:
    """Return a positive integer, falling back to the provided default for invalid inputs."""

    if isinstance(value, bool):
        logger.warning("Ignoring boolean configuration value %r and using default %s.", value, default)
        return default
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            logger.warning("Ignoring blank configuration value and using default %s.", default)
            return default
        try:
            return max(1, int(stripped, 10))
        except ValueError:
            logger.warning("Ignoring non-integer configuration value %r and using default %s.", value, default)
            return default
    logger.warning("Ignoring unsupported configuration value %r and using default %s.", value, default)
    return default


def _validate_spec(spec: SkillSpec) -> tuple[_ValidatedSkillSpec, tuple[str, ...]]:
    reasons: list[str] = []

    raw_capabilities = getattr(spec, "capabilities", None)
    if not _is_non_string_sequence(raw_capabilities):
        reasons.append("Skill spec field `capabilities` must be a sequence of non-empty strings.")
        capabilities: tuple[str, ...] = ()
    else:
        # AUDIT-FIX(#6): Deduplicate capability identifiers so duplicate entries do not create false denials.
        capabilities = tuple(_unique_strings(raw_capabilities, field_name="capabilities", reasons=reasons))

    raw_trigger = getattr(spec, "trigger", None)
    if raw_trigger is None:
        reasons.append("Skill spec field `trigger` is required.")
        trigger_mode = ""
        trigger_conditions: tuple[Any, ...] = ()
    else:
        trigger_mode = _require_non_empty_string(getattr(raw_trigger, "mode", None), field_name="trigger.mode", reasons=reasons)
        trigger_conditions = _require_sequence(
            getattr(raw_trigger, "conditions", None),
            field_name="trigger.conditions",
            reasons=reasons,
        )

    constraints = _require_sequence(getattr(spec, "constraints", None), field_name="constraints", reasons=reasons)

    raw_scope = getattr(spec, "scope", None)
    if not isinstance(raw_scope, Mapping):
        reasons.append("Skill spec field `scope` must be a mapping.")
        scope: Mapping[str, Any] = {}
    else:
        scope = raw_scope

    return (
        _ValidatedSkillSpec(
            capabilities=capabilities,
            trigger_mode=trigger_mode,
            trigger_conditions=trigger_conditions,
            constraints=constraints,
            scope=scope,
        ),
        tuple(reasons),
    )


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _unique_strings(
    values: Sequence[Any],
    *,
    field_name: str | None = None,
    reasons: list[str] | None = None,
) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            if field_name is not None and reasons is not None:
                reasons.append(f"Skill spec field `{field_name}[{index}]` must be a non-empty string.")
            continue
        normalized_value = value.strip()
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        result.append(normalized_value)
    return result


def _require_non_empty_string(value: Any, *, field_name: str, reasons: list[str]) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    reasons.append(f"Skill spec field `{field_name}` must be a non-empty string.")
    return ""


def _require_sequence(value: Any, *, field_name: str, reasons: list[str]) -> tuple[Any, ...]:
    if _is_non_string_sequence(value):
        return tuple(value)
    reasons.append(f"Skill spec field `{field_name}` must be a sequence.")
    return ()


def _status_reason(status: Any) -> str:
    if isinstance(status, CapabilityStatus):
        return f"Capability status is {status.value}."
    return "Capability status is unavailable."


def _scope_leaf_count(payload: Mapping[str, Any], *, max_count: int | None = None) -> int:
    total = 0
    # AUDIT-FIX(#1): Use explicit enter/exit frames so active-path cycles are rejected without undercounting shared aliases.
    stack: list[tuple[bool, Any]] = [(False, payload)]
    active_container_ids: set[int] = set()

    while stack:
        is_exit_frame, current = stack.pop()
        if is_exit_frame:
            active_container_ids.discard(id(current))
            continue

        if isinstance(current, Mapping):
            current_id = id(current)
            if current_id in active_container_ids:
                raise _ScopeTraversalError("Skill spec field `scope` must not contain cyclic containers.")
            active_container_ids.add(current_id)
            stack.append((True, current))
            if not current:
                total += 1
                if max_count is not None and total >= max_count:
                    return total
                continue
            for value in current.values():
                stack.append((False, value))
            continue

        if isinstance(current, (list, tuple)):
            current_id = id(current)
            if current_id in active_container_ids:
                raise _ScopeTraversalError("Skill spec field `scope` must not contain cyclic containers.")
            active_container_ids.add(current_id)
            stack.append((True, current))
            if not current:
                total += 1
                if max_count is not None and total >= max_count:
                    return total
                continue
            for value in current:
                stack.append((False, value))
            continue

        total += 1
        if max_count is not None and total >= max_count:
            return total

    return total