"""Define the stable status enums used by the Adaptive Skill Engine core."""

from __future__ import annotations

from enum import Enum

# AUDIT-FIX(#3): Explicit exports keep the module's public contract stable and
# prevent helper symbols from leaking via wildcard imports.
__all__ = [
    "CapabilityStatus",
    "CapabilityRiskClass",
    "FeasibilityOutcome",
    "RequirementsDialogueStatus",
    "CompileTarget",
    "CompileJobStatus",
    "LearnedSkillStatus",
    "ArtifactKind",
    "TERMINAL_COMPILE_JOB_STATUSES",
    "ACTIVE_LEARNED_SKILL_STATUSES",
    "is_terminal_compile_job_status",
    "is_active_learned_skill_status",
]


class _TypedStringEnum(str, Enum):
    """String-valued enum with exact-type equality semantics."""

    # AUDIT-FIX(#1): Preserve readable string coercion for logging, serialization,
    # and JSON encoders that treat enum members as string subclasses.
    def __str__(self) -> str:
        return self.value

    # AUDIT-FIX(#1): Prevent silent cross-enum equality when different status
    # families reuse the same string literal, e.g. "failed" or "draft".
    def __eq__(self, other: object) -> bool:
        return other.__class__ is self.__class__ and self is other

    # AUDIT-FIX(#1): Include the enum class in the hash so dict/set membership
    # cannot collide across different status families with the same value.
    def __hash__(self) -> int:
        return hash((self.__class__, self.value))


class CapabilityStatus(_TypedStringEnum):
    """Describe whether a capability can currently be used by ASE."""

    READY = "ready"
    UNCONFIGURED = "unconfigured"
    MISSING = "missing"
    BLOCKED = "blocked"


class CapabilityRiskClass(_TypedStringEnum):
    """Group capabilities by the amount of operator and user risk they carry."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class FeasibilityOutcome(_TypedStringEnum):
    """Represent the user-facing buildability outcome for a skill request."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class RequirementsDialogueStatus(_TypedStringEnum):
    """Represent the lifecycle of one requirements-gathering dialogue."""

    QUESTIONING = "questioning"
    CONFIRMING = "confirming"
    READY_FOR_COMPILE = "ready_for_compile"
    CANCELLED = "cancelled"


class CompileTarget(_TypedStringEnum):
    """Describe the artifact family that the coding worker should produce."""

    AUTOMATION_MANIFEST = "automation_manifest"
    SKILL_PACKAGE = "skill_package"


class CompileJobStatus(_TypedStringEnum):
    """Represent the lifecycle of one self-coding compile job."""

    DRAFT = "draft"
    QUESTIONING = "questioning"
    QUEUED = "queued"
    COMPILING = "compiling"
    VALIDATING = "validating"
    SOFT_LAUNCH_READY = "soft_launch_ready"
    FAILED = "failed"


class LearnedSkillStatus(_TypedStringEnum):
    """Represent the lifecycle of one learned skill version."""

    DRAFT = "draft"
    SOFT_LAUNCH_READY = "soft_launch_ready"
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    FAILED = "failed"
    RETIRED = "retired"


class ArtifactKind(_TypedStringEnum):
    """Describe the persisted artifact produced by a self-coding job."""

    AUTOMATION_MANIFEST = "automation_manifest"
    SKILL_PACKAGE = "skill_package"
    TEST_SUITE = "test_suite"
    REVIEW = "review"
    LOG = "log"


# AUDIT-FIX(#2): Keep legacy tuple exports for backward compatibility, but back
# them with exact-type helper predicates to avoid scattered lifecycle logic.
TERMINAL_COMPILE_JOB_STATUSES: tuple[CompileJobStatus, ...] = (
    CompileJobStatus.SOFT_LAUNCH_READY,
    CompileJobStatus.FAILED,
)
_TERMINAL_COMPILE_JOB_STATUS_SET = frozenset(TERMINAL_COMPILE_JOB_STATUSES)

ACTIVE_LEARNED_SKILL_STATUSES: tuple[LearnedSkillStatus, ...] = (
    LearnedSkillStatus.ACTIVE,
    LearnedSkillStatus.PAUSED,
)
_ACTIVE_LEARNED_SKILL_STATUS_SET = frozenset(ACTIVE_LEARNED_SKILL_STATUSES)


def is_terminal_compile_job_status(value: object) -> bool:
    """Return whether the value is a terminal compile-job status."""

    # AUDIT-FIX(#2): Centralize exact-type lifecycle checks for callers.
    return isinstance(value, CompileJobStatus) and value in _TERMINAL_COMPILE_JOB_STATUS_SET


def is_active_learned_skill_status(value: object) -> bool:
    """Return whether the value is an active learned-skill status."""

    # AUDIT-FIX(#2): Centralize exact-type lifecycle checks for callers.
    return isinstance(value, LearnedSkillStatus) and value in _ACTIVE_LEARNED_SKILL_STATUS_SET