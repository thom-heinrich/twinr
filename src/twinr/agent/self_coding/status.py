"""Define the stable status enums used by the Adaptive Skill Engine core."""

from __future__ import annotations

from enum import Enum


class CapabilityStatus(str, Enum):
    """Describe whether a capability can currently be used by ASE."""

    READY = "ready"
    UNCONFIGURED = "unconfigured"
    MISSING = "missing"
    BLOCKED = "blocked"


class CapabilityRiskClass(str, Enum):
    """Group capabilities by the amount of operator and user risk they carry."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class FeasibilityOutcome(str, Enum):
    """Represent the user-facing buildability outcome for a skill request."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class RequirementsDialogueStatus(str, Enum):
    """Represent the lifecycle of one requirements-gathering dialogue."""

    QUESTIONING = "questioning"
    CONFIRMING = "confirming"
    READY_FOR_COMPILE = "ready_for_compile"
    CANCELLED = "cancelled"


class CompileTarget(str, Enum):
    """Describe the artifact family that the coding worker should produce."""

    AUTOMATION_MANIFEST = "automation_manifest"
    SKILL_PACKAGE = "skill_package"


class CompileJobStatus(str, Enum):
    """Represent the lifecycle of one self-coding compile job."""

    DRAFT = "draft"
    QUESTIONING = "questioning"
    QUEUED = "queued"
    COMPILING = "compiling"
    VALIDATING = "validating"
    SOFT_LAUNCH_READY = "soft_launch_ready"
    FAILED = "failed"


class LearnedSkillStatus(str, Enum):
    """Represent the lifecycle of one learned skill version."""

    DRAFT = "draft"
    SOFT_LAUNCH_READY = "soft_launch_ready"
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    FAILED = "failed"
    RETIRED = "retired"


class ArtifactKind(str, Enum):
    """Describe the persisted artifact produced by a self-coding job."""

    AUTOMATION_MANIFEST = "automation_manifest"
    SKILL_PACKAGE = "skill_package"
    TEST_SUITE = "test_suite"
    REVIEW = "review"
    LOG = "log"


TERMINAL_COMPILE_JOB_STATUSES: tuple[CompileJobStatus, ...] = (
    CompileJobStatus.SOFT_LAUNCH_READY,
    CompileJobStatus.FAILED,
)

ACTIVE_LEARNED_SKILL_STATUSES: tuple[LearnedSkillStatus, ...] = (
    LearnedSkillStatus.ACTIVE,
    LearnedSkillStatus.PAUSED,
)
