"""Expose the core Adaptive Skill Engine package surface.

This package currently owns the versioned contract objects, status enums,
file-backed store, and deterministic capability registry used by the
`self_coding` subsystem. Workflow integration stays thin here; compile workers
and local Codex drivers now live behind focused modules and sandboxed skill
execution still belongs to later slices.
"""

from twinr.agent.self_coding.activation import SelfCodingActivationService
from twinr.agent.self_coding.capability_registry import SelfCodingCapabilityRegistry
from twinr.agent.self_coding.codex_driver import CodexSdkDriver
from twinr.agent.self_coding.contracts import (
    ActivationRecord,
    CapabilityAvailability,
    CapabilityDefinition,
    CompileArtifactRecord,
    CompileJobRecord,
    CompileRunStatusRecord,
    FeasibilityResult,
    RequirementsDialogueSession,
    SkillSpec,
    SkillTriggerSpec,
)
from twinr.agent.self_coding.feasibility import SelfCodingFeasibilityChecker
from twinr.agent.self_coding.learning_flow import SelfCodingLearningFlow, SelfCodingLearningUpdate
from twinr.agent.self_coding.operator_status import SelfCodingOperatorStatus, build_self_coding_operator_status
from twinr.agent.self_coding.status import (
    ArtifactKind,
    CapabilityRiskClass,
    CapabilityStatus,
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    LearnedSkillStatus,
    RequirementsDialogueStatus,
)
from twinr.agent.self_coding.requirements_dialogue import SelfCodingRequirementsDialogue
from twinr.agent.self_coding.store import SelfCodingStore, self_coding_store_root
from twinr.agent.self_coding.worker import LocalCodexCompileDriver, SelfCodingCompileWorker

__all__ = [
    "ActivationRecord",
    "ArtifactKind",
    "build_self_coding_operator_status",
    "CapabilityAvailability",
    "CapabilityDefinition",
    "CapabilityRiskClass",
    "CapabilityStatus",
    "CompileArtifactRecord",
    "CompileJobRecord",
    "CompileRunStatusRecord",
    "CompileJobStatus",
    "CompileTarget",
    "CodexSdkDriver",
    "FeasibilityOutcome",
    "RequirementsDialogueStatus",
    "FeasibilityResult",
    "LearnedSkillStatus",
    "LocalCodexCompileDriver",
    "SelfCodingActivationService",
    "SelfCodingCapabilityRegistry",
    "SelfCodingCompileWorker",
    "SelfCodingFeasibilityChecker",
    "SelfCodingLearningFlow",
    "SelfCodingLearningUpdate",
    "SelfCodingOperatorStatus",
    "SelfCodingRequirementsDialogue",
    "SelfCodingStore",
    "RequirementsDialogueSession",
    "SkillSpec",
    "SkillTriggerSpec",
    "self_coding_store_root",
]
