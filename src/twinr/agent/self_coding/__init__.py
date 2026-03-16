"""Expose the core Adaptive Skill Engine package surface.

This package currently owns the versioned contract objects, status enums,
file-backed store, and deterministic capability registry used by the
`self_coding` subsystem. Workflow integration, compile workers, and sandboxed
skill execution live in later slices and should not be added here by default.
"""

from twinr.agent.self_coding.capability_registry import SelfCodingCapabilityRegistry
from twinr.agent.self_coding.contracts import (
    ActivationRecord,
    CapabilityAvailability,
    CapabilityDefinition,
    CompileArtifactRecord,
    CompileJobRecord,
    FeasibilityResult,
    RequirementsDialogueSession,
    SkillSpec,
    SkillTriggerSpec,
)
from twinr.agent.self_coding.feasibility import SelfCodingFeasibilityChecker
from twinr.agent.self_coding.learning_flow import SelfCodingLearningFlow, SelfCodingLearningUpdate
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

__all__ = [
    "ActivationRecord",
    "ArtifactKind",
    "CapabilityAvailability",
    "CapabilityDefinition",
    "CapabilityRiskClass",
    "CapabilityStatus",
    "CompileArtifactRecord",
    "CompileJobRecord",
    "CompileJobStatus",
    "CompileTarget",
    "FeasibilityOutcome",
    "RequirementsDialogueStatus",
    "FeasibilityResult",
    "LearnedSkillStatus",
    "SelfCodingCapabilityRegistry",
    "SelfCodingFeasibilityChecker",
    "SelfCodingLearningFlow",
    "SelfCodingLearningUpdate",
    "SelfCodingRequirementsDialogue",
    "SelfCodingStore",
    "RequirementsDialogueSession",
    "SkillSpec",
    "SkillTriggerSpec",
    "self_coding_store_root",
]
