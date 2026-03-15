from twinr.memory.longterm.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.conflicts import LongTermConflictResolver
from twinr.memory.longterm.extract import LongTermTurnExtractor
from twinr.memory.longterm.midterm import (
    LongTermStructuredReflectionProgram,
    OpenAIStructuredReflectionProgram,
    structured_reflection_program_from_config,
)
from twinr.memory.longterm.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.multimodal import LongTermMultimodalExtractor
from twinr.memory.longterm.models import (
    LongTermConsolidationResultV1,
    LongTermConflictOptionV1,
    LongTermConflictQueueItemV1,
    LongTermConflictResolutionV1,
    LongTermConversationTurn,
    LongTermEnqueueResult,
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryContext,
    LongTermMemoryConflictV1,
    LongTermMemoryMutationResultV1,
    LongTermMemoryObjectV1,
    LongTermMemoryReviewItemV1,
    LongTermMemoryReviewResultV1,
    LongTermMidtermPacketV1,
    LongTermMultimodalEvidence,
    LongTermProactiveCandidateV1,
    LongTermProactivePlanV1,
    LongTermRetentionResultV1,
    LongTermReflectionResultV1,
    LongTermSourceRefV1,
    LongTermTurnExtractionV1,
)
from twinr.memory.longterm.planner import LongTermProactivePlanner
from twinr.memory.longterm.propositions import (
    LongTermTurnPropositionBundleV1,
    LongTermTurnPropositionCompiler,
)
from twinr.memory.longterm.proactive import (
    LongTermProactiveHistoryEntryV1,
    LongTermProactivePolicy,
    LongTermProactiveReservationV1,
    LongTermProactiveStateStore,
)
from twinr.memory.longterm.reflect import LongTermMemoryReflector
from twinr.memory.longterm.retriever import LongTermRetriever
from twinr.memory.longterm.retention import LongTermRetentionPolicy
from twinr.memory.longterm.sensor_memory import LongTermSensorMemoryCompiler
from twinr.memory.longterm.service import LongTermMemoryService
from twinr.memory.longterm.store import LongTermStructuredStore
from twinr.memory.longterm.subtext import LongTermSubtextBuilder
from twinr.memory.longterm.truth import LongTermTruthMaintainer
from twinr.memory.longterm.worker import AsyncLongTermMemoryWriter, AsyncLongTermMultimodalWriter

__all__ = [
    "AsyncLongTermMemoryWriter",
    "LongTermConsolidationResultV1",
    "LongTermConflictOptionV1",
    "LongTermConflictQueueItemV1",
    "LongTermConflictResolutionV1",
    "LongTermConflictResolver",
    "LongTermEvalCase",
    "LongTermEvalCaseResult",
    "LongTermEvalResult",
    "LongTermEvalSeedStats",
    "LongTermEvalSummary",
    "LongTermGraphEdgeCandidateV1",
    "LongTermMemoryConflictV1",
    "LongTermConversationTurn",
    "LongTermMemoryConsolidator",
    "LongTermEnqueueResult",
    "LongTermMemoryContext",
    "LongTermMemoryObjectV1",
    "LongTermMemoryMutationResultV1",
    "LongTermMemoryReviewItemV1",
    "LongTermMemoryReviewResultV1",
    "LongTermMidtermPacketV1",
    "LongTermMidtermStore",
    "LongTermMultimodalEvidence",
    "LongTermMultimodalExtractor",
    "MultimodalEvalCase",
    "MultimodalEvalCaseResult",
    "MultimodalEvalResult",
    "MultimodalEvalSeedStats",
    "MultimodalEvalSummary",
    "LongTermProactiveCandidateV1",
    "LongTermProactiveHistoryEntryV1",
    "LongTermProactivePolicy",
    "LongTermProactivePlanV1",
    "LongTermProactivePlanner",
    "LongTermProactiveReservationV1",
    "LongTermProactiveStateStore",
    "LongTermTurnPropositionBundleV1",
    "LongTermTurnPropositionCompiler",
    "LongTermMemoryReflector",
    "LongTermRetriever",
    "LongTermRetentionPolicy",
    "LongTermSensorMemoryCompiler",
    "LongTermRetentionResultV1",
    "LongTermReflectionResultV1",
    "LongTermMemoryService",
    "LongTermStructuredStore",
    "LongTermSourceRefV1",
    "LongTermSubtextBuilder",
    "LongTermStructuredReflectionProgram",
    "LongTermTruthMaintainer",
    "LongTermTurnExtractionV1",
    "LongTermTurnExtractor",
    "OpenAIStructuredReflectionProgram",
    "AsyncLongTermMultimodalWriter",
    "structured_reflection_program_from_config",
    "run_multimodal_longterm_eval",
    "run_synthetic_longterm_eval",
]


def __getattr__(name: str):
    if name in {
        "LongTermEvalCase",
        "LongTermEvalCaseResult",
        "LongTermEvalResult",
        "LongTermEvalSeedStats",
        "LongTermEvalSummary",
        "run_synthetic_longterm_eval",
    }:
        from twinr.memory.longterm import eval as eval_module

        return getattr(eval_module, name)
    if name in {
        "MultimodalEvalCase",
        "MultimodalEvalCaseResult",
        "MultimodalEvalResult",
        "MultimodalEvalSeedStats",
        "MultimodalEvalSummary",
        "run_multimodal_longterm_eval",
    }:
        from twinr.memory.longterm import multimodal_eval as multimodal_eval_module

        return getattr(multimodal_eval_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
