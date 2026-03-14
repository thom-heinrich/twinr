from twinr.memory.longterm.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.extract import LongTermTurnExtractor
from twinr.memory.longterm.models import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
    LongTermEnqueueResult,
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryContext,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
    LongTermTurnExtractionV1,
)
from twinr.memory.longterm.service import LongTermMemoryService
from twinr.memory.longterm.subtext import LongTermSubtextBuilder
from twinr.memory.longterm.truth import LongTermTruthMaintainer
from twinr.memory.longterm.worker import AsyncLongTermMemoryWriter

__all__ = [
    "AsyncLongTermMemoryWriter",
    "LongTermConsolidationResultV1",
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
    "LongTermMemoryService",
    "LongTermSourceRefV1",
    "LongTermSubtextBuilder",
    "LongTermTruthMaintainer",
    "LongTermTurnExtractionV1",
    "LongTermTurnExtractor",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
