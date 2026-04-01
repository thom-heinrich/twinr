"""Main service type for long-term runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import threading

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.context_store import PromptContextStore
from twinr.memory.query_normalization import LongTermQueryRewriter
from twinr.memory.longterm.ingestion.backfill import LongTermOpsEventBackfiller
from twinr.memory.longterm.ingestion.extract import LongTermTurnExtractor
from twinr.memory.longterm.ingestion.multimodal import LongTermMultimodalExtractor
from twinr.memory.longterm.ingestion.sensor_memory import LongTermSensorMemoryCompiler
from twinr.memory.longterm.proactive.planner import LongTermProactivePlanner
from twinr.memory.longterm.proactive.state import LongTermProactivePolicy
from twinr.memory.longterm.reasoning.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.reasoning.conflicts import LongTermConflictResolver
from twinr.memory.longterm.reasoning.reflect import LongTermMemoryReflector
from twinr.memory.longterm.reasoning.retention import LongTermRetentionPolicy
from twinr.memory.longterm.reasoning.truth import LongTermTruthMaintainer
from twinr.memory.longterm.reasoning.turn_continuity import LongTermTurnContinuityCompiler
from twinr.memory.longterm.retrieval.fast_topic import LongTermFastTopicContextBuilder
from twinr.memory.longterm.retrieval.restart_recall_policy import (
    LongTermRestartRecallPolicyCompiler,
)
from twinr.memory.longterm.retrieval.retriever import LongTermRetriever
from twinr.memory.longterm.runtime.context_snapshot import LongTermContextSnapshot
from twinr.memory.longterm.runtime.prepared_context import PreparedLongTermContextFront
from twinr.memory.longterm.runtime.provider_answer_front import MaterializedProviderAnswerFront
from twinr.memory.longterm.storage.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.storage.store import LongTermStructuredStore
from twinr.memory.longterm.runtime.worker import (
    AsyncLongTermMemoryWriter,
    AsyncLongTermMultimodalWriter,
)

from .builder import LongTermMemoryServiceBuilderMixin
from .context import LongTermMemoryServiceContextMixin
from .ingestion import LongTermMemoryServiceIngestionMixin
from .lifecycle import LongTermMemoryServiceLifecycleMixin
from .maintenance import LongTermMemoryServiceMaintenanceMixin
from .mutations import LongTermMemoryServiceMutationMixin
from .persistence import LongTermMemoryServicePersistenceMixin
from .proactive import LongTermMemoryServiceProactiveMixin
from .readiness import LongTermMemoryServiceReadinessMixin

if TYPE_CHECKING:
    from twinr.agent.personality.learning import PersonalityLearningService


@dataclass(slots=True)
class LongTermMemoryService(
    LongTermMemoryServiceBuilderMixin,
    LongTermMemoryServiceReadinessMixin,
    LongTermMemoryServiceContextMixin,
    LongTermMemoryServiceIngestionMixin,
    LongTermMemoryServiceMaintenanceMixin,
    LongTermMemoryServiceProactiveMixin,
    LongTermMemoryServiceMutationMixin,
    LongTermMemoryServiceLifecycleMixin,
    LongTermMemoryServicePersistenceMixin,
):
    """Coordinate Twinr long-term memory runtime flows."""

    config: TwinrConfig
    prompt_context_store: PromptContextStore
    graph_store: TwinrPersonalGraphStore
    object_store: LongTermStructuredStore
    midterm_store: LongTermMidtermStore
    query_rewriter: LongTermQueryRewriter
    retriever: LongTermRetriever
    extractor: LongTermTurnExtractor
    multimodal_extractor: LongTermMultimodalExtractor
    truth_maintainer: LongTermTruthMaintainer
    consolidator: LongTermMemoryConsolidator
    conflict_resolver: LongTermConflictResolver
    reflector: LongTermMemoryReflector
    sensor_memory: LongTermSensorMemoryCompiler
    ops_backfiller: LongTermOpsEventBackfiller
    planner: LongTermProactivePlanner
    proactive_policy: LongTermProactivePolicy
    retention_policy: LongTermRetentionPolicy
    prepared_context_front: PreparedLongTermContextFront | None = None
    provider_answer_front: MaterializedProviderAnswerFront | None = None
    fast_topic_builder: LongTermFastTopicContextBuilder | None = None
    turn_continuity_compiler: LongTermTurnContinuityCompiler = field(default_factory=LongTermTurnContinuityCompiler)
    restart_recall_policy_compiler: LongTermRestartRecallPolicyCompiler | None = None
    personality_learning: PersonalityLearningService | None = None
    writer: AsyncLongTermMemoryWriter | None = None
    multimodal_writer: AsyncLongTermMultimodalWriter | None = None
    _store_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _context_snapshot_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _latest_context_snapshots: dict[str, LongTermContextSnapshot] = field(default_factory=dict, repr=False, compare=False)
