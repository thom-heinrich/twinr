"""Factory wiring for the long-term runtime service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
import threading

from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.context_store import PromptContextStore
from twinr.memory.longterm.core.models import LongTermConversationTurn
from twinr.memory.longterm.ingestion.backfill import LongTermOpsEventBackfiller
from twinr.memory.longterm.ingestion.extract import LongTermTurnExtractor
from twinr.memory.longterm.ingestion.multimodal import LongTermMultimodalExtractor
from twinr.memory.longterm.ingestion.sensor_memory import LongTermSensorMemoryCompiler
from twinr.memory.longterm.proactive.planner import LongTermProactivePlanner
from twinr.memory.longterm.proactive.state import (
    LongTermProactivePolicy,
    LongTermProactiveStateStore,
)
from twinr.memory.longterm.reasoning.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.reasoning.conflicts import LongTermConflictResolver
from twinr.memory.longterm.reasoning.reflect import LongTermMemoryReflector
from twinr.memory.longterm.reasoning.retention import LongTermRetentionPolicy
from twinr.memory.longterm.reasoning.truth import LongTermTruthMaintainer
from twinr.memory.longterm.reasoning.turn_continuity import LongTermTurnContinuityCompiler
from twinr.memory.longterm.retrieval.adaptive_policy import LongTermAdaptivePolicyBuilder
from twinr.memory.longterm.retrieval.fast_topic import LongTermFastTopicContextBuilder
from twinr.memory.longterm.retrieval.restart_recall_policy import (
    LongTermRestartRecallPolicyCompiler,
)
from twinr.memory.longterm.retrieval.retriever import LongTermRetriever
from twinr.memory.longterm.retrieval.subtext import (
    LongTermSubtextBuilder,
    LongTermSubtextCompiler,
)
from twinr.memory.longterm.runtime.prepared_context import PreparedLongTermContextFront
from twinr.memory.longterm.runtime.provider_answer_front import MaterializedProviderAnswerFront
from twinr.memory.longterm.storage.provider_answer_front_store import LongTermProviderAnswerFrontStore
from twinr.memory.longterm.storage.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.storage.store import LongTermStructuredStore
from twinr.memory.longterm.runtime.worker import (
    AsyncLongTermMemoryWriter,
    AsyncLongTermMultimodalWriter,
)
from twinr.memory.query_normalization import LongTermQueryRewriter

from .compat import _MAX_QUEUE_SIZE, _coerce_positive_int

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig
    from .main import LongTermMemoryService


class LongTermMemoryServiceBuilderMixin:
    """Build a fully wired long-term runtime service from config."""

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        graph_store: TwinrPersonalGraphStore | None = None,
        prompt_context_store: PromptContextStore | None = None,
        extractor: LongTermTurnExtractor | None = None,
    ) -> LongTermMemoryService:
        """Build the runtime service and its bounded background writers."""

        # `cls` is the real runtime service class; keep the cast runtime-safe so
        # the compatibility wrapper can call this factory without importing the
        # concrete class here at module import time.
        factory = cast(Any, cls)
        store = prompt_context_store or PromptContextStore.from_config(config)
        graph = graph_store or TwinrPersonalGraphStore.from_config(config)
        object_store = cast(LongTermStructuredStore, LongTermStructuredStore.from_config(config))
        midterm_store = LongTermMidtermStore.from_config(config)
        extractor = extractor or LongTermTurnExtractor.from_config(config)
        multimodal_extractor = LongTermMultimodalExtractor(timezone_name=config.local_timezone_name)
        truth_maintainer = LongTermTruthMaintainer()
        consolidator = LongTermMemoryConsolidator(truth_maintainer=truth_maintainer)
        conflict_resolver = LongTermConflictResolver()
        reflector = LongTermMemoryReflector.from_config(config)
        turn_continuity_compiler = LongTermTurnContinuityCompiler()
        sensor_memory = LongTermSensorMemoryCompiler.from_config(config)
        ops_backfiller = LongTermOpsEventBackfiller()
        planner = LongTermProactivePlanner(timezone_name=config.local_timezone_name)
        proactive_state_store = LongTermProactiveStateStore.from_config(config)
        proactive_policy = LongTermProactivePolicy(
            config=config,
            state_store=proactive_state_store,
        )
        remote_state = getattr(store.memory_store, "remote_state", None)
        retention_policy = LongTermRetentionPolicy(
            timezone_name=config.local_timezone_name,
            mode=config.long_term_memory_retention_mode,
            archive_enabled=config.long_term_memory_archive_enabled,
        )
        from twinr.agent.personality.learning import PersonalityLearningService

        personality_learning = PersonalityLearningService.from_config(
            config,
            remote_state=remote_state,
        )
        subtext_builder = LongTermSubtextBuilder(
            config=config,
            graph_store=graph,
            compiler=LongTermSubtextCompiler.from_config(config),
        )
        retriever = LongTermRetriever(
            config=config,
            prompt_context_store=store,
            graph_store=graph,
            object_store=object_store,
            midterm_store=midterm_store,
            conflict_resolver=conflict_resolver,
            subtext_builder=subtext_builder,
            adaptive_policy_builder=LongTermAdaptivePolicyBuilder(
                proactive_state_store=proactive_state_store,
            ),
        )
        fast_topic_builder = LongTermFastTopicContextBuilder(
            config=config,
            object_store=object_store,
        )
        prepared_context_front = PreparedLongTermContextFront() if config.long_term_memory_enabled else None
        provider_answer_front = (
            MaterializedProviderAnswerFront(
                LongTermProviderAnswerFrontStore(remote_state),
            )
            if config.long_term_memory_enabled and remote_state is not None
            else None
        )
        store_lock = threading.RLock()
        queue_size = _coerce_positive_int(
            getattr(config, "long_term_memory_write_queue_size", 1),
            default=1,
            maximum=_MAX_QUEUE_SIZE,
        )
        service = factory(
            config=config,
            prompt_context_store=store,
            graph_store=graph,
            object_store=object_store,
            midterm_store=midterm_store,
            query_rewriter=LongTermQueryRewriter.from_config(config),
            fast_topic_builder=fast_topic_builder,
            retriever=retriever,
            extractor=extractor,
            multimodal_extractor=multimodal_extractor,
            truth_maintainer=truth_maintainer,
            consolidator=consolidator,
            conflict_resolver=conflict_resolver,
            reflector=reflector,
            turn_continuity_compiler=turn_continuity_compiler,
            sensor_memory=sensor_memory,
            ops_backfiller=ops_backfiller,
            planner=planner,
            proactive_policy=proactive_policy,
            retention_policy=retention_policy,
            prepared_context_front=prepared_context_front,
            provider_answer_front=provider_answer_front,
            restart_recall_policy_compiler=LongTermRestartRecallPolicyCompiler(),
            personality_learning=personality_learning,
            writer=None,
            multimodal_writer=None,
            _store_lock=store_lock,
        )
        if config.long_term_memory_enabled and config.long_term_memory_background_store_turns:
            # Build the service first so background writers receive the bound
            # invalidator instead of an unbound mixin method.
            def _write_turn(item: LongTermConversationTurn) -> None:
                service._persist_longterm_turn(
                    config=config,
                    store=store,
                    graph_store=graph,
                    object_store=object_store,
                    midterm_store=midterm_store,
                    extractor=extractor,
                    consolidator=consolidator,
                    reflector=reflector,
                    turn_continuity_compiler=turn_continuity_compiler,
                    sensor_memory=sensor_memory,
                    retention_policy=retention_policy,
                    personality_learning=personality_learning,
                    prepared_context_invalidator=service._invalidate_prepared_contexts,
                    store_lock=store_lock,
                    timezone_name=config.local_timezone_name,
                    item=item,
                )

            service.writer = AsyncLongTermMemoryWriter(
                write_callback=_write_turn,
                max_queue_size=queue_size,
            )
            service.multimodal_writer = AsyncLongTermMultimodalWriter(
                write_callback=lambda item: service._persist_multimodal_evidence(
                    object_store=object_store,
                    midterm_store=midterm_store,
                    multimodal_extractor=multimodal_extractor,
                    consolidator=consolidator,
                    reflector=reflector,
                    sensor_memory=sensor_memory,
                    retention_policy=retention_policy,
                    prepared_context_invalidator=service._invalidate_prepared_contexts,
                    store_lock=store_lock,
                    timezone_name=config.local_timezone_name,
                    item=item,
                ),
                max_queue_size=queue_size,
            )
        return service
