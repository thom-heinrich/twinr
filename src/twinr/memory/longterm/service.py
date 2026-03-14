from __future__ import annotations

from dataclasses import dataclass
import json

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.context_store import ManagedContextEntry, PersistentMemoryEntry, PromptContextStore
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.query_normalization import LongTermQueryRewriter
from twinr.memory.longterm.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.conflicts import LongTermConflictResolver
from twinr.memory.longterm.extract import LongTermTurnExtractor
from twinr.memory.longterm.multimodal import LongTermMultimodalExtractor
from twinr.memory.longterm.models import (
    LongTermConsolidationResultV1,
    LongTermConflictQueueItemV1,
    LongTermConflictResolutionV1,
    LongTermConversationTurn,
    LongTermEnqueueResult,
    LongTermMemoryContext,
    LongTermMemoryMutationResultV1,
    LongTermMemoryReviewResultV1,
    LongTermMultimodalEvidence,
    LongTermProactivePlanV1,
    LongTermRetentionResultV1,
    LongTermReflectionResultV1,
)
from twinr.memory.longterm.planner import LongTermProactivePlanner
from twinr.memory.longterm.proactive import LongTermProactivePolicy, LongTermProactiveReservationV1, LongTermProactiveStateStore
from twinr.memory.longterm.reflect import LongTermMemoryReflector
from twinr.memory.longterm.retriever import LongTermRetriever
from twinr.memory.longterm.retention import LongTermRetentionPolicy
from twinr.memory.longterm.store import LongTermStructuredStore
from twinr.memory.longterm.subtext import LongTermSubtextBuilder
from twinr.memory.longterm.truth import LongTermTruthMaintainer
from twinr.memory.longterm.worker import AsyncLongTermMemoryWriter, AsyncLongTermMultimodalWriter


def _normalize_text(value: str, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


@dataclass(slots=True)
class LongTermMemoryService:
    config: TwinrConfig
    prompt_context_store: PromptContextStore
    graph_store: TwinrPersonalGraphStore
    object_store: LongTermStructuredStore
    query_rewriter: LongTermQueryRewriter
    retriever: LongTermRetriever
    extractor: LongTermTurnExtractor
    multimodal_extractor: LongTermMultimodalExtractor
    truth_maintainer: LongTermTruthMaintainer
    consolidator: LongTermMemoryConsolidator
    conflict_resolver: LongTermConflictResolver
    reflector: LongTermMemoryReflector
    planner: LongTermProactivePlanner
    proactive_policy: LongTermProactivePolicy
    retention_policy: LongTermRetentionPolicy
    writer: AsyncLongTermMemoryWriter | None = None
    multimodal_writer: AsyncLongTermMultimodalWriter | None = None

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        graph_store: TwinrPersonalGraphStore | None = None,
        prompt_context_store: PromptContextStore | None = None,
        extractor: LongTermTurnExtractor | None = None,
    ) -> "LongTermMemoryService":
        store = prompt_context_store or PromptContextStore.from_config(config)
        graph = graph_store or TwinrPersonalGraphStore.from_config(config)
        object_store = LongTermStructuredStore.from_config(config)
        extractor = extractor or LongTermTurnExtractor.from_config(config)
        multimodal_extractor = LongTermMultimodalExtractor(timezone_name=config.local_timezone_name)
        truth_maintainer = LongTermTruthMaintainer()
        consolidator = LongTermMemoryConsolidator(truth_maintainer=truth_maintainer)
        conflict_resolver = LongTermConflictResolver()
        reflector = LongTermMemoryReflector()
        planner = LongTermProactivePlanner(timezone_name=config.local_timezone_name)
        proactive_state_store = LongTermProactiveStateStore.from_config(config)
        proactive_policy = LongTermProactivePolicy(
            config=config,
            state_store=proactive_state_store,
        )
        retention_policy = LongTermRetentionPolicy(timezone_name=config.local_timezone_name)
        subtext_builder = LongTermSubtextBuilder(config=config, graph_store=graph)
        retriever = LongTermRetriever(
            config=config,
            prompt_context_store=store,
            graph_store=graph,
            object_store=object_store,
            conflict_resolver=conflict_resolver,
            subtext_builder=subtext_builder,
        )
        writer: AsyncLongTermMemoryWriter | None = None
        multimodal_writer: AsyncLongTermMultimodalWriter | None = None
        if config.long_term_memory_enabled and config.long_term_memory_background_store_turns:
            writer = AsyncLongTermMemoryWriter(
                write_callback=lambda item: cls._persist_longterm_turn(
                    store=store,
                    object_store=object_store,
                    extractor=extractor,
                    consolidator=consolidator,
                    reflector=reflector,
                    retention_policy=retention_policy,
                    item=item,
                ),
                max_queue_size=config.long_term_memory_write_queue_size,
            )
            multimodal_writer = AsyncLongTermMultimodalWriter(
                write_callback=lambda item: cls._persist_multimodal_evidence(
                    object_store=object_store,
                    multimodal_extractor=multimodal_extractor,
                    consolidator=consolidator,
                    reflector=reflector,
                    retention_policy=retention_policy,
                    item=item,
                ),
                max_queue_size=config.long_term_memory_write_queue_size,
            )
        return cls(
            config=config,
            prompt_context_store=store,
            graph_store=graph,
            object_store=object_store,
            query_rewriter=LongTermQueryRewriter.from_config(config),
            retriever=retriever,
            extractor=extractor,
            multimodal_extractor=multimodal_extractor,
            truth_maintainer=truth_maintainer,
            consolidator=consolidator,
            conflict_resolver=conflict_resolver,
            reflector=reflector,
            planner=planner,
            proactive_policy=proactive_policy,
            retention_policy=retention_policy,
            writer=writer,
            multimodal_writer=multimodal_writer,
        )

    def build_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        query = self.query_rewriter.profile(query_text)
        return self.retriever.build_context(
            query=query,
            original_query_text=query_text,
        )

    def enqueue_conversation_turn(
        self,
        *,
        transcript: str,
        response: str,
        source: str = "conversation",
    ) -> LongTermEnqueueResult | None:
        clean_transcript = _normalize_text(transcript, limit=320)
        clean_response = _normalize_text(response, limit=320)
        if not clean_transcript or not clean_response:
            return None
        if self.writer is None:
            return None
        return self.writer.enqueue(
            LongTermConversationTurn(
                transcript=clean_transcript,
                response=clean_response,
                source=source,
            )
        )

    def analyze_conversation_turn(
        self,
        *,
        transcript: str,
        response: str,
    ) -> LongTermConsolidationResultV1:
        extraction = self.extractor.extract_conversation_turn(
            transcript=transcript,
            response=response,
        )
        return self.consolidator.consolidate(
            extraction=extraction,
            existing_objects=self.object_store.load_objects(),
        )

    def analyze_multimodal_evidence(
        self,
        *,
        event_name: str,
        modality: str,
        source: str = "device_event",
        message: str | None = None,
        data: dict[str, object] | None = None,
    ) -> LongTermConsolidationResultV1:
        extraction = self.multimodal_extractor.extract_evidence(
            LongTermMultimodalEvidence(
                event_name=event_name,
                modality=modality,
                source=source,
                message=message,
                data=dict(data or {}),
            )
        )
        return self.consolidator.consolidate(
            extraction=extraction,
            existing_objects=self.object_store.load_objects(),
        )

    def enqueue_multimodal_evidence(
        self,
        *,
        event_name: str,
        modality: str,
        source: str = "device_event",
        message: str | None = None,
        data: dict[str, object] | None = None,
    ) -> LongTermEnqueueResult | None:
        if self.multimodal_writer is None:
            return None
        return self.multimodal_writer.enqueue(
            LongTermMultimodalEvidence(
                event_name=event_name,
                modality=modality,
                source=source,
                message=message,
                data=dict(data or {}),
            )
        )

    def run_reflection(self) -> LongTermReflectionResultV1:
        result = self.reflector.reflect(objects=self.object_store.load_objects())
        self.object_store.apply_reflection(result)
        return result

    def plan_proactive_candidates(self) -> LongTermProactivePlanV1:
        return self.planner.plan(objects=self.object_store.load_objects())

    def reserve_proactive_candidate(
        self,
        *,
        now: datetime | None = None,
    ) -> LongTermProactiveReservationV1 | None:
        plan = self.plan_proactive_candidates()
        return self.proactive_policy.reserve_candidate(plan=plan, now=now)

    def reserve_specific_proactive_candidate(
        self,
        candidate: LongTermProactiveCandidateV1,
        *,
        now: datetime | None = None,
    ) -> LongTermProactiveReservationV1:
        return self.proactive_policy.reserve_specific_candidate(candidate, now=now)

    def preview_proactive_candidate(
        self,
        *,
        now: datetime | None = None,
    ) -> LongTermProactiveCandidateV1 | None:
        plan = self.plan_proactive_candidates()
        return self.proactive_policy.preview_candidate(plan=plan, now=now)

    def mark_proactive_candidate_delivered(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        delivered_at: datetime | None = None,
        prompt_text: str | None = None,
    ):
        return self.proactive_policy.mark_delivered(
            reservation,
            delivered_at=delivered_at,
            prompt_text=prompt_text,
        )

    def mark_proactive_candidate_skipped(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        reason: str,
        skipped_at: datetime | None = None,
    ):
        return self.proactive_policy.mark_skipped(
            reservation,
            reason=reason,
            skipped_at=skipped_at,
        )

    def run_retention(self) -> LongTermRetentionResultV1:
        result = self.retention_policy.apply(objects=self.object_store.load_objects())
        self.object_store.apply_retention(result)
        return result

    def select_conflict_queue(
        self,
        query_text: str | None,
        *,
        limit: int | None = None,
    ) -> tuple[LongTermConflictQueueItemV1, ...]:
        query = self.query_rewriter.profile(query_text)
        return self.retriever.select_conflict_queue(
            query=query,
            limit=limit,
        )

    def resolve_conflict(
        self,
        *,
        slot_key: str,
        selected_memory_id: str,
    ) -> LongTermConflictResolutionV1:
        conflicts = self.object_store.load_conflicts()
        conflict = next((item for item in conflicts if item.slot_key == slot_key), None)
        if conflict is None:
            raise ValueError(f"No open long-term memory conflict found for slot {slot_key!r}.")
        result = self.conflict_resolver.resolve(
            conflict=conflict,
            objects=self.object_store.load_objects(),
            remaining_conflicts=conflicts,
            selected_memory_id=selected_memory_id,
        )
        self.object_store.apply_conflict_resolution(result)
        return result

    def review_memory(
        self,
        *,
        query_text: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        include_episodes: bool = False,
        limit: int = 12,
    ) -> LongTermMemoryReviewResultV1:
        return self.object_store.review_objects(
            query_text=query_text,
            status=status,
            kind=kind,
            include_episodes=include_episodes,
            limit=limit,
        )

    def confirm_memory(self, *, memory_id: str) -> LongTermMemoryMutationResultV1 | LongTermConflictResolutionV1:
        conflicts = self.object_store.load_conflicts()
        conflict = next(
            (
                item
                for item in conflicts
                if memory_id == item.candidate_memory_id or memory_id in item.existing_memory_ids
            ),
            None,
        )
        if conflict is not None:
            result = self.conflict_resolver.resolve(
                conflict=conflict,
                objects=self.object_store.load_objects(),
                remaining_conflicts=conflicts,
                selected_memory_id=memory_id,
            )
            self.object_store.apply_conflict_resolution(result)
            return result
        result = self.object_store.confirm_object(memory_id)
        self.object_store.apply_memory_mutation(result)
        return result

    def invalidate_memory(
        self,
        *,
        memory_id: str,
        reason: str | None = None,
    ) -> LongTermMemoryMutationResultV1:
        result = self.object_store.invalidate_object(memory_id, reason=reason)
        self.object_store.apply_memory_mutation(result)
        return result

    def delete_memory(self, *, memory_id: str) -> LongTermMemoryMutationResultV1:
        result = self.object_store.delete_object(memory_id)
        self.object_store.apply_memory_mutation(result)
        return result

    def store_explicit_memory(
        self,
        *,
        kind: str,
        summary: str,
        details: str | None = None,
    ) -> PersistentMemoryEntry:
        return self.prompt_context_store.memory_store.remember(
            kind=kind,
            summary=summary,
            details=details,
        )

    def update_user_profile(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        return self.prompt_context_store.user_store.upsert(
            category=category,
            instruction=instruction,
        )

    def update_personality(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        return self.prompt_context_store.personality_store.upsert(
            category=category,
            instruction=instruction,
        )

    def flush(self, *, timeout_s: float = 2.0) -> bool:
        writer_ok = True if self.writer is None else self.writer.flush(timeout_s=timeout_s)
        multimodal_ok = True if self.multimodal_writer is None else self.multimodal_writer.flush(timeout_s=timeout_s)
        return writer_ok and multimodal_ok

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        if self.writer is not None:
            self.writer.shutdown(timeout_s=timeout_s)
        if self.multimodal_writer is not None:
            self.multimodal_writer.shutdown(timeout_s=timeout_s)

    @staticmethod
    def _persist_longterm_turn(
        *,
        store: PromptContextStore,
        object_store: LongTermStructuredStore,
        extractor: LongTermTurnExtractor,
        consolidator: LongTermMemoryConsolidator,
        reflector: LongTermMemoryReflector,
        retention_policy: LongTermRetentionPolicy,
        item: LongTermConversationTurn,
    ) -> PersistentMemoryEntry:
        extraction = extractor.extract_conversation_turn(
            transcript=item.transcript,
            response=item.response,
            occurred_at=item.created_at,
        )
        result = consolidator.consolidate(
            extraction=extraction,
            existing_objects=object_store.load_objects(),
        )
        object_store.apply_consolidation(result)
        object_store.apply_reflection(reflector.reflect(objects=object_store.load_objects()))
        object_store.apply_retention(retention_policy.apply(objects=object_store.load_objects()))
        return LongTermMemoryService._persist_episodic_turn(store=store, item=item)

    @staticmethod
    def _persist_multimodal_evidence(
        *,
        object_store: LongTermStructuredStore,
        multimodal_extractor: LongTermMultimodalExtractor,
        consolidator: LongTermMemoryConsolidator,
        reflector: LongTermMemoryReflector,
        retention_policy: LongTermRetentionPolicy,
        item: LongTermMultimodalEvidence,
    ) -> None:
        extraction = multimodal_extractor.extract_evidence(item)
        result = consolidator.consolidate(
            extraction=extraction,
            existing_objects=object_store.load_objects(),
        )
        object_store.apply_consolidation(result)
        object_store.apply_reflection(reflector.reflect(objects=object_store.load_objects()))
        object_store.apply_retention(retention_policy.apply(objects=object_store.load_objects()))

    @staticmethod
    def _persist_episodic_turn(*, store: PromptContextStore, item: LongTermConversationTurn) -> PersistentMemoryEntry:
        quoted_transcript = json.dumps(item.transcript, ensure_ascii=False)
        quoted_response = json.dumps(item.response, ensure_ascii=False)
        return store.memory_store.remember(
            kind="episodic_turn",
            summary=f"Conversation about {quoted_transcript}",
            details=f"User said: {quoted_transcript} Twinr answered: {quoted_response}",
        )
