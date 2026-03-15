from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
import json

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.backfill import (
    LongTermOpsBackfillRunResult,
    LongTermOpsEventBackfiller,
)
from twinr.memory.context_store import ManagedContextEntry, PersistentMemoryEntry, PromptContextStore
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.query_normalization import LongTermQueryRewriter
from twinr.memory.longterm.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.conflicts import LongTermConflictResolver
from twinr.memory.longterm.extract import LongTermTurnExtractor
from twinr.memory.longterm.multimodal import LongTermMultimodalExtractor
from twinr.memory.longterm.ontology import kind_matches
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
from twinr.memory.longterm.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.planner import LongTermProactivePlanner
from twinr.memory.longterm.proactive import LongTermProactivePolicy, LongTermProactiveReservationV1, LongTermProactiveStateStore
from twinr.memory.longterm.reflect import LongTermMemoryReflector
from twinr.memory.longterm.retriever import LongTermRetriever
from twinr.memory.longterm.retention import LongTermRetentionPolicy
from twinr.memory.longterm.remote_state import LongTermRemoteUnavailableError
from twinr.memory.longterm.sensor_memory import LongTermSensorMemoryCompiler
from twinr.memory.longterm.store import LongTermStructuredStore
from twinr.memory.longterm.subtext import LongTermSubtextBuilder, LongTermSubtextCompiler
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
        midterm_store = LongTermMidtermStore.from_config(config)
        extractor = extractor or LongTermTurnExtractor.from_config(config)
        multimodal_extractor = LongTermMultimodalExtractor(timezone_name=config.local_timezone_name)
        truth_maintainer = LongTermTruthMaintainer()
        consolidator = LongTermMemoryConsolidator(truth_maintainer=truth_maintainer)
        conflict_resolver = LongTermConflictResolver()
        reflector = LongTermMemoryReflector.from_config(config)
        sensor_memory = LongTermSensorMemoryCompiler.from_config(config)
        ops_backfiller = LongTermOpsEventBackfiller()
        planner = LongTermProactivePlanner(timezone_name=config.local_timezone_name)
        proactive_state_store = LongTermProactiveStateStore.from_config(config)
        proactive_policy = LongTermProactivePolicy(
            config=config,
            state_store=proactive_state_store,
        )
        retention_policy = LongTermRetentionPolicy(
            timezone_name=config.local_timezone_name,
            mode=config.long_term_memory_retention_mode,
            archive_enabled=config.long_term_memory_archive_enabled,
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
        )
        writer: AsyncLongTermMemoryWriter | None = None
        multimodal_writer: AsyncLongTermMultimodalWriter | None = None
        if config.long_term_memory_enabled and config.long_term_memory_background_store_turns:
            writer = AsyncLongTermMemoryWriter(
                write_callback=lambda item: cls._persist_longterm_turn(
                    config=config,
                    store=store,
                    graph_store=graph,
                    object_store=object_store,
                    midterm_store=midterm_store,
                    extractor=extractor,
                    consolidator=consolidator,
                    reflector=reflector,
                    sensor_memory=sensor_memory,
                    retention_policy=retention_policy,
                    item=item,
                ),
                max_queue_size=config.long_term_memory_write_queue_size,
            )
            multimodal_writer = AsyncLongTermMultimodalWriter(
                write_callback=lambda item: cls._persist_multimodal_evidence(
                    object_store=object_store,
                    midterm_store=midterm_store,
                    multimodal_extractor=multimodal_extractor,
                    consolidator=consolidator,
                    reflector=reflector,
                    sensor_memory=sensor_memory,
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
            midterm_store=midterm_store,
            query_rewriter=LongTermQueryRewriter.from_config(config),
            retriever=retriever,
            extractor=extractor,
            multimodal_extractor=multimodal_extractor,
            truth_maintainer=truth_maintainer,
            consolidator=consolidator,
            conflict_resolver=conflict_resolver,
            reflector=reflector,
            sensor_memory=sensor_memory,
            ops_backfiller=ops_backfiller,
            planner=planner,
            proactive_policy=proactive_policy,
            retention_policy=retention_policy,
            writer=writer,
            multimodal_writer=multimodal_writer,
        )

    def build_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        try:
            query = self.query_rewriter.profile(query_text)
            return self.retriever.build_context(
                query=query,
                original_query_text=query_text,
            )
        except LongTermRemoteUnavailableError:
            return LongTermMemoryContext()

    def build_tool_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        try:
            query = self.query_rewriter.profile(query_text)
            context = self.retriever.build_context(
                query=query,
                original_query_text=query_text,
            )
            conflict_queue = self.select_conflict_queue(query_text=query.retrieval_text)
            conflicting_memory_ids = {
                option.memory_id
                for item in conflict_queue
                for option in item.options
            }
            durable_objects = self.object_store.select_relevant_objects(
                query_text=query.retrieval_text,
                limit=max(1, self.config.long_term_memory_recall_limit),
            )
            filtered_durable_objects = tuple(
                item
                for item in durable_objects
                if not kind_matches(item.kind, "fact", item.attributes, attr_key="fact_type", attr_value="contact_method")
                and item.memory_id not in conflicting_memory_ids
            )
            return LongTermMemoryContext(
                subtext_context=context.subtext_context,
                midterm_context=context.midterm_context,
                durable_context=self.retriever._render_durable_context(filtered_durable_objects),
                episodic_context=context.episodic_context,
                graph_context=self.graph_store.build_prompt_context(
                    query.retrieval_text,
                    include_contact_methods=False,
                ),
                conflict_context=None,
            )
        except LongTermRemoteUnavailableError:
            return LongTermMemoryContext()

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
        try:
            result = self.reflector.reflect(objects=self.object_store.load_objects())
        except LongTermRemoteUnavailableError:
            return LongTermReflectionResultV1(reflected_objects=(), created_summaries=(), midterm_packets=())
        self.object_store.apply_reflection(result)
        self.midterm_store.apply_reflection(result)
        sensor_memory_result = self.run_sensor_memory()
        if sensor_memory_result.created_summaries or sensor_memory_result.reflected_objects:
            return LongTermReflectionResultV1(
                reflected_objects=tuple((*result.reflected_objects, *sensor_memory_result.reflected_objects)),
                created_summaries=tuple((*result.created_summaries, *sensor_memory_result.created_summaries)),
                midterm_packets=result.midterm_packets,
            )
        return result

    def run_sensor_memory(self, *, now: datetime | None = None) -> LongTermReflectionResultV1:
        try:
            result = self.sensor_memory.compile(objects=self.object_store.load_objects(), now=now)
        except LongTermRemoteUnavailableError:
            return LongTermReflectionResultV1(reflected_objects=(), created_summaries=(), midterm_packets=())
        if result.created_summaries or result.reflected_objects:
            self.object_store.apply_reflection(result)
        return result

    def backfill_ops_multimodal_history(
        self,
        *,
        entries: Iterable[Mapping[str, object]] | None = None,
        now: datetime | None = None,
    ) -> LongTermOpsBackfillRunResult:
        if entries is None:
            from twinr.ops.events import TwinrOpsEventStore

        raw_entries = (
            tuple(entries)
            if entries is not None
            else self.ops_backfiller.load_entries(TwinrOpsEventStore.from_config(self.config).path)
        )
        build = self.ops_backfiller.build_evidence(raw_entries)
        objects_by_id = {item.memory_id: item for item in self.object_store.load_objects()}
        conflicts_by_slot = {item.slot_key: item for item in self.object_store.load_conflicts()}
        seen_turn_ids = {
            event_id
            for item in objects_by_id.values()
            for event_id in item.source.event_ids
        }
        applied_evidence = 0
        skipped_existing = 0
        for evidence in build.evidence:
            extraction = self.multimodal_extractor.extract_evidence(evidence)
            if extraction.turn_id in seen_turn_ids:
                skipped_existing += 1
                continue
            result = self.consolidator.consolidate(
                extraction=extraction,
                existing_objects=tuple(objects_by_id.values()),
            )
            for item in (*result.episodic_objects, *result.durable_objects, *result.deferred_objects):
                objects_by_id[item.memory_id] = self.object_store._merge_object(
                    existing=objects_by_id.get(item.memory_id),
                    incoming=item,
                    increment_support=True,
                )
            for conflict in result.conflicts:
                conflicts_by_slot[conflict.slot_key] = conflict
            seen_turn_ids.add(extraction.turn_id)
            applied_evidence += 1

        reflected_objects = 0
        created_summaries = 0
        reflection_error: str | None = None
        if applied_evidence:
            self.object_store.write_snapshot(
                objects=tuple(objects_by_id.values()),
                conflicts=tuple(conflicts_by_slot.values()),
            )
            try:
                reflection = self.reflector.reflect(objects=self.object_store.load_objects())
            except Exception as exc:
                reflection_error = f"{type(exc).__name__}: {exc}"
            else:
                self.object_store.apply_reflection(reflection)
                self.midterm_store.apply_reflection(reflection)
                reflected_objects += len(reflection.reflected_objects)
                created_summaries += len(reflection.created_summaries)
        sensor_result = self.run_sensor_memory(now=now)
        reflected_objects += len(sensor_result.reflected_objects)
        created_summaries += len(sensor_result.created_summaries)
        self.object_store.apply_retention(self.retention_policy.apply(objects=self.object_store.load_objects()))
        return LongTermOpsBackfillRunResult(
            scanned_events=build.scanned_events,
            generated_evidence=build.generated_evidence,
            applied_evidence=applied_evidence,
            skipped_existing=skipped_existing,
            sensor_observations=build.sensor_observations,
            button_interactions=build.button_interactions,
            print_completions=build.print_completions,
            reflected_objects=reflected_objects,
            created_summaries=created_summaries,
            reflection_error=reflection_error,
        )

    def plan_proactive_candidates(
        self,
        *,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactivePlanV1:
        try:
            return self.planner.plan(
                objects=self.object_store.load_objects(),
                live_facts=live_facts,
            )
        except LongTermRemoteUnavailableError:
            return LongTermProactivePlanV1(candidates=())

    def reserve_proactive_candidate(
        self,
        *,
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactiveReservationV1 | None:
        plan = self.plan_proactive_candidates(live_facts=live_facts)
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
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactiveCandidateV1 | None:
        plan = self.plan_proactive_candidates(live_facts=live_facts)
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
        try:
            result = self.retention_policy.apply(objects=self.object_store.load_objects())
            self.object_store.apply_retention(result)
            return result
        except LongTermRemoteUnavailableError:
            return LongTermRetentionResultV1(
                kept_objects=(),
                expired_objects=(),
                pruned_memory_ids=(),
                archived_objects=(),
            )

    def select_conflict_queue(
        self,
        query_text: str | None,
        *,
        limit: int | None = None,
    ) -> tuple[LongTermConflictQueueItemV1, ...]:
        try:
            query = self.query_rewriter.profile(query_text)
            return self.retriever.select_conflict_queue(
                query=query,
                limit=limit,
            )
        except LongTermRemoteUnavailableError:
            return ()

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
        config: TwinrConfig,
        store: PromptContextStore,
        graph_store: TwinrPersonalGraphStore,
        object_store: LongTermStructuredStore,
        midterm_store: LongTermMidtermStore,
        extractor: LongTermTurnExtractor,
        consolidator: LongTermMemoryConsolidator,
        reflector: LongTermMemoryReflector,
        sensor_memory: LongTermSensorMemoryCompiler,
        retention_policy: LongTermRetentionPolicy,
        item: LongTermConversationTurn,
    ) -> PersistentMemoryEntry:
        try:
            existing_objects = tuple(object_store.load_objects())
            existing_conflicts = tuple(object_store.load_conflicts())
            existing_archived = tuple(object_store.load_archived_objects())
            extraction = extractor.extract_conversation_turn(
                transcript=item.transcript,
                response=item.response,
                occurred_at=item.created_at,
            )
            result = consolidator.consolidate(
                extraction=extraction,
                existing_objects=existing_objects,
            )
            current_objects, current_conflicts = LongTermMemoryService._merge_consolidation_state(
                object_store=object_store,
                existing_objects=existing_objects,
                existing_conflicts=existing_conflicts,
                result=result,
            )
            graph_store.apply_candidate_edges(result.graph_edges)
            reflection = reflector.reflect(objects=current_objects)
            current_objects = LongTermMemoryService._merge_reflection_objects(
                object_store=object_store,
                current_objects=current_objects,
                reflection=reflection,
            )
            midterm_store.apply_reflection(reflection)
            sensor_reflection = sensor_memory.compile(objects=current_objects, now=item.created_at)
            current_objects = LongTermMemoryService._merge_reflection_objects(
                object_store=object_store,
                current_objects=current_objects,
                reflection=sensor_reflection,
            )
            retention = retention_policy.apply(objects=current_objects)
            archived = {item.memory_id: item for item in existing_archived}
            for archived_item in retention.archived_objects:
                archived[archived_item.memory_id] = archived_item
            object_store.write_snapshot(
                objects=retention.kept_objects,
                conflicts=current_conflicts,
                archived_objects=tuple(sorted(archived.values(), key=lambda row: row.memory_id)),
            )
            if config.long_term_memory_mode == "remote_primary":
                return None
        except LongTermRemoteUnavailableError:
            pass
        return LongTermMemoryService._persist_episodic_turn(store=store, item=item)

    @staticmethod
    def _persist_multimodal_evidence(
        *,
        object_store: LongTermStructuredStore,
        midterm_store: LongTermMidtermStore,
        multimodal_extractor: LongTermMultimodalExtractor,
        consolidator: LongTermMemoryConsolidator,
        reflector: LongTermMemoryReflector,
        sensor_memory: LongTermSensorMemoryCompiler,
        retention_policy: LongTermRetentionPolicy,
        item: LongTermMultimodalEvidence,
    ) -> None:
        try:
            extraction = multimodal_extractor.extract_evidence(item)
            result = consolidator.consolidate(
                extraction=extraction,
                existing_objects=object_store.load_objects(),
            )
            object_store.apply_consolidation(result)
            reflection = reflector.reflect(objects=object_store.load_objects())
            object_store.apply_reflection(reflection)
            midterm_store.apply_reflection(reflection)
            object_store.apply_reflection(sensor_memory.compile(objects=object_store.load_objects(), now=item.created_at))
            object_store.apply_retention(retention_policy.apply(objects=object_store.load_objects()))
        except LongTermRemoteUnavailableError:
            return None

    @staticmethod
    def _persist_episodic_turn(*, store: PromptContextStore, item: LongTermConversationTurn) -> PersistentMemoryEntry:
        quoted_transcript = json.dumps(item.transcript, ensure_ascii=False)
        quoted_response = json.dumps(item.response, ensure_ascii=False)
        local_memory_store = type(store.memory_store)(store.memory_store.path)
        return local_memory_store.remember(
            kind="episodic_turn",
            summary=f"Conversation about {quoted_transcript}",
            details=f"User said: {quoted_transcript} Twinr answered: {quoted_response}",
        )

    @staticmethod
    def _merge_consolidation_state(
        *,
        object_store: LongTermStructuredStore,
        existing_objects: tuple,
        existing_conflicts: tuple,
        result: LongTermConsolidationResultV1,
    ) -> tuple[tuple, tuple]:
        merged_objects = {item.memory_id: item for item in existing_objects}
        for item in (*result.episodic_objects, *result.durable_objects, *result.deferred_objects):
            merged_objects[item.memory_id] = object_store._merge_object(
                existing=merged_objects.get(item.memory_id),
                incoming=item,
                increment_support=True,
            )
        merged_conflicts = {item.slot_key: item for item in existing_conflicts}
        for conflict in result.conflicts:
            merged_conflicts[conflict.slot_key] = conflict
        return (
            tuple(sorted(merged_objects.values(), key=lambda row: row.memory_id)),
            tuple(sorted(merged_conflicts.values(), key=lambda row: (row.slot_key, row.candidate_memory_id))),
        )

    @staticmethod
    def _merge_reflection_objects(
        *,
        object_store: LongTermStructuredStore,
        current_objects: tuple,
        reflection: LongTermReflectionResultV1,
    ) -> tuple:
        merged = {item.memory_id: item for item in current_objects}
        for item in (*reflection.reflected_objects, *reflection.created_summaries):
            merged[item.memory_id] = object_store._merge_object(
                existing=merged.get(item.memory_id),
                incoming=item,
                increment_support=False,
            )
        return tuple(sorted(merged.values(), key=lambda row: row.memory_id))
