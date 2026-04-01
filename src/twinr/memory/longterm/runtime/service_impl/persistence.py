# mypy: disable-error-code=attr-defined
"""Persistence helpers for the long-term runtime service."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any
import json
import threading

from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.context_store import PersistentMemoryEntry, PromptContextStore
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
    LongTermMultimodalEvidence,
    LongTermReflectionResultV1,
    LongTermRetentionResultV1,
)
from twinr.memory.longterm.core.ontology import kind_matches
from twinr.memory.longterm.ingestion.extract import LongTermTurnExtractor
from twinr.memory.longterm.ingestion.multimodal import LongTermMultimodalExtractor
from twinr.memory.longterm.ingestion.sensor_memory import LongTermSensorMemoryCompiler
from twinr.memory.longterm.reasoning.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.reasoning.reflect import LongTermMemoryReflector
from twinr.memory.longterm.reasoning.retention import LongTermRetentionPolicy
from twinr.memory.longterm.reasoning.turn_continuity import LongTermTurnContinuityCompiler
from twinr.memory.longterm.runtime.live_object_selectors import (
    select_reflection_neighborhood_objects,
    select_sensor_memory_neighborhood_objects,
)
from twinr.memory.longterm.storage.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.memory.longterm.storage.store import LongTermStructuredStore

from ._typing import ServiceMixinBase
from .compat import (
    _MULTIMODAL_MESSAGE_LIMIT,
    _MULTIMODAL_NAME_LIMIT,
    _SOURCE_LIMIT,
    _normalize_datetime,
    _normalize_text,
    _sanitize_jsonish,
    _sort_conflicts,
    _sort_objects_by_memory_id,
    _validate_regular_file_path,
    logger,
)

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig
    from twinr.agent.personality.learning import PersonalityLearningService


class LongTermMemoryServicePersistenceMixin(ServiceMixinBase):
    """Persist conversation turns and multimodal evidence through long-term stores."""

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
        turn_continuity_compiler: LongTermTurnContinuityCompiler | None = None,
        sensor_memory: LongTermSensorMemoryCompiler,
        retention_policy: LongTermRetentionPolicy,
        personality_learning: PersonalityLearningService | None = None,
        prepared_context_invalidator: Callable[..., None] | None = None,
        store_lock: threading.RLock | None = None,
        timezone_name: str | None = None,
        item: LongTermConversationTurn,
        episode_attributes: Mapping[str, object] | None = None,
    ) -> PersistentMemoryEntry | None:
        """Persist one conversation turn through the full long-term pipeline."""

        effective_store_lock = store_lock or threading.RLock()
        effective_timezone_name = timezone_name or config.local_timezone_name
        effective_turn_continuity_compiler = turn_continuity_compiler or LongTermTurnContinuityCompiler()
        occurred_at = _normalize_datetime(item.created_at, timezone_name=effective_timezone_name) or item.created_at
        reflection = LongTermMemoryServicePersistenceMixin._empty_reflection_result()
        sensor_reflection = LongTermMemoryServicePersistenceMixin._empty_reflection_result()

        try:
            with effective_store_lock:
                continuity_packet = effective_turn_continuity_compiler.compile_packet(turn=item)
                if continuity_packet is not None:
                    midterm_store.save_packets_preserving_attribute(
                        packets=(continuity_packet,),
                        attribute_key="persistence_scope",
                        attribute_value="restart_recall",
                    )
                extraction = extractor.extract_conversation_turn(
                    transcript=item.transcript,
                    response=item.response,
                    occurred_at=occurred_at,
                    source=item.source,
                    modality=item.modality,
                    episode_attributes=episode_attributes,
                )
                working_set = object_store.load_active_working_set(
                    candidate_objects=(extraction.episode, *extraction.candidate_objects),
                    event_ids=(extraction.turn_id,),
                )
                existing_objects = working_set.objects
                existing_conflicts = working_set.conflicts
                result = consolidator.consolidate(
                    extraction=extraction,
                    existing_objects=existing_objects,
                )
                current_objects, current_conflicts = LongTermMemoryServicePersistenceMixin._merge_consolidation_state(
                    object_store=object_store,
                    existing_objects=existing_objects,
                    existing_conflicts=existing_conflicts,
                    result=result,
                )
                try:
                    reflection = reflector.reflect(
                        objects=tuple(
                            select_reflection_neighborhood_objects(
                                object_store,
                                seed_objects=current_objects,
                            )
                        )
                    )
                except Exception:
                    logger.exception("Long-term reflection failed during conversation-turn persistence.")
                else:
                    current_objects = LongTermMemoryServicePersistenceMixin._merge_reflection_objects(
                        object_store=object_store,
                        current_objects=current_objects,
                        reflection=reflection,
                    )
                try:
                    sensor_reflection = sensor_memory.compile(
                        objects=tuple(
                            select_sensor_memory_neighborhood_objects(
                                object_store,
                                seed_objects=current_objects,
                            )
                        ),
                        now=occurred_at,
                    )
                except Exception:
                    logger.exception("Sensor-memory compilation failed during conversation-turn persistence.")
                else:
                    if LongTermMemoryServicePersistenceMixin._has_reflection_payload(sensor_reflection):
                        current_objects = LongTermMemoryServicePersistenceMixin._merge_reflection_objects(
                            object_store=object_store,
                            current_objects=current_objects,
                            reflection=sensor_reflection,
                        )
                retention = LongTermMemoryServicePersistenceMixin._apply_retention_or_keep(
                    retention_policy=retention_policy,
                    objects=current_objects,
                )
                object_store.commit_active_delta(
                    object_upserts=_sort_objects_by_memory_id(retention.kept_objects),
                    object_delete_ids=LongTermMemoryServicePersistenceMixin._retention_deleted_memory_ids(
                        current_objects=current_objects,
                        retention=retention,
                    ),
                    conflict_upserts=current_conflicts,
                    archive_upserts=retention.archived_objects,
                )
                try:
                    graph_store.apply_candidate_edges(result.graph_edges)
                except Exception:
                    logger.exception("Graph-store update failed after conversation-turn snapshot commit.")
                try:
                    if reflection.midterm_packets:
                        midterm_store.apply_reflection(reflection)
                    if sensor_reflection.midterm_packets:
                        midterm_store.apply_reflection(sensor_reflection)
                except Exception:
                    logger.exception("Midterm-store update failed after conversation-turn snapshot commit.")
                if personality_learning is not None:
                    learning_consolidation = LongTermMemoryServicePersistenceMixin._merge_reflection_into_consolidation(
                        result=result,
                        reflection_batches=(reflection, sensor_reflection),
                    )
                    personality_learning.record_conversation_consolidation(
                        turn=item,
                        consolidation=learning_consolidation,
                    )
                if callable(prepared_context_invalidator):
                    prepared_context_invalidator(reason="conversation_turn_persisted")
                if config.long_term_memory_mode == "remote_primary":
                    return None
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term conversation-turn persistence failed; falling back to episodic memory.")

        return LongTermMemoryServicePersistenceMixin._persist_episodic_turn(store=store, item=item)

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
        prepared_context_invalidator: Callable[..., None] | None = None,
        store_lock: threading.RLock | None = None,
        timezone_name: str | None = None,
        item: LongTermMultimodalEvidence,
    ) -> None:
        """Persist one multimodal evidence item through the long-term pipeline."""

        effective_store_lock = store_lock or threading.RLock()
        effective_timezone_name = timezone_name or "UTC"
        created_at = _normalize_datetime(item.created_at, timezone_name=effective_timezone_name) or item.created_at
        reflection = LongTermMemoryServicePersistenceMixin._empty_reflection_result()
        sensor_reflection = LongTermMemoryServicePersistenceMixin._empty_reflection_result()

        try:
            with effective_store_lock:
                extraction = multimodal_extractor.extract_evidence(item)
                working_set = object_store.load_active_working_set(
                    candidate_objects=(extraction.episode, *extraction.candidate_objects),
                    event_ids=(extraction.turn_id,),
                )
                existing_objects = working_set.objects
                existing_conflicts = working_set.conflicts
                result = consolidator.consolidate(
                    extraction=extraction,
                    existing_objects=existing_objects,
                )
                current_objects, current_conflicts = LongTermMemoryServicePersistenceMixin._merge_consolidation_state(
                    object_store=object_store,
                    existing_objects=existing_objects,
                    existing_conflicts=existing_conflicts,
                    result=result,
                )
                try:
                    reflection = reflector.reflect(
                        objects=tuple(
                            select_reflection_neighborhood_objects(
                                object_store,
                                seed_objects=current_objects,
                            )
                        ),
                        include_midterm=LongTermMemoryServicePersistenceMixin._should_include_midterm_in_multimodal_reflection(result),
                    )
                except Exception:
                    logger.exception("Long-term reflection failed during multimodal persistence.")
                else:
                    current_objects = LongTermMemoryServicePersistenceMixin._merge_reflection_objects(
                        object_store=object_store,
                        current_objects=current_objects,
                        reflection=reflection,
                    )
                try:
                    sensor_reflection = sensor_memory.compile(
                        objects=tuple(
                            select_sensor_memory_neighborhood_objects(
                                object_store,
                                seed_objects=current_objects,
                            )
                        ),
                        now=created_at,
                    )
                except Exception:
                    logger.exception("Sensor-memory compilation failed during multimodal persistence.")
                else:
                    if LongTermMemoryServicePersistenceMixin._has_reflection_payload(sensor_reflection):
                        current_objects = LongTermMemoryServicePersistenceMixin._merge_reflection_objects(
                            object_store=object_store,
                            current_objects=current_objects,
                            reflection=sensor_reflection,
                        )
                retention = LongTermMemoryServicePersistenceMixin._apply_retention_or_keep(
                    retention_policy=retention_policy,
                    objects=current_objects,
                )
                object_store.commit_active_delta(
                    object_upserts=_sort_objects_by_memory_id(retention.kept_objects),
                    object_delete_ids=LongTermMemoryServicePersistenceMixin._retention_deleted_memory_ids(
                        current_objects=current_objects,
                        retention=retention,
                    ),
                    conflict_upserts=current_conflicts,
                    archive_upserts=retention.archived_objects,
                )
                if LongTermMemoryServicePersistenceMixin._has_reflection_payload(reflection):
                    midterm_store.apply_reflection(reflection)
                if LongTermMemoryServicePersistenceMixin._has_reflection_payload(sensor_reflection):
                    midterm_store.apply_reflection(sensor_reflection)
                if callable(prepared_context_invalidator):
                    prepared_context_invalidator(reason="multimodal_evidence_persisted")
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term multimodal persistence failed.")
            return None

    @staticmethod
    def _persist_episodic_turn(*, store: PromptContextStore, item: LongTermConversationTurn) -> PersistentMemoryEntry | None:
        """Write an episodic fallback record into prompt memory."""

        quoted_transcript = json.dumps(item.transcript, ensure_ascii=False)
        quoted_response = json.dumps(item.response, ensure_ascii=False)
        summary = f"Conversation about {quoted_transcript}"
        details = f"User said: {quoted_transcript} Twinr answered: {quoted_response}"

        try:
            return store.memory_store.remember(
                kind="episodic_turn",
                summary=summary,
                details=details,
            )
        except LongTermRemoteUnavailableError:
            logger.info("Configured episodic memory store is remote-unavailable; attempting local fallback.")
        except Exception:
            logger.exception("Primary episodic-memory write failed; attempting local fallback.")

        local_memory_store = LongTermMemoryServicePersistenceMixin._clone_local_memory_store(store.memory_store)
        if local_memory_store is None:
            logger.error("Failed to construct local episodic-memory fallback store.")
            return None
        try:
            return local_memory_store.remember(
                kind="episodic_turn",
                summary=summary,
                details=details,
            )
        except Exception:
            logger.exception("Failed to persist episodic fallback memory locally.")
            return None

    @staticmethod
    def _merge_consolidation_state(
        *,
        object_store: LongTermStructuredStore,
        existing_objects: tuple,
        existing_conflicts: tuple,
        result: LongTermConsolidationResultV1,
    ) -> tuple[tuple, tuple]:
        """Merge newly consolidated objects and conflicts into current state."""

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
            _sort_objects_by_memory_id(merged_objects.values()),
            _sort_conflicts(merged_conflicts.values()),
        )

    @staticmethod
    def _merge_reflection_objects(
        *,
        object_store: LongTermStructuredStore,
        current_objects: tuple,
        reflection: LongTermReflectionResultV1,
    ) -> tuple:
        """Merge reflected objects and summaries into the current object set."""

        merged = {item.memory_id: item for item in current_objects}
        for item in (*reflection.reflected_objects, *reflection.created_summaries):
            merged[item.memory_id] = object_store._merge_object(
                existing=merged.get(item.memory_id),
                incoming=item,
                increment_support=False,
            )
        return _sort_objects_by_memory_id(merged.values())

    @staticmethod
    def _empty_reflection_result() -> LongTermReflectionResultV1:
        """Return an empty reflection payload for fail-closed callers."""

        return LongTermReflectionResultV1(reflected_objects=(), created_summaries=(), midterm_packets=())

    @staticmethod
    def _has_reflection_payload(result: LongTermReflectionResultV1) -> bool:
        """Report whether a reflection payload contains any state updates."""

        return bool(result.reflected_objects or result.created_summaries or result.midterm_packets)

    @staticmethod
    def _merge_reflection_into_consolidation(
        *,
        result: LongTermConsolidationResultV1,
        reflection_batches: Iterable[LongTermReflectionResultV1],
    ) -> LongTermConsolidationResultV1:
        """Return one consolidation view enriched with reflection-side object updates."""

        durable_by_id = {item.memory_id: item for item in result.durable_objects}
        deferred_by_id = {item.memory_id: item for item in result.deferred_objects}
        for reflection in reflection_batches:
            for item in reflection.reflected_objects:
                if item.memory_id in deferred_by_id:
                    deferred_by_id[item.memory_id] = item
                else:
                    durable_by_id[item.memory_id] = item
            for item in reflection.created_summaries:
                deferred_by_id.pop(item.memory_id, None)
                durable_by_id[item.memory_id] = item
        return LongTermConsolidationResultV1(
            turn_id=result.turn_id,
            occurred_at=result.occurred_at,
            episodic_objects=result.episodic_objects,
            durable_objects=_sort_objects_by_memory_id(durable_by_id.values()),
            deferred_objects=_sort_objects_by_memory_id(deferred_by_id.values()),
            conflicts=result.conflicts,
            graph_edges=result.graph_edges,
        )

    @staticmethod
    def _should_include_midterm_in_multimodal_reflection(result: LongTermConsolidationResultV1) -> bool:
        """Return whether one multimodal batch needs optional midterm compilation."""

        for item in (*result.episodic_objects, *result.durable_objects, *result.deferred_objects):
            if kind_matches(item.kind, "episode", item.attributes):
                continue
            if kind_matches(item.kind, "observation", item.attributes):
                continue
            if kind_matches(item.kind, "pattern", item.attributes):
                continue
            return True
        return False

    @staticmethod
    def _apply_retention_or_keep(
        *,
        retention_policy: LongTermRetentionPolicy,
        objects: tuple,
    ) -> LongTermRetentionResultV1:
        """Apply retention, or keep current objects if retention fails."""

        try:
            return retention_policy.apply(objects=objects)
        except Exception:
            logger.exception("Long-term retention failed; keeping current objects unpruned.")
            return LongTermRetentionResultV1(
                kept_objects=objects,
                expired_objects=(),
                pruned_memory_ids=(),
                archived_objects=(),
            )

    @staticmethod
    def _merge_archived_objects(
        *,
        existing_archived: Iterable[Any],
        archived_updates: Iterable[Any],
    ) -> tuple[Any, ...]:
        """Merge archived-object updates by memory ID."""

        archived = {item.memory_id: item for item in existing_archived}
        for archived_item in archived_updates:
            archived[archived_item.memory_id] = archived_item
        return _sort_objects_by_memory_id(archived.values())

    @staticmethod
    def _merge_unique_objects(
        *object_groups: Iterable[Any],
    ) -> tuple[Any, ...]:
        """Return unique objects keyed by memory id while preserving first-seen order."""

        merged: dict[str, Any] = {}
        for group in object_groups:
            for item in group:
                memory_id = getattr(item, "memory_id", None)
                if not isinstance(memory_id, str) or not memory_id:
                    continue
                if memory_id not in merged:
                    merged[memory_id] = item
        return tuple(merged.values())

    @staticmethod
    def _retention_deleted_memory_ids(
        *,
        current_objects: Iterable[Any],
        retention: LongTermRetentionResultV1,
    ) -> tuple[str, ...]:
        """Return touched current memory ids removed by one retention pass."""

        kept_ids = {item.memory_id for item in retention.kept_objects}
        deleted_ids = [
            item.memory_id
            for item in current_objects
            if item.memory_id not in kept_ids
        ]
        deleted_ids.extend(retention.pruned_memory_ids)
        return tuple(
            dict.fromkeys(
                memory_id
                for memory_id in deleted_ids
                if isinstance(memory_id, str) and memory_id
            )
        )

    @staticmethod
    def _clone_local_memory_store(memory_store: Any) -> Any | None:
        """Clone a local prompt-memory store when episodic fallback is needed."""

        path = _validate_regular_file_path(getattr(memory_store, "path", None), allow_missing=True)
        if path is None:
            return None
        try:
            return type(memory_store)(str(path))
        except Exception:
            logger.exception("Failed to clone local prompt memory store for episodic fallback.")
            return None

    def _load_backfill_entries(self) -> tuple[Mapping[str, object], ...]:
        """Load validated ops-event entries for multimodal backfill."""

        try:
            from twinr.ops.events import TwinrOpsEventStore
        except Exception:
            logger.exception("Failed to import TwinrOpsEventStore for ops-event backfill.")
            return ()

        try:
            store = TwinrOpsEventStore.from_config(self.config)
            path = _validate_regular_file_path(getattr(store, "path", None), allow_missing=False)
            if path is None:
                logger.warning("Skipping ops-event backfill because the configured path is unsafe or missing.")
                return ()
            return tuple(self.ops_backfiller.load_entries(str(path)))
        except Exception:
            logger.exception("Failed to load ops-event history for multimodal backfill.")
            return ()

    def _build_multimodal_evidence(
        self,
        *,
        event_name: str,
        modality: str,
        source: str = "device_event",
        message: str | None = None,
        data: Mapping[str, object] | None = None,
    ) -> LongTermMultimodalEvidence:
        """Build a bounded multimodal evidence payload from device data."""

        sanitized_message = _normalize_text(message, limit=_MULTIMODAL_MESSAGE_LIMIT) or None
        sanitized_data = _sanitize_jsonish(
            dict(data or {}),
            timezone_name=self.config.local_timezone_name,
        )
        payload = sanitized_data if isinstance(sanitized_data, dict) else {"value": sanitized_data}
        return LongTermMultimodalEvidence(
            event_name=_normalize_text(event_name, limit=_MULTIMODAL_NAME_LIMIT) or "unknown_event",
            modality=_normalize_text(modality, limit=_SOURCE_LIMIT) or "unknown_modality",
            source=_normalize_text(source, limit=_SOURCE_LIMIT) or "device_event",
            message=sanitized_message,
            data=payload,
        )
