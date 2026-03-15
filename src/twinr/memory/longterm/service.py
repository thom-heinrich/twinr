from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field  # AUDIT-FIX(#10): keep 3.11-safe dataclass field support for bounded defaults and locks.
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any  # AUDIT-FIX(#12): make fallback return types explicit and 3.11-safe.
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # AUDIT-FIX(#9): normalize naive/aware datetimes into the configured local timezone.
import json
import logging
import math
import threading

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
    LongTermProactiveCandidateV1,  # AUDIT-FIX(#5): import the proactive candidate model used in public type hints.
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


logger = logging.getLogger(__name__)

_TEXT_LIMIT = 320
_SOURCE_LIMIT = 64
_MULTIMODAL_NAME_LIMIT = 128
_MULTIMODAL_MESSAGE_LIMIT = 512
_JSON_STRING_LIMIT = 512
_JSON_ITEMS_LIMIT = 64
_JSON_DEPTH_LIMIT = 5
_DEFAULT_REVIEW_LIMIT = 12
_MAX_REVIEW_LIMIT = 100
_MAX_QUEUE_SIZE = 4096


# AUDIT-FIX(#10): harden text normalization against zero/negative limits and silent off-by-one truncation.
def _normalize_text(value: object, *, limit: int) -> str:
    if limit <= 0:
        return ""
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1].rstrip() + "…"


# AUDIT-FIX(#10): clamp queue sizes and review limits to sane positive values on misconfigured .env input.
def _coerce_positive_int(value: object, *, default: int, maximum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if result <= 0:
        result = default
    if maximum is not None:
        result = min(result, maximum)
    return result


def _coerce_timeout_s(value: object, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result) or result < 0:
        return default
    return result


@lru_cache(maxsize=16)
def _load_timezone(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        logger.warning("Unknown timezone %r; falling back to UTC.", timezone_name)
        return ZoneInfo("UTC")


# AUDIT-FIX(#9): convert public datetime inputs to timezone-aware local datetimes before scheduling or retention logic.
def _normalize_datetime(value: datetime | None, *, timezone_name: str) -> datetime | None:
    if value is None:
        return None
    timezone = _load_timezone(timezone_name)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone)
    return value.astimezone(timezone)


def _serialize_datetime(value: datetime, *, timezone_name: str) -> str:
    normalized = _normalize_datetime(value, timezone_name=timezone_name)
    if normalized is None:
        return ""
    return normalized.isoformat()


# AUDIT-FIX(#6): bound and sanitize multimodal payloads so malformed sensor data cannot poison background persistence.
def _sanitize_jsonish(
    value: Any,
    *,
    timezone_name: str,
    depth: int = 0,
) -> object:
    if depth >= _JSON_DEPTH_LIMIT:
        return _normalize_text(repr(value), limit=_JSON_STRING_LIMIT)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _normalize_text(value, limit=_JSON_STRING_LIMIT)
    if isinstance(value, datetime):
        return _serialize_datetime(value, timezone_name=timezone_name)
    if isinstance(value, bytes):
        return {"__bytes_len__": len(value)}
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for index, (key, nested_value) in enumerate(value.items()):
            if index >= _JSON_ITEMS_LIMIT:
                sanitized["__truncated__"] = True
                break
            sanitized[_normalize_text(key, limit=64) or f"key_{index}"] = _sanitize_jsonish(
                nested_value,
                timezone_name=timezone_name,
                depth=depth + 1,
            )
        return sanitized
    if isinstance(value, Iterable):
        sanitized_items: list[object] = []
        for index, item in enumerate(value):
            if index >= _JSON_ITEMS_LIMIT:
                sanitized_items.append("…")
                break
            sanitized_items.append(
                _sanitize_jsonish(
                    item,
                    timezone_name=timezone_name,
                    depth=depth + 1,
                )
            )
        return sanitized_items
    return _normalize_text(repr(value), limit=_JSON_STRING_LIMIT)


# AUDIT-FIX(#7): reject unsafe config-backed file paths used for backfill and local fallback store recreation.
def _validate_regular_file_path(path_value: object, *, allow_missing: bool) -> Path | None:
    try:
        path = Path(path_value).expanduser()
    except (TypeError, ValueError):
        return None
    try:
        if path.exists():
            if path.is_symlink() or not path.is_file():
                return None
        elif not allow_missing:
            return None
        parent = path.parent
        if parent.exists():
            if parent.is_symlink() or not parent.is_dir():
                return None
    except OSError:
        return None
    return path


def _sort_objects_by_memory_id(items: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(items, key=lambda row: row.memory_id))


def _sort_conflicts(items: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(items, key=lambda row: (row.slot_key, row.candidate_memory_id)))


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
    _store_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)  # AUDIT-FIX(#1): serialize file-backed state mutations across foreground calls and background workers.

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
        store_lock = threading.RLock()  # AUDIT-FIX(#1): share one re-entrant mutation lock with both background writers.
        # AUDIT-FIX(#10): guard startup against zero/negative queue sizes from configuration.
        queue_size = _coerce_positive_int(
            getattr(config, "long_term_memory_write_queue_size", 1),
            default=1,
            maximum=_MAX_QUEUE_SIZE,
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
                    store_lock=store_lock,
                    timezone_name=config.local_timezone_name,
                    item=item,
                ),
                max_queue_size=queue_size,
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
                    store_lock=store_lock,
                    timezone_name=config.local_timezone_name,
                    item=item,
                ),
                max_queue_size=queue_size,
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
            _store_lock=store_lock,
        )

    def ensure_remote_ready(self) -> None:
        remote_state = getattr(self.prompt_context_store.memory_store, "remote_state", None)
        if remote_state is None or not remote_state.enabled:
            return
        status = remote_state.status()
        if not status.ready:
            if remote_state.required:
                raise LongTermRemoteUnavailableError(
                    status.detail or "Remote-primary long-term memory is not ready."
                )
            return
        with self._store_lock:
            self.prompt_context_store.ensure_remote_snapshots()
            self.graph_store.ensure_remote_snapshot()
            self.object_store.ensure_remote_snapshots()
            self.midterm_store.ensure_remote_snapshot()

    def build_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        try:
            query = self.query_rewriter.profile(query_text)
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                return self.retriever.build_context(
                    query=query,
                    original_query_text=query_text,
                )
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to build long-term provider context.")  # AUDIT-FIX(#11): degrade to empty context instead of crashing the turn.
            return LongTermMemoryContext()

    def build_tool_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        try:
            query = self.query_rewriter.profile(query_text)
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
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
                    limit=max(
                        1,
                        _coerce_positive_int(
                            getattr(self.config, "long_term_memory_recall_limit", 1),
                            default=1,
                            maximum=_MAX_REVIEW_LIMIT,
                        ),
                    ),
                )
                filtered_durable_objects = tuple(
                    item
                    for item in durable_objects
                    if not kind_matches(
                        item.kind,
                        "fact",
                        item.attributes,
                        attr_key="fact_type",
                        attr_value="contact_method",
                    )
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
            raise
        except Exception:
            logger.exception("Failed to build tool-facing long-term provider context.")  # AUDIT-FIX(#11): tool context failure must not take down the request path.
            return LongTermMemoryContext()

    def enqueue_conversation_turn(
        self,
        *,
        transcript: str,
        response: str,
        source: str = "conversation",
    ) -> LongTermEnqueueResult | None:
        clean_transcript = _normalize_text(transcript, limit=_TEXT_LIMIT)
        clean_response = _normalize_text(response, limit=_TEXT_LIMIT)
        clean_source = _normalize_text(source, limit=_SOURCE_LIMIT) or "conversation"
        if not clean_transcript or not clean_response:
            return None
        # AUDIT-FIX(#6): normalize source text early so queue fallback stores a bounded record.
        item = LongTermConversationTurn(
            transcript=clean_transcript,
            response=clean_response,
            source=clean_source,
        )
        if self.writer is None:
            return None
        try:
            return self.writer.enqueue(item)
        except Exception:
            logger.exception("Failed to enqueue conversation turn; persisting synchronously.")  # AUDIT-FIX(#6): queue failure should not silently drop a senior interaction.
            self._persist_longterm_turn(
                config=self.config,
                store=self.prompt_context_store,
                graph_store=self.graph_store,
                object_store=self.object_store,
                midterm_store=self.midterm_store,
                extractor=self.extractor,
                consolidator=self.consolidator,
                reflector=self.reflector,
                sensor_memory=self.sensor_memory,
                retention_policy=self.retention_policy,
                store_lock=self._store_lock,
                timezone_name=self.config.local_timezone_name,
                item=item,
            )
            return None

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
        with self._store_lock:
            existing_objects = tuple(self.object_store.load_objects())
        return self.consolidator.consolidate(
            extraction=extraction,
            existing_objects=existing_objects,
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
        # AUDIT-FIX(#6): sanitize multimodal inputs before extraction so malformed payloads remain JSON-safe and bounded.
        evidence = self._build_multimodal_evidence(
            event_name=event_name,
            modality=modality,
            source=source,
            message=message,
            data=data,
        )
        extraction = self.multimodal_extractor.extract_evidence(evidence)
        with self._store_lock:
            existing_objects = tuple(self.object_store.load_objects())
        return self.consolidator.consolidate(
            extraction=extraction,
            existing_objects=existing_objects,
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
        # AUDIT-FIX(#6): sanitize multimodal inputs before extraction so malformed payloads remain JSON-safe and bounded.
        evidence = self._build_multimodal_evidence(
            event_name=event_name,
            modality=modality,
            source=source,
            message=message,
            data=data,
        )
        try:
            return self.multimodal_writer.enqueue(evidence)
        except Exception:
            logger.exception("Failed to enqueue multimodal evidence; persisting synchronously.")  # AUDIT-FIX(#6): keep device events durable even when the async queue is unavailable.
            self._persist_multimodal_evidence(
                object_store=self.object_store,
                midterm_store=self.midterm_store,
                multimodal_extractor=self.multimodal_extractor,
                consolidator=self.consolidator,
                reflector=self.reflector,
                sensor_memory=self.sensor_memory,
                retention_policy=self.retention_policy,
                store_lock=self._store_lock,
                timezone_name=self.config.local_timezone_name,
                item=evidence,
            )
            return None

    # AUDIT-FIX(#4): persist sensor-memory midterm packets everywhere reflection results are applied.
    def run_reflection(self) -> LongTermReflectionResultV1:
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                result = self.reflector.reflect(objects=self.object_store.load_objects())
                if self._has_reflection_payload(result):
                    self.object_store.apply_reflection(result)
                    self.midterm_store.apply_reflection(result)
                sensor_memory_result = self.sensor_memory.compile(objects=self.object_store.load_objects())
                if self._has_reflection_payload(sensor_memory_result):
                    self.object_store.apply_reflection(sensor_memory_result)
                    self.midterm_store.apply_reflection(sensor_memory_result)
                    return LongTermReflectionResultV1(
                        reflected_objects=tuple((*result.reflected_objects, *sensor_memory_result.reflected_objects)),
                        created_summaries=tuple((*result.created_summaries, *sensor_memory_result.created_summaries)),
                        midterm_packets=tuple((*result.midterm_packets, *sensor_memory_result.midterm_packets)),
                    )
                return result
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term reflection failed.")
            return self._empty_reflection_result()

    # AUDIT-FIX(#4): sensor-only reflections now update both object and midterm stores.
    def run_sensor_memory(self, *, now: datetime | None = None) -> LongTermReflectionResultV1:
        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                result = self.sensor_memory.compile(objects=self.object_store.load_objects(), now=normalized_now)
                if self._has_reflection_payload(result):
                    self.object_store.apply_reflection(result)
                    self.midterm_store.apply_reflection(result)
                return result
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Sensor-memory compilation failed.")
            return self._empty_reflection_result()

    # AUDIT-FIX(#8): process backfill batches defensively so one bad item does not abort the entire recovery run.
    def backfill_ops_multimodal_history(
        self,
        *,
        entries: Iterable[Mapping[str, object]] | None = None,
        now: datetime | None = None,
    ) -> LongTermOpsBackfillRunResult:
        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        scanned_events = 0
        generated_evidence = 0
        sensor_observations = 0
        button_interactions = 0
        print_completions = 0
        applied_evidence = 0
        skipped_existing = 0
        reflected_objects = 0
        created_summaries = 0
        reflection_error: str | None = None

        try:
            raw_entries = tuple(entries) if entries is not None else self._load_backfill_entries()
            build = self.ops_backfiller.build_evidence(raw_entries)
            scanned_events = build.scanned_events
            generated_evidence = build.generated_evidence
            sensor_observations = build.sensor_observations
            button_interactions = build.button_interactions
            print_completions = build.print_completions

            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                existing_objects = tuple(self.object_store.load_objects())
                existing_conflicts = tuple(self.object_store.load_conflicts())
                existing_archived = tuple(self.object_store.load_archived_objects())
                objects_by_id = {item.memory_id: item for item in existing_objects}
                conflicts_by_slot = {item.slot_key: item for item in existing_conflicts}
                seen_turn_ids = {
                    event_id
                    for item in objects_by_id.values()
                    for event_id in item.source.event_ids
                }

                for evidence in build.evidence:
                    try:
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
                    except Exception as exc:
                        logger.exception("Failed to backfill one multimodal evidence item.")
                        if reflection_error is None:
                            reflection_error = f"{type(exc).__name__}: {exc}"

                current_objects = _sort_objects_by_memory_id(objects_by_id.values())
                current_conflicts = _sort_conflicts(conflicts_by_slot.values())

                reflection_result = self._empty_reflection_result()
                if applied_evidence:
                    try:
                        reflection_result = self.reflector.reflect(objects=current_objects)
                    except Exception as exc:
                        logger.exception("Long-term backfill reflection failed.")
                        if reflection_error is None:
                            reflection_error = f"{type(exc).__name__}: {exc}"
                    else:
                        current_objects = self._merge_reflection_objects(
                            object_store=self.object_store,
                            current_objects=current_objects,
                            reflection=reflection_result,
                        )
                        reflected_objects += len(reflection_result.reflected_objects)
                        created_summaries += len(reflection_result.created_summaries)

                sensor_result = self._empty_reflection_result()
                try:
                    sensor_result = self.sensor_memory.compile(objects=current_objects, now=normalized_now)
                except Exception as exc:
                    logger.exception("Long-term backfill sensor-memory compilation failed.")
                    if reflection_error is None:
                        reflection_error = f"{type(exc).__name__}: {exc}"
                else:
                    if self._has_reflection_payload(sensor_result):
                        current_objects = self._merge_reflection_objects(
                            object_store=self.object_store,
                            current_objects=current_objects,
                            reflection=sensor_result,
                        )
                    reflected_objects += len(sensor_result.reflected_objects)
                    created_summaries += len(sensor_result.created_summaries)

                retention = self._apply_retention_or_keep(
                    retention_policy=self.retention_policy,
                    objects=current_objects,
                )
                should_write_snapshot = (
                    applied_evidence > 0
                    or self._has_reflection_payload(reflection_result)
                    or self._has_reflection_payload(sensor_result)
                    or bool(retention.expired_objects or retention.pruned_memory_ids or retention.archived_objects)
                )
                if should_write_snapshot:
                    self.object_store.write_snapshot(
                        objects=_sort_objects_by_memory_id(retention.kept_objects),
                        conflicts=current_conflicts,
                        archived_objects=self._merge_archived_objects(
                            existing_archived=existing_archived,
                            archived_updates=retention.archived_objects,
                        ),
                    )
                if self._has_reflection_payload(reflection_result):
                    self.midterm_store.apply_reflection(reflection_result)
                if self._has_reflection_payload(sensor_result):
                    self.midterm_store.apply_reflection(sensor_result)
        except LongTermRemoteUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Ops multimodal backfill failed.")
            reflection_error = f"{type(exc).__name__}: {exc}"

        return LongTermOpsBackfillRunResult(
            scanned_events=scanned_events,
            generated_evidence=generated_evidence,
            applied_evidence=applied_evidence,
            skipped_existing=skipped_existing,
            sensor_observations=sensor_observations,
            button_interactions=button_interactions,
            print_completions=print_completions,
            reflected_objects=reflected_objects,
            created_summaries=created_summaries,
            reflection_error=reflection_error,
        )

    def plan_proactive_candidates(
        self,
        *,
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactivePlanV1:
        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                return self.planner.plan(
                    objects=self.object_store.load_objects(),
                    now=normalized_now,
                    live_facts=live_facts,
                )
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Proactive planning failed.")  # AUDIT-FIX(#11): proactive failure is optional and should fail closed.
            return LongTermProactivePlanV1(candidates=())

    def reserve_proactive_candidate(
        self,
        *,
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactiveReservationV1 | None:
        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                plan = self.planner.plan(
                    objects=self.object_store.load_objects(),
                    now=normalized_now,
                    live_facts=live_facts,
                )
                return self.proactive_policy.reserve_candidate(plan=plan, now=normalized_now)
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Proactive candidate reservation failed.")
            return None

    # AUDIT-FIX(#9): proactive reservation now runs on normalized local datetimes.
    def reserve_specific_proactive_candidate(
        self,
        candidate: LongTermProactiveCandidateV1,
        *,
        now: datetime | None = None,
    ) -> LongTermProactiveReservationV1:
        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        with self._store_lock:
            return self.proactive_policy.reserve_specific_candidate(candidate, now=normalized_now)

    def preview_proactive_candidate(
        self,
        *,
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactiveCandidateV1 | None:
        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                plan = self.planner.plan(
                    objects=self.object_store.load_objects(),
                    now=normalized_now,
                    live_facts=live_facts,
                )
                return self.proactive_policy.preview_candidate(plan=plan, now=normalized_now)
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Proactive candidate preview failed.")
            return None

    # AUDIT-FIX(#12): make best-effort proactive state updates explicit in the return type.
    def mark_proactive_candidate_delivered(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        delivered_at: datetime | None = None,
        prompt_text: str | None = None,
    ) -> Any | None:
        normalized_delivered_at = _normalize_datetime(delivered_at, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                return self.proactive_policy.mark_delivered(
                    reservation,
                    delivered_at=normalized_delivered_at,
                    prompt_text=prompt_text,
                )
        except Exception:
            logger.exception("Failed to mark proactive candidate as delivered.")
            return None

    def mark_proactive_candidate_skipped(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        reason: str,
        skipped_at: datetime | None = None,
    ) -> Any | None:
        normalized_skipped_at = _normalize_datetime(skipped_at, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                return self.proactive_policy.mark_skipped(
                    reservation,
                    reason=reason,
                    skipped_at=normalized_skipped_at,
                )
        except Exception:
            logger.exception("Failed to mark proactive candidate as skipped.")
            return None

    def run_retention(self) -> LongTermRetentionResultV1:
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                result = self.retention_policy.apply(objects=self.object_store.load_objects())
                self.object_store.apply_retention(result)
                return result
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term retention run failed.")  # AUDIT-FIX(#11): scheduled maintenance should degrade instead of crashing the service.
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
            normalized_limit = None if limit is None else _coerce_positive_int(limit, default=1, maximum=_MAX_REVIEW_LIMIT)
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                return self.retriever.select_conflict_queue(
                    query=query,
                    limit=normalized_limit,
                )
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to select long-term conflict queue.")  # AUDIT-FIX(#11): retrieval helpers must fail safe.
            return ()

    def resolve_conflict(
        self,
        *,
        slot_key: str,
        selected_memory_id: str,
    ) -> LongTermConflictResolutionV1:
        with self._store_lock:
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
        limit: int = _DEFAULT_REVIEW_LIMIT,
    ) -> LongTermMemoryReviewResultV1:
        normalized_limit = _coerce_positive_int(limit, default=_DEFAULT_REVIEW_LIMIT, maximum=_MAX_REVIEW_LIMIT)
        with self._store_lock:
            return self.object_store.review_objects(
                query_text=query_text,
                status=status,
                kind=kind,
                include_episodes=include_episodes,
                limit=normalized_limit,
            )

    def confirm_memory(self, *, memory_id: str) -> LongTermMemoryMutationResultV1 | LongTermConflictResolutionV1:
        with self._store_lock:
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
        with self._store_lock:
            result = self.object_store.invalidate_object(memory_id, reason=reason)
            self.object_store.apply_memory_mutation(result)
            return result

    def delete_memory(self, *, memory_id: str) -> LongTermMemoryMutationResultV1:
        with self._store_lock:
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
        with self._store_lock:
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
        with self._store_lock:
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
        with self._store_lock:
            return self.prompt_context_store.personality_store.upsert(
                category=category,
                instruction=instruction,
            )

    def flush(self, *, timeout_s: float = 2.0) -> bool:
        resolved_timeout_s = _coerce_timeout_s(timeout_s, default=2.0)  # AUDIT-FIX(#11): harden lifecycle calls against invalid timeout input and writer exceptions.
        writer_ok = True
        multimodal_ok = True
        if self.writer is not None:
            try:
                writer_ok = self.writer.flush(timeout_s=resolved_timeout_s)
            except Exception:
                logger.exception("Failed to flush long-term conversation writer.")
                writer_ok = False
        if self.multimodal_writer is not None:
            try:
                multimodal_ok = self.multimodal_writer.flush(timeout_s=resolved_timeout_s)
            except Exception:
                logger.exception("Failed to flush long-term multimodal writer.")
                multimodal_ok = False
        return writer_ok and multimodal_ok

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        resolved_timeout_s = _coerce_timeout_s(timeout_s, default=2.0)  # AUDIT-FIX(#11): harden lifecycle calls against invalid timeout input and writer exceptions.
        if self.writer is not None:
            try:
                self.writer.shutdown(timeout_s=resolved_timeout_s)
            except Exception:
                logger.exception("Failed to shutdown long-term conversation writer cleanly.")
        if self.multimodal_writer is not None:
            try:
                self.multimodal_writer.shutdown(timeout_s=resolved_timeout_s)
            except Exception:
                logger.exception("Failed to shutdown long-term multimodal writer cleanly.")

    @staticmethod
    # AUDIT-FIX(#2): compute reflection/sensor/retention in memory, commit one object snapshot, then update dependent stores.
    # AUDIT-FIX(#3): catch broad callback failures so background workers do not die on one bad turn.
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
        store_lock: threading.RLock | None = None,
        timezone_name: str | None = None,
        item: LongTermConversationTurn,
    ) -> PersistentMemoryEntry | None:
        effective_store_lock = store_lock or threading.RLock()
        effective_timezone_name = timezone_name or config.local_timezone_name
        occurred_at = _normalize_datetime(item.created_at, timezone_name=effective_timezone_name) or item.created_at
        reflection = LongTermMemoryService._empty_reflection_result()
        sensor_reflection = LongTermMemoryService._empty_reflection_result()

        try:
            with effective_store_lock:
                existing_objects = tuple(object_store.load_objects())
                existing_conflicts = tuple(object_store.load_conflicts())
                existing_archived = tuple(object_store.load_archived_objects())
                extraction = extractor.extract_conversation_turn(
                    transcript=item.transcript,
                    response=item.response,
                    occurred_at=occurred_at,
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
                try:
                    reflection = reflector.reflect(objects=current_objects)
                except Exception:
                    logger.exception("Long-term reflection failed during conversation-turn persistence.")
                else:
                    current_objects = LongTermMemoryService._merge_reflection_objects(
                        object_store=object_store,
                        current_objects=current_objects,
                        reflection=reflection,
                    )
                try:
                    sensor_reflection = sensor_memory.compile(objects=current_objects, now=occurred_at)
                except Exception:
                    logger.exception("Sensor-memory compilation failed during conversation-turn persistence.")
                else:
                    if LongTermMemoryService._has_reflection_payload(sensor_reflection):
                        current_objects = LongTermMemoryService._merge_reflection_objects(
                            object_store=object_store,
                            current_objects=current_objects,
                            reflection=sensor_reflection,
                        )
                retention = LongTermMemoryService._apply_retention_or_keep(
                    retention_policy=retention_policy,
                    objects=current_objects,
                )
                object_store.write_snapshot(
                    objects=_sort_objects_by_memory_id(retention.kept_objects),
                    conflicts=current_conflicts,
                    archived_objects=LongTermMemoryService._merge_archived_objects(
                        existing_archived=existing_archived,
                        archived_updates=retention.archived_objects,
                    ),
                )
                try:
                    graph_store.apply_candidate_edges(result.graph_edges)  # AUDIT-FIX(#2): graph write happens only after the durable object snapshot commits.
                except Exception:
                    logger.exception("Graph-store update failed after conversation-turn snapshot commit.")
                try:
                    if LongTermMemoryService._has_reflection_payload(reflection):
                        midterm_store.apply_reflection(reflection)
                    if LongTermMemoryService._has_reflection_payload(sensor_reflection):
                        midterm_store.apply_reflection(sensor_reflection)
                except Exception:
                    logger.exception("Midterm-store update failed after conversation-turn snapshot commit.")
                if config.long_term_memory_mode == "remote_primary":
                    return None
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term conversation-turn persistence failed; falling back to episodic memory.")

        return LongTermMemoryService._persist_episodic_turn(store=store, item=item)

    @staticmethod
    # AUDIT-FIX(#2): multimodal writes now use the same single-snapshot commit pattern as conversation turns.
    # AUDIT-FIX(#3): broad callback handling prevents worker death on malformed evidence or store faults.
    def _persist_multimodal_evidence(
        *,
        object_store: LongTermStructuredStore,
        midterm_store: LongTermMidtermStore,
        multimodal_extractor: LongTermMultimodalExtractor,
        consolidator: LongTermMemoryConsolidator,
        reflector: LongTermMemoryReflector,
        sensor_memory: LongTermSensorMemoryCompiler,
        retention_policy: LongTermRetentionPolicy,
        store_lock: threading.RLock | None = None,
        timezone_name: str | None = None,
        item: LongTermMultimodalEvidence,
    ) -> None:
        effective_store_lock = store_lock or threading.RLock()
        effective_timezone_name = timezone_name or "UTC"
        created_at = _normalize_datetime(item.created_at, timezone_name=effective_timezone_name) or item.created_at
        reflection = LongTermMemoryService._empty_reflection_result()
        sensor_reflection = LongTermMemoryService._empty_reflection_result()

        try:
            with effective_store_lock:
                existing_objects = tuple(object_store.load_objects())
                existing_conflicts = tuple(object_store.load_conflicts())
                existing_archived = tuple(object_store.load_archived_objects())
                extraction = multimodal_extractor.extract_evidence(item)
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
                try:
                    reflection = reflector.reflect(objects=current_objects)
                except Exception:
                    logger.exception("Long-term reflection failed during multimodal persistence.")
                else:
                    current_objects = LongTermMemoryService._merge_reflection_objects(
                        object_store=object_store,
                        current_objects=current_objects,
                        reflection=reflection,
                    )
                try:
                    sensor_reflection = sensor_memory.compile(objects=current_objects, now=created_at)
                except Exception:
                    logger.exception("Sensor-memory compilation failed during multimodal persistence.")
                else:
                    if LongTermMemoryService._has_reflection_payload(sensor_reflection):
                        current_objects = LongTermMemoryService._merge_reflection_objects(
                            object_store=object_store,
                            current_objects=current_objects,
                            reflection=sensor_reflection,
                        )
                retention = LongTermMemoryService._apply_retention_or_keep(
                    retention_policy=retention_policy,
                    objects=current_objects,
                )
                object_store.write_snapshot(
                    objects=_sort_objects_by_memory_id(retention.kept_objects),
                    conflicts=current_conflicts,
                    archived_objects=LongTermMemoryService._merge_archived_objects(
                        existing_archived=existing_archived,
                        archived_updates=retention.archived_objects,
                    ),
                )
                if LongTermMemoryService._has_reflection_payload(reflection):
                    midterm_store.apply_reflection(reflection)
                if LongTermMemoryService._has_reflection_payload(sensor_reflection):
                    midterm_store.apply_reflection(sensor_reflection)
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term multimodal persistence failed.")
            return None

    @staticmethod
    # AUDIT-FIX(#7): prefer the configured store, then fall back to a guarded local clone instead of blindly reconstructing it from an unsafe path.
    def _persist_episodic_turn(*, store: PromptContextStore, item: LongTermConversationTurn) -> PersistentMemoryEntry | None:
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

        local_memory_store = LongTermMemoryService._clone_local_memory_store(store.memory_store)
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
        return LongTermReflectionResultV1(reflected_objects=(), created_summaries=(), midterm_packets=())

    @staticmethod
    def _has_reflection_payload(result: LongTermReflectionResultV1) -> bool:
        return bool(result.reflected_objects or result.created_summaries or result.midterm_packets)

    @staticmethod
    # AUDIT-FIX(#8): preserve newly consolidated state even when retention logic fails; do not discard the whole batch.
    def _apply_retention_or_keep(
        *,
        retention_policy: LongTermRetentionPolicy,
        objects: tuple,
    ) -> LongTermRetentionResultV1:
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
        archived = {item.memory_id: item for item in existing_archived}
        for archived_item in archived_updates:
            archived[archived_item.memory_id] = archived_item
        return _sort_objects_by_memory_id(archived.values())

    @staticmethod
    def _clone_local_memory_store(memory_store: Any) -> Any | None:
        path = _validate_regular_file_path(getattr(memory_store, "path", None), allow_missing=True)
        if path is None:
            return None
        try:
            return type(memory_store)(str(path))
        except Exception:
            logger.exception("Failed to clone local prompt memory store for episodic fallback.")
            return None

    # AUDIT-FIX(#7): only read ops-event history from validated regular files.
    def _load_backfill_entries(self) -> tuple[Mapping[str, object], ...]:
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

    # AUDIT-FIX(#6): normalize and bound sensor evidence before it touches extractors or persistent stores.
    def _build_multimodal_evidence(
        self,
        *,
        event_name: str,
        modality: str,
        source: str = "device_event",
        message: str | None = None,
        data: Mapping[str, object] | None = None,
    ) -> LongTermMultimodalEvidence:
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
