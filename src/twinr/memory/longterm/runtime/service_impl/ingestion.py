"""Ingress and analysis entry points for the long-term runtime service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from twinr.memory.context_store import PersistentMemoryEntry
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
    LongTermEnqueueResult,
)

from ._typing import ServiceMixinBase
from .compat import _SOURCE_LIMIT, _TEXT_LIMIT, _normalize_text, logger

if TYPE_CHECKING:
    from twinr.agent.personality.intelligence import WorldIntelligenceConfigRequest


class LongTermMemoryServiceIngestionMixin(ServiceMixinBase):
    """Foreground enqueue, import, and dry-run analysis helpers."""

    def _persist_conversation_turn_sync(
        self,
        *,
        item: LongTermConversationTurn,
    ) -> LongTermEnqueueResult:
        """Persist one conversation turn immediately when no background writer exists.

        Evaluation and foreground service modes can disable background writers
        intentionally. In that configuration enqueue-style callers still mean
        "store this turn now", so failing closed with ``None`` would silently
        drop episodic memory writes.
        """

        self._persist_longterm_turn(
            config=self.config,
            store=self.prompt_context_store,
            graph_store=self.graph_store,
            object_store=self.object_store,
            midterm_store=self.midterm_store,
            extractor=self.extractor,
            consolidator=self.consolidator,
            reflector=self.reflector,
            turn_continuity_compiler=self.turn_continuity_compiler,
            sensor_memory=self.sensor_memory,
            retention_policy=self.retention_policy,
            personality_learning=self.personality_learning,
            prepared_context_invalidator=self._invalidate_prepared_contexts,
            store_lock=self._store_lock,
            timezone_name=self.config.local_timezone_name,
            item=item,
        )
        return LongTermEnqueueResult(
            accepted=True,
            pending_count=0,
            dropped_count=0,
        )

    def _persist_multimodal_evidence_sync(
        self,
        *,
        evidence,
    ) -> LongTermEnqueueResult:
        """Persist one multimodal evidence item immediately without a worker."""

        self._persist_multimodal_evidence(
            object_store=self.object_store,
            midterm_store=self.midterm_store,
            multimodal_extractor=self.multimodal_extractor,
            consolidator=self.consolidator,
            reflector=self.reflector,
            sensor_memory=self.sensor_memory,
            retention_policy=self.retention_policy,
            prepared_context_invalidator=self._invalidate_prepared_contexts,
            store_lock=self._store_lock,
            timezone_name=self.config.local_timezone_name,
            item=evidence,
        )
        return LongTermEnqueueResult(
            accepted=True,
            pending_count=0,
            dropped_count=0,
        )

    def enqueue_conversation_turn(
        self,
        *,
        transcript: str,
        response: str,
        source: str = "conversation",
        modality: str = "voice",
    ) -> LongTermEnqueueResult | None:
        """Queue one conversation turn for bounded long-term persistence."""

        clean_transcript = _normalize_text(transcript, limit=_TEXT_LIMIT)
        clean_response = _normalize_text(response, limit=_TEXT_LIMIT)
        clean_source = _normalize_text(source, limit=_SOURCE_LIMIT) or "conversation"
        clean_modality = _normalize_text(modality, limit=_SOURCE_LIMIT) or "voice"
        if not clean_transcript or not clean_response:
            return None
        item = LongTermConversationTurn(
            transcript=clean_transcript,
            response=clean_response,
            source=clean_source,
            modality=clean_modality,
        )
        if self.writer is None:
            return self._persist_conversation_turn_sync(item=item)
        try:
            return self.writer.enqueue(item)
        except Exception:
            logger.exception("Failed to enqueue conversation turn; persisting synchronously.")
            return self._persist_conversation_turn_sync(item=item)

    def import_external_conversation_turn(
        self,
        *,
        transcript: str,
        response: str,
        source: str,
        modality: str,
        created_at: datetime,
        allow_personality_learning: bool = False,
    ) -> PersistentMemoryEntry | None:
        """Persist one historical or external turn directly into shared memory."""

        item = LongTermConversationTurn(
            transcript=_normalize_text(transcript, limit=_TEXT_LIMIT),
            response=_normalize_text(response, limit=_TEXT_LIMIT),
            source=_normalize_text(source, limit=_SOURCE_LIMIT) or "conversation",
            modality=_normalize_text(modality, limit=_SOURCE_LIMIT) or "text",
            created_at=created_at,
        )
        if not item.transcript or not item.response:
            return None
        return self._persist_longterm_turn(
            config=self.config,
            store=self.prompt_context_store,
            graph_store=self.graph_store,
            object_store=self.object_store,
            midterm_store=self.midterm_store,
            extractor=self.extractor,
            consolidator=self.consolidator,
            reflector=self.reflector,
            turn_continuity_compiler=self.turn_continuity_compiler,
            sensor_memory=self.sensor_memory,
            retention_policy=self.retention_policy,
            personality_learning=self.personality_learning if allow_personality_learning else None,
            prepared_context_invalidator=self._invalidate_prepared_contexts,
            store_lock=self._store_lock,
            timezone_name=self.config.local_timezone_name,
            item=item,
            episode_attributes={
                "ingestion_origin": "external_history",
                "retention_policy": "preserve",
            },
        )

    def record_personality_tool_history(
        self,
        *,
        tool_calls: Sequence[AgentToolCall],
        tool_results: Sequence[AgentToolResult],
    ) -> None:
        """Record structured tool-history signals for personality learning."""

        if self.personality_learning is None:
            return
        normalized_tool_calls = tuple(tool_calls)
        normalized_tool_results = tuple(tool_results)
        if not normalized_tool_calls and not normalized_tool_results:
            return
        with self._store_lock:
            self.personality_learning.record_tool_history(
                tool_calls=normalized_tool_calls,
                tool_results=normalized_tool_results,
            )

    def enqueue_personality_tool_history(
        self,
        *,
        tool_calls: Sequence[AgentToolCall],
        tool_results: Sequence[AgentToolResult],
    ) -> None:
        """Queue structured tool-history signals for the next background commit."""

        if self.personality_learning is None:
            return
        normalized_tool_calls = tuple(tool_calls)
        normalized_tool_results = tuple(tool_results)
        if not normalized_tool_calls and not normalized_tool_results:
            return
        self.personality_learning.enqueue_tool_history(
            tool_calls=normalized_tool_calls,
            tool_results=normalized_tool_results,
        )

    def configure_world_intelligence(
        self,
        *,
        request: WorldIntelligenceConfigRequest,
        search_backend: object | None = None,
    ):
        """Apply one explicit RSS/world-intelligence configuration change."""

        if self.personality_learning is None:
            raise RuntimeError("personality learning is not configured")
        with self._store_lock:
            return self.personality_learning.configure_world_intelligence(
                request=request,
                search_backend=search_backend,
            )

    def analyze_conversation_turn(
        self,
        *,
        transcript: str,
        response: str,
        source: str = "conversation",
        modality: str = "voice",
    ) -> LongTermConsolidationResultV1:
        """Run extraction and consolidation without mutating stored state."""

        extraction = self.extractor.extract_conversation_turn(
            transcript=transcript,
            response=response,
            source=source,
            modality=modality,
        )
        with self._store_lock:
            working_set = self.object_store.load_active_working_set(
                candidate_objects=(extraction.episode, *extraction.candidate_objects),
                event_ids=(extraction.turn_id,),
            )
            existing_objects = working_set.objects
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
        """Run multimodal extraction and consolidation without persisting."""

        evidence = self._build_multimodal_evidence(
            event_name=event_name,
            modality=modality,
            source=source,
            message=message,
            data=data,
        )
        extraction = self.multimodal_extractor.extract_evidence(evidence)
        with self._store_lock:
            working_set = self.object_store.load_active_working_set(
                candidate_objects=(extraction.episode, *extraction.candidate_objects),
                event_ids=(extraction.turn_id,),
            )
            existing_objects = working_set.objects
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
        data: Mapping[str, object] | None = None,
    ) -> LongTermEnqueueResult | None:
        """Queue one multimodal evidence item for bounded persistence."""

        if self.multimodal_writer is None:
            return None
        evidence = self._build_multimodal_evidence(
            event_name=event_name,
            modality=modality,
            source=source,
            message=message,
            data=data,
        )
        if self.multimodal_writer is None:
            return self._persist_multimodal_evidence_sync(evidence=evidence)
        try:
            return self.multimodal_writer.enqueue(evidence)
        except Exception:
            logger.exception("Failed to enqueue multimodal evidence; persisting synchronously.")
            return self._persist_multimodal_evidence_sync(evidence=evidence)
