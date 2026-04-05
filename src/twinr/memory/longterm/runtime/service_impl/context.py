"""Provider-context assembly for the long-term runtime service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import threading
import time
from typing import cast

from twinr.agent.workflows.forensics import workflow_event, workflow_span
from twinr.agent.workflows.voice_turn_latency import record_voice_turn_remote_memory_ready
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.longterm.core.models import LongTermMemoryContext
from twinr.memory.longterm.runtime.context_snapshot import LongTermContextSnapshot
from twinr.memory.longterm.runtime.prepared_context import (
    PreparedContextProfile,
    PreparedLongTermContextResolution,
)
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteReadFailedError,
    LongTermRemoteUnavailableError,
)
from twinr.memory.query_normalization import LongTermQueryProfile

from ._typing import ServiceMixinBase
from .compat import (
    _context_details,
    _writer_state_details,
    logger,
)

# Bound the first non-English provider-context build long enough for the
# background canonical rewrite to finish in the common case. Live probes on
# April 3, 2026 showed real rewrite latency around 1.1s to 2.8s; the previous
# 350ms budget systematically missed first-turn graph/subtext personalization.
_PROVIDER_QUERY_REWRITE_WAIT_S = 3.0
_PREWARM_QUERY_REWRITE_WAIT_S = 3.0


class LongTermMemoryServiceContextMixin(ServiceMixinBase):
    """Build normal, fast, and tool-facing provider contexts."""

    def latest_context_snapshot(
        self,
        *,
        profile: PreparedContextProfile,
    ) -> LongTermContextSnapshot | None:
        """Return the newest built long-term context snapshot for one profile."""

        lock, snapshots = self._context_snapshot_state()
        with lock:
            return snapshots.get(profile)

    def prewarm_foreground_read_cache(self) -> None:
        """Warm remote-backed read caches for the first foreground text turn."""

        with self._store_lock:
            with self._temporary_remote_probe_cache():
                remote_catalog = getattr(getattr(self, "object_store", None), "_remote_catalog", None)
                load_catalog_entries = cast(
                    Callable[..., object] | None,
                    getattr(remote_catalog, "load_catalog_entries", None),
                )
                if load_catalog_entries is not None:
                    with workflow_span(name="longterm_prewarm_object_catalog", kind="cache"):
                        load_catalog_entries(snapshot_kind="objects")
                    with workflow_span(name="longterm_prewarm_conflict_catalog", kind="cache"):
                        load_catalog_entries(snapshot_kind="conflicts")
                load_packets = getattr(self.midterm_store, "load_packets", None)
                if callable(load_packets):
                    load_packets()
                load_document = getattr(self.graph_store, "load_document", None)
                if callable(load_document):
                    load_document()
        prepared_context_front = getattr(self, "prepared_context_front", None)
        if prepared_context_front is not None:
            refreshed = prepared_context_front.refresh_recent_async(
                build_for_query=self._schedule_prepared_context_refresh,
            )
            workflow_event(
                kind="cache",
                msg="longterm_prewarm_prepared_context_refresh_scheduled",
                details={"sticky_queries": refreshed},
            )
        provider_answer_front = getattr(self, "provider_answer_front", None)
        if provider_answer_front is not None:
            refreshed = provider_answer_front.refresh_recent_async(
                build_for_query=self._schedule_provider_answer_front_refresh,
            )
            workflow_event(
                kind="cache",
                msg="longterm_prewarm_materialized_provider_front_refresh_scheduled",
                details={"sticky_queries": refreshed},
            )

    def build_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        """Build the normal long-term context injected into provider prompts."""

        started_at = time.monotonic()
        try:
            query = self.query_rewriter.profile(
                query_text,
                wait_for_rewrite_s=_PROVIDER_QUERY_REWRITE_WAIT_S,
            )
            writer_details = _writer_state_details(getattr(self, "writer", None))
            multimodal_writer_details = _writer_state_details(getattr(self, "multimodal_writer", None))
            with workflow_span(
                name="longterm_service_build_provider_context",
                kind="retrieval",
                details={
                    "query_present": bool(str(query_text or "").strip()),
                    "conversation_writer": writer_details,
                    "multimodal_writer": multimodal_writer_details,
                },
            ):
                workflow_event(
                    kind="branch",
                    msg="longterm_provider_context_shared_lock_skipped",
                    details={
                        "lock_used": False,
                        "conversation_writer": writer_details,
                        "multimodal_writer": multimodal_writer_details,
                    },
                )
                try:
                    topic_context = self._build_quick_topic_context(
                        query_text=query_text,
                        workflow_event_name="longterm_provider_context_quick_memory_remote_read_failed",
                    )
                except LongTermRemoteReadFailedError as exc:
                    # Quick-topic hints are an auxiliary low-latency lane. When that
                    # bounded read times out, still let the main remote retriever build
                    # the authoritative provider context for the turn.
                    workflow_event(
                        kind="warning",
                        msg="longterm_provider_context_quick_memory_skipped",
                        details=dict(exc.details),
                    )
                    topic_context = None
                resolution = self._consume_materialized_provider_context(
                    query=query,
                    strict_missing=False,
                )
                if resolution is None:
                    resolution = self._consume_prepared_context(
                        profile="provider",
                        query=query,
                        build_context=lambda: self._build_provider_context_uncached(
                            query=query,
                            original_query_text=query_text,
                        ),
                    )
                    self._persist_materialized_provider_context(
                        query=query,
                        context=resolution.context,
                    )
                context = resolution.context
                if topic_context:
                    context = replace(context, topic_context=topic_context)
                self._remember_context_snapshot(
                    profile="provider",
                    query=query,
                    context=context,
                    source=resolution.source,
                )
                workflow_event(
                    kind="retrieval",
                    msg="longterm_provider_context_built",
                    details={
                        **_context_details(context),
                        "prepared_source": resolution.source,
                    },
                )
                return context
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to build long-term provider context.")
            return LongTermMemoryContext()
        finally:
            record_voice_turn_remote_memory_ready(
                duration_ms=max(0.0, (time.monotonic() - started_at) * 1000.0),
            )

    def build_live_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        """Build the live provider context strictly from materialized answer fronts.

        The transcript-first live path is not allowed to resurrect the broad
        retriever synchronously. If the required materialized front is missing or
        stale, runtime must fail closed instead of rebuilding the full provider
        context inline.
        """

        started_at = time.monotonic()
        try:
            query = self.query_rewriter.profile(query_text)
            writer_details = _writer_state_details(getattr(self, "writer", None))
            multimodal_writer_details = _writer_state_details(getattr(self, "multimodal_writer", None))
            with workflow_span(
                name="longterm_service_build_live_provider_context",
                kind="retrieval",
                details={
                    "query_present": bool(str(query_text or "").strip()),
                    "conversation_writer": writer_details,
                    "multimodal_writer": multimodal_writer_details,
                },
            ):
                try:
                    topic_context = self._build_quick_topic_context(
                        query_text=query_text,
                        workflow_event_name="longterm_live_provider_context_quick_memory_remote_read_failed",
                    )
                except LongTermRemoteReadFailedError as exc:
                    workflow_event(
                        kind="warning",
                        msg="longterm_live_provider_context_quick_memory_skipped",
                        details=dict(exc.details),
                    )
                    topic_context = None
                resolution = self._consume_materialized_provider_context(
                    query=query,
                    strict_missing=True,
                )
                assert resolution is not None
                context = resolution.context
                if topic_context:
                    context = replace(context, topic_context=topic_context)
                self._remember_context_snapshot(
                    profile="provider",
                    query=query,
                    context=context,
                    source=resolution.source,
                )
                workflow_event(
                    kind="retrieval",
                    msg="longterm_live_provider_context_built",
                    details={
                        **_context_details(context),
                        "prepared_source": resolution.source,
                    },
                )
                return context
        finally:
            record_voice_turn_remote_memory_ready(
                duration_ms=max(0.0, (time.monotonic() - started_at) * 1000.0),
            )

    def materialize_live_provider_context(
        self,
        query_text: str | None,
    ) -> PreparedLongTermContextResolution:
        """Synchronously build and persist one live provider front for non-live callers.

        The transcript-first runtime path must stay strict and consume only an
        already materialized provider-answer front. Non-streaming callers such
        as evaluations can use this explicit helper to build that front ahead of
        `build_live_provider_context(...)` without weakening the live path into
        a hidden synchronous retriever fallback.
        """

        query = self.query_rewriter.profile(
            query_text,
            wait_for_rewrite_s=_PROVIDER_QUERY_REWRITE_WAIT_S,
        )
        resolution = self._consume_materialized_provider_context(
            query=query,
            strict_missing=False,
        )
        if resolution is not None:
            return resolution
        provider_answer_front = getattr(self, "provider_answer_front", None)
        if provider_answer_front is None or not provider_answer_front.enabled():
            raise LongTermRemoteUnavailableError(
                "Required materialized provider answer front is unavailable for explicit materialization."
            )
        context = self._build_provider_context_uncached(
            query=query,
            original_query_text=query_text,
        )
        materialized = provider_answer_front.persist_built_context(
            query_keys=self._prepared_context_key_texts(query),
            sticky_query_text=query.original_text or query.retrieval_text,
            context=context,
        )
        resolved = PreparedLongTermContextResolution(
            context=materialized.context,
            source=materialized.source,
        )
        self._remember_context_snapshot(
            profile="provider",
            query=query,
            context=resolved.context,
            source=resolved.source,
        )
        workflow_event(
            kind="cache",
            msg="longterm_materialized_provider_front_built_sync",
            details={
                **_context_details(resolved.context),
                "prepared_source": resolved.source,
                "query_present": bool(str(query_text or "").strip()),
            },
        )
        return resolved

    def build_fast_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        """Build a tiny topic-memory context for latency-sensitive answer paths."""

        if not self.config.long_term_memory_enabled or not self.config.long_term_memory_fast_topic_enabled:
            return LongTermMemoryContext()
        started_at = time.monotonic()
        try:
            query = LongTermQueryProfile.from_text(query_text)
            writer_details = _writer_state_details(getattr(self, "writer", None))
            multimodal_writer_details = _writer_state_details(getattr(self, "multimodal_writer", None))
            with workflow_span(
                name="longterm_service_build_fast_provider_context",
                kind="retrieval",
                details={
                    "query_present": bool(str(query_text or "").strip()),
                    "conversation_writer": writer_details,
                    "multimodal_writer": multimodal_writer_details,
                },
            ):
                try:
                    topic_context = self._build_quick_topic_context(
                        query_text=query.original_text or query.retrieval_text,
                        workflow_event_name="longterm_fast_provider_context_remote_read_failed",
                    )
                except LongTermRemoteReadFailedError as exc:
                    workflow_event(
                        kind="warning",
                        msg="longterm_fast_provider_context_quick_memory_skipped",
                        details=dict(exc.details),
                    )
                    topic_context = None
                context = LongTermMemoryContext(topic_context=topic_context)
                workflow_event(
                    kind="retrieval",
                    msg="longterm_fast_provider_context_built",
                    details=_context_details(context),
                )
                return context
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to build fast long-term provider context.")
            return LongTermMemoryContext()
        finally:
            record_voice_turn_remote_memory_ready(
                duration_ms=max(0.0, (time.monotonic() - started_at) * 1000.0),
            )

    def _build_quick_topic_context(
        self,
        *,
        query_text: str | None,
        workflow_event_name: str,
    ) -> str | None:
        """Build the compact quick-memory topic block for one answer turn."""

        if (
            not self.config.long_term_memory_enabled
            or not self.config.long_term_memory_fast_topic_enabled
            or self.fast_topic_builder is None
        ):
            return None
        try:
            query = LongTermQueryProfile.from_text(query_text)
            return self.fast_topic_builder.build(query_profile=query)
        except LongTermRemoteReadFailedError as exc:
            workflow_event(
                kind="warning",
                msg=workflow_event_name,
                details=dict(exc.details),
            )
            raise
        except LongTermRemoteUnavailableError:
            raise
        except ChonkyDBError as exc:
            details: dict[str, object] = {
                "operation": "fast_provider_context",
                "request_kind": "read",
                "outcome": "failed",
                "classification": "unexpected_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            workflow_event(
                kind="warning",
                msg=workflow_event_name,
                details=details,
            )
            raise LongTermRemoteReadFailedError(
                "Required remote long-term fast-topic retrieval failed.",
                details=details,
            ) from exc
        except Exception as exc:
            logger.warning(
                "Failed to build fast-topic memory hints inside the quick-memory path. error_type=%s",
                type(exc).__name__,
            )
            return None

    def build_tool_provider_context(
        self,
        query_text: str | None,
        *,
        include_graph_fallback: bool = True,
    ) -> LongTermMemoryContext:
        """Build a tool-facing context with sensitive details redacted."""

        started_at = time.monotonic()
        try:
            query = self.query_rewriter.profile(query_text)
            writer_details = _writer_state_details(getattr(self, "writer", None))
            multimodal_writer_details = _writer_state_details(getattr(self, "multimodal_writer", None))
            with workflow_span(
                name="longterm_service_build_tool_provider_context",
                kind="retrieval",
                details={
                    "query_present": bool(str(query_text or "").strip()),
                    "conversation_writer": writer_details,
                    "multimodal_writer": multimodal_writer_details,
                },
            ):
                workflow_event(
                    kind="branch",
                    msg="longterm_tool_provider_context_shared_lock_skipped",
                    details={
                        "lock_used": False,
                        "conversation_writer": writer_details,
                        "multimodal_writer": multimodal_writer_details,
                    },
                )
                resolution = self._consume_prepared_context(
                    profile="tool",
                    query=query,
                    build_context=lambda: self._build_tool_provider_context_uncached(
                        query=query,
                        original_query_text=query_text,
                        include_graph_fallback=include_graph_fallback,
                    ),
                )
                tool_context = resolution.context
                self._remember_context_snapshot(
                    profile="tool",
                    query=query,
                    context=tool_context,
                    source=resolution.source,
                )
                workflow_event(
                    kind="retrieval",
                    msg="longterm_tool_provider_context_built",
                    details={
                        **_context_details(tool_context),
                        "prepared_source": resolution.source,
                    },
                )
                return tool_context
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to build tool-facing long-term provider context.")
            return LongTermMemoryContext()
        finally:
            record_voice_turn_remote_memory_ready(
                duration_ms=max(0.0, (time.monotonic() - started_at) * 1000.0),
            )

    def prewarm_provider_context(
        self,
        query_text: str | None,
        *,
        tool_context: bool = False,
        sticky: bool = False,
        rewrite_query: bool = False,
    ) -> bool:
        """Schedule a prepared provider-context build outside the answer hot path."""

        if not self.config.long_term_memory_enabled:
            return False
        normalized_text = str(query_text or "").strip()
        if not normalized_text:
            return False
        profile_name: PreparedContextProfile = "tool" if tool_context else "provider"
        query = (
            self.query_rewriter.profile(
                query_text,
                wait_for_rewrite_s=_PREWARM_QUERY_REWRITE_WAIT_S,
            )
            if rewrite_query
            else LongTermQueryProfile.from_text(query_text)
        )
        key_texts = self._prepared_context_key_texts(query)
        if not key_texts:
            return False
        if tool_context:
            prepared_context_front = getattr(self, "prepared_context_front", None)
            if prepared_context_front is None:
                return False
            def build_context() -> LongTermMemoryContext:
                return self._build_tool_provider_context_uncached(
                    query=query,
                    original_query_text=query_text,
                    include_graph_fallback=True,
                )
            scheduled = prepared_context_front.prime_async(
                profile=profile_name,
                cache_key_texts=key_texts,
                sticky_query_text=query.original_text or query.retrieval_text,
                build_context=build_context,
                sticky=sticky,
            )
        else:
            provider_answer_front = getattr(self, "provider_answer_front", None)
            def build_context() -> LongTermMemoryContext:
                return self._build_provider_context_uncached(
                    query=query,
                    original_query_text=query_text,
                )
            if provider_answer_front is not None and provider_answer_front.enabled():
                scheduled = provider_answer_front.prime_async(
                    query_keys=key_texts,
                    sticky_query_text=query.original_text or query.retrieval_text,
                    build_context=build_context,
                    sticky=sticky,
                )
            else:
                prepared_context_front = getattr(self, "prepared_context_front", None)
                if prepared_context_front is None:
                    return False
                scheduled = prepared_context_front.prime_async(
                    profile=profile_name,
                    cache_key_texts=key_texts,
                    sticky_query_text=query.original_text or query.retrieval_text,
                    build_context=build_context,
                    sticky=sticky,
                )
        workflow_event(
            kind="cache",
            msg="longterm_prepared_context_prewarm_requested",
            details={
                "profile": profile_name,
                "scheduled": scheduled,
                "sticky": sticky,
                "rewrite_query": rewrite_query,
                "query_present": bool(normalized_text),
            },
        )
        return scheduled

    def _build_provider_context_uncached(
        self,
        *,
        query: LongTermQueryProfile,
        original_query_text: str | None,
    ) -> LongTermMemoryContext:
        with self._temporary_remote_probe_cache():
            with workflow_span(
                name="longterm_service_provider_context_retrieval",
                kind="retrieval",
                details={
                    "query_present": bool(query.retrieval_text),
                    "prepared_query": True,
                },
            ):
                return self.retriever.build_context(
                    query=query,
                    original_query_text=original_query_text,
                )

    def _build_tool_provider_context_uncached(
        self,
        *,
        query: LongTermQueryProfile,
        original_query_text: str | None,
        include_graph_fallback: bool = True,
    ) -> LongTermMemoryContext:
        with self._temporary_remote_probe_cache():
            return self.retriever.build_tool_context(
                query=query,
                original_query_text=original_query_text,
                include_graph_fallback=include_graph_fallback,
            )

    def _consume_prepared_context(
        self,
        *,
        profile: PreparedContextProfile,
        query: LongTermQueryProfile,
        build_context: Callable[[], LongTermMemoryContext],
    ) -> PreparedLongTermContextResolution:
        prepared_context_front = getattr(self, "prepared_context_front", None)
        if prepared_context_front is None:
            return PreparedLongTermContextResolution(context=build_context(), source="built_sync")
        return prepared_context_front.consume_or_build(
            profile=profile,
            cache_key_texts=self._prepared_context_key_texts(query),
            sticky_query_text=query.original_text or query.retrieval_text,
            build_context=build_context,
        )

    def _prepared_context_key_texts(self, query: LongTermQueryProfile) -> tuple[str, ...]:
        candidates = (
            query.retrieval_text,
            query.original_text,
            query.canonical_english_text or "",
        )
        ordered: list[str] = []
        for candidate in candidates:
            normalized = str(candidate or "").strip()
            if normalized and normalized not in ordered:
                ordered.append(normalized)
        return tuple(ordered)

    def _schedule_prepared_context_refresh(
        self,
        profile: PreparedContextProfile,
        query_text: str,
    ) -> None:
        self.prewarm_provider_context(
            query_text,
            tool_context=profile == "tool",
            sticky=True,
            rewrite_query=True,
        )

    def _schedule_provider_answer_front_refresh(self, query_text: str) -> None:
        """Refresh one sticky live-provider front after invalidation/startup."""

        self.prewarm_provider_context(
            query_text,
            tool_context=False,
            sticky=True,
            rewrite_query=True,
        )

    def _invalidate_prepared_contexts(
        self,
        *,
        reason: str,
        refresh_recent: bool = True,
    ) -> None:
        prepared_context_front = getattr(self, "prepared_context_front", None)
        if prepared_context_front is not None:
            generation = prepared_context_front.invalidate()
            refreshed = 0
            if refresh_recent:
                refreshed = prepared_context_front.refresh_recent_async(
                    build_for_query=self._schedule_prepared_context_refresh,
                )
            workflow_event(
                kind="cache",
                msg="longterm_prepared_context_invalidated",
                details={
                    "reason": reason,
                    "generation": generation,
                    "refresh_recent": refresh_recent,
                    "sticky_queries": prepared_context_front.sticky_query_count(),
                    "refresh_scheduled": refreshed,
                },
            )
        provider_answer_front = getattr(self, "provider_answer_front", None)
        if provider_answer_front is not None and provider_answer_front.enabled():
            generation = provider_answer_front.invalidate()
            refreshed = 0
            if refresh_recent:
                refreshed = provider_answer_front.refresh_recent_async(
                    build_for_query=self._schedule_provider_answer_front_refresh,
                )
            workflow_event(
                kind="cache",
                msg="longterm_materialized_provider_front_invalidated",
                details={
                    "reason": reason,
                    "generation": generation,
                    "refresh_recent": refresh_recent,
                    "sticky_queries": provider_answer_front.sticky_query_count(),
                    "refresh_scheduled": refreshed,
                },
            )

    def _context_snapshot_state(
        self,
    ) -> tuple[threading.RLock, dict[str, LongTermContextSnapshot]]:
        """Return the mutable store that tracks the newest built context snapshots."""

        lock = getattr(self, "_context_snapshot_lock", None)
        if lock is None:
            lock = threading.RLock()
            setattr(self, "_context_snapshot_lock", lock)
        snapshots = getattr(self, "_latest_context_snapshots", None)
        if snapshots is None:
            snapshots = {}
            setattr(self, "_latest_context_snapshots", snapshots)
        return lock, cast(dict[str, LongTermContextSnapshot], snapshots)

    def _remember_context_snapshot(
        self,
        *,
        profile: PreparedContextProfile,
        query: LongTermQueryProfile,
        context: LongTermMemoryContext,
        source: str,
    ) -> None:
        """Persist the latest built provider/tool context for operator inspection."""

        lock, snapshots = self._context_snapshot_state()
        with lock:
            snapshots[profile] = LongTermContextSnapshot(
                profile=profile,
                query_profile=query,
                context=context,
                source=str(source or "built_sync"),
            )

    def _consume_materialized_provider_context(
        self,
        *,
        query: LongTermQueryProfile,
        strict_missing: bool,
    ) -> PreparedLongTermContextResolution | None:
        """Consume the materialized live-provider answer front when available."""

        provider_answer_front = getattr(self, "provider_answer_front", None)
        if provider_answer_front is None or not provider_answer_front.enabled():
            return None
        resolution = provider_answer_front.consume(
            query_keys=self._prepared_context_key_texts(query),
            strict_missing=strict_missing,
        )
        if resolution is None:
            return None
        return PreparedLongTermContextResolution(
            context=resolution.context,
            source=resolution.source,
        )

    def _persist_materialized_provider_context(
        self,
        *,
        query: LongTermQueryProfile,
        context: LongTermMemoryContext,
    ) -> None:
        """Schedule persistence of one built provider context into the live front.

        The compatibility provider-context path may still build the full
        retriever result synchronously, but it must not also wait on the remote
        current-head write for the live materialized front. Live transcript-first
        prewarms own the authoritative front build; this compatibility path only
        seeds the same front in the background when no materialized version is
        ready yet.
        """

        provider_answer_front = getattr(self, "provider_answer_front", None)
        if provider_answer_front is None or not provider_answer_front.enabled():
            return
        provider_answer_front.schedule_persist_built_context(
            query_keys=self._prepared_context_key_texts(query),
            sticky_query_text=query.original_text or query.retrieval_text,
            context=context,
        )
