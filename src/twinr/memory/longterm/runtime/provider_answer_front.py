"""Runtime manager for materialized live provider answer fronts.

The live provider path should consume current-head-backed answer fronts instead
of rebuilding a full long-term retriever context inline. This manager keeps a
small in-process cache for hot consumption, persists freshly built fronts to
remote current-head storage, and coalesces transcript-first speculative
prewarms so partial transcripts do not trigger one rebuild per token.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time

from twinr.memory.longterm.core.models import LongTermMemoryContext
from twinr.memory.longterm.storage.provider_answer_front_store import (
    LongTermProviderAnswerFrontStore,
    MaterializedProviderAnswerFrontRecord,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

_DEFAULT_MAX_CACHE_ENTRIES = 12
_DEFAULT_MAX_STICKY_QUERIES = 6
_DEFAULT_MAX_WORKERS = 2


def _normalize_keys(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize one ordered query-key set without destroying wording."""

    normalized: list[str] = []
    for raw_value in values:
        candidate = " ".join(str(raw_value or "").split()).strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class MaterializedProviderAnswerFrontResolution:
    """Describe how one materialized provider context was obtained."""

    context: LongTermMemoryContext
    source: str


@dataclass(frozen=True, slots=True)
class _CachedProviderAnswerFront:
    """Keep one completed answer front together with generation metadata."""

    record: MaterializedProviderAnswerFrontRecord
    cached_at_monotonic: float


@dataclass(frozen=True, slots=True)
class _PendingProviderAnswerBuild:
    """Carry the newest queued speculative provider-front build."""

    query_keys: tuple[str, ...]
    sticky_query_text: str | None
    build_context: Callable[[], LongTermMemoryContext]
    sticky: bool


class MaterializedProviderAnswerFront:
    """Maintain live provider fronts backed by remote current-head storage."""

    def __init__(
        self,
        store: LongTermProviderAnswerFrontStore,
        *,
        max_cache_entries: int = _DEFAULT_MAX_CACHE_ENTRIES,
        max_sticky_queries: int = _DEFAULT_MAX_STICKY_QUERIES,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        self.store = store
        self._lock = threading.RLock()
        # Service construction must stay side-effect free. The authoritative
        # generation is learned lazily from persisted fronts or invalidations
        # instead of probing remote current heads during builder wiring.
        self._generation = 0
        self._max_cache_entries = max(1, int(max_cache_entries))
        self._max_sticky_queries = max(1, int(max_sticky_queries))
        self._entries: OrderedDict[str, _CachedProviderAnswerFront] = OrderedDict()
        self._recent_queries: OrderedDict[str, str] = OrderedDict()
        self._inflight: dict[str, Future[LongTermMemoryContext]] = {}
        self._speculative_inflight: str | None = None
        self._pending_speculative: _PendingProviderAnswerBuild | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="twinr-provider-front",
        )
        self._accepting_work = True

    def enabled(self) -> bool:
        """Return whether remote-authoritative provider fronts are usable."""

        return self.store.enabled()

    def shutdown(self, *, wait: bool = True) -> None:
        """Release background resources and reject new answer-front jobs."""

        executor: ThreadPoolExecutor | None
        with self._lock:
            self._accepting_work = False
            executor = self._executor
            if wait:
                self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)

    def generation(self) -> int:
        """Return the newest known current-head generation."""

        with self._lock:
            return self._generation

    def sticky_query_count(self) -> int:
        """Return the number of remembered sticky query texts."""

        with self._lock:
            return len(self._recent_queries)

    def consume(
        self,
        *,
        query_keys: Sequence[str],
        strict_missing: bool,
    ) -> MaterializedProviderAnswerFrontResolution | None:
        """Return a materialized provider context without running the retriever."""

        normalized_keys = _normalize_keys(query_keys)
        if not normalized_keys or not self.enabled():
            return None
        cached = self._lookup_cached(normalized_keys)
        if cached is not None:
            return MaterializedProviderAnswerFrontResolution(
                context=cached.record.context,
                source="materialized_local_cache_hit",
            )
        record = self.store.load_front(query_keys=normalized_keys)
        if record is not None:
            self._store_cached_record(record)
            with self._lock:
                self._generation = max(self._generation, record.generation)
            return MaterializedProviderAnswerFrontResolution(
                context=record.context,
                source="materialized_remote_front_hit",
            )
        if strict_missing:
            raise LongTermRemoteUnavailableError(
                "Required materialized provider answer front is missing or stale for the live runtime path."
            )
        return None

    def persist_built_context(
        self,
        *,
        query_keys: Sequence[str],
        sticky_query_text: str | None,
        context: LongTermMemoryContext,
    ) -> MaterializedProviderAnswerFrontResolution | None:
        """Persist one already-built provider context as a materialized front."""

        normalized_keys = _normalize_keys(query_keys)
        if not normalized_keys or not self.enabled():
            return None
        return self._persist_built_context_sync(
            query_keys=normalized_keys,
            sticky_query_text=sticky_query_text,
            context=context,
            source="materialized_built_sync",
        )

    def prime_async(
        self,
        *,
        query_keys: Sequence[str],
        sticky_query_text: str | None,
        build_context: Callable[[], LongTermMemoryContext],
        sticky: bool,
    ) -> bool:
        """Schedule one remote-backed provider-front materialization build."""

        normalized_keys = _normalize_keys(query_keys)
        if not normalized_keys or not self.enabled():
            return False
        if sticky:
            self._remember_sticky_query(normalized_keys=normalized_keys, sticky_query_text=sticky_query_text)
        primary_key = normalized_keys[0]
        if self._lookup_cached(normalized_keys) is not None:
            return False
        with self._lock:
            if not self._accepting_work:
                return False
            if primary_key in self._inflight:
                return False
            if not sticky and self._speculative_inflight is not None and self._speculative_inflight != primary_key:
                self._pending_speculative = _PendingProviderAnswerBuild(
                    query_keys=normalized_keys,
                    sticky_query_text=sticky_query_text,
                    build_context=build_context,
                    sticky=sticky,
                )
                return False
            executor = self._executor
            if executor is None:
                return False
            try:
                future = executor.submit(
                    self._build_and_materialize_context,
                    query_keys=normalized_keys,
                    sticky_query_text=sticky_query_text,
                    build_context=build_context,
                )
            except RuntimeError:
                self._accepting_work = False
                return False
            self._inflight[primary_key] = future
            if not sticky:
                self._speculative_inflight = primary_key
            future.add_done_callback(
                lambda finished, *, callback_keys=normalized_keys, callback_sticky=sticky: self._complete_async_materialization(
                    query_keys=callback_keys,
                    future=finished,
                    sticky=callback_sticky,
                )
            )
        return True

    def schedule_persist_built_context(
        self,
        *,
        query_keys: Sequence[str],
        sticky_query_text: str | None,
        context: LongTermMemoryContext,
    ) -> bool:
        """Persist one already-built provider context in the background.

        Compatibility callers may still need to build a broad provider context
        synchronously. They must not then block the same foreground answer path
        on the remote current-head write for the materialized live front.
        """

        normalized_keys = _normalize_keys(query_keys)
        if not normalized_keys or not self.enabled():
            return False
        if self._lookup_cached(normalized_keys) is not None:
            return False
        primary_key = normalized_keys[0]
        self._remember_sticky_query(normalized_keys=normalized_keys, sticky_query_text=sticky_query_text)
        with self._lock:
            if not self._accepting_work:
                return False
            if primary_key in self._inflight:
                return False
            executor = self._executor
            if executor is None:
                return False
            try:
                future = executor.submit(
                    self._materialize_context,
                    query_keys=normalized_keys,
                    sticky_query_text=sticky_query_text,
                    context=context,
                )
            except RuntimeError:
                self._accepting_work = False
                return False
            self._inflight[primary_key] = future
            future.add_done_callback(
                lambda finished, *, callback_keys=normalized_keys: self._complete_async_materialization(
                    query_keys=callback_keys,
                    future=finished,
                    sticky=True,
                )
            )
        return True

    def invalidate(self) -> int:
        """Advance to the next generation and clear completed local fronts."""

        if not self.enabled():
            return 0
        generation = self.store.invalidate(reason="runtime_provider_context_invalidated")
        with self._lock:
            self._generation = generation
            self._entries.clear()
            self._inflight.clear()
            self._speculative_inflight = None
            self._pending_speculative = None
        return generation

    def refresh_recent_async(
        self,
        *,
        build_for_query: Callable[[str], None],
    ) -> int:
        """Request eager rebuilds for sticky queries after invalidation/startup."""

        with self._lock:
            if not self._accepting_work:
                return 0
            recent_queries = tuple(self._recent_queries.values())
        for query_text in recent_queries:
            build_for_query(query_text)
        return len(recent_queries)

    def _complete_async_materialization(
        self,
        *,
        query_keys: tuple[str, ...],
        sticky: bool,
        future: Future[LongTermMemoryContext],
    ) -> None:
        primary_key = query_keys[0]
        with self._lock:
            self._inflight.pop(primary_key, None)
            if not sticky and self._speculative_inflight == primary_key:
                self._speculative_inflight = None
        try:
            future.result()
        except Exception:
            pass
        pending = None
        with self._lock:
            if self._pending_speculative is not None and self._speculative_inflight is None:
                pending = self._pending_speculative
                self._pending_speculative = None
        if pending is not None:
            self.prime_async(
                query_keys=pending.query_keys,
                sticky_query_text=pending.sticky_query_text,
                build_context=pending.build_context,
                sticky=pending.sticky,
            )

    def _lookup_cached(self, normalized_keys: Sequence[str]) -> _CachedProviderAnswerFront | None:
        """Return one matching local cache entry for the active generation."""

        with self._lock:
            for front_id, entry in tuple(self._entries.items()):
                if entry.record.generation != self._generation:
                    self._entries.pop(front_id, None)
                    continue
                if {entry.record.front_key, *entry.record.aliases}.intersection(normalized_keys):
                    self._entries.move_to_end(front_id)
                    return entry
        return None

    def _store_cached_record(self, record: MaterializedProviderAnswerFrontRecord) -> None:
        """Store one loaded or freshly persisted front in the local hot cache."""

        with self._lock:
            self._entries[record.front_id] = _CachedProviderAnswerFront(
                record=record,
                cached_at_monotonic=time.monotonic(),
            )
            self._entries.move_to_end(record.front_id)
            while len(self._entries) > self._max_cache_entries:
                self._entries.popitem(last=False)

    def _remember_sticky_query(
        self,
        *,
        normalized_keys: Sequence[str],
        sticky_query_text: str | None,
    ) -> None:
        """Remember the newest sticky query text for later refreshes."""

        resolved_query = " ".join(str(sticky_query_text or "").split()).strip()
        if not resolved_query:
            return
        primary_key = normalized_keys[0]
        with self._lock:
            self._recent_queries[primary_key] = resolved_query
            self._recent_queries.move_to_end(primary_key)
            while len(self._recent_queries) > self._max_sticky_queries:
                self._recent_queries.popitem(last=False)

    def _build_and_materialize_context(
        self,
        *,
        query_keys: tuple[str, ...],
        sticky_query_text: str | None,
        build_context: Callable[[], LongTermMemoryContext],
    ) -> LongTermMemoryContext:
        """Build one provider context and persist the materialized front once."""

        context = build_context()
        return self._materialize_context(
            query_keys=query_keys,
            sticky_query_text=sticky_query_text,
            context=context,
        )

    def _materialize_context(
        self,
        *,
        query_keys: tuple[str, ...],
        sticky_query_text: str | None,
        context: LongTermMemoryContext,
    ) -> LongTermMemoryContext:
        """Persist one already-built context and return it for callers/tests."""

        self._persist_built_context_sync(
            query_keys=query_keys,
            sticky_query_text=sticky_query_text,
            context=context,
            source="materialized_built_async",
        )
        return context

    def _persist_built_context_sync(
        self,
        *,
        query_keys: tuple[str, ...],
        sticky_query_text: str | None,
        context: LongTermMemoryContext,
        source: str,
    ) -> MaterializedProviderAnswerFrontResolution:
        """Persist one front synchronously and update the local cache state."""

        self._remember_sticky_query(normalized_keys=query_keys, sticky_query_text=sticky_query_text)
        record = self.store.save_front(query_keys=query_keys, context=context)
        self._store_cached_record(record)
        with self._lock:
            self._generation = max(self._generation, record.generation)
        return MaterializedProviderAnswerFrontResolution(
            context=context,
            source=source,
        )
