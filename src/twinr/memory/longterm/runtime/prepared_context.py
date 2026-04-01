"""Prepared long-term prompt fronts for live runtime consumption.

The live answer path should not rebuild the full long-term provider context
from raw remote memory every time the user finishes speaking. This module owns
small in-process prepared prompt artifacts instead: completed
``LongTermMemoryContext`` values keyed by query/profile plus a tiny amount of
speculative scheduling state so transcript-first partials can prewarm the next
turn without flooding the remote store.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Literal

from twinr.memory.longterm.core.models import LongTermMemoryContext


PreparedContextProfile = Literal["provider", "tool"]
_DEFAULT_MAX_CACHE_ENTRIES = 12
_DEFAULT_MAX_STICKY_QUERIES = 6
_DEFAULT_MAX_WORKERS = 2


@dataclass(frozen=True, slots=True)
class PreparedLongTermContextResolution:
    """Describe how one prepared long-term context was obtained."""

    context: LongTermMemoryContext
    source: Literal[
        "cache_hit",
        "inflight_hit",
        "built_sync",
        "materialized_local_cache_hit",
        "materialized_remote_front_hit",
        "materialized_built_sync",
    ]


@dataclass(frozen=True, slots=True)
class _PreparedContextEntry:
    """Store one prepared context together with its validity metadata."""

    profile: PreparedContextProfile
    generation: int
    built_at_monotonic: float
    context: LongTermMemoryContext


@dataclass(frozen=True, slots=True)
class _PendingPreparedBuild:
    """Describe one deferred speculative build for the newest transcript."""

    profile: PreparedContextProfile
    primary_key: tuple[PreparedContextProfile, str]
    alias_keys: tuple[tuple[PreparedContextProfile, str], ...]
    sticky_query_text: str | None
    build_context: Callable[[], LongTermMemoryContext]
    sticky: bool


class PreparedLongTermContextFront:
    """Maintain prepared full-context fronts for live long-term retrieval.

    The front keeps a bounded set of completed contexts plus a small speculative
    queue per profile. Only the newest speculative request for a profile is
    retained while another speculative build is already in flight, which keeps
    transcript-first partials from spawning one remote rebuild per token.
    """

    def __init__(
        self,
        *,
        max_cache_entries: int = _DEFAULT_MAX_CACHE_ENTRIES,
        max_sticky_queries: int = _DEFAULT_MAX_STICKY_QUERIES,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        self._lock = threading.RLock()
        self._generation = 0
        self._max_cache_entries = max(1, int(max_cache_entries))
        self._max_sticky_queries = max(1, int(max_sticky_queries))
        self._entries: OrderedDict[tuple[PreparedContextProfile, str], _PreparedContextEntry] = OrderedDict()
        self._inflight: dict[tuple[PreparedContextProfile, str], Future[LongTermMemoryContext]] = {}
        self._inflight_aliases: dict[tuple[PreparedContextProfile, str], tuple[tuple[PreparedContextProfile, str], ...]] = {}
        self._recent_queries: OrderedDict[tuple[PreparedContextProfile, str], str] = OrderedDict()
        self._speculative_inflight: dict[PreparedContextProfile, tuple[PreparedContextProfile, str]] = {}
        self._pending_speculative: dict[PreparedContextProfile, _PendingPreparedBuild] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="twinr-longterm-front",
        )

    def shutdown(self, *, wait: bool = True) -> None:
        """Release background resources for prepared-context builds."""

        self._executor.shutdown(wait=wait, cancel_futures=True)

    def generation(self) -> int:
        """Return the current invalidation generation."""

        with self._lock:
            return self._generation

    def sticky_query_count(self) -> int:
        """Return the number of remembered sticky queries."""

        with self._lock:
            return len(self._recent_queries)

    def consume_or_build(
        self,
        *,
        profile: PreparedContextProfile,
        cache_key_texts: Sequence[str],
        sticky_query_text: str | None,
        build_context: Callable[[], LongTermMemoryContext],
    ) -> PreparedLongTermContextResolution:
        """Return a prepared context, waiting on identical in-flight work when possible."""

        primary_key, alias_keys = self._normalize_keys(profile=profile, cache_key_texts=cache_key_texts)
        if primary_key is None:
            return PreparedLongTermContextResolution(
                context=build_context(),
                source="built_sync",
            )

        self._remember_sticky_query(primary_key=primary_key, sticky_query_text=sticky_query_text)
        cached = self._lookup_cached(alias_keys)
        if cached is not None:
            return PreparedLongTermContextResolution(context=cached.context, source="cache_hit")

        with self._lock:
            future = self._lookup_inflight_unlocked(alias_keys)
        if future is not None:
            try:
                return PreparedLongTermContextResolution(context=future.result(), source="inflight_hit")
            except Exception:
                pass

        context = build_context()
        self._store_completed_context(
            profile=profile,
            primary_key=primary_key,
            alias_keys=alias_keys,
            context=context,
            generation=self.generation(),
        )
        return PreparedLongTermContextResolution(context=context, source="built_sync")

    def prime_async(
        self,
        *,
        profile: PreparedContextProfile,
        cache_key_texts: Sequence[str],
        sticky_query_text: str | None,
        build_context: Callable[[], LongTermMemoryContext],
        sticky: bool,
    ) -> bool:
        """Schedule one prepared-context build when the current generation needs it."""

        primary_key, alias_keys = self._normalize_keys(profile=profile, cache_key_texts=cache_key_texts)
        if primary_key is None:
            return False
        if sticky:
            self._remember_sticky_query(primary_key=primary_key, sticky_query_text=sticky_query_text)
        if self._lookup_cached(alias_keys) is not None:
            return False
        with self._lock:
            if self._lookup_inflight_unlocked(alias_keys) is not None:
                return False
            if not sticky:
                speculative_key = self._speculative_inflight.get(profile)
                if speculative_key is not None and speculative_key != primary_key:
                    self._pending_speculative[profile] = _PendingPreparedBuild(
                        profile=profile,
                        primary_key=primary_key,
                        alias_keys=alias_keys,
                        sticky_query_text=sticky_query_text,
                        build_context=build_context,
                        sticky=sticky,
                    )
                    return False
            generation = self._generation
            future = self._executor.submit(build_context)
            self._inflight[primary_key] = future
            self._inflight_aliases[primary_key] = alias_keys
            if not sticky:
                self._speculative_inflight[profile] = primary_key
            future.add_done_callback(
                lambda finished, *, callback_profile=profile, callback_primary=primary_key, callback_aliases=alias_keys, callback_generation=generation, callback_sticky=sticky: self._complete_async_build(
                    profile=callback_profile,
                    primary_key=callback_primary,
                    alias_keys=callback_aliases,
                    generation=callback_generation,
                    sticky=callback_sticky,
                    future=finished,
                )
            )
            return True

    def invalidate(self) -> int:
        """Drop completed prepared contexts and advance to the next generation."""

        with self._lock:
            self._generation += 1
            self._entries.clear()
            self._inflight.clear()
            self._inflight_aliases.clear()
            self._speculative_inflight.clear()
            self._pending_speculative.clear()
            return self._generation

    def refresh_recent_async(
        self,
        *,
        build_for_query: Callable[[PreparedContextProfile, str], None],
    ) -> int:
        """Request eager refreshes for the remembered sticky queries."""

        with self._lock:
            recent_queries = tuple(self._recent_queries.items())
        for (profile, _normalized_key), query_text in recent_queries:
            build_for_query(profile, query_text)
        return len(recent_queries)

    def _complete_async_build(
        self,
        *,
        profile: PreparedContextProfile,
        primary_key: tuple[PreparedContextProfile, str],
        alias_keys: tuple[tuple[PreparedContextProfile, str], ...],
        generation: int,
        sticky: bool,
        future: Future[LongTermMemoryContext],
    ) -> None:
        with self._lock:
            self._inflight.pop(primary_key, None)
            self._inflight_aliases.pop(primary_key, None)
            if not sticky and self._speculative_inflight.get(profile) == primary_key:
                self._speculative_inflight.pop(profile, None)
        try:
            context = future.result()
        except Exception:
            context = None
        if context is not None:
            self._store_completed_context(
                profile=profile,
                primary_key=primary_key,
                alias_keys=alias_keys,
                context=context,
                generation=generation,
            )
        pending = None
        with self._lock:
            if profile in self._pending_speculative and profile not in self._speculative_inflight:
                pending = self._pending_speculative.pop(profile)
        if pending is not None:
            self.prime_async(
                profile=pending.profile,
                cache_key_texts=tuple(key_text for _, key_text in pending.alias_keys),
                sticky_query_text=pending.sticky_query_text,
                build_context=pending.build_context,
                sticky=pending.sticky,
            )

    def _store_completed_context(
        self,
        *,
        profile: PreparedContextProfile,
        primary_key: tuple[PreparedContextProfile, str],
        alias_keys: tuple[tuple[PreparedContextProfile, str], ...],
        context: LongTermMemoryContext,
        generation: int,
    ) -> None:
        if generation != self.generation():
            return
        entry = _PreparedContextEntry(
            profile=profile,
            generation=generation,
            built_at_monotonic=time.monotonic(),
            context=context,
        )
        with self._lock:
            if generation != self._generation:
                return
            for key in alias_keys:
                self._entries[key] = entry
                self._entries.move_to_end(key)
            while len(self._entries) > self._max_cache_entries:
                self._entries.popitem(last=False)

    def _lookup_cached(
        self,
        alias_keys: tuple[tuple[PreparedContextProfile, str], ...],
    ) -> _PreparedContextEntry | None:
        with self._lock:
            for key in alias_keys:
                entry = self._entries.get(key)
                if entry is None or entry.generation != self._generation:
                    continue
                self._entries.move_to_end(key)
                return entry
        return None

    def _lookup_inflight_unlocked(
        self,
        alias_keys: tuple[tuple[PreparedContextProfile, str], ...],
    ) -> Future[LongTermMemoryContext] | None:
        for key in alias_keys:
            future = self._inflight.get(key)
            if future is not None:
                return future
        for primary_key, inflight_aliases in self._inflight_aliases.items():
            if any(alias in inflight_aliases for alias in alias_keys):
                future = self._inflight.get(primary_key)
                if future is not None:
                    return future
        return None

    def _remember_sticky_query(
        self,
        *,
        primary_key: tuple[PreparedContextProfile, str],
        sticky_query_text: str | None,
    ) -> None:
        normalized_query = self._normalize_query_text(sticky_query_text)
        if not normalized_query:
            return
        with self._lock:
            self._recent_queries[primary_key] = normalized_query
            self._recent_queries.move_to_end(primary_key)
            while len(self._recent_queries) > self._max_sticky_queries:
                self._recent_queries.popitem(last=False)

    def _normalize_keys(
        self,
        *,
        profile: PreparedContextProfile,
        cache_key_texts: Sequence[str],
    ) -> tuple[
        tuple[PreparedContextProfile, str] | None,
        tuple[tuple[PreparedContextProfile, str], ...],
    ]:
        normalized_keys: list[tuple[PreparedContextProfile, str]] = []
        seen: set[str] = set()
        for candidate in cache_key_texts:
            normalized = self._normalize_key_text(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_keys.append((profile, normalized))
        if not normalized_keys:
            return None, ()
        return normalized_keys[0], tuple(normalized_keys)

    @staticmethod
    def _normalize_key_text(value: str | None) -> str:
        return " ".join(str(value or "").split()).strip().lower()

    @staticmethod
    def _normalize_query_text(value: str | None) -> str:
        return " ".join(str(value or "").split()).strip()
