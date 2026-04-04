"""Catalog cache and snapshot-assembly helpers for remote catalog storage."""

from __future__ import annotations

from collections.abc import Mapping
import time

from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

from ._typing import RemoteCatalogMixinBase
from .shared import (
    LongTermRemoteCatalogAssemblyResult,
    LongTermRemoteCatalogEntry,
    _CATALOG_VERSION,
    _CachedCatalogEntries,
    _CachedItemPayload,
    _CachedLocalSearchSelector,
    _CachedSearchResult,
    _LEGACY_CATALOG_VERSION,
    _RemoteCollectionDefinition,
)


class RemoteCatalogCatalogMixin(RemoteCatalogMixinBase):
    def probe_catalog_payload_result(
        self,
        *,
        snapshot_kind: str,
        fast_fail: bool = False,
    ) -> tuple[str, dict[str, object] | None]:
        """Probe one fixed current head and preserve missing-vs-invalid status.

        Callers that bootstrap or repair remote state need to distinguish a
        fresh missing current head from an already-existing but malformed head.
        The latter should trigger a repair write instead of being treated as a
        harmless empty namespace forever.
        """

        result = self._load_catalog_head_result(
            snapshot_kind=snapshot_kind,
            metadata_only=True,
            fast_fail=fast_fail,
        )
        if isinstance(result.payload, Mapping):
            payload = dict(result.payload)
            self._store_recent_catalog_head_payload(snapshot_kind=snapshot_kind, payload=payload)
            return result.status, payload
        cached = self._recent_catalog_head_payload(snapshot_kind=snapshot_kind)
        if isinstance(cached, Mapping):
            return "found", dict(cached)
        return result.status, None

    def load_catalog_payload(self, *, snapshot_kind: str) -> dict[str, object] | None:
        """Load the authoritative current catalog payload for one collection."""

        payload = self._load_catalog_payload(snapshot_kind=snapshot_kind)
        if isinstance(payload, Mapping):
            self._store_recent_catalog_head_payload(snapshot_kind=snapshot_kind, payload=payload)
            return dict(payload)
        cached = self._recent_catalog_head_payload(snapshot_kind=snapshot_kind)
        if isinstance(cached, Mapping):
            return dict(cached)
        return None

    def probe_catalog_payload(self, *, snapshot_kind: str) -> dict[str, object] | None:
        """Probe the fixed-URI current head document without falling back to snapshots."""

        _status, payload = self.probe_catalog_payload_result(snapshot_kind=snapshot_kind)
        if isinstance(payload, Mapping):
            return dict(payload)
        return None

    def _recent_catalog_head_payload(self, *, snapshot_kind: str) -> dict[str, object] | None:
        """Return one bounded same-process current-head payload bridge."""

        now = time.monotonic()
        with self._cache_lock:
            cached = self._recent_catalog_head_cache.get(snapshot_kind)
            if cached is None:
                return None
            if cached.expires_at_monotonic <= now:
                self._recent_catalog_head_cache.pop(snapshot_kind, None)
                return None
            return dict(cached.payload)

    def _store_recent_catalog_head_payload(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> None:
        """Keep the latest same-process catalog head long enough for read-after-write probes."""

        with self._cache_lock:
            self._invalid_catalog_head_cache.pop(snapshot_kind, None)
            self._recent_catalog_head_cache[snapshot_kind] = _CachedItemPayload(
                payload=dict(payload),
                expires_at_monotonic=time.monotonic() + self._recent_catalog_head_ttl_s(),
            )

    def _cached_catalog_entries(self, *, snapshot_kind: str) -> tuple[LongTermRemoteCatalogEntry, ...] | None:
        ttl_s = self._persistent_read_cache_ttl_s()
        if ttl_s <= 0.0:
            return None
        now = time.monotonic()
        with self._cache_lock:
            cached = self._catalog_entries_cache.get(snapshot_kind)
            if cached is None:
                return None
            if cached.expires_at_monotonic <= now:
                self._catalog_entries_cache.pop(snapshot_kind, None)
                self._local_search_selector_cache.pop(snapshot_kind, None)
                return None
            return cached.entries

    def _recent_catalog_entries(self, *, snapshot_kind: str) -> tuple[LongTermRemoteCatalogEntry, ...] | None:
        """Return one bounded same-process current-catalog bridge."""

        now = time.monotonic()
        with self._cache_lock:
            cached = self._recent_catalog_entries_cache.get(snapshot_kind)
            if cached is None:
                return None
            if cached.expires_at_monotonic <= now:
                self._recent_catalog_entries_cache.pop(snapshot_kind, None)
                return None
            return cached.entries

    def _store_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        entries: tuple[LongTermRemoteCatalogEntry, ...],
    ) -> None:
        ttl_s = self._persistent_read_cache_ttl_s()
        recent_ttl_s = self._recent_catalog_entries_ttl_s()
        if ttl_s <= 0.0 and recent_ttl_s <= 0.0:
            return
        expires_at_monotonic = time.monotonic()
        with self._cache_lock:
            self._unsupported_scope_search_cache.pop(snapshot_kind, None)
            if ttl_s > 0.0:
                self._catalog_entries_cache[snapshot_kind] = _CachedCatalogEntries(
                    entries=entries,
                    expires_at_monotonic=expires_at_monotonic + ttl_s,
                )
                self._local_search_selector_cache[snapshot_kind] = _CachedLocalSearchSelector(
                    selector=self._build_local_search_selector(entries=entries),
                    by_item_id={entry.item_id: entry for entry in entries},
                    expires_at_monotonic=expires_at_monotonic + ttl_s,
                )
            if recent_ttl_s > 0.0:
                self._recent_catalog_entries_cache[snapshot_kind] = _CachedCatalogEntries(
                    entries=entries,
                    expires_at_monotonic=expires_at_monotonic + recent_ttl_s,
                )

    def _build_local_search_selector(
        self,
        *,
        entries: tuple[LongTermRemoteCatalogEntry, ...],
    ) -> FullTextSelector:
        """Build one local full-text selector over the current catalog entries."""

        return FullTextSelector(
            tuple(
                FullTextDocument(
                    doc_id=entry.item_id,
                    category="remote_catalog",
                    content=self._catalog_entry_search_text(entry),
                )
                for entry in entries
            )
        )

    def _local_search_selector(
        self,
        *,
        snapshot_kind: str,
        entries: tuple[LongTermRemoteCatalogEntry, ...],
    ) -> _CachedLocalSearchSelector | None:
        """Return one cached local selector for the current snapshot entries."""

        ttl_s = self._persistent_read_cache_ttl_s()
        if ttl_s <= 0.0:
            return None
        now = time.monotonic()
        with self._cache_lock:
            cached = self._local_search_selector_cache.get(snapshot_kind)
            if cached is None:
                return None
            if cached.expires_at_monotonic <= now:
                self._local_search_selector_cache.pop(snapshot_kind, None)
                return None
            if len(cached.by_item_id) != len(entries):
                self._local_search_selector_cache.pop(snapshot_kind, None)
                return None
            return cached

    def _cached_item_payload(self, *, snapshot_kind: str, item_id: str) -> dict[str, object] | None:
        ttl_s = self._persistent_read_cache_ttl_s()
        if ttl_s <= 0.0:
            return None
        cache_key = (snapshot_kind, item_id)
        now = time.monotonic()
        with self._cache_lock:
            cached = self._item_payload_cache.get(cache_key)
            if cached is None:
                return None
            if cached.expires_at_monotonic <= now:
                self._item_payload_cache.pop(cache_key, None)
                return None
            return dict(cached.payload)

    def _store_item_payload(
        self,
        *,
        snapshot_kind: str,
        item_id: str,
        payload: Mapping[str, object],
    ) -> None:
        ttl_s = self._persistent_read_cache_ttl_s()
        if ttl_s <= 0.0:
            return
        payload_dict = dict(payload)
        with self._cache_lock:
            self._item_payload_cache[(snapshot_kind, item_id)] = _CachedItemPayload(
                payload=payload_dict,
                expires_at_monotonic=time.monotonic() + ttl_s,
            )

    def _cached_search_result(
        self,
        *,
        snapshot_kind: str,
        query_text: str,
        limit: int,
        filtered_item_ids: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        ttl_s = self._persistent_read_cache_ttl_s()
        if ttl_s <= 0.0:
            return None
        cache_key = (snapshot_kind, query_text, limit, filtered_item_ids)
        now = time.monotonic()
        with self._cache_lock:
            cached = self._search_result_cache.get(cache_key)
            if cached is None:
                return None
            if cached.expires_at_monotonic <= now:
                self._search_result_cache.pop(cache_key, None)
                return None
            return cached.item_ids

    def _store_search_result(
        self,
        *,
        snapshot_kind: str,
        query_text: str,
        limit: int,
        filtered_item_ids: tuple[str, ...],
        selected_item_ids: tuple[str, ...],
    ) -> None:
        ttl_s = self._persistent_read_cache_ttl_s()
        if ttl_s <= 0.0:
            return
        cache_key = (snapshot_kind, query_text, limit, filtered_item_ids)
        with self._cache_lock:
            self._search_result_cache[cache_key] = _CachedSearchResult(
                item_ids=selected_item_ids,
                expires_at_monotonic=time.monotonic() + ttl_s,
            )

    def _clear_read_cache(self, *, snapshot_kind: str) -> None:
        with self._cache_lock:
            self._catalog_entries_cache.pop(snapshot_kind, None)
            self._local_search_selector_cache.pop(snapshot_kind, None)
            self._recent_catalog_entries_cache.pop(snapshot_kind, None)
            self._recent_catalog_head_cache.pop(snapshot_kind, None)
            self._unsupported_scope_search_cache.pop(snapshot_kind, None)
            self._invalid_catalog_head_cache.pop(snapshot_kind, None)
            self._item_payload_cache = {
                cache_key: cached
                for cache_key, cached in self._item_payload_cache.items()
                if cache_key[0] != snapshot_kind
            }
            self._search_result_cache = {
                cache_key: cached
                for cache_key, cached in self._search_result_cache.items()
                if cache_key[0] != snapshot_kind
            }

    def is_catalog_payload(self, *, snapshot_kind: str, payload: Mapping[str, object] | None) -> bool:
        """Return whether one remote snapshot payload is a supported item catalog."""

        definition = self._definition(snapshot_kind)
        if definition is None or not isinstance(payload, Mapping):
            return False
        return self._is_segmented_catalog_payload(definition=definition, payload=payload) or self._is_legacy_catalog_payload(
            definition=definition,
            payload=payload,
        )

    def _is_segmented_catalog_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
    ) -> bool:
        segments = payload.get("segments")
        if not isinstance(segments, list):
            return False
        items_count = self._normalize_segment_index(payload.get("items_count"))
        if items_count is None:
            return False
        return (
            payload.get("schema") == definition.catalog_schema
            and payload.get("version") == _CATALOG_VERSION
            and (items_count <= 0 or bool(segments))
        )

    def _is_legacy_catalog_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
    ) -> bool:
        return (
            payload.get("schema") == definition.legacy_catalog_schema
            and payload.get("version") == _LEGACY_CATALOG_VERSION
            and isinstance(payload.get("items"), list)
        )

    def load_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object] | None = None,
        bypass_cache: bool = False,
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        """Load and normalize the current catalog entries for one collection."""

        definition = self._require_definition(snapshot_kind)
        if payload is None and not bypass_cache:
            cached_entries = self._cached_catalog_entries(snapshot_kind=snapshot_kind)
            if cached_entries is not None:
                return cached_entries
        if payload is None:
            payload = self.load_catalog_payload(snapshot_kind=snapshot_kind)
        if not isinstance(payload, Mapping):
            return ()
        if self._is_segmented_catalog_payload(definition=definition, payload=payload):
            entries = self._load_segmented_catalog_entries(definition=definition, payload=payload)
            if payload is not None:
                self._store_catalog_entries(snapshot_kind=snapshot_kind, entries=entries)
            return entries
        if self._is_legacy_catalog_payload(definition=definition, payload=payload):
            entries = self._load_legacy_catalog_entries(definition=definition, payload=payload)
            if payload is not None:
                self._store_catalog_entries(snapshot_kind=snapshot_kind, entries=entries)
            return entries
        return ()

    def catalog_available(self, *, snapshot_kind: str) -> bool:
        """Return whether the current remote snapshot is already a fine-grained catalog."""

        return self.is_catalog_payload(
            snapshot_kind=snapshot_kind,
            payload=self.load_catalog_payload(snapshot_kind=snapshot_kind),
        )

    def catalog_item_count(self, *, snapshot_kind: str) -> int | None:
        """Return the current catalog item count when the remote head exposes it."""

        payload = self.load_catalog_payload(snapshot_kind=snapshot_kind)
        if not isinstance(payload, Mapping):
            return None
        raw_count = payload.get("items_count")
        parsed = self._normalize_segment_index(raw_count)
        if parsed is None:
            return None
        return max(0, parsed)

    def assemble_snapshot_from_catalog(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object] | None,
        bypass_cache: bool = False,
    ) -> dict[str, object] | None:
        """Load every current remote item referenced by one catalog payload."""

        definition = self._require_definition(snapshot_kind)
        entries = self.load_catalog_entries(
            snapshot_kind=snapshot_kind,
            payload=payload,
            bypass_cache=bypass_cache,
        )
        item_payloads: list[dict[str, object]] = []
        loaded_payloads = self._load_item_payloads_from_entries(
            snapshot_kind=definition.snapshot_kind,
            entries=entries,
        )
        for entry, loaded in zip(entries, loaded_payloads, strict=False):
            if loaded is None:
                raise LongTermRemoteUnavailableError(
                    f"Required remote long-term {snapshot_kind!r} item {entry.item_id!r} is unavailable."
                )
            item_payloads.append(loaded)
        if snapshot_kind == "conflicts":
            assembled = {
                "schema": "twinr_memory_conflict_store",
                "version": 1,
                "conflicts": item_payloads,
            }
        else:
            assembled = {
                "schema": "twinr_memory_object_store" if snapshot_kind == "objects" else "twinr_memory_archive_store",
                "version": 1,
                "objects": item_payloads,
            }
        written_at = payload.get("written_at") if isinstance(payload, Mapping) else None
        if isinstance(written_at, str):
            assembled["written_at"] = written_at
        return assembled

    def assemble_snapshot_from_catalog_result(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object] | None,
        bypass_cache: bool = False,
    ) -> LongTermRemoteCatalogAssemblyResult:
        """Load a startup-compatible snapshot and report whether it stayed catalog-only."""

        definition = self._require_definition(snapshot_kind)
        entries = self.load_catalog_entries(
            snapshot_kind=snapshot_kind,
            payload=payload,
            bypass_cache=bypass_cache,
        )
        item_payloads: list[dict[str, object]] = []
        upgraded_entries: list[LongTermRemoteCatalogEntry] = []
        loaded_payloads, direct_catalog_complete = self._load_compat_item_payloads_from_entries(
            snapshot_kind=definition.snapshot_kind,
            entries=entries,
        )
        for entry, loaded in zip(entries, loaded_payloads, strict=False):
            if loaded is None:
                raise LongTermRemoteUnavailableError(
                    f"Required remote long-term {snapshot_kind!r} item {entry.item_id!r} is unavailable."
                )
            item_payloads.append(loaded)
            upgraded_entries.append(
                self._enrich_catalog_entry(
                    snapshot_kind=definition.snapshot_kind,
                    entry=entry,
                    payload=loaded,
                )
            )
        if snapshot_kind == "conflicts":
            assembled = {
                "schema": "twinr_memory_conflict_store",
                "version": 1,
                "conflicts": item_payloads,
            }
        else:
            assembled = {
                "schema": "twinr_memory_object_store" if snapshot_kind == "objects" else "twinr_memory_archive_store",
                "version": 1,
                "objects": item_payloads,
            }
        written_at = payload.get("written_at") if isinstance(payload, Mapping) else None
        if isinstance(written_at, str):
            assembled["written_at"] = written_at
        return LongTermRemoteCatalogAssemblyResult(
            payload=assembled,
            direct_catalog_complete=direct_catalog_complete,
            entries=tuple(upgraded_entries),
        )


__all__ = [
    "RemoteCatalogCatalogMixin",
]
