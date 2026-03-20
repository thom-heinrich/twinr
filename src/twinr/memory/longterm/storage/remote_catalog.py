"""Persist fine-grained remote long-term memory documents plus compact catalogs.

This module replaces the old remote manifest+shard pattern for long-term
objects, conflicts, and archives. Each memory item is stored as its own
ChonkyDB document, while the existing remote-state snapshot path keeps only
the small current catalog that maps stable Twinr ids to the active ChonkyDB
document ids.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import logging
from threading import RLock
import time
from urllib.parse import quote
from uuid import uuid4

from twinr.memory.chonkydb.models import (
    ChonkyDBBulkRecordRequest,
    ChonkyDBRecordItem,
    ChonkyDBRetrieveRequest,
    ChonkyDBTopKRecordsRequest,
)
from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.longterm.core.models import LongTermMemoryConflictV1, LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.storage.remote_read_diagnostics import (
    LongTermRemoteReadContext,
    LongTermRemoteWriteContext,
    record_remote_read_diagnostic,
    record_remote_write_diagnostic,
)
from twinr.memory.longterm.storage.remote_read_observability import record_remote_read_observation
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore, LongTermRemoteUnavailableError


_LOG = logging.getLogger(__name__)

_CATALOG_VERSION = 3
_LEGACY_CATALOG_VERSION = 2
_SEGMENT_VERSION = 1
_ITEM_VERSION = 1
_DEFAULT_MAX_ITEM_CONTENT_CHARS = 256_000
_DEFAULT_BULK_BATCH_SIZE = 64
_DEFAULT_BULK_REQUEST_MAX_BYTES = 512 * 1024
_DEFAULT_RETRIEVE_BATCH_SIZE = 500
_DEFAULT_REMOTE_READ_MAX_WORKERS = 4
_ALLOWED_DOC_IDS_RETRIEVE_QUERY = "__allowed_doc_ids__"
_CATALOG_ENTRY_TEXT_FIELDS = (
    "kind",
    "status",
    "summary",
    "slot_key",
    "value_key",
    "created_at",
    "updated_at",
    "archived_at",
    "candidate_memory_id",
    "question",
    "reason",
)
_CATALOG_ENTRY_LIST_FIELDS = ("existing_memory_ids",)


@dataclass(frozen=True, slots=True)
class LongTermRemoteCatalogEntry:
    """Describe one current fine-grained remote memory item."""

    snapshot_kind: str
    item_id: str
    document_id: str | None
    uri: str
    metadata: dict[str, object]

    def updated_at(self) -> str:
        """Return the best available update timestamp string for sorting."""

        value = self.metadata.get("updated_at")
        return str(value).strip() if isinstance(value, str) else ""


@dataclass(frozen=True, slots=True)
class LongTermRemoteCatalogAssemblyResult:
    """Describe one catalog assembly pass and whether it stayed catalog-only."""

    payload: dict[str, object] | None
    direct_catalog_complete: bool
    entries: tuple[LongTermRemoteCatalogEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class _RemoteCollectionDefinition:
    snapshot_kind: str
    catalog_schema: str
    legacy_catalog_schema: str
    segment_schema: str
    item_schema: str
    envelope_key: str
    uri_segment: str


@dataclass(frozen=True, slots=True)
class _CachedCatalogEntries:
    entries: tuple[LongTermRemoteCatalogEntry, ...]
    expires_at_monotonic: float


@dataclass(frozen=True, slots=True)
class _CachedItemPayload:
    payload: dict[str, object]
    expires_at_monotonic: float


@dataclass(frozen=True, slots=True)
class _CachedSearchResult:
    item_ids: tuple[str, ...]
    expires_at_monotonic: float


@dataclass(frozen=True, slots=True)
class _CachedLocalSearchSelector:
    selector: FullTextSelector
    by_item_id: dict[str, LongTermRemoteCatalogEntry]
    expires_at_monotonic: float


_DEFINITIONS: dict[str, _RemoteCollectionDefinition] = {
    "objects": _RemoteCollectionDefinition(
        snapshot_kind="objects",
        catalog_schema="twinr_memory_object_catalog_v3",
        legacy_catalog_schema="twinr_memory_object_catalog_v2",
        segment_schema="twinr_memory_object_catalog_segment_v1",
        item_schema="twinr_memory_object_record_v2",
        envelope_key="object",
        uri_segment="objects",
    ),
    "conflicts": _RemoteCollectionDefinition(
        snapshot_kind="conflicts",
        catalog_schema="twinr_memory_conflict_catalog_v3",
        legacy_catalog_schema="twinr_memory_conflict_catalog_v2",
        segment_schema="twinr_memory_conflict_catalog_segment_v1",
        item_schema="twinr_memory_conflict_record_v2",
        envelope_key="conflict",
        uri_segment="conflicts",
    ),
    "archive": _RemoteCollectionDefinition(
        snapshot_kind="archive",
        catalog_schema="twinr_memory_archive_catalog_v3",
        legacy_catalog_schema="twinr_memory_archive_catalog_v2",
        segment_schema="twinr_memory_archive_catalog_segment_v1",
        item_schema="twinr_memory_archive_record_v2",
        envelope_key="object",
        uri_segment="archive",
    ),
}

class LongTermRemoteCatalogStore:
    """Store and query remote long-term memory items as individual documents."""

    def __init__(self, remote_state: LongTermRemoteStateStore | None) -> None:
        self.remote_state = remote_state
        self._cache_lock = RLock()
        self._catalog_entries_cache: dict[str, _CachedCatalogEntries] = {}
        self._item_payload_cache: dict[tuple[str, str], _CachedItemPayload] = {}
        self._search_result_cache: dict[tuple[str, str, int, tuple[str, ...]], _CachedSearchResult] = {}
        self._local_search_selector_cache: dict[str, _CachedLocalSearchSelector] = {}

    def enabled(self) -> bool:
        """Return whether fine-grained remote storage is available."""

        return bool(self.remote_state is not None and self.remote_state.enabled)

    def _persistent_read_cache_ttl_s(self) -> float:
        remote_state = self.remote_state
        config = None if remote_state is None else getattr(remote_state, "config", None)
        try:
            ttl_s = float(getattr(config, "long_term_memory_remote_read_cache_ttl_s", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return ttl_s if ttl_s > 0.0 else 0.0

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

    def _store_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        entries: tuple[LongTermRemoteCatalogEntry, ...],
    ) -> None:
        ttl_s = self._persistent_read_cache_ttl_s()
        if ttl_s <= 0.0:
            return
        with self._cache_lock:
            self._catalog_entries_cache[snapshot_kind] = _CachedCatalogEntries(
                entries=entries,
                expires_at_monotonic=time.monotonic() + ttl_s,
            )
            self._local_search_selector_cache[snapshot_kind] = _CachedLocalSearchSelector(
                selector=self._build_local_search_selector(entries=entries),
                by_item_id={entry.item_id: entry for entry in entries},
                expires_at_monotonic=time.monotonic() + ttl_s,
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
        return (
            payload.get("schema") == definition.catalog_schema
            and payload.get("version") == _CATALOG_VERSION
            and isinstance(payload.get("segments"), list)
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
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        """Load and normalize the current catalog entries for one collection."""

        definition = self._require_definition(snapshot_kind)
        if payload is None:
            cached_entries = self._cached_catalog_entries(snapshot_kind=snapshot_kind)
            if cached_entries is not None:
                return cached_entries
        if payload is None:
            payload = self._load_catalog_payload(snapshot_kind=snapshot_kind)
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
            payload=self._load_catalog_payload(snapshot_kind=snapshot_kind),
        )

    def assemble_snapshot_from_catalog(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        """Load every current remote item referenced by one catalog payload."""

        definition = self._require_definition(snapshot_kind)
        entries = self.load_catalog_entries(snapshot_kind=snapshot_kind, payload=payload)
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
    ) -> LongTermRemoteCatalogAssemblyResult:
        """Load a startup-compatible snapshot and report whether it stayed catalog-only."""

        definition = self._require_definition(snapshot_kind)
        entries = self.load_catalog_entries(snapshot_kind=snapshot_kind, payload=payload)
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

    def load_item_payload(
        self,
        *,
        snapshot_kind: str,
        item_id: str,
        document_id: str | None = None,
        uri: str | None = None,
    ) -> dict[str, object] | None:
        """Load one remote item document exactly by document id or URI."""

        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        definition = self._require_definition(snapshot_kind)
        normalized_item_id = self._normalize_item_id(item_id)
        if not normalized_item_id:
            return None
        cached_payload = self._cached_item_payload(snapshot_kind=snapshot_kind, item_id=normalized_item_id)
        if cached_payload is not None:
            return cached_payload
        resolved_uri = uri or self.item_uri(snapshot_kind=snapshot_kind, item_id=normalized_item_id)
        started_monotonic = time.monotonic()
        try:
            envelope = read_client.fetch_full_document(
                document_id=document_id,
                origin_uri=None if document_id else resolved_uri,
                include_content=True,
                max_content_chars=self._max_content_chars(),
            )
        except Exception as exc:
            if document_id:
                try:
                    envelope = read_client.fetch_full_document(
                        origin_uri=resolved_uri,
                        include_content=True,
                        max_content_chars=self._max_content_chars(),
                    )
                except Exception as fallback_exc:
                    record_remote_read_diagnostic(
                        remote_state=remote_state,
                        context=LongTermRemoteReadContext(
                            snapshot_kind=snapshot_kind,
                            operation="fetch_item_document",
                            item_id=normalized_item_id,
                            document_id_hint=document_id,
                            uri_hint=resolved_uri,
                        ),
                        exc=fallback_exc,
                        started_monotonic=started_monotonic,
                        outcome="failed",
                    )
                    raise LongTermRemoteUnavailableError(
                        f"Failed to read remote long-term {snapshot_kind!r} item {normalized_item_id!r}."
                    ) from fallback_exc
            else:
                record_remote_read_diagnostic(
                    remote_state=remote_state,
                    context=LongTermRemoteReadContext(
                        snapshot_kind=snapshot_kind,
                        operation="fetch_item_document",
                        item_id=normalized_item_id,
                        document_id_hint=document_id,
                        uri_hint=resolved_uri,
                    ),
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="failed",
                )
                raise LongTermRemoteUnavailableError(
                    f"Failed to read remote long-term {snapshot_kind!r} item {normalized_item_id!r}."
                ) from exc
        item_payload = self._extract_item_payload(
            definition=definition,
            item_id=normalized_item_id,
            payload=envelope,
        )
        if item_payload is None:
            return None
        self._store_item_payload(
            snapshot_kind=snapshot_kind,
            item_id=normalized_item_id,
            payload=item_payload,
        )
        return dict(item_payload)

    def load_item_payloads(
        self,
        *,
        snapshot_kind: str,
        item_ids: Iterable[str],
    ) -> tuple[dict[str, object], ...]:
        """Load a bounded set of current item payloads by Twinr item id."""

        entry_by_id = {
            entry.item_id: entry
            for entry in self.load_catalog_entries(snapshot_kind=snapshot_kind)
        }
        ordered_item_ids: list[str] = []
        uncached_entries: list[LongTermRemoteCatalogEntry] = []
        loaded_by_item_id: dict[str, dict[str, object]] = {}
        for raw_item_id in item_ids:
            item_id = self._normalize_item_id(raw_item_id)
            if not item_id:
                continue
            entry = entry_by_id.get(item_id)
            if entry is None:
                continue
            ordered_item_ids.append(item_id)
            cached_payload = self._cached_item_payload(snapshot_kind=snapshot_kind, item_id=item_id)
            if cached_payload is not None:
                loaded_by_item_id[item_id] = cached_payload
                continue
            uncached_entries.append(entry)
        loaded_uncached = self._load_item_payloads_from_entries(
            snapshot_kind=snapshot_kind,
            entries=uncached_entries,
        )
        for entry, payload in zip(uncached_entries, loaded_uncached, strict=False):
            if payload is not None:
                loaded_by_item_id[entry.item_id] = payload
        return tuple(loaded_by_item_id[item_id] for item_id in ordered_item_ids if item_id in loaded_by_item_id)

    def prewarm_item_payload_cache(self, *, snapshot_kind: str) -> None:
        """Warm the full current snapshot payload set into the read cache."""

        if self._persistent_read_cache_ttl_s() <= 0.0:
            return
        entries = self.load_catalog_entries(snapshot_kind=snapshot_kind)
        if not entries:
            return
        self.load_item_payloads(
            snapshot_kind=snapshot_kind,
            item_ids=(entry.item_id for entry in entries),
        )

    def search_current_item_payloads(
        self,
        *,
        snapshot_kind: str,
        query_text: str,
        limit: int,
        eligible: Callable[[LongTermRemoteCatalogEntry], bool] | None = None,
    ) -> tuple[dict[str, object], ...] | None:
        """Search the current remote scope directly without hydrating the full catalog.

        Returns ``None`` when the current backend cannot satisfy one-shot
        ``scope_ref`` searches and callers should fall back to the older
        catalog-hydration path. When supported, this keeps query-driven reads
        bounded to the ranked candidate window instead of fetching every
        current catalog segment up front.
        """

        clean_query = self._normalize_text(query_text)
        if not clean_query:
            return ()
        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        topk_records = getattr(read_client, "topk_records", None)
        supports_topk_records = bool(getattr(read_client, "supports_topk_records", callable(topk_records)))
        if not (supports_topk_records and callable(topk_records)):
            return None
        scope_ref, namespace = self._current_scope_request_context(snapshot_kind=snapshot_kind)
        if not scope_ref or not namespace:
            return None
        definition = self._require_definition(snapshot_kind)
        bounded_limit = max(1, int(limit))
        max_request_limit = max(bounded_limit, self._retrieve_batch_size())
        request_limit = self._initial_scope_search_limit(
            bounded_limit=bounded_limit,
            max_request_limit=max_request_limit,
            requires_client_filter=eligible is not None,
        )
        selected_payloads: list[dict[str, object]] = []
        seen_item_ids: set[str] = set()
        while True:
            candidates = self._search_remote_candidates(
                snapshot_kind=snapshot_kind,
                read_client=read_client,
                query_text=clean_query,
                result_limit=request_limit,
                allowed_doc_ids=(),
                scope_ref=scope_ref,
                namespace=namespace,
                catalog_entry_count=0,
            )
            candidate_count = 0
            for candidate in candidates:
                candidate_count += 1
                entry = self._candidate_catalog_entry(definition=definition, payload=candidate)
                if entry is None or entry.item_id in seen_item_ids:
                    continue
                if eligible is not None and not eligible(entry):
                    continue
                payload = self._extract_item_payload(
                    definition=definition,
                    item_id=entry.item_id,
                    payload=candidate,
                )
                if not isinstance(payload, Mapping):
                    continue
                payload_dict = dict(payload)
                self._store_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=entry.item_id,
                    payload=payload_dict,
                )
                selected_payloads.append(payload_dict)
                seen_item_ids.add(entry.item_id)
                if len(selected_payloads) >= bounded_limit:
                    return tuple(selected_payloads)
            if candidate_count < request_limit or request_limit >= max_request_limit:
                break
            next_limit = min(max_request_limit, request_limit * 2)
            if next_limit <= request_limit:
                break
            request_limit = next_limit
        return tuple(selected_payloads)

    def _initial_scope_search_limit(
        self,
        *,
        bounded_limit: int,
        max_request_limit: int,
        requires_client_filter: bool,
    ) -> int:
        """Choose the first one-shot top-k window for current-scope searches.

        Remote current-scope queries with client-side eligibility filters pay a
        full Pi-to-ChonkyDB roundtrip for every widening step. Starting with a
        wider bounded window cuts that latency without hydrating the entire
        catalog into the foreground turn.
        """

        if not requires_client_filter:
            return bounded_limit
        return min(
            max_request_limit,
            max(bounded_limit * 8, min(max_request_limit, 16)),
        )

    def search_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        query_text: str,
        limit: int,
        eligible: Callable[[LongTermRemoteCatalogEntry], bool] | None = None,
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        """Search one loaded catalog locally against the current remote projection."""

        use_local_cached_search = self._persistent_read_cache_ttl_s() > 0.0
        all_entries = self.load_catalog_entries(snapshot_kind=snapshot_kind)
        if not all_entries:
            return ()
        entries = tuple(entry for entry in all_entries if eligible is None or eligible(entry))
        if not entries:
            return ()
        result_limit = max(1, int(limit))
        if len(entries) <= result_limit:
            return tuple(entries[:result_limit])
        filtered_item_ids = tuple(entry.item_id for entry in entries)
        by_item_id = {entry.item_id: entry for entry in entries}
        cached_search = self._cached_search_result(
            snapshot_kind=snapshot_kind,
            query_text=query_text,
            limit=result_limit,
            filtered_item_ids=filtered_item_ids,
        )
        if cached_search is not None:
            return tuple(by_item_id[item_id] for item_id in cached_search if item_id in by_item_id)
        if use_local_cached_search:
            selected = self._local_search_catalog_entries(
                snapshot_kind=snapshot_kind,
                entries=all_entries,
                query_text=query_text,
                limit=result_limit,
                eligible=eligible,
            )
            self._store_search_result(
                snapshot_kind=snapshot_kind,
                query_text=query_text,
                limit=result_limit,
                filtered_item_ids=filtered_item_ids,
                selected_item_ids=tuple(entry.item_id for entry in selected),
            )
            return tuple(selected)

        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        allowed_doc_ids = [
            entry.document_id
            for entry in entries
            if isinstance(entry.document_id, str) and entry.document_id
        ]
        if not allowed_doc_ids:
            return ()
        topk_scope_ref, topk_namespace = self._topk_scope_request_context(
            snapshot_kind=snapshot_kind,
            entries=entries,
            all_entries=all_entries,
        )
        started_monotonic = time.monotonic()
        definition = self._require_definition(snapshot_kind)
        try:
            candidates = self._search_remote_candidates(
                snapshot_kind=snapshot_kind,
                read_client=read_client,
                query_text=query_text,
                result_limit=result_limit,
                allowed_doc_ids=tuple(allowed_doc_ids),
                scope_ref=topk_scope_ref,
                namespace=topk_namespace,
                catalog_entry_count=len(entries),
            )
        except Exception:
            return self._local_search_catalog_entries(
                snapshot_kind=snapshot_kind,
                entries=all_entries,
                query_text=query_text,
                limit=result_limit,
                eligible=eligible,
            )

        selected: list[LongTermRemoteCatalogEntry] = []
        seen: set[str] = set()
        for candidate in candidates:
            metadata = candidate.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            item_id = self._normalize_item_id(metadata.get("twinr_memory_item_id"))
            if not item_id or item_id in seen:
                continue
            entry = by_item_id.get(item_id)
            if entry is None:
                continue
            cached_payload = self._extract_item_payload(
                definition=definition,
                item_id=item_id,
                payload=candidate,
            )
            if cached_payload is not None:
                self._store_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=item_id,
                    payload=cached_payload,
                )
            selected.append(entry)
            seen.add(item_id)
            if len(selected) >= max(1, int(limit)):
                break
        self._store_search_result(
            snapshot_kind=snapshot_kind,
            query_text=query_text,
            limit=result_limit,
            filtered_item_ids=filtered_item_ids,
            selected_item_ids=tuple(entry.item_id for entry in selected),
        )
        return tuple(selected)

    def _search_remote_candidates(
        self,
        *,
        snapshot_kind: str,
        read_client: object,
        query_text: str,
        result_limit: int,
        allowed_doc_ids: tuple[str, ...],
        scope_ref: str | None = None,
        namespace: str | None = None,
        catalog_entry_count: int,
    ) -> tuple[Mapping[str, object], ...]:
        """Run remote search, preferring one-shot structured top-k responses."""

        remote_state = self._require_remote_state()
        topk_records = getattr(read_client, "topk_records", None)
        supports_topk_records = bool(getattr(read_client, "supports_topk_records", callable(topk_records)))
        if supports_topk_records and callable(topk_records):
            started_monotonic = time.monotonic()
            try:
                response = topk_records(
                    ChonkyDBTopKRecordsRequest(
                        query_text=query_text,
                        result_limit=result_limit,
                        include_content=False,
                        include_metadata=True,
                        allowed_doc_ids=None if scope_ref and namespace else allowed_doc_ids,
                        namespace=namespace,
                        scope_ref=scope_ref,
                    )
                )
                record_remote_read_observation(
                    remote_state=remote_state,
                    context=LongTermRemoteReadContext(
                        snapshot_kind=snapshot_kind,
                        operation="topk_search",
                        query_text=query_text,
                        catalog_entry_count=catalog_entry_count,
                        allowed_doc_count=len(allowed_doc_ids),
                        result_limit=result_limit,
                    ),
                    latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
                    outcome="ok",
                    classification="ok",
                )
                return tuple(self._iter_retrieve_result_candidates(response))
            except Exception as exc:
                record_remote_read_diagnostic(
                    remote_state=remote_state,
                    context=LongTermRemoteReadContext(
                        snapshot_kind=snapshot_kind,
                        operation="topk_search",
                        query_text=query_text,
                        catalog_entry_count=catalog_entry_count,
                        allowed_doc_count=len(allowed_doc_ids),
                        result_limit=result_limit,
                    ),
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="fallback",
                )

        started_monotonic = time.monotonic()
        try:
            response = read_client.retrieve(
                ChonkyDBRetrieveRequest(
                    query_text=query_text,
                    result_limit=result_limit,
                    include_content=False,
                    include_metadata=True,
                    allowed_doc_ids=allowed_doc_ids,
                )
            )
        except Exception as exc:
            record_remote_read_diagnostic(
                remote_state=remote_state,
                context=LongTermRemoteReadContext(
                    snapshot_kind=snapshot_kind,
                    operation="retrieve_search",
                    query_text=query_text,
                    catalog_entry_count=catalog_entry_count,
                    allowed_doc_count=len(allowed_doc_ids),
                    result_limit=result_limit,
                ),
                exc=exc,
                started_monotonic=started_monotonic,
                outcome="degraded",
            )
            raise
        record_remote_read_observation(
            remote_state=remote_state,
            context=LongTermRemoteReadContext(
                snapshot_kind=snapshot_kind,
                operation="retrieve_search",
                query_text=query_text,
                catalog_entry_count=catalog_entry_count,
                allowed_doc_count=len(allowed_doc_ids),
                result_limit=result_limit,
            ),
            latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
            outcome="ok",
            classification="ok",
        )
        return tuple(self._iter_retrieve_result_candidates(response))

    def _topk_scope_request_context(
        self,
        *,
        snapshot_kind: str,
        entries: tuple[LongTermRemoteCatalogEntry, ...],
        all_entries: tuple[LongTermRemoteCatalogEntry, ...],
    ) -> tuple[str | None, str | None]:
        """Return the server-side scope context for full current-snapshot searches only."""

        if len(entries) != len(all_entries):
            return None, None
        return self._current_scope_request_context(snapshot_kind=snapshot_kind)

    def _local_search_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        entries: tuple[LongTermRemoteCatalogEntry, ...],
        query_text: str,
        limit: int,
        eligible: Callable[[LongTermRemoteCatalogEntry], bool] | None = None,
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        """Rank already-loaded catalog entries locally through the cached selector."""

        bounded_limit = max(1, int(limit))
        cached_selector = self._local_search_selector(snapshot_kind=snapshot_kind, entries=entries)
        selector = self._build_local_search_selector(entries=entries) if cached_selector is None else cached_selector.selector
        by_item_id = {entry.item_id: entry for entry in entries} if cached_selector is None else cached_selector.by_item_id
        search_limit = bounded_limit if eligible is None else len(entries)
        selected_ids = selector.search(
            query_text,
            limit=max(1, search_limit),
            category="remote_catalog",
            allow_fallback=False,
        )
        selected: list[LongTermRemoteCatalogEntry] = []
        for item_id in selected_ids:
            entry = by_item_id.get(item_id)
            if entry is None:
                continue
            if eligible is not None and not eligible(entry):
                continue
            selected.append(entry)
            if len(selected) >= bounded_limit:
                break
        return tuple(selected)

    def _catalog_entry_search_text(self, entry: LongTermRemoteCatalogEntry) -> str:
        """Build one local fallback search document from catalog metadata."""

        parts: list[str] = [entry.item_id]
        for field_name in _CATALOG_ENTRY_TEXT_FIELDS:
            value = self._normalize_text(entry.metadata.get(field_name))
            if value:
                parts.append(value)
        for field_name in _CATALOG_ENTRY_LIST_FIELDS:
            values = self._normalize_text_list(entry.metadata.get(field_name))
            if values:
                parts.extend(values)
        return " ".join(parts)

    def top_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        limit: int,
        eligible: Callable[[LongTermRemoteCatalogEntry], bool] | None = None,
        preserve_order: bool = False,
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        """Return current catalog entries ordered either by recency or catalog order."""

        entries = [
            entry
            for entry in self.load_catalog_entries(snapshot_kind=snapshot_kind)
            if eligible is None or eligible(entry)
        ]
        if not preserve_order:
            entries.sort(key=lambda entry: (entry.updated_at(), entry.item_id), reverse=True)
        return tuple(entries[: max(1, int(limit))])

    def build_catalog_payload(
        self,
        *,
        snapshot_kind: str,
        item_payloads: Iterable[Mapping[str, object]],
        item_id_getter: Callable[[Mapping[str, object]], str | None],
        metadata_builder: Callable[[Mapping[str, object]], Mapping[str, object]],
        content_builder: Callable[[Mapping[str, object]], str],
    ) -> dict[str, object]:
        """Persist individual remote items and return the small current catalog."""

        self._clear_read_cache(snapshot_kind=snapshot_kind)
        remote_state = self._require_remote_state()
        write_client = self._require_client(getattr(remote_state, "write_client", None), operation="write")
        definition = self._require_definition(snapshot_kind)
        existing_entries = {
            entry.item_id: entry
            for entry in self.load_catalog_entries(snapshot_kind=snapshot_kind)
        }
        record_items: list[ChonkyDBRecordItem] = []
        staged: list[tuple[str, dict[str, object], str]] = []
        catalog_entries: list[dict[str, object]] = []
        for raw_payload in item_payloads:
            payload = dict(raw_payload)
            item_id = self._normalize_item_id(item_id_getter(payload))
            if not item_id:
                continue
            payload_sha256 = self._payload_sha256(payload)
            user_metadata = metadata_builder(payload)
            catalog_metadata = self._catalog_entry_metadata(
                payload_sha256=payload_sha256,
                user_metadata=user_metadata,
            )
            existing_entry = existing_entries.get(item_id)
            if self._can_reuse_existing_entry(
                snapshot_kind=definition.snapshot_kind,
                entry=existing_entry,
                item_id=item_id,
                payload_sha256=payload_sha256,
            ):
                catalog_entries.append(
                    self._build_catalog_entry(
                        item_id=item_id,
                        document_id=existing_entry.document_id if existing_entry is not None else None,
                        metadata=catalog_metadata,
                    )
                )
                continue
            uri = self.item_uri(snapshot_kind=snapshot_kind, item_id=item_id)
            metadata = self._build_item_record_metadata(
                definition=definition,
                item_id=item_id,
                payload=payload,
                payload_sha256=payload_sha256,
            )
            if isinstance(user_metadata, Mapping):
                for key, value in user_metadata.items():
                    if value is not None:
                        metadata[str(key)] = value
            content_text = content_builder(payload)
            record_items.append(
                ChonkyDBRecordItem(
                    # Keep a per-item Twinr envelope while also mirroring the
                    # public metadata/content fields that real ChonkyDB
                    # full-document reads materialize back to clients.
                    payload={
                        "schema": definition.item_schema,
                        "version": _ITEM_VERSION,
                        "snapshot_kind": definition.snapshot_kind,
                        "item_id": item_id,
                        "metadata": {
                            "twinr_snapshot_kind": definition.snapshot_kind,
                            "twinr_memory_item_id": item_id,
                        },
                        "content": content_text,
                    },
                    metadata=metadata,
                    content=content_text,
                    uri=uri,
                    enable_chunking=False,
                    include_insights_in_response=False,
                )
            )
            staged.append((item_id, catalog_metadata, payload_sha256))
        document_ids = self._store_record_items(
            write_client,
            snapshot_kind=definition.snapshot_kind,
            record_items=record_items,
        )
        for index, (item_id, metadata, _payload_sha256) in enumerate(staged):
            catalog_entries.append(
                self._build_catalog_entry(
                    item_id=item_id,
                    document_id=document_ids[index] if index < len(document_ids) else None,
                    metadata=metadata,
                )
            )
        segment_refs = self._persist_catalog_segments(
            write_client,
            definition=definition,
            catalog_entries=catalog_entries,
        )
        return {
            "schema": definition.catalog_schema,
            "version": _CATALOG_VERSION,
            "items_count": len(catalog_entries),
            "segments": list(segment_refs),
        }

    def persist_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        entries: Iterable[LongTermRemoteCatalogEntry],
        written_at: str | None = None,
    ) -> dict[str, object]:
        """Persist only the current catalog head/segments for existing item docs."""

        self._clear_read_cache(snapshot_kind=snapshot_kind)
        remote_state = self._require_remote_state()
        write_client = self._require_client(getattr(remote_state, "write_client", None), operation="write")
        definition = self._require_definition(snapshot_kind)
        catalog_entries = [
            self._build_catalog_entry(
                item_id=entry.item_id,
                document_id=entry.document_id,
                metadata=entry.metadata,
            )
            for entry in entries
        ]
        segment_refs = self._persist_catalog_segments(
            write_client,
            definition=definition,
            catalog_entries=catalog_entries,
        )
        payload = {
            "schema": definition.catalog_schema,
            "version": _CATALOG_VERSION,
            "items_count": len(catalog_entries),
            "segments": list(segment_refs),
        }
        if isinstance(written_at, str) and written_at.strip():
            payload["written_at"] = written_at
        remote_state.save_snapshot(snapshot_kind=snapshot_kind, payload=payload)
        return payload

    def _build_item_record_metadata(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        payload: Mapping[str, object],
        payload_sha256: str,
    ) -> dict[str, object]:
        return {
            "twinr_snapshot_kind": definition.snapshot_kind,
            "twinr_memory_item_id": item_id,
            "twinr_payload_sha256": payload_sha256,
            "twinr_payload": dict(payload),
        }

    def _catalog_entry_metadata(
        self,
        *,
        payload_sha256: str,
        user_metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {"payload_sha256": payload_sha256}
        if isinstance(user_metadata, Mapping):
            for field_name in _CATALOG_ENTRY_TEXT_FIELDS:
                value = self._normalize_text(user_metadata.get(field_name))
                if value:
                    metadata[field_name] = value
            for field_name in _CATALOG_ENTRY_LIST_FIELDS:
                values = self._normalize_text_list(user_metadata.get(field_name))
                if values:
                    metadata[field_name] = list(values)
        return metadata

    def _build_catalog_entry(
        self,
        *,
        item_id: str,
        document_id: str | None,
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        catalog_entry: dict[str, object] = {
            "item_id": item_id,
            "document_id": document_id,
        }
        for field_name in (*_CATALOG_ENTRY_TEXT_FIELDS, "payload_sha256"):
            value = self._normalize_text(metadata.get(field_name))
            if value:
                catalog_entry[field_name] = value
        for field_name in _CATALOG_ENTRY_LIST_FIELDS:
            values = self._normalize_text_list(metadata.get(field_name))
            if values:
                catalog_entry[field_name] = list(values)
        return catalog_entry

    def _can_reuse_existing_entry(
        self,
        *,
        snapshot_kind: str,
        entry: LongTermRemoteCatalogEntry | None,
        item_id: str,
        payload_sha256: str,
    ) -> bool:
        del snapshot_kind
        del item_id
        if entry is None:
            return False
        existing_sha256 = self._normalize_text(entry.metadata.get("payload_sha256"))
        if not existing_sha256 or existing_sha256 != payload_sha256:
            return False
        return bool(self._normalize_text(entry.document_id) or self._normalize_text(entry.uri))

    def _payload_sha256(self, payload: Mapping[str, object]) -> str:
        serialized = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def item_uri(self, *, snapshot_kind: str, item_id: str) -> str:
        """Return the canonical URI for one fine-grained remote item."""

        remote_state = self._require_remote_state()
        definition = self._require_definition(snapshot_kind)
        namespace = quote(str(getattr(remote_state, "namespace", "") or "twinr_longterm_v1"), safe="")
        encoded_id = quote(item_id, safe="")
        return f"twinr://longterm/{namespace}/{definition.uri_segment}/{encoded_id}"

    def _load_legacy_catalog_entries(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        items = payload.get("items")
        assert isinstance(items, list)
        entries: list[LongTermRemoteCatalogEntry] = []
        for raw_entry in items:
            if not isinstance(raw_entry, Mapping):
                continue
            item_id = self._normalize_item_id(raw_entry.get("item_id"))
            uri = self._normalize_text(raw_entry.get("uri"))
            if not item_id or not uri:
                continue
            metadata = raw_entry.get("metadata")
            entries.append(
                LongTermRemoteCatalogEntry(
                    snapshot_kind=definition.snapshot_kind,
                    item_id=item_id,
                    document_id=self._normalize_text(raw_entry.get("document_id")),
                    uri=uri,
                    metadata=self._catalog_entry_metadata_from_mapping(metadata if isinstance(metadata, Mapping) else {}),
                )
            )
        return tuple(entries)

    def _load_segmented_catalog_entries(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        segments = payload.get("segments")
        assert isinstance(segments, list)
        entries: list[LongTermRemoteCatalogEntry] = []
        segment_requests: list[tuple[int | None, str | None, str | None]] = []
        for raw_segment in segments:
            if not isinstance(raw_segment, Mapping):
                continue
            segment_index = self._normalize_segment_index(raw_segment.get("segment_index"))
            document_id = self._normalize_text(raw_segment.get("document_id"))
            uri = self._normalize_text(raw_segment.get("uri"))
            if document_id is None and uri is None:
                continue
            segment_requests.append((segment_index, document_id, uri))
        for envelope in self._load_catalog_segment_payloads(
            definition=definition,
            segment_requests=tuple(segment_requests),
        ):
            entries.extend(self._extract_segment_entries(definition=definition, payload=envelope))
        return tuple(entries)

    def _load_catalog_segment_payloads(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        segment_requests: tuple[tuple[int | None, str | None, str | None], ...],
    ) -> tuple[Mapping[str, object], ...]:
        """Load segmented catalog documents with bounded parallel reads."""

        if not segment_requests:
            return ()

        def load_one(request: tuple[int | None, str | None, str | None]) -> Mapping[str, object]:
            segment_index, document_id, uri = request
            return self._load_catalog_segment_payload(
                definition=definition,
                segment_index=segment_index,
                document_id=document_id,
                uri=uri,
            )

        max_workers = self._remote_read_max_workers(len(segment_requests))
        if max_workers <= 1:
            return tuple(load_one(request) for request in segment_requests)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return tuple(executor.map(load_one, segment_requests))

    def _load_catalog_segment_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        segment_index: int | None,
        document_id: str | None,
        uri: str | None,
    ) -> Mapping[str, object]:
        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        fallback_uri = uri or self._catalog_segment_uri(
            snapshot_kind=definition.snapshot_kind,
            segment_index=segment_index or 0,
        )
        started_monotonic = time.monotonic()
        try:
            return read_client.fetch_full_document(
                document_id=document_id,
                origin_uri=None if document_id else fallback_uri,
                include_content=True,
                max_content_chars=self._max_content_chars(),
            )
        except Exception as exc:
            if document_id:
                try:
                    return read_client.fetch_full_document(
                        origin_uri=fallback_uri,
                        include_content=True,
                        max_content_chars=self._max_content_chars(),
                    )
                except Exception as fallback_exc:
                    record_remote_read_diagnostic(
                        remote_state=remote_state,
                        context=LongTermRemoteReadContext(
                            snapshot_kind=definition.snapshot_kind,
                            operation="fetch_catalog_segment",
                            document_id_hint=document_id,
                            uri_hint=fallback_uri,
                            segment_index=segment_index,
                        ),
                        exc=fallback_exc,
                        started_monotonic=started_monotonic,
                        outcome="failed",
                    )
                    raise LongTermRemoteUnavailableError(
                        f"Failed to read remote long-term {definition.snapshot_kind!r} catalog segment."
                    ) from fallback_exc
            record_remote_read_diagnostic(
                remote_state=remote_state,
                context=LongTermRemoteReadContext(
                    snapshot_kind=definition.snapshot_kind,
                    operation="fetch_catalog_segment",
                    document_id_hint=document_id,
                    uri_hint=fallback_uri,
                    segment_index=segment_index,
                ),
                exc=exc,
                started_monotonic=started_monotonic,
                outcome="failed",
            )
            raise LongTermRemoteUnavailableError(
                f"Failed to read remote long-term {definition.snapshot_kind!r} catalog segment."
            ) from exc

    def _extract_segment_entries(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        entries: list[LongTermRemoteCatalogEntry] = []
        for candidate in self._iter_record_candidates(payload):
            if candidate.get("schema") != definition.segment_schema:
                continue
            if candidate.get("version") != _SEGMENT_VERSION:
                continue
            if candidate.get("snapshot_kind") != definition.snapshot_kind:
                continue
            raw_items = candidate.get("items")
            if not isinstance(raw_items, list):
                continue
            for raw_entry in raw_items:
                if not isinstance(raw_entry, Mapping):
                    continue
                item_id = self._normalize_item_id(raw_entry.get("item_id"))
                if not item_id:
                    continue
                entries.append(
                    LongTermRemoteCatalogEntry(
                        snapshot_kind=definition.snapshot_kind,
                        item_id=item_id,
                        document_id=self._normalize_text(raw_entry.get("document_id")),
                        uri=self.item_uri(snapshot_kind=definition.snapshot_kind, item_id=item_id),
                        metadata=self._catalog_entry_metadata_from_mapping(raw_entry),
                    )
                )
            break
        return tuple(entries)

    def _persist_catalog_segments(
        self,
        write_client: object,
        *,
        definition: _RemoteCollectionDefinition,
        catalog_entries: list[dict[str, object]],
    ) -> tuple[dict[str, object], ...]:
        segment_batches = self._split_catalog_segment_entries(catalog_entries)
        if not segment_batches:
            return ()
        record_items: list[ChonkyDBRecordItem] = []
        for segment_index, segment_entries in enumerate(segment_batches):
            segment_payload = {
                "schema": definition.segment_schema,
                "version": _SEGMENT_VERSION,
                "snapshot_kind": definition.snapshot_kind,
                "segment_index": segment_index,
                "items": list(segment_entries),
            }
            record_items.append(
                ChonkyDBRecordItem(
                    payload=segment_payload,
                    metadata={
                        "twinr_snapshot_kind": definition.snapshot_kind,
                        "twinr_catalog_segment_index": segment_index,
                        "twinr_catalog_segment_items": len(segment_entries),
                    },
                    content=json.dumps(segment_payload, ensure_ascii=False),
                    uri=self._catalog_segment_uri(snapshot_kind=definition.snapshot_kind, segment_index=segment_index),
                    enable_chunking=False,
                    include_insights_in_response=False,
                )
            )
        document_ids = self._store_record_items(
            write_client,
            snapshot_kind=definition.snapshot_kind,
            record_items=record_items,
        )
        refs: list[dict[str, object]] = []
        for segment_index, segment_entries in enumerate(segment_batches):
            refs.append(
                {
                    "segment_index": segment_index,
                    "document_id": document_ids[segment_index] if segment_index < len(document_ids) else None,
                    "uri": self._catalog_segment_uri(
                        snapshot_kind=definition.snapshot_kind,
                        segment_index=segment_index,
                    ),
                    "entry_count": len(segment_entries),
                }
            )
        return tuple(refs)

    def _split_catalog_segment_entries(
        self,
        catalog_entries: list[dict[str, object]],
    ) -> tuple[tuple[dict[str, object], ...], ...]:
        if not catalog_entries:
            return ()
        max_segment_bytes = self._catalog_segment_max_bytes()
        empty_segment_bytes = len(
            json.dumps(
                {
                    "schema": "segment",
                    "version": _SEGMENT_VERSION,
                    "snapshot_kind": "snapshot",
                    "segment_index": 0,
                    "items": [],
                },
                ensure_ascii=False,
            ).encode("utf-8")
        )
        segments: list[tuple[dict[str, object], ...]] = []
        current_segment: list[dict[str, object]] = []
        current_bytes = empty_segment_bytes
        for entry in catalog_entries:
            entry_bytes = len(json.dumps(entry, ensure_ascii=False).encode("utf-8"))
            separator_bytes = 2 if current_segment else 0
            if current_segment and current_bytes + separator_bytes + entry_bytes > max_segment_bytes:
                segments.append(tuple(current_segment))
                current_segment = []
                current_bytes = empty_segment_bytes
                separator_bytes = 0
            current_segment.append(dict(entry))
            current_bytes += separator_bytes + entry_bytes
        if current_segment:
            segments.append(tuple(current_segment))
        return tuple(segments)

    def _catalog_segment_uri(self, *, snapshot_kind: str, segment_index: int) -> str:
        remote_state = self._require_remote_state()
        definition = self._require_definition(snapshot_kind)
        namespace = quote(str(getattr(remote_state, "namespace", "") or "twinr_longterm_v1"), safe="")
        return f"twinr://longterm/{namespace}/{definition.uri_segment}/catalog/segment/{segment_index:04d}"

    def _store_record_items(
        self,
        write_client: object,
        *,
        snapshot_kind: str,
        record_items: list[ChonkyDBRecordItem],
    ) -> tuple[str | None, ...]:
        if not record_items:
            return ()
        batches = self._split_bulk_record_batches(record_items)
        document_ids: list[str | None] = []
        total_batches = len(batches)
        for index, batch in enumerate(batches):
            request_correlation_id = self._new_remote_write_correlation_id()
            request = ChonkyDBBulkRecordRequest(
                items=tuple(batch),
                timeout_seconds=self._remote_write_timeout_s(),
                client_request_id=request_correlation_id,
                finalize_vector_segments=index + 1 >= len(batches),
            )
            request_bytes = self._bulk_request_bytes(request)
            started_monotonic = time.monotonic()
            try:
                result = getattr(write_client, "store_records_bulk")(request)
            except Exception as exc:
                remote_state = self._require_remote_state()
                remote_write_context = {
                    "snapshot_kind": snapshot_kind,
                    "operation": "store_records_bulk",
                    "request_correlation_id": request_correlation_id,
                    "batch_index": index + 1,
                    "batch_count": total_batches,
                    "request_item_count": len(batch),
                    "request_bytes": request_bytes,
                }
                record_remote_write_diagnostic(
                    remote_state=remote_state,
                    context=LongTermRemoteWriteContext(
                        snapshot_kind=snapshot_kind,
                        operation="store_records_bulk",
                        attempt_count=1,
                        request_item_count=len(batch),
                        request_correlation_id=request_correlation_id,
                        batch_index=index + 1,
                        batch_count=total_batches,
                        request_bytes=request_bytes,
                    ),
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="failed",
                )
                unavailable = LongTermRemoteUnavailableError(
                    "Failed to persist fine-grained remote long-term memory items "
                    f"(request_id={request_correlation_id}, batch={index + 1}/{total_batches}, "
                    f"items={len(batch)}, bytes={request_bytes})."
                )
                setattr(unavailable, "remote_write_context", remote_write_context)
                raise unavailable from exc
            document_ids.extend(self._extract_document_ids(result, expected=len(batch)))
        return tuple(document_ids)

    @staticmethod
    def _new_remote_write_correlation_id() -> str:
        return f"ltw-{uuid4().hex[:12]}"

    @staticmethod
    def _bulk_request_bytes(request: ChonkyDBBulkRecordRequest) -> int:
        return len(json.dumps(request.to_payload(), ensure_ascii=False).encode("utf-8"))

    def _remote_write_timeout_s(self) -> float | None:
        config = getattr(getattr(self, "remote_state", None), "config", None)
        try:
            timeout_s = float(getattr(config, "long_term_memory_remote_write_timeout_s", None))
        except (TypeError, ValueError):
            return None
        return timeout_s if timeout_s > 0.0 else None

    def _load_item_payloads_from_entries(
        self,
        *,
        snapshot_kind: str,
        entries: Iterable[LongTermRemoteCatalogEntry],
    ) -> tuple[dict[str, object] | None, ...]:
        """Load multiple current item payloads with batch retrieve fallback."""

        ordered_entries = tuple(entries)
        if not ordered_entries:
            return ()
        loaded_by_item_id: dict[str, dict[str, object]] = {}
        unresolved_entries: list[LongTermRemoteCatalogEntry] = []
        for entry in ordered_entries:
            cached_payload = self._cached_item_payload(snapshot_kind=snapshot_kind, item_id=entry.item_id)
            if cached_payload is not None:
                loaded_by_item_id[entry.item_id] = cached_payload
                continue
            unresolved_entries.append(entry)
        loaded_by_item_id.update(self._load_item_payloads_via_retrieve(
            snapshot_kind=snapshot_kind,
            entries=tuple(unresolved_entries),
        ))
        resolved: list[dict[str, object] | None] = []
        for entry in ordered_entries:
            payload = loaded_by_item_id.get(entry.item_id)
            if payload is None:
                payload = self.load_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=entry.item_id,
                    document_id=entry.document_id,
                    uri=entry.uri,
                )
            elif isinstance(payload, Mapping):
                self._store_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=entry.item_id,
                    payload=payload,
                )
            resolved.append(dict(payload) if isinstance(payload, Mapping) else None)
        return tuple(resolved)

    def _load_compat_item_payloads_from_entries(
        self,
        *,
        snapshot_kind: str,
        entries: Iterable[LongTermRemoteCatalogEntry],
    ) -> tuple[tuple[dict[str, object] | None, ...], bool]:
        """Load startup-compatible payloads, preferring direct catalog reconstruction."""

        ordered_entries = tuple(entries)
        if not ordered_entries:
            return (), snapshot_kind in {"objects", "archive"}
        loaded_by_item_id: dict[str, dict[str, object]] = {}
        unresolved_entries: list[LongTermRemoteCatalogEntry] = []
        direct_catalog_complete = snapshot_kind in {"objects", "archive"}
        if direct_catalog_complete:
            for entry in ordered_entries:
                payload = self._catalog_entry_item_payload(snapshot_kind=snapshot_kind, entry=entry)
                if payload is None:
                    direct_catalog_complete = False
                    unresolved_entries.append(entry)
                    continue
                loaded_by_item_id[entry.item_id] = payload
        else:
            unresolved_entries.extend(ordered_entries)
        if unresolved_entries:
            loaded_by_item_id.update(
                self._load_item_payloads_via_retrieve(
                    snapshot_kind=snapshot_kind,
                    entries=tuple(unresolved_entries),
                )
            )
        resolved: list[dict[str, object] | None] = []
        for entry in ordered_entries:
            payload = loaded_by_item_id.get(entry.item_id)
            if payload is None:
                payload = self.load_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=entry.item_id,
                    document_id=entry.document_id,
                    uri=entry.uri,
                )
            resolved.append(dict(payload) if isinstance(payload, Mapping) else None)
        return tuple(resolved), direct_catalog_complete

    def _load_item_payloads_via_retrieve(
        self,
        *,
        snapshot_kind: str,
        entries: tuple[LongTermRemoteCatalogEntry, ...],
    ) -> dict[str, dict[str, object]]:
        """Load many current item payloads through ChonkyDB retrieve batches."""

        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        definition = self._require_definition(snapshot_kind)
        entries_by_document_id: dict[str, list[LongTermRemoteCatalogEntry]] = {}
        for entry in entries:
            document_id = self._normalize_text(entry.document_id)
            if document_id:
                entries_by_document_id.setdefault(document_id, []).append(entry)
        if not entries_by_document_id:
            return {}
        loaded: dict[str, dict[str, object]] = {}
        for batch, candidates in self._load_retrieve_batch_candidates(
            snapshot_kind=snapshot_kind,
            read_client=read_client,
            batches=self._split_retrieve_doc_id_batches(tuple(entries_by_document_id)),
        ):
            for candidate in candidates:
                document_id = self._normalize_text(candidate.get("payload_id")) or self._normalize_text(candidate.get("chonky_id"))
                if not document_id:
                    continue
                batch_entries = entries_by_document_id.get(document_id)
                if not batch_entries:
                    continue
                for entry in batch_entries:
                    payload = self._extract_item_payload(
                        definition=definition,
                        item_id=entry.item_id,
                        payload=candidate,
                    )
                    if payload is not None:
                        loaded[entry.item_id] = dict(payload)
                        self._store_item_payload(
                            snapshot_kind=snapshot_kind,
                            item_id=entry.item_id,
                            payload=payload,
                        )
        return loaded

    def _load_retrieve_batch_candidates(
        self,
        *,
        snapshot_kind: str,
        read_client: object,
        batches: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[tuple[str, ...], tuple[Mapping[str, object], ...]], ...]:
        """Load retrieve batches with bounded parallel reads."""

        if not batches:
            return ()
        remote_state = self._require_remote_state()

        def load_one(batch: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...]]:
            try:
                return batch, self._load_remote_batch_candidates(
                    snapshot_kind=snapshot_kind,
                    read_client=read_client,
                    batch=batch,
                )
            except Exception:
                return batch, ()

        max_workers = self._remote_read_max_workers(len(batches))
        if max_workers <= 1:
            return tuple(load_one(batch) for batch in batches)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return tuple(executor.map(load_one, batches))

    def _load_remote_batch_candidates(
        self,
        *,
        snapshot_kind: str,
        read_client: object,
        batch: tuple[str, ...],
    ) -> tuple[Mapping[str, object], ...]:
        """Load one document-id batch, preferring one-shot structured reads."""

        remote_state = self._require_remote_state()
        topk_records = getattr(read_client, "topk_records", None)
        supports_topk_records = bool(getattr(read_client, "supports_topk_records", callable(topk_records)))
        if supports_topk_records and callable(topk_records):
            started_monotonic = time.monotonic()
            try:
                response = topk_records(
                    ChonkyDBTopKRecordsRequest(
                        # ChonkyDB currently requires a query string even when
                        # allowed_doc_ids already narrows the candidate set.
                        query_text=_ALLOWED_DOC_IDS_RETRIEVE_QUERY,
                        result_limit=len(batch),
                        include_content=False,
                        include_metadata=True,
                        allowed_doc_ids=batch,
                    )
                )
                record_remote_read_observation(
                    remote_state=remote_state,
                    context=LongTermRemoteReadContext(
                        snapshot_kind=snapshot_kind,
                        operation="topk_batch",
                        allowed_doc_count=len(batch),
                        result_limit=len(batch),
                        batch_size=len(batch),
                    ),
                    latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
                    outcome="ok",
                    classification="ok",
                )
                return tuple(self._iter_retrieve_result_candidates(response))
            except Exception as exc:
                record_remote_read_diagnostic(
                    remote_state=remote_state,
                    context=LongTermRemoteReadContext(
                        snapshot_kind=snapshot_kind,
                        operation="topk_batch",
                        allowed_doc_count=len(batch),
                        result_limit=len(batch),
                        batch_size=len(batch),
                    ),
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="fallback",
                )

        started_monotonic = time.monotonic()
        try:
            response = read_client.retrieve(
                ChonkyDBRetrieveRequest(
                    # ChonkyDB currently rejects empty query_text even when
                    # allowed_doc_ids already fully constrains the candidate set.
                    query_text=_ALLOWED_DOC_IDS_RETRIEVE_QUERY,
                    result_limit=len(batch),
                    include_content=False,
                    include_metadata=True,
                    allowed_doc_ids=batch,
                )
            )
        except Exception as exc:
            record_remote_read_diagnostic(
                remote_state=remote_state,
                context=LongTermRemoteReadContext(
                    snapshot_kind=snapshot_kind,
                    operation="retrieve_batch",
                    allowed_doc_count=len(batch),
                    result_limit=len(batch),
                    batch_size=len(batch),
                ),
                exc=exc,
                started_monotonic=started_monotonic,
                outcome="degraded",
            )
            raise
        record_remote_read_observation(
            remote_state=remote_state,
            context=LongTermRemoteReadContext(
                snapshot_kind=snapshot_kind,
                operation="retrieve_batch",
                allowed_doc_count=len(batch),
                result_limit=len(batch),
                batch_size=len(batch),
            ),
            latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
            outcome="ok",
            classification="ok",
        )
        return tuple(self._iter_retrieve_result_candidates(response))

    def _iter_retrieve_result_candidates(self, payload: object) -> Iterable[Mapping[str, object]]:
        """Yield mapping-shaped raw retrieval hits from typed or fake clients."""

        raw = getattr(payload, "raw", None)
        if isinstance(raw, Mapping):
            raw_results = raw.get("results")
            if isinstance(raw_results, list):
                for item in raw_results:
                    if isinstance(item, Mapping):
                        yield item
                return
        results = getattr(payload, "results", None)
        if not isinstance(results, Iterable) or isinstance(results, (str, bytes, bytearray, Mapping)):
            return
        for hit in results:
            if isinstance(hit, Mapping):
                yield hit
                continue
            hit_raw = getattr(hit, "raw", None)
            if isinstance(hit_raw, Mapping):
                yield hit_raw
                continue
            candidate: dict[str, object] = {}
            payload_id = self._normalize_text(getattr(hit, "payload_id", None))
            if payload_id:
                candidate["payload_id"] = payload_id
            metadata = getattr(hit, "metadata", None)
            if isinstance(metadata, Mapping):
                candidate["metadata"] = dict(metadata)
            if candidate:
                yield candidate

    def _split_retrieve_doc_id_batches(
        self,
        document_ids: tuple[str, ...],
    ) -> tuple[tuple[str, ...], ...]:
        """Split retrieve doc ids into bounded batches."""

        if not document_ids:
            return ()
        batch_size = self._retrieve_batch_size()
        return tuple(
            tuple(document_ids[index : index + batch_size])
            for index in range(0, len(document_ids), batch_size)
        )

    def _split_bulk_record_batches(
        self,
        record_items: list[ChonkyDBRecordItem],
    ) -> tuple[tuple[ChonkyDBRecordItem, ...], ...]:
        if not record_items:
            return ()
        batch_size = self._bulk_write_batch_size()
        max_request_bytes = self._bulk_request_max_bytes()
        empty_request_bytes = len(
            json.dumps(
                ChonkyDBBulkRecordRequest(items=(), finalize_vector_segments=False).to_payload(),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        batches: list[tuple[ChonkyDBRecordItem, ...]] = []
        current_batch: list[ChonkyDBRecordItem] = []
        current_bytes = empty_request_bytes
        for item in record_items:
            item_bytes = len(json.dumps(item.to_payload(), ensure_ascii=False).encode("utf-8"))
            separator_bytes = 2 if current_batch else 0
            if current_batch and (
                len(current_batch) >= batch_size
                or current_bytes + separator_bytes + item_bytes > max_request_bytes
            ):
                batches.append(tuple(current_batch))
                current_batch = []
                current_bytes = empty_request_bytes
                separator_bytes = 0
            current_batch.append(item)
            current_bytes += separator_bytes + item_bytes
        if current_batch:
            batches.append(tuple(current_batch))
        return tuple(batches)

    def _extract_document_ids(self, payload: object, *, expected: int) -> tuple[str | None, ...]:
        if not isinstance(payload, Mapping):
            return tuple(None for _ in range(expected))
        items = payload.get("items")
        if not isinstance(items, list):
            return tuple(None for _ in range(expected))
        document_ids: list[str | None] = []
        for item in items[:expected]:
            if not isinstance(item, Mapping):
                document_ids.append(None)
                continue
            document_ids.append(
                self._normalize_text(item.get("document_id"))
                or self._normalize_text(item.get("payload_id"))
                or self._normalize_text(item.get("chonky_id"))
            )
        while len(document_ids) < expected:
            document_ids.append(None)
        return tuple(document_ids)

    def _candidate_catalog_entry(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
    ) -> LongTermRemoteCatalogEntry | None:
        """Build a minimal catalog entry from one live retrieval candidate."""

        item_id: str | None = None
        metadata: dict[str, object] = {}
        for candidate in self._iter_record_candidates(payload):
            if item_id is None:
                item_id = self._normalize_item_id(candidate.get("twinr_memory_item_id"))
            candidate_metadata = self._catalog_entry_metadata_from_mapping(candidate)
            if candidate_metadata:
                metadata.update(candidate_metadata)
        if not item_id:
            return None
        document_id = (
            self._normalize_text(payload.get("document_id"))
            or self._normalize_text(payload.get("payload_id"))
            or self._normalize_text(payload.get("chonky_id"))
        )
        return LongTermRemoteCatalogEntry(
            snapshot_kind=definition.snapshot_kind,
            item_id=item_id,
            document_id=document_id,
            uri=self.item_uri(snapshot_kind=definition.snapshot_kind, item_id=item_id),
            metadata=metadata,
        )

    def _extract_item_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        payload: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        for candidate in self._iter_record_candidates(payload):
            normalized = self._extract_public_item_payload(
                definition=definition,
                item_id=item_id,
                candidate=candidate,
            )
            if normalized is not None:
                return normalized
            metadata_only = self._extract_legacy_metadata_only_item_payload(
                definition=definition,
                item_id=item_id,
                candidate=candidate,
            )
            if metadata_only is not None:
                return metadata_only
            if candidate.get("schema") != definition.item_schema:
                continue
            if candidate.get("version") != _ITEM_VERSION:
                continue
            if candidate.get("snapshot_kind") != definition.snapshot_kind:
                continue
            if self._normalize_item_id(candidate.get("item_id")) != item_id:
                continue
            raw_item = candidate.get(definition.envelope_key)
            if isinstance(raw_item, Mapping):
                return raw_item
        return None

    def _extract_public_item_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        candidate: Mapping[str, object],
    ) -> dict[str, object] | None:
        normalized_item_id = self._normalize_item_id(item_id)
        if not normalized_item_id:
            return None
        if definition.snapshot_kind in {"objects", "archive"}:
            try:
                obj = LongTermMemoryObjectV1.from_payload(candidate)
            except Exception:
                return None
            if self._normalize_text(obj.memory_id) != normalized_item_id:
                return None
            return obj.to_payload()
        if definition.snapshot_kind == "conflicts":
            try:
                conflict = LongTermMemoryConflictV1.from_payload(candidate)
            except Exception:
                return None
            if self._conflict_doc_id(conflict) != normalized_item_id:
                return None
            return conflict.to_payload()
        return None

    def _conflict_doc_id(self, conflict: LongTermMemoryConflictV1) -> str:
        """Return the canonical remote item id for one conflict payload."""

        return conflict.catalog_item_id()

    def _catalog_entry_item_payload(
        self,
        *,
        snapshot_kind: str,
        entry: LongTermRemoteCatalogEntry,
    ) -> dict[str, object] | None:
        definition = self._require_definition(snapshot_kind)
        candidate = dict(entry.metadata)
        candidate["twinr_snapshot_kind"] = definition.snapshot_kind
        candidate["twinr_memory_item_id"] = entry.item_id
        candidate["document_id"] = entry.document_id
        return self._build_compat_object_payload_from_metadata(
            definition=definition,
            item_id=entry.item_id,
            candidate=candidate,
            source_type="remote_catalog_entry",
            compatibility_attribute="remote_catalog_entry_compatibility",
        )

    def _enrich_catalog_entry(
        self,
        *,
        snapshot_kind: str,
        entry: LongTermRemoteCatalogEntry,
        payload: Mapping[str, object],
    ) -> LongTermRemoteCatalogEntry:
        metadata = dict(entry.metadata)
        for field_name in _CATALOG_ENTRY_TEXT_FIELDS:
            value = self._normalize_text(payload.get(field_name))
            if value:
                metadata[field_name] = value
        for field_name in _CATALOG_ENTRY_LIST_FIELDS:
            values = self._normalize_text_list(payload.get(field_name))
            if values:
                metadata[field_name] = list(values)
        payload_sha256 = self._normalize_text(entry.metadata.get("payload_sha256"))
        if payload_sha256:
            metadata["payload_sha256"] = payload_sha256
        return LongTermRemoteCatalogEntry(
            snapshot_kind=snapshot_kind,
            item_id=entry.item_id,
            document_id=entry.document_id,
            uri=entry.uri,
            metadata=metadata,
        )

    def _extract_legacy_metadata_only_item_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        candidate: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Rebuild minimal object payloads from legacy metadata-only live reads.

        Older ChonkyDB documents can surface only the record metadata and an
        empty content field. Those reads no longer carry the original Twinr
        payload or public JSON body, but they still expose enough stable object
        identity and summary fields to reconstruct a compatibility payload.
        """

        if definition.snapshot_kind not in {"objects", "archive"}:
            return None
        if isinstance(candidate.get("twinr_payload"), Mapping):
            return None
        if self._normalize_text(candidate.get("twinr_snapshot_kind")) != definition.snapshot_kind:
            return None
        return self._build_compat_object_payload_from_metadata(
            definition=definition,
            item_id=item_id,
            candidate=candidate,
            source_type="legacy_remote_catalog_metadata",
            compatibility_attribute="legacy_remote_catalog_metadata_only",
        )

    def _build_compat_object_payload_from_metadata(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        candidate: Mapping[str, object],
        source_type: str,
        compatibility_attribute: str,
    ) -> dict[str, object] | None:
        if definition.snapshot_kind not in {"objects", "archive"}:
            return None
        normalized_item_id = self._normalize_item_id(candidate.get("twinr_memory_item_id"))
        if normalized_item_id != item_id:
            return None
        kind = self._normalize_text(candidate.get("kind"))
        summary = self._normalize_text(candidate.get("summary"))
        created_at = self._normalize_text(candidate.get("created_at"))
        updated_at = self._normalize_text(candidate.get("updated_at")) or created_at
        if not kind or not summary or not created_at or not updated_at:
            return None
        attributes: dict[str, object] = {compatibility_attribute: True}
        origin_uri = self._normalize_text(candidate.get("origin_uri"))
        if origin_uri:
            attributes["legacy_origin_uri"] = origin_uri
        raw_timestamp = candidate.get("timestamp")
        if isinstance(raw_timestamp, int):
            attributes["legacy_timestamp"] = raw_timestamp
        document_id = self._normalize_text(candidate.get("document_id"))
        if document_id:
            attributes["remote_document_id"] = document_id
        try:
            return LongTermMemoryObjectV1(
                memory_id=item_id,
                kind=kind,
                summary=summary,
                source=LongTermSourceRefV1(
                    source_type=source_type,
                    event_ids=(item_id,),
                ),
                status=self._normalize_text(candidate.get("status")) or "candidate",
                slot_key=self._normalize_text(candidate.get("slot_key")),
                value_key=self._normalize_text(candidate.get("value_key")),
                archived_at=self._normalize_text(candidate.get("archived_at")),
                created_at=created_at,
                updated_at=updated_at,
                attributes=attributes,
            ).to_payload()
        except Exception:
            return None

    def _iter_record_candidates(self, payload: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
        yield payload
        direct = payload.get("payload")
        if isinstance(direct, Mapping):
            yield direct
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            yield metadata
            nested_payload = metadata.get("twinr_payload")
            if isinstance(nested_payload, Mapping):
                yield nested_payload
        nested = payload.get("record")
        if isinstance(nested, Mapping):
            yield nested
            nested_payload = nested.get("payload")
            if isinstance(nested_payload, Mapping):
                yield nested_payload
            nested_metadata = nested.get("metadata")
            if isinstance(nested_metadata, Mapping):
                yield nested_metadata
                nested_inner_payload = nested_metadata.get("twinr_payload")
                if isinstance(nested_inner_payload, Mapping):
                    yield nested_inner_payload
            nested_content = nested.get("content")
            if isinstance(nested_content, str):
                parsed = self._parse_json_mapping(nested_content)
                if parsed is not None:
                    yield parsed
            nested_document = nested.get("document")
            if isinstance(nested_document, Mapping):
                yield from self._iter_record_candidates(nested_document)
        document = payload.get("document")
        if isinstance(document, Mapping):
            yield from self._iter_record_candidates(document)
        content = payload.get("content")
        if isinstance(content, str):
            parsed = self._parse_json_mapping(content)
            if parsed is not None:
                yield parsed
        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    continue
                chunk_metadata = chunk.get("metadata")
                if isinstance(chunk_metadata, Mapping):
                    yield chunk_metadata
                    chunk_inner_payload = chunk_metadata.get("twinr_payload")
                    if isinstance(chunk_inner_payload, Mapping):
                        yield chunk_inner_payload
                chunk_payload = chunk.get("payload")
                if isinstance(chunk_payload, Mapping):
                    yield chunk_payload
                chunk_content = chunk.get("content")
                if isinstance(chunk_content, str):
                    parsed = self._parse_json_mapping(chunk_content)
                    if parsed is not None:
                        yield parsed

    def _parse_json_mapping(self, value: str) -> Mapping[str, object] | None:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, Mapping) else None

    def _load_catalog_payload(self, *, snapshot_kind: str) -> Mapping[str, object] | None:
        remote_state = self._require_remote_state()
        payload = remote_state.load_snapshot(snapshot_kind=snapshot_kind)
        return payload if isinstance(payload, Mapping) else None

    def _max_content_chars(self) -> int:
        remote_state = self._require_remote_state()
        value = getattr(getattr(remote_state, "config", None), "long_term_memory_remote_max_content_chars", None)
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            return _DEFAULT_MAX_ITEM_CONTENT_CHARS
        return max(1, resolved)

    def _bulk_write_batch_size(self) -> int:
        remote_state = self._require_remote_state()
        value = getattr(getattr(remote_state, "config", None), "long_term_memory_migration_batch_size", None)
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            return _DEFAULT_BULK_BATCH_SIZE
        return max(1, resolved)

    def _bulk_request_max_bytes(self) -> int:
        remote_state = self._require_remote_state()
        value = getattr(getattr(remote_state, "config", None), "long_term_memory_remote_bulk_request_max_bytes", None)
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            return _DEFAULT_BULK_REQUEST_MAX_BYTES
        return max(1, resolved)

    def _retrieve_batch_size(self) -> int:
        return _DEFAULT_RETRIEVE_BATCH_SIZE

    def _remote_read_max_workers(self, batch_count: int) -> int:
        if batch_count <= 1:
            return 1
        return max(1, min(_DEFAULT_REMOTE_READ_MAX_WORKERS, batch_count))

    def _catalog_segment_max_bytes(self) -> int:
        return max(1, min(self._bulk_request_max_bytes() // 4, self._max_content_chars()))

    def _definition(self, snapshot_kind: str) -> _RemoteCollectionDefinition | None:
        return _DEFINITIONS.get(self._normalize_text(snapshot_kind))

    def _require_definition(self, snapshot_kind: str) -> _RemoteCollectionDefinition:
        definition = self._definition(snapshot_kind)
        if definition is None:
            raise ValueError(f"Unsupported fine-grained remote snapshot kind {snapshot_kind!r}.")
        return definition

    def _require_remote_state(self) -> LongTermRemoteStateStore:
        remote_state = self.remote_state
        if remote_state is None or not remote_state.enabled:
            raise LongTermRemoteUnavailableError("Required remote long-term memory state is not configured.")
        return remote_state

    def _require_client(self, client: object | None, *, operation: str) -> object:
        if client is not None:
            return client
        raise LongTermRemoteUnavailableError(
            f"Remote-primary long-term memory is enabled but ChonkyDB is not configured for {operation} operations."
        )

    def _catalog_entry_metadata_from_mapping(self, payload: Mapping[str, object]) -> dict[str, object]:
        metadata: dict[str, object] = {}
        for field_name in (*_CATALOG_ENTRY_TEXT_FIELDS, "payload_sha256"):
            value = self._normalize_text(payload.get(field_name))
            if value:
                metadata[field_name] = value
        for field_name in _CATALOG_ENTRY_LIST_FIELDS:
            values = self._normalize_text_list(payload.get(field_name))
            if values:
                metadata[field_name] = list(values)
        return metadata

    def _current_scope_request_context(self, *, snapshot_kind: str) -> tuple[str | None, str | None]:
        """Return the current-snapshot scope and namespace for one collection."""

        remote_state = self.remote_state
        namespace = self._normalize_text(None if remote_state is None else getattr(remote_state, "namespace", None))
        if not namespace:
            return None, None
        definition = self._definition(snapshot_kind)
        if definition is None:
            return None, None
        return f"longterm:{definition.snapshot_kind}:current", namespace

    @staticmethod
    def _normalize_text(value: object) -> str | None:
        if value is None:
            return None
        normalized = " ".join(str(value).split()).strip()
        return normalized or None

    @staticmethod
    def _normalize_item_id(value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _normalize_text_list(self, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        normalized: list[str] = []
        for item in value:
            text = self._normalize_text(item)
            if text:
                normalized.append(text)
        return tuple(normalized)

    @staticmethod
    def _normalize_segment_index(value: object) -> int | None:
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            return None
        return resolved if resolved >= 0 else None


__all__ = [
    "LongTermRemoteCatalogAssemblyResult",
    "LongTermRemoteCatalogEntry",
    "LongTermRemoteCatalogStore",
]
