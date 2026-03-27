"""Small base class that composes the focused remote-catalog mixins."""

from __future__ import annotations

import hashlib
import json
import math
from threading import RLock
from urllib.parse import quote

from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore, LongTermRemoteUnavailableError

from .catalog import RemoteCatalogCatalogMixin
from .documents import RemoteCatalogDocumentMixin
from .search import RemoteCatalogSearchMixin
from .shared import (
    _CachedCatalogEntries,
    _CachedItemPayload,
    _CachedLocalSearchSelector,
    _CachedSearchResult,
    _DEFAULT_ASYNC_ATTESTATION_POLL_S,
    _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S,
    _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_S,
    _DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_S,
    _DEFAULT_BULK_BATCH_SIZE,
    _DEFAULT_BULK_REQUEST_MAX_BYTES,
    _DEFAULT_MAX_ITEM_CONTENT_CHARS,
    _DEFAULT_REMOTE_FLUSH_TIMEOUT_S,
    _DEFAULT_REMOTE_READ_MAX_WORKERS,
    _DEFAULT_REMOTE_READ_TIMEOUT_S,
    _DEFAULT_REMOTE_RETRY_ATTEMPTS,
    _DEFAULT_REMOTE_RETRY_BACKOFF_S,
    _DEFAULT_REMOTE_WRITE_TIMEOUT_S,
    _DEFAULT_RETRIEVE_BATCH_SIZE,
    _DEFINITIONS,
    _RemoteCollectionDefinition,
)
from .writes import RemoteCatalogWriteMixin


class LongTermRemoteCatalogStoreBase(
    RemoteCatalogCatalogMixin,
    RemoteCatalogSearchMixin,
    RemoteCatalogWriteMixin,
    RemoteCatalogDocumentMixin,
):
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

    def _payload_sha256(self, payload) -> str:
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

    def _catalog_segment_uri(self, *, snapshot_kind: str, segment_index: int) -> str:
        remote_state = self._require_remote_state()
        definition = self._require_definition(snapshot_kind)
        namespace = quote(str(getattr(remote_state, "namespace", "") or "twinr_longterm_v1"), safe="")
        return f"twinr://longterm/{namespace}/{definition.uri_segment}/catalog/segment/{segment_index:04d}"

    def _load_catalog_payload(self, *, snapshot_kind: str):
        remote_state = self._require_remote_state()
        payload = remote_state.load_snapshot(snapshot_kind=snapshot_kind)
        return payload if isinstance(payload, dict) else None

    def _remote_write_timeout_s(self) -> float | None:
        """Return the HTTP transport timeout for write-bound ChonkyDB calls."""

        config = getattr(getattr(self, "remote_state", None), "config", None)
        try:
            timeout_s = float(getattr(config, "long_term_memory_remote_write_timeout_s", None))
        except (TypeError, ValueError):
            return None
        return timeout_s if timeout_s > 0.0 else None

    def _remote_flush_timeout_s(self) -> float:
        """Return the end-to-end remote flush budget for one memory persist."""

        config = getattr(getattr(self, "remote_state", None), "config", None)
        try:
            timeout_s = float(
                getattr(
                    config,
                    "long_term_memory_remote_flush_timeout_s",
                    _DEFAULT_REMOTE_FLUSH_TIMEOUT_S,
                )
            )
        except (TypeError, ValueError):
            timeout_s = _DEFAULT_REMOTE_FLUSH_TIMEOUT_S
        return max(0.05, float(timeout_s))

    def _remote_async_job_timeout_s(self) -> float | None:
        """Return the server-side async job budget for accepted bulk writes."""

        transport_timeout_s = self._remote_write_timeout_s()
        flush_timeout_s = self._remote_flush_timeout_s()
        candidate = max(flush_timeout_s, transport_timeout_s or 0.0)
        return candidate if candidate > 0.0 else None

    def _remote_retry_attempts(self) -> int:
        config = getattr(getattr(self, "remote_state", None), "config", None)
        try:
            resolved = int(getattr(config, "long_term_memory_remote_retry_attempts", _DEFAULT_REMOTE_RETRY_ATTEMPTS))
        except (TypeError, ValueError):
            return _DEFAULT_REMOTE_RETRY_ATTEMPTS
        return min(10, max(1, resolved))

    def _remote_retry_backoff_s(self) -> float:
        config = getattr(getattr(self, "remote_state", None), "config", None)
        try:
            resolved = float(getattr(config, "long_term_memory_remote_retry_backoff_s", _DEFAULT_REMOTE_RETRY_BACKOFF_S))
        except (TypeError, ValueError):
            return _DEFAULT_REMOTE_RETRY_BACKOFF_S
        return min(30.0, max(0.0, resolved))

    def _async_attestation_visibility_timeout_s(self) -> float:
        """Return the bounded same-URI readback window for accepted async writes."""

        config = getattr(getattr(self, "remote_state", None), "config", None)
        try:
            read_timeout_s = float(getattr(config, "long_term_memory_remote_read_timeout_s", _DEFAULT_REMOTE_READ_TIMEOUT_S))
        except (TypeError, ValueError):
            read_timeout_s = _DEFAULT_REMOTE_READ_TIMEOUT_S
        write_timeout_s = self._remote_write_timeout_s() or _DEFAULT_REMOTE_WRITE_TIMEOUT_S
        candidate = max(
            _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_S,
            read_timeout_s,
            write_timeout_s,
            min(write_timeout_s * 2.0, _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S),
        )
        return min(
            _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S,
            max(_DEFAULT_ASYNC_ATTESTATION_POLL_S, candidate),
        )

    def _async_job_visibility_timeout_s(self) -> float:
        """Return the bounded window for async job completion plus document-id readback."""

        config = getattr(getattr(self, "remote_state", None), "config", None)
        try:
            read_timeout_s = float(getattr(config, "long_term_memory_remote_read_timeout_s", _DEFAULT_REMOTE_READ_TIMEOUT_S))
        except (TypeError, ValueError):
            read_timeout_s = _DEFAULT_REMOTE_READ_TIMEOUT_S
        flush_timeout_s = self._remote_flush_timeout_s()
        async_job_timeout_s = self._remote_async_job_timeout_s() or flush_timeout_s
        candidate = max(
            _DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_S,
            read_timeout_s,
            flush_timeout_s,
            async_job_timeout_s + read_timeout_s,
        )
        return min(
            _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S,
            max(_DEFAULT_ASYNC_ATTESTATION_POLL_S, candidate),
        )

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

    def _client_with_timeout(self, client: object, *, timeout_s: float | None) -> object:
        """Clone a client with a smaller transport timeout when supported."""

        if timeout_s is None:
            return client
        try:
            resolved_timeout_s = float(timeout_s)
        except (TypeError, ValueError):
            return client
        if not math.isfinite(resolved_timeout_s) or resolved_timeout_s <= 0.0:
            return client
        clone = getattr(client, "clone_with_timeout", None)
        if not callable(clone):
            return client
        try:
            return clone(max(0.05, resolved_timeout_s))
        except Exception:
            return client

    def _catalog_entry_metadata_from_mapping(self, payload) -> dict[str, object]:
        metadata: dict[str, object] = {}
        for field_name in (*self._catalog_entry_text_fields(), "payload_sha256"):
            value = self._normalize_text(payload.get(field_name))
            if value:
                metadata[field_name] = value
        for field_name in self._catalog_entry_list_fields():
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
    def _catalog_entry_text_fields() -> tuple[str, ...]:
        from .shared import _CATALOG_ENTRY_TEXT_FIELDS

        return _CATALOG_ENTRY_TEXT_FIELDS

    @staticmethod
    def _catalog_entry_list_fields() -> tuple[str, ...]:
        from .shared import _CATALOG_ENTRY_LIST_FIELDS

        return _CATALOG_ENTRY_LIST_FIELDS

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
    "LongTermRemoteCatalogStoreBase",
]
