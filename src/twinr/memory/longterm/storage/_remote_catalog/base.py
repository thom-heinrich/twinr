"""Small base class that composes the focused remote-catalog mixins."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import math
from threading import RLock
import time
from urllib.parse import quote

from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.longterm.storage.remote_read_diagnostics import (
    LongTermRemoteReadContext,
    record_remote_read_diagnostic,
)
from twinr.memory.longterm.storage.remote_read_observability import record_remote_read_observation
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore, LongTermRemoteUnavailableError
from twinr.memory.longterm.storage._remote_retry import (
    remote_read_retry_delay_s,
    should_retry_remote_read_error,
)

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
    _DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_CAP_S,
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

_LOG = logging.getLogger("twinr.memory.longterm.storage.remote_catalog")
_RECENT_CATALOG_HEAD_TTL_S = 120.0
_RECENT_CATALOG_ENTRIES_TTL_S = 120.0
_NEGATIVE_REMOTE_CONTRACT_TTL_S = 120.0
_MUTABLE_CURRENT_HEAD_SNAPSHOT_KINDS = frozenset({"objects", "conflicts", "archive"})
_ASYNC_JOB_TIMEOUT_FLOOR_S = 180.0
_SELECTION_SEARCH_ALLOWED_INDEXES = ("fulltext", "temporal", "tags")
_SELECTION_HYDRATION_ALLOWED_INDEXES = ("fulltext",)
_SEARCHABLE_WRITE_TARGET_INDEXES = ("fulltext", "temporal", "tags")
_CONTROL_PLANE_WRITE_TARGET_INDEXES: tuple[str, ...] = ()
_ASYNC_JOB_TIMEOUT_FLOORED_SNAPSHOT_KINDS = frozenset(
    {
        "objects",
        "conflicts",
        "archive",
        "graph_nodes",
        "graph_edges",
        "provider_answer_fronts",
    }
)


@dataclass(frozen=True, slots=True)
class _CatalogHeadLoadResult:
    """Describe one current-head read without forcing snapshot compat."""

    status: str
    payload: dict[str, object] | None = None


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
        self._recent_catalog_head_cache: dict[str, _CachedItemPayload] = {}
        self._recent_catalog_entries_cache: dict[str, _CachedCatalogEntries] = {}
        self._unsupported_scope_search_cache: dict[str, float] = {}
        self._invalid_catalog_head_cache: dict[str, float] = {}

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

    def _recent_catalog_head_ttl_s(self) -> float:
        """Return the bounded same-process bridge window for current-head writes."""

        return _RECENT_CATALOG_HEAD_TTL_S

    def _recent_catalog_entries_ttl_s(self) -> float:
        """Return the bounded same-process bridge window for current catalog entries."""

        return _RECENT_CATALOG_ENTRIES_TTL_S

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

    def _catalog_segment_uri(
        self,
        *,
        snapshot_kind: str,
        segment_index: int,
        segment_token: str | None = None,
    ) -> str:
        remote_state = self._require_remote_state()
        definition = self._require_definition(snapshot_kind)
        namespace = quote(str(getattr(remote_state, "namespace", "") or "twinr_longterm_v1"), safe="")
        base_uri = f"twinr://longterm/{namespace}/{definition.uri_segment}/catalog/segment/{segment_index:04d}"
        normalized_token = str(segment_token or "").strip()
        if not normalized_token:
            return base_uri
        return f"{base_uri}/{quote(normalized_token, safe='')}"

    def _catalog_head_uri(self, *, snapshot_kind: str) -> str:
        """Return the canonical URI for the current catalog head document."""

        remote_state = self._require_remote_state()
        definition = self._require_definition(snapshot_kind)
        namespace = quote(str(getattr(remote_state, "namespace", "") or "twinr_longterm_v1"), safe="")
        return f"twinr://longterm/{namespace}/{definition.uri_segment}/catalog/current"

    def _load_catalog_payload(self, *, snapshot_kind: str):
        recent_head = self._recent_catalog_head_payload(snapshot_kind=snapshot_kind)
        if self.is_catalog_payload(snapshot_kind=snapshot_kind, payload=recent_head):
            return dict(recent_head)
        if self._catalog_head_known_invalid(snapshot_kind=snapshot_kind):
            payload = self._load_remote_state_snapshot_fallback(snapshot_kind=snapshot_kind)
            if not isinstance(payload, Mapping):
                return None
            payload_dict = dict(payload)
            if not self.is_catalog_payload(snapshot_kind=snapshot_kind, payload=payload_dict):
                return None
            self._maybe_promote_legacy_catalog_head(snapshot_kind=snapshot_kind, payload=payload_dict)
            return payload_dict
        direct_result = self._load_catalog_head_result(snapshot_kind=snapshot_kind)
        direct_payload = direct_result.payload
        if isinstance(direct_payload, dict):
            return direct_payload
        payload = self._load_remote_state_snapshot_fallback(snapshot_kind=snapshot_kind)
        if not isinstance(payload, Mapping):
            return None
        payload_dict = dict(payload)
        if not self.is_catalog_payload(snapshot_kind=snapshot_kind, payload=payload_dict):
            return None
        self._maybe_promote_legacy_catalog_head(snapshot_kind=snapshot_kind, payload=payload_dict)
        return payload_dict

    def _load_remote_state_snapshot_fallback(
        self,
        *,
        snapshot_kind: str,
    ) -> object:
        """Load one mutable current head without pinning reads to a stale document id.

        Structured-memory current heads mutate behind a stable URI. Reusing an
        older exact document id for those heads can pin a fresh reader to an
        already superseded empty/current catalog document after another process
        writes a new head. For these mutable heads we therefore force the
        fallback snapshot loader through pointer/origin resolution instead of a
        remembered exact-document shortcut.
        """

        remote_state = self._require_remote_state()
        load_snapshot = getattr(remote_state, "load_snapshot")
        try:
            parameters = inspect.signature(load_snapshot).parameters
        except (TypeError, ValueError):
            parameters = {}
        kwargs: dict[str, object] = {"snapshot_kind": snapshot_kind}
        if "prefer_cached_document_id" in parameters:
            kwargs["prefer_cached_document_id"] = snapshot_kind not in _MUTABLE_CURRENT_HEAD_SNAPSHOT_KINDS
        return load_snapshot(**kwargs)

    def _load_catalog_head_payload(self, *, snapshot_kind: str, metadata_only: bool = False) -> dict[str, object] | None:
        """Load the fixed-URI current catalog head without going through snapshot pointers."""

        result = self._load_catalog_head_result(snapshot_kind=snapshot_kind, metadata_only=metadata_only)
        return dict(result.payload) if isinstance(result.payload, Mapping) else None

    def _load_catalog_head_result(
        self,
        *,
        snapshot_kind: str,
        metadata_only: bool = False,
        fast_fail: bool = False,
    ) -> _CatalogHeadLoadResult:
        """Read one current-head document while preserving not-found semantics."""

        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        retry_attempts = 1 if fast_fail else max(1, self._remote_retry_attempts())
        retry_backoff_s = 0.0 if fast_fail else self._remote_retry_backoff_s()
        effective_read_client = read_client if fast_fail else self._catalog_head_read_client(read_client=read_client)
        context = LongTermRemoteReadContext(
            snapshot_kind=snapshot_kind,
            operation="load_catalog_current_head",
            request_method="GET",
            request_payload_kind="catalog_current_head_document",
            uri_hint=self._catalog_head_uri(snapshot_kind=snapshot_kind),
            request_path="/v1/external/documents/full",
            namespace=self._normalize_text(getattr(remote_state, "namespace", None)),
            access_classification="catalog_current_head",
            attempt_count=retry_attempts,
            retry_attempts_configured=self._remote_retry_attempts(),
            retry_backoff_s=retry_backoff_s,
            retry_mode="single_probe" if fast_fail else "bounded_transient_retry",
        )
        started_monotonic = time.monotonic()
        attempt_index = 0
        while True:
            try:
                envelope = self._fetch_catalog_head_envelope(
                    read_client=effective_read_client,
                    snapshot_kind=snapshot_kind,
                    metadata_only=metadata_only,
                )
                break
            except Exception as exc:
                if self._status_code_from_exception(exc) == 404:
                    self._clear_invalid_catalog_head(snapshot_kind=snapshot_kind)
                    record_remote_read_observation(
                        remote_state=remote_state,
                        context=context,
                        latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
                        outcome="missing",
                        classification="not_found",
                    )
                    return _CatalogHeadLoadResult(status="not_found")
                if should_retry_remote_read_error(exc) and attempt_index + 1 < retry_attempts:
                    delay_s = remote_read_retry_delay_s(
                        exc,
                        default_backoff_s=retry_backoff_s,
                        attempt_index=attempt_index,
                    )
                    if delay_s > 0.0:
                        time.sleep(delay_s)
                    attempt_index += 1
                    continue
                record_remote_read_diagnostic(
                    remote_state=remote_state,
                    context=context,
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="fallback",
                )
                return _CatalogHeadLoadResult(status="unavailable")
        payload = self._extract_catalog_payload_from_document(snapshot_kind=snapshot_kind, payload=envelope)
        if payload is None and metadata_only:
            # Metadata-only current-head reads are an optimization. Non-empty
            # heads can still require one content-bearing read when the backend
            # omits enough envelope detail that the catalog cannot be rebuilt
            # from metadata alone.
            try:
                envelope = self._fetch_catalog_head_envelope(
                    read_client=effective_read_client,
                    snapshot_kind=snapshot_kind,
                    metadata_only=False,
                )
            except Exception as exc:
                record_remote_read_diagnostic(
                    remote_state=remote_state,
                    context=context,
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="fallback",
                )
                return _CatalogHeadLoadResult(status="unavailable")
            payload = self._extract_catalog_payload_from_document(snapshot_kind=snapshot_kind, payload=envelope)
        if payload is None:
            record_remote_read_diagnostic(
                remote_state=remote_state,
                context=context,
                exc=ValueError(
                    f"Remote long-term {snapshot_kind!r} current-head document did not contain a supported catalog payload."
                ),
                started_monotonic=started_monotonic,
                outcome="fallback",
            )
            self._remember_invalid_catalog_head(snapshot_kind=snapshot_kind)
            return _CatalogHeadLoadResult(status="invalid")
        self._clear_invalid_catalog_head(snapshot_kind=snapshot_kind)
        record_remote_read_observation(
            remote_state=remote_state,
            context=context,
            latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
            outcome="ok",
            classification="ok",
        )
        return _CatalogHeadLoadResult(status="found", payload=payload)

    def _catalog_head_read_client(self, *, read_client: object) -> object:
        """Return the client used for fixed current-head reads.

        Fixed `.../catalog/current` heads are same-URI moving targets, just like
        cold `origin_uri` snapshot resolution. Some live ChonkyDB builds reject
        the lighter metadata-only form first and only serve the subsequent full
        document read, which can exceed the hot-path exact-document timeout on a
        fresh reader. Reuse the remote state's bootstrap timeout tuning here so
        fail-closed current-head loads do not time out while the same backend
        would still satisfy the bounded cold-start budget.
        """

        remote_state = self.remote_state
        origin_resolution_client = None if remote_state is None else getattr(remote_state, "_origin_resolution_client", None)
        if not callable(origin_resolution_client):
            return read_client
        try:
            return origin_resolution_client(read_client)
        except Exception:
            return read_client

    def _negative_remote_contract_ttl_s(self) -> float:
        """Return how long one per-process negative remote contract stays valid."""

        return max(0.0, _NEGATIVE_REMOTE_CONTRACT_TTL_S)

    def _scope_search_supported(self, *, snapshot_kind: str) -> bool:
        """Return whether current-scope top-k remains worth probing in-process."""

        now = time.monotonic()
        with self._cache_lock:
            expires_at = self._unsupported_scope_search_cache.get(snapshot_kind)
            if expires_at is None:
                return True
            if expires_at <= now:
                self._unsupported_scope_search_cache.pop(snapshot_kind, None)
                return True
            return False

    def _remember_unsupported_scope_search(self, *, snapshot_kind: str) -> None:
        """Suppress repeated unsupported/false-empty scope probes for one process window."""

        ttl_s = self._negative_remote_contract_ttl_s()
        if ttl_s <= 0.0:
            return
        with self._cache_lock:
            self._unsupported_scope_search_cache[snapshot_kind] = time.monotonic() + ttl_s

    def _clear_unsupported_scope_search(self, *, snapshot_kind: str) -> None:
        """Forget one previous scope-search contract failure."""

        with self._cache_lock:
            self._unsupported_scope_search_cache.pop(snapshot_kind, None)

    def _catalog_head_known_invalid(self, *, snapshot_kind: str) -> bool:
        """Return whether the fixed current-head document was already proven invalid in-process."""

        now = time.monotonic()
        with self._cache_lock:
            expires_at = self._invalid_catalog_head_cache.get(snapshot_kind)
            if expires_at is None:
                return False
            if expires_at <= now:
                self._invalid_catalog_head_cache.pop(snapshot_kind, None)
                return False
            return True

    def _remember_invalid_catalog_head(self, *, snapshot_kind: str) -> None:
        """Cache one invalid current-head shape so repeated reads go straight to the real fallback."""

        ttl_s = self._negative_remote_contract_ttl_s()
        if ttl_s <= 0.0:
            return
        with self._cache_lock:
            self._invalid_catalog_head_cache[snapshot_kind] = time.monotonic() + ttl_s

    def _clear_invalid_catalog_head(self, *, snapshot_kind: str) -> None:
        """Forget one previous invalid current-head result."""

        with self._cache_lock:
            self._invalid_catalog_head_cache.pop(snapshot_kind, None)

    def _fetch_catalog_head_envelope(
        self,
        *,
        read_client: object,
        snapshot_kind: str,
        metadata_only: bool,
    ) -> object:
        """Read one fixed current head, retrying metadata-only contract rejects once.

        Some live ChonkyDB builds reject metadata-only ``documents/full``
        current-head reads with a 400 validation response even though the same
        fixed URI is otherwise readable. Snapshot probes already treat that as
        a backend contract quirk and retry with a content-bearing read; keep
        catalog current-head probes on the same contract so Graph/Object/Prompt
        readers do not degrade into the legacy fallback path for a readable
        head.
        """

        origin_uri = self._catalog_head_uri(snapshot_kind=snapshot_kind)
        try:
            return read_client.fetch_full_document(
                origin_uri=origin_uri,
                include_content=not metadata_only,
                max_content_chars=self._metadata_only_max_content_chars() if metadata_only else self._max_content_chars(),
            )
        except ChonkyDBError as exc:
            if not metadata_only or int(exc.status_code or 0) != 400:
                raise
        return read_client.fetch_full_document(
            origin_uri=origin_uri,
            include_content=True,
            max_content_chars=self._max_content_chars(),
        )

    def _load_catalog_entries_for_write(
        self,
        *,
        snapshot_kind: str,
    ) -> tuple[object, ...]:
        """Return reusable current entries without reviving legacy snapshot heads.

        Write paths may reuse the fixed-URI current head and same-process
        bridges, but they must not silently revive pointer-era snapshot reads.
        A missing current head is a legitimate empty starting point; an
        unavailable current head remains a hard remote-memory failure.
        """

        cached_entries = self._cached_catalog_entries(snapshot_kind=snapshot_kind)
        if cached_entries is not None:
            return cached_entries
        recent_entries = self._recent_catalog_entries(snapshot_kind=snapshot_kind)
        if recent_entries is not None:
            return recent_entries
        recent_head = self._recent_catalog_head_payload(snapshot_kind=snapshot_kind)
        if self.is_catalog_payload(snapshot_kind=snapshot_kind, payload=recent_head):
            return self.load_catalog_entries(snapshot_kind=snapshot_kind, payload=recent_head)
        head_result = self._load_catalog_head_result(snapshot_kind=snapshot_kind)
        if head_result.status == "found" and isinstance(head_result.payload, Mapping):
            return self.load_catalog_entries(snapshot_kind=snapshot_kind, payload=head_result.payload)
        if head_result.status == "not_found":
            return ()
        if head_result.status == "invalid":
            raise LongTermRemoteUnavailableError(
                f"Required remote long-term {snapshot_kind!r} current catalog head is invalid."
            )
        if self._remote_is_required():
            raise LongTermRemoteUnavailableError(
                f"Required remote long-term {snapshot_kind!r} current catalog head is unavailable."
            )
        return ()

    def _extract_catalog_payload_from_document(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Return the catalog payload embedded in one current-head document envelope."""

        newest_payload: dict[str, object] | None = None
        newest_rank: tuple[datetime, int] | None = None
        for position, candidate in enumerate(self._iter_record_candidates(payload)):
            candidate_payload = self._catalog_payload_candidate(
                snapshot_kind=snapshot_kind,
                candidate=candidate,
            )
            if candidate_payload is None:
                continue
            candidate_rank = (
                self._catalog_payload_written_at(candidate_payload),
                position,
            )
            if newest_rank is None or candidate_rank > newest_rank:
                newest_rank = candidate_rank
                newest_payload = candidate_payload
        return newest_payload

    def _maybe_promote_legacy_catalog_head(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> None:
        """Write the fixed-URI head document when legacy snapshot heads are still in use."""

        try:
            self.persist_catalog_payload(snapshot_kind=snapshot_kind, payload=payload)
        except Exception:
            _LOG.warning(
                "Failed promoting legacy remote %s catalog snapshot into the fixed current-head document.",
                snapshot_kind,
                exc_info=True,
            )

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

    def _remote_async_job_timeout_s(self, *, snapshot_kind: str | None = None) -> float | None:
        """Return the server-side async job budget for accepted bulk writes.

        Live incidents show that accepted current-head/catalog writes can
        legitimately outlive the generic flush budget while ChonkyDB is still
        warming payload/fulltext indexes for fresh namespaces or post-restart
        cold paths. The HTTP transport timeout is handled separately by the
        write path; this value only controls the server-side execution window
        attached to accepted async jobs.
        """

        transport_timeout_s = self._remote_write_timeout_s()
        flush_timeout_s = self._remote_flush_timeout_s()
        candidate = max(flush_timeout_s, transport_timeout_s or 0.0)
        if snapshot_kind in _ASYNC_JOB_TIMEOUT_FLOORED_SNAPSHOT_KINDS:
            candidate = max(candidate, _ASYNC_JOB_TIMEOUT_FLOOR_S)
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

    def _async_job_visibility_timeout_s(self, *, snapshot_kind: str | None = None) -> float:
        """Return the bounded window for async job completion plus document-id readback."""

        config = getattr(getattr(self, "remote_state", None), "config", None)
        try:
            read_timeout_s = float(getattr(config, "long_term_memory_remote_read_timeout_s", _DEFAULT_REMOTE_READ_TIMEOUT_S))
        except (TypeError, ValueError):
            read_timeout_s = _DEFAULT_REMOTE_READ_TIMEOUT_S
        flush_timeout_s = self._remote_flush_timeout_s()
        async_job_timeout_s = self._remote_async_job_timeout_s(snapshot_kind=snapshot_kind) or flush_timeout_s
        candidate = max(
            _DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_S,
            read_timeout_s,
            flush_timeout_s,
            async_job_timeout_s + read_timeout_s,
        )
        return min(
            _DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_CAP_S,
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

    @staticmethod
    def _selection_search_allowed_indexes() -> tuple[str, ...]:
        """Keep query-time catalog searches on lightweight non-ANN indexes."""

        return _SELECTION_SEARCH_ALLOWED_INDEXES

    @staticmethod
    def _selection_hydration_allowed_indexes() -> tuple[str, ...]:
        """Keep exact allowlist hydration on the cheapest content-bearing lane."""

        return _SELECTION_HYDRATION_ALLOWED_INDEXES

    @staticmethod
    def _searchable_write_target_indexes() -> tuple[str, ...]:
        """Persist searchable Twinr items only into the indexes the runtime uses."""

        return _SEARCHABLE_WRITE_TARGET_INDEXES

    @staticmethod
    def _control_plane_write_target_indexes() -> tuple[str, ...]:
        """Persist fixed-URI catalog control-plane docs off the search/index lanes."""

        return _CONTROL_PLANE_WRITE_TARGET_INDEXES

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

    def _remote_is_required(self) -> bool:
        """Return whether this store must fail closed on remote-memory errors."""

        remote_state = self.remote_state
        if remote_state is None or not getattr(remote_state, "enabled", False):
            return False
        required = getattr(remote_state, "required", None)
        if callable(required):
            try:
                return bool(required())
            except Exception:
                return True
        if required is None:
            return True
        return bool(required)

    def _require_client(self, client: object | None, *, operation: str) -> object:
        if client is not None:
            return client
        raise LongTermRemoteUnavailableError(
            f"Remote-primary long-term memory is enabled but ChonkyDB is not configured for {operation} operations."
        )

    def _client_with_timeout(self, client: object, *, timeout_s: float | None) -> object:
        """Clone a client with one specific transport timeout when supported."""

        if timeout_s is None:
            return client
        try:
            resolved_timeout_s = float(timeout_s)
        except (TypeError, ValueError):
            return client
        if not math.isfinite(resolved_timeout_s) or resolved_timeout_s <= 0.0:
            return client
        current_config = getattr(client, "config", None)
        try:
            current_timeout_s = float(getattr(current_config, "timeout_s"))
        except (AttributeError, TypeError, ValueError):
            current_timeout_s = None
        if current_timeout_s is not None and math.isfinite(current_timeout_s):
            if abs(current_timeout_s - resolved_timeout_s) <= 1e-9:
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
        for field_name in self._catalog_entry_object_fields():
            value = payload.get(field_name)
            if isinstance(value, Mapping):
                metadata[field_name] = dict(value)
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

    def _catalog_payload_candidate(
        self,
        *,
        snapshot_kind: str,
        candidate: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Return one usable catalog payload from a live current-head candidate shape."""

        if self.is_catalog_payload(snapshot_kind=snapshot_kind, payload=candidate):
            return dict(candidate)
        definition = self._definition(snapshot_kind)
        if definition is None:
            return None
        if candidate.get("twinr_catalog_current_head") is not True:
            return None
        if self._normalize_text(candidate.get("twinr_catalog_schema")) != definition.catalog_schema:
            return None
        try:
            items_count = int(candidate.get("twinr_catalog_items_count") or 0)
        except (TypeError, ValueError):
            items_count = 0
        if items_count > 0:
            return None
        payload: dict[str, object] = {
            "schema": definition.catalog_schema,
            "version": 3,
            "items_count": max(0, items_count),
            "segments": [],
        }
        written_at = self._normalize_text(candidate.get("twinr_catalog_written_at"))
        if written_at:
            payload["written_at"] = written_at
        return payload

    @staticmethod
    def _catalog_payload_written_at(payload: Mapping[str, object]) -> datetime:
        """Return one sortable written_at timestamp for current-head disambiguation."""

        raw_written_at = " ".join(str(payload.get("written_at") or "").split()).strip()
        if not raw_written_at:
            return datetime.min.replace(tzinfo=timezone.utc)
        normalized = raw_written_at.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _metadata_only_max_content_chars() -> int:
        """Return the backend-compatible floor for metadata-only head probes."""

        return 100

    @staticmethod
    def _catalog_entry_text_fields() -> tuple[str, ...]:
        from .shared import _CATALOG_ENTRY_TEXT_FIELDS

        return _CATALOG_ENTRY_TEXT_FIELDS

    @staticmethod
    def _catalog_entry_list_fields() -> tuple[str, ...]:
        from .shared import _CATALOG_ENTRY_LIST_FIELDS

        return _CATALOG_ENTRY_LIST_FIELDS

    @staticmethod
    def _catalog_entry_object_fields() -> tuple[str, ...]:
        from .shared import _CATALOG_ENTRY_OBJECT_FIELDS

        return _CATALOG_ENTRY_OBJECT_FIELDS

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

    def _status_code_from_exception(self, exc: BaseException) -> int | None:
        """Extract one ChonkyDB HTTP status code from an exception chain."""

        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, ChonkyDBError) and current.status_code is not None:
                try:
                    return int(current.status_code)
                except (TypeError, ValueError):
                    return None
            current = current.__cause__ or current.__context__
        return None

    def _response_bits_from_exception(self, exc: BaseException) -> tuple[str, ...]:
        """Collect compact response fragments from one exception chain."""

        response_bits: list[str] = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, ChonkyDBError):
                if isinstance(current.response_text, str):
                    response_bits.append(current.response_text)
                if isinstance(current.response_json, Mapping):
                    for field_name in ("detail", "error", "error_type", "title"):
                        value = current.response_json.get(field_name)
                        if value is not None:
                            response_bits.append(str(value))
            current = current.__cause__ or current.__context__
        return tuple(response_bits)

    def _scope_search_failure_mode(self, *, snapshot_kind: str, exc: BaseException) -> str:
        """Classify one scope-search failure without conflating fresh empty heads and bad contracts."""

        status_code = self._status_code_from_exception(exc)
        if status_code not in {400, 404}:
            return "other"
        compact_bits = " ".join(self._response_bits_from_exception(exc)).strip().lower()
        if "unsupported scope_ref" in compact_bits:
            return "unsupported_scope_search"
        if status_code != 404 or "document_not_found" not in compact_bits:
            return "other"
        head_result = self._load_catalog_head_result(snapshot_kind=snapshot_kind, metadata_only=True)
        if head_result.status == "not_found":
            return "missing_current_head"
        if head_result.status == "found" and isinstance(head_result.payload, Mapping):
            self._store_recent_catalog_head_payload(snapshot_kind=snapshot_kind, payload=head_result.payload)
            return "unsupported_scope_search"
        return "other"

    def _should_disable_scope_search_from_exception(self, *, snapshot_kind: str, exc: BaseException) -> bool:
        """Return whether one scope-search failure proves the current backend contract is unusable here."""

        return self._scope_search_failure_mode(snapshot_kind=snapshot_kind, exc=exc) == "unsupported_scope_search"


__all__ = [
    "LongTermRemoteCatalogStoreBase",
]
