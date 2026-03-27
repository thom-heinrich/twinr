"""Shared types, defaults, and trace helpers for remote catalog storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

from twinr.agent.workflows.forensics import workflow_event
from twinr.memory.fulltext import FullTextSelector

_CATALOG_VERSION = 3
_LEGACY_CATALOG_VERSION = 2
_SEGMENT_VERSION = 1
_ITEM_VERSION = 1
_DEFAULT_MAX_ITEM_CONTENT_CHARS = 256_000
_DEFAULT_BULK_BATCH_SIZE = 64
_DEFAULT_BULK_REQUEST_MAX_BYTES = 512 * 1024
_DEFAULT_RETRIEVE_BATCH_SIZE = 500
_DEFAULT_REMOTE_READ_MAX_WORKERS = 4
_DEFAULT_REMOTE_READ_TIMEOUT_S = 8.0
_DEFAULT_REMOTE_WRITE_TIMEOUT_S = 15.0
_DEFAULT_REMOTE_FLUSH_TIMEOUT_S = 60.0
_DEFAULT_REMOTE_RETRY_ATTEMPTS = 3
_DEFAULT_REMOTE_RETRY_BACKOFF_S = 1.0
_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_S = 20.0
_DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_S = 20.0
_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S = 180.0
_DEFAULT_ASYNC_ATTESTATION_POLL_S = 0.05
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


def _run_timed_workflow_step(
    *,
    name: str,
    kind: str,
    details: dict[str, object],
    operation: Callable[[], object],
) -> object:
    """Emit timing events for one remote-catalog step without contextmanager rethrow issues."""

    workflow_event(kind="span_start", msg=name, details={"kind": kind, **details})
    started = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:
        workflow_event(
            kind="exception",
            msg=f"{name}_exception",
            level="ERROR",
            details={
                "span": name,
                "kind": kind,
                "exception": {"type": type(exc).__name__},
            },
            kpi={"duration_ms": round((time.perf_counter() - started) * 1000.0, 3)},
        )
        raise
    workflow_event(
        kind="span_end",
        msg=name,
        details={"kind": kind, **details},
        kpi={"duration_ms": round((time.perf_counter() - started) * 1000.0, 3)},
    )
    return result


def _trace_search_details(
    *,
    snapshot_kind: str,
    query_text: str | None = None,
    result_limit: int | None = None,
    allowed_doc_count: int | None = None,
    candidate_limit: int | None = None,
    batch_size: int | None = None,
    scope_ref: str | None = None,
    namespace: str | None = None,
    catalog_entry_count: int | None = None,
    cached_catalog_entries: int | None = None,
) -> dict[str, object]:
    """Summarize remote-catalog request shape for workflow traces."""

    normalized_query = " ".join(str(query_text or "").split()).strip()
    details: dict[str, object] = {
        "snapshot_kind": str(snapshot_kind or "").strip(),
        "query_chars": len(normalized_query),
        "scope_search": bool(scope_ref and namespace),
        "has_namespace": bool(namespace),
        "has_scope_ref": bool(scope_ref),
    }
    if result_limit is not None:
        details["result_limit"] = max(0, int(result_limit))
    if allowed_doc_count is not None:
        details["allowed_doc_count"] = max(0, int(allowed_doc_count))
    if candidate_limit is not None:
        details["candidate_limit"] = max(0, int(candidate_limit))
    if batch_size is not None:
        details["batch_size"] = max(0, int(batch_size))
    if catalog_entry_count is not None:
        details["catalog_entry_count"] = max(0, int(catalog_entry_count))
    if cached_catalog_entries is not None:
        details["cached_catalog_entries"] = max(0, int(cached_catalog_entries))
    return details


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


__all__ = [
    "LongTermRemoteCatalogAssemblyResult",
    "LongTermRemoteCatalogEntry",
]
