"""Shared types, defaults, and trace helpers for remote catalog storage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import TypeVar

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
_DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_CAP_S = 240.0
_DEFAULT_ASYNC_ATTESTATION_POLL_S = 0.05

_ResultT = TypeVar("_ResultT")
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
_CATALOG_ENTRY_OBJECT_FIELDS = ("selection_projection",)
_KNOWN_ITEM_ENVELOPE_KEYS = (
    "object",
    "conflict",
    "packet",
    "node",
    "edge",
    "answer_front_block",
    "memory_entry",
    "managed_context_entry",
    "personality_snapshot",
    "interaction_signal",
    "place_signal",
    "world_signal",
    "personality_delta",
    "world_feed_subscription",
    "world_intelligence_state",
)


def _run_timed_workflow_step(
    *,
    name: str,
    kind: str,
    details: dict[str, object],
    operation: Callable[[], _ResultT],
) -> _ResultT:
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


def _iter_known_item_envelopes(candidate: object) -> tuple[Mapping[str, object], ...]:
    """Return nested public item envelopes carried inside a record wrapper.

    Remote catalog records wrap the first-class Twinr payload inside a typed
    ChonkyDB record envelope. Some read paths surface that wrapper, while others
    surface only a nested payload fragment. Keeping the known envelope keys in
    one helper lets write attestation and live-document parsing accept either
    shape without duplicating ad-hoc key checks.
    """

    if not isinstance(candidate, Mapping):
        return ()
    envelopes: list[Mapping[str, object]] = []
    for field_name in _KNOWN_ITEM_ENVELOPE_KEYS:
        value = candidate.get(field_name)
        if isinstance(value, dict):
            envelopes.append(dict(value))
    return tuple(envelopes)


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
    "midterm": _RemoteCollectionDefinition(
        snapshot_kind="midterm",
        catalog_schema="twinr_memory_midterm_catalog_v3",
        legacy_catalog_schema="twinr_memory_midterm_catalog_v2",
        segment_schema="twinr_memory_midterm_catalog_segment_v1",
        item_schema="twinr_memory_midterm_packet_record_v1",
        envelope_key="packet",
        uri_segment="midterm",
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
    "graph_nodes": _RemoteCollectionDefinition(
        snapshot_kind="graph_nodes",
        catalog_schema="twinr_graph_node_catalog_v3",
        legacy_catalog_schema="twinr_graph_node_catalog_v2",
        segment_schema="twinr_graph_node_catalog_segment_v1",
        item_schema="twinr_graph_node_record_v1",
        envelope_key="node",
        uri_segment="graph_nodes",
    ),
    "graph_edges": _RemoteCollectionDefinition(
        snapshot_kind="graph_edges",
        catalog_schema="twinr_graph_edge_catalog_v3",
        legacy_catalog_schema="twinr_graph_edge_catalog_v2",
        segment_schema="twinr_graph_edge_catalog_segment_v1",
        item_schema="twinr_graph_edge_record_v1",
        envelope_key="edge",
        uri_segment="graph_edges",
    ),
    "provider_answer_fronts": _RemoteCollectionDefinition(
        snapshot_kind="provider_answer_fronts",
        catalog_schema="twinr_provider_answer_front_catalog_v3",
        legacy_catalog_schema="twinr_provider_answer_front_catalog_v2",
        segment_schema="twinr_provider_answer_front_catalog_segment_v1",
        item_schema="twinr_provider_answer_front_record_v1",
        envelope_key="answer_front_block",
        uri_segment="provider_answer_fronts",
    ),
    "prompt_memory": _RemoteCollectionDefinition(
        snapshot_kind="prompt_memory",
        catalog_schema="twinr_prompt_memory_catalog_v3",
        legacy_catalog_schema="twinr_prompt_memory_catalog_v2",
        segment_schema="twinr_prompt_memory_catalog_segment_v1",
        item_schema="twinr_prompt_memory_record_v1",
        envelope_key="memory_entry",
        uri_segment="prompt_memory",
    ),
    "user_context": _RemoteCollectionDefinition(
        snapshot_kind="user_context",
        catalog_schema="twinr_user_context_catalog_v3",
        legacy_catalog_schema="twinr_user_context_catalog_v2",
        segment_schema="twinr_user_context_catalog_segment_v1",
        item_schema="twinr_user_context_record_v1",
        envelope_key="managed_context_entry",
        uri_segment="user_context",
    ),
    "personality_context": _RemoteCollectionDefinition(
        snapshot_kind="personality_context",
        catalog_schema="twinr_personality_context_catalog_v3",
        legacy_catalog_schema="twinr_personality_context_catalog_v2",
        segment_schema="twinr_personality_context_catalog_segment_v1",
        item_schema="twinr_personality_context_record_v1",
        envelope_key="managed_context_entry",
        uri_segment="personality_context",
    ),
    "agent_personality_context_v1": _RemoteCollectionDefinition(
        snapshot_kind="agent_personality_context_v1",
        catalog_schema="twinr_agent_personality_snapshot_catalog_v3",
        legacy_catalog_schema="twinr_agent_personality_snapshot_catalog_v2",
        segment_schema="twinr_agent_personality_snapshot_catalog_segment_v1",
        item_schema="twinr_agent_personality_snapshot_record_v1",
        envelope_key="personality_snapshot",
        uri_segment="agent_personality_context",
    ),
    "agent_personality_interaction_signals_v1": _RemoteCollectionDefinition(
        snapshot_kind="agent_personality_interaction_signals_v1",
        catalog_schema="twinr_agent_personality_interaction_signal_catalog_v3",
        legacy_catalog_schema="twinr_agent_personality_interaction_signal_catalog_v2",
        segment_schema="twinr_agent_personality_interaction_signal_catalog_segment_v1",
        item_schema="twinr_agent_personality_interaction_signal_record_v1",
        envelope_key="interaction_signal",
        uri_segment="agent_personality_interaction_signals",
    ),
    "agent_personality_place_signals_v1": _RemoteCollectionDefinition(
        snapshot_kind="agent_personality_place_signals_v1",
        catalog_schema="twinr_agent_personality_place_signal_catalog_v3",
        legacy_catalog_schema="twinr_agent_personality_place_signal_catalog_v2",
        segment_schema="twinr_agent_personality_place_signal_catalog_segment_v1",
        item_schema="twinr_agent_personality_place_signal_record_v1",
        envelope_key="place_signal",
        uri_segment="agent_personality_place_signals",
    ),
    "agent_personality_world_signals_v1": _RemoteCollectionDefinition(
        snapshot_kind="agent_personality_world_signals_v1",
        catalog_schema="twinr_agent_personality_world_signal_catalog_v3",
        legacy_catalog_schema="twinr_agent_personality_world_signal_catalog_v2",
        segment_schema="twinr_agent_personality_world_signal_catalog_segment_v1",
        item_schema="twinr_agent_personality_world_signal_record_v1",
        envelope_key="world_signal",
        uri_segment="agent_personality_world_signals",
    ),
    "agent_personality_deltas_v1": _RemoteCollectionDefinition(
        snapshot_kind="agent_personality_deltas_v1",
        catalog_schema="twinr_agent_personality_delta_catalog_v3",
        legacy_catalog_schema="twinr_agent_personality_delta_catalog_v2",
        segment_schema="twinr_agent_personality_delta_catalog_segment_v1",
        item_schema="twinr_agent_personality_delta_record_v1",
        envelope_key="personality_delta",
        uri_segment="agent_personality_deltas",
    ),
    "agent_world_intelligence_subscriptions_v1": _RemoteCollectionDefinition(
        snapshot_kind="agent_world_intelligence_subscriptions_v1",
        catalog_schema="twinr_world_intelligence_subscription_catalog_v3",
        legacy_catalog_schema="twinr_world_intelligence_subscription_catalog_v2",
        segment_schema="twinr_world_intelligence_subscription_catalog_segment_v1",
        item_schema="twinr_world_intelligence_subscription_record_v1",
        envelope_key="world_feed_subscription",
        uri_segment="agent_world_intelligence_subscriptions",
    ),
    "agent_world_intelligence_state_v1": _RemoteCollectionDefinition(
        snapshot_kind="agent_world_intelligence_state_v1",
        catalog_schema="twinr_world_intelligence_state_catalog_v3",
        legacy_catalog_schema="twinr_world_intelligence_state_catalog_v2",
        segment_schema="twinr_world_intelligence_state_catalog_segment_v1",
        item_schema="twinr_world_intelligence_state_record_v1",
        envelope_key="world_intelligence_state",
        uri_segment="agent_world_intelligence_state",
    ),
}


__all__ = [
    "LongTermRemoteCatalogAssemblyResult",
    "LongTermRemoteCatalogEntry",
]
