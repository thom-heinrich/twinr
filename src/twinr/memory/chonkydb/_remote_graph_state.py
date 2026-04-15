"""Remote-authoritative current-view and topology sync for Twinr personal graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import logging
import time
from typing import TYPE_CHECKING, cast

from twinr.agent.workflows.forensics import workflow_decision, workflow_event
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.chonkydb.models import (
    ChonkyDBGraphNeighborsRequest,
    ChonkyDBGraphPathRequest,
    ChonkyDBGraphStoreManyEdge,
    ChonkyDBGraphStoreManyNode,
    ChonkyDBGraphStoreManyRequest,
)
from twinr.memory.chonkydb.schema import TwinrGraphDocumentV1, TwinrGraphEdgeV1, TwinrGraphNodeV1
from twinr.text_utils import folded_lookup_text, slugify_identifier, truncate_text

if TYPE_CHECKING:
    from twinr.memory.longterm.storage._remote_catalog.shared import LongTermRemoteCatalogEntry
    from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore


LOGGER = logging.getLogger(__name__)
_GRAPH_NODE_SNAPSHOT_KIND = "graph_nodes"
_GRAPH_EDGE_SNAPSHOT_KIND = "graph_edges"
_MIN_GRAPH_STORE_TIMEOUT_S = 10
_MAX_GRAPH_QUERY_RETRY_ATTEMPTS = 2
_MAX_GRAPH_QUERY_RETRY_BACKOFF_S = 0.25
_MAX_GRAPH_STORE_RETRY_ATTEMPTS = 10
_READINESS_CURRENT_HEAD_TIMEOUT_S = 5.0
_GRAPH_NODE_CATALOG_SEARCH_TEXT_LIMIT = 512
_GRAPH_NODE_RERANK_DEBUG_LIMIT = 8
_GRAPH_CONTACT_QUERY_TERMS = frozenset(
    {
        "email",
        "mail",
        "address",
        "adresse",
        "phone",
        "telefon",
        "number",
        "nummer",
    }
)


@dataclass(frozen=True, slots=True)
class TwinrRemoteGraphSelection:
    """Hold one query-first remote graph subgraph plus its explainable plan."""

    document: TwinrGraphDocumentV1
    query_plan: dict[str, object]


def _remote_unavailable_error(message: str) -> Exception:
    from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

    return LongTermRemoteUnavailableError(message)


def _remote_unavailable_error_type():
    from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

    return LongTermRemoteUnavailableError


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _lookup_terms(value: object | None) -> tuple[str, ...]:
    normalized = " ".join(folded_lookup_text(_normalize_text(value)).split())
    if not normalized:
        return ()
    seen: set[str] = set()
    terms: list[str] = []
    for term in normalized.split():
        clean_term = _normalize_text(term)
        if clean_term and clean_term not in seen:
            seen.add(clean_term)
            terms.append(clean_term)
    return tuple(terms)


def _phrase_fingerprint(value: object | None) -> str:
    return " ".join(_lookup_terms(value))


def _edge_item_id(edge: TwinrGraphEdgeV1) -> str:
    return f"{edge.source_node_id}|{edge.edge_type}|{edge.target_node_id}"


def _edge_item_id_from_payload(payload: Mapping[str, object]) -> str | None:
    source = _normalize_text(payload.get("source"))
    edge_type = _normalize_text(payload.get("type"))
    target = _normalize_text(payload.get("target"))
    if not source or not edge_type or not target:
        return None
    return f"{source}|{edge_type}|{target}"


def _parse_json_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _iter_direct_graph_head_candidates(
    payload: Mapping[str, object],
    *,
    _seen: set[int] | None = None,
):
    """Yield payload/body/content candidates without trusting metadata shadows.

    Graph current-head repair must look at the advertised fixed-URI head
    document itself. Metadata-carried ``twinr_payload`` shadows can preserve a
    stale complete head behind an already generic/incomplete direct
    ``catalog/current`` record and would therefore suppress the repair path we
    need when the compatible snapshot head is also missing.
    """

    seen = _seen if _seen is not None else set()
    payload_id = id(payload)
    if payload_id in seen:
        return
    seen.add(payload_id)
    yield payload
    for field_name in ("body", "payload", "record", "document"):
        nested = payload.get(field_name)
        if isinstance(nested, Mapping):
            yield from _iter_direct_graph_head_candidates(nested, _seen=seen)
    parsed_content = _parse_json_mapping(payload.get("content"))
    if isinstance(parsed_content, Mapping):
        yield from _iter_direct_graph_head_candidates(parsed_content, _seen=seen)
    chunks = payload.get("chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if isinstance(chunk, Mapping):
                yield from _iter_direct_graph_head_candidates(chunk, _seen=seen)


def _looks_like_graph_catalog_payload(
    *,
    snapshot_kind: str,
    payload: Mapping[str, object],
    catalog: object,
) -> bool:
    definition_getter = getattr(catalog, "_definition", None)
    if not callable(definition_getter):
        return False
    definition = definition_getter(snapshot_kind)
    if definition is None:
        return False
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return False
    try:
        version = int(_normalize_text(payload.get("version")) or "0")
        int(_normalize_text(payload.get("items_count")) or "0")
    except (TypeError, ValueError):
        return False
    return payload.get("schema") == definition.catalog_schema and version == 3


def _node_content(node_payload: Mapping[str, object]) -> str:
    parts: list[str] = [
        _normalize_text(node_payload.get("label")),
        _normalize_text(node_payload.get("type")),
    ]
    aliases = node_payload.get("aliases")
    if isinstance(aliases, (list, tuple)):
        parts.extend(_normalize_text(item) for item in aliases if _normalize_text(item))
    attributes = node_payload.get("attributes")
    if isinstance(attributes, Mapping):
        for key, value in attributes.items():
            key_text = _normalize_text(key)
            if key_text:
                parts.append(key_text)
            if isinstance(value, str):
                value_text = _normalize_text(value)
                if value_text:
                    parts.append(value_text)
            elif isinstance(value, (list, tuple)):
                parts.extend(_normalize_text(item) for item in value if _normalize_text(item))
    return " ".join(part for part in parts if part)


def _node_related_contact_fragments(
    document: TwinrGraphDocumentV1,
    node_payload: Mapping[str, object],
) -> tuple[str, ...]:
    """Return one-hop contact fragments that improve person/contact-method recall.

    Query-first graph selection ranks raw node records. Person nodes therefore
    need some contact-method vocabulary, and contact-method nodes need their
    owning person label, otherwise queries such as "Anna Becker email address"
    can incorrectly prefer another Becker person that lacks the requested
    contact method.
    """

    node_id = _normalize_text(node_payload.get("id"))
    node_type = _normalize_text(node_payload.get("type")).lower()
    if not node_id or not node_type:
        return ()
    nodes_by_id = {node.node_id: node for node in document.nodes}
    fragments: list[str] = []
    if node_type == "person":
        for edge in document.edges:
            if edge.source_node_id != node_id or edge.edge_type != "general_has_contact_method":
                continue
            target = nodes_by_id.get(edge.target_node_id)
            if target is None:
                continue
            kind = _normalize_text((edge.attributes or {}).get("kind") or target.node_type).lower()
            target_label = _normalize_text(target.label)
            canonical_value = _normalize_text((target.attributes or {}).get("canonical") or target.label)
            if kind == "email":
                fragments.extend(("email", "email address"))
            elif kind == "phone":
                fragments.extend(("phone", "phone number"))
            if kind:
                fragments.append(kind)
            if target_label:
                fragments.append(target_label)
            if canonical_value and canonical_value != target_label:
                fragments.append(canonical_value)
    elif node_type in {"email", "phone"}:
        if node_type == "email":
            fragments.extend(("email", "email address"))
        elif node_type == "phone":
            fragments.extend(("phone", "phone number"))
        for edge in document.edges:
            if edge.target_node_id != node_id or edge.edge_type != "general_has_contact_method":
                continue
            source = nodes_by_id.get(edge.source_node_id)
            if source is None:
                continue
            source_label = _normalize_text(source.label)
            if source_label:
                fragments.append(source_label)
            aliases = getattr(source, "aliases", ()) or ()
            for alias in aliases:
                alias_text = _normalize_text(alias)
                if alias_text:
                    fragments.append(alias_text)
    deduped: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        normalized = _normalize_text(fragment)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return tuple(deduped)


def _node_content_for_document(
    document: TwinrGraphDocumentV1,
    node_payload: Mapping[str, object],
) -> str:
    """Return search text for one graph node plus adjacent contact-method cues."""

    base_content = _node_content(node_payload)
    related_contact_content = " ".join(_node_related_contact_fragments(document, node_payload))
    return " ".join(part for part in (base_content, related_contact_content) if part)


def _node_search_text_for_catalog(
    document: TwinrGraphDocumentV1,
    node_payload: Mapping[str, object],
) -> str:
    """Return bounded search text for projection-only graph-node catalog entries.

    Projection-only graph current views deliberately skip a second fine-grained
    item-document write lane. The catalog metadata therefore has to preserve a
    small searchable projection of the enriched node text, otherwise ambiguous
    contact queries lose the email/phone fragments needed for disambiguation.
    """

    return truncate_text(
        _node_content_for_document(document, node_payload),
        limit=_GRAPH_NODE_CATALOG_SEARCH_TEXT_LIMIT,
    )


def _node_aliases_from_payload(node_payload: Mapping[str, object]) -> tuple[str, ...]:
    aliases = node_payload.get("aliases")
    if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes, bytearray)):
        return ()
    normalized_aliases: list[str] = []
    for alias in aliases:
        alias_text = _normalize_text(alias)
        if alias_text:
            normalized_aliases.append(alias_text)
    return tuple(normalized_aliases)


def _node_search_text_from_entry(entry: object | None) -> str:
    metadata = getattr(entry, "metadata", None)
    if not isinstance(metadata, Mapping):
        return ""
    return _normalize_text(metadata.get("search_text"))


def _node_owner_person_ids(
    document: TwinrGraphDocumentV1,
    node_payload: Mapping[str, object],
) -> tuple[str, ...]:
    """Return owning person ids for one contact-method node."""

    node_id = _normalize_text(node_payload.get("id"))
    node_type = _normalize_text(node_payload.get("type")).lower()
    if not node_id or node_type not in {"email", "phone"}:
        return ()
    person_ids: list[str] = []
    seen: set[str] = set()
    nodes_by_id = {node.node_id: node for node in document.nodes}
    for edge in document.edges:
        if edge.target_node_id != node_id or edge.edge_type != "general_has_contact_method":
            continue
        source = nodes_by_id.get(edge.source_node_id)
        if source is None or _normalize_text(source.node_type).lower() != "person":
            continue
        if source.node_id in seen:
            continue
        seen.add(source.node_id)
        person_ids.append(source.node_id)
    return tuple(person_ids)


def _node_owner_person_ids_from_entry(entry: object | None) -> tuple[str, ...]:
    metadata = getattr(entry, "metadata", None)
    if not isinstance(metadata, Mapping):
        return ()
    owner_ids = metadata.get("owner_person_node_ids")
    if not isinstance(owner_ids, Sequence) or isinstance(owner_ids, (str, bytes, bytearray)):
        return ()
    normalized_owner_ids: list[str] = []
    seen: set[str] = set()
    for owner_id in owner_ids:
        normalized_owner_id = _normalize_text(owner_id)
        if not normalized_owner_id or normalized_owner_id in seen:
            continue
        seen.add(normalized_owner_id)
        normalized_owner_ids.append(normalized_owner_id)
    return tuple(normalized_owner_ids)


def _catalog_entry_projection_payload(entry: object | None) -> dict[str, object] | None:
    if entry is None:
        return None
    metadata = getattr(entry, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    projection = metadata.get("selection_projection")
    if not isinstance(projection, Mapping):
        return None
    return dict(projection)


def _promote_owner_person_candidate_payloads(
    *,
    candidate_payloads: Sequence[Mapping[str, object]],
    current_entries_by_item_id: Mapping[str, object],
    origins_by_item_id: Mapping[str, Sequence[str]],
) -> tuple[tuple[dict[str, object], ...], dict[str, tuple[str, ...]], tuple[dict[str, object], ...]]:
    """Inject owning person nodes for direct contact-method hits.

    Ambiguous queries such as "Anna Becker email" can retrieve a direct contact
    node (`email:anna...`) without the owning person node appearing in the same
    bounded candidate pool. If that happens, later unified selection sees only
    the distractor person (`Chris Becker`) as a focal person match. Promote the
    owning person into the same rerank pool so entity disambiguation is decided
    before graph seed expansion.
    """

    promoted_payloads: list[dict[str, object]] = [
        dict(payload)
        for payload in candidate_payloads
        if isinstance(payload, Mapping)
    ]
    promoted_origins: dict[str, tuple[str, ...]] = {
        item_id: tuple(origins)
        for item_id, origins in origins_by_item_id.items()
    }
    known_item_ids = {
        _normalize_text(payload.get("id"))
        for payload in promoted_payloads
        if _normalize_text(payload.get("id"))
    }
    promotion_debug: list[dict[str, object]] = []
    for payload in candidate_payloads:
        if not isinstance(payload, Mapping):
            continue
        node_id = _normalize_text(payload.get("id"))
        if not node_id:
            continue
        entry = current_entries_by_item_id.get(node_id)
        owner_person_ids = _node_owner_person_ids_from_entry(entry)
        if not owner_person_ids:
            continue
        for owner_person_id in owner_person_ids:
            if owner_person_id in known_item_ids:
                continue
            owner_entry = current_entries_by_item_id.get(owner_person_id)
            owner_payload = _catalog_entry_projection_payload(owner_entry)
            if not isinstance(owner_payload, Mapping):
                continue
            promoted_payloads.append(dict(owner_payload))
            known_item_ids.add(owner_person_id)
            prior_origins = list(promoted_origins.get(owner_person_id, ()))
            if "owner_person_promotion" not in prior_origins:
                prior_origins.append("owner_person_promotion")
            promoted_origins[owner_person_id] = tuple(prior_origins)
            promotion_debug.append(
                {
                    "from_node_id": node_id,
                    "from_node_type": _normalize_text(payload.get("type")).lower() or "unknown",
                    "promoted_owner_node_id": owner_person_id,
                    "promoted_owner_label": _normalize_text(owner_payload.get("label")) or owner_person_id,
                    "reason": "contact_method_owner_seed",
                }
            )
    return tuple(promoted_payloads), promoted_origins, tuple(promotion_debug)


def _merge_graph_node_candidate_payloads(
    *,
    initial_payloads: Sequence[Mapping[str, object]],
    local_entries: Sequence[object],
) -> tuple[tuple[dict[str, object], ...], dict[str, tuple[str, ...]]]:
    """Merge scope-topk and current-catalog candidates into one bounded rerank pool."""

    payloads_by_item_id: dict[str, dict[str, object]] = {}
    origins_by_item_id: dict[str, list[str]] = {}
    order_by_item_id: dict[str, int] = {}

    def register_payload(
        *,
        payload: Mapping[str, object] | None,
        origin: str,
        order: int,
    ) -> None:
        if not isinstance(payload, Mapping):
            return
        item_id = _normalize_text(payload.get("id"))
        if not item_id:
            return
        payloads_by_item_id.setdefault(item_id, dict(payload))
        order_by_item_id.setdefault(item_id, order)
        origins = origins_by_item_id.setdefault(item_id, [])
        if origin not in origins:
            origins.append(origin)

    for index, payload in enumerate(initial_payloads):
        register_payload(payload=payload, origin="scope_topk_records", order=index)
    for index, entry in enumerate(local_entries, start=len(order_by_item_id)):
        register_payload(
            payload=_catalog_entry_projection_payload(entry),
            origin="local_catalog_search",
            order=index,
        )

    ordered_payloads = tuple(
        payload
        for _item_id, payload in sorted(payloads_by_item_id.items(), key=lambda item: order_by_item_id[item[0]])
    )
    origins = {
        item_id: tuple(origin_list)
        for item_id, origin_list in origins_by_item_id.items()
    }
    return ordered_payloads, origins


def _graph_node_rerank_scorecard(
    *,
    query_text: str,
    query_terms: Sequence[str],
    node_payload: Mapping[str, object],
    entry: object | None,
    remote_rank: int,
    origins: Sequence[str],
) -> dict[str, object]:
    """Return one bounded scorecard for local graph-node reranking."""

    node_id = _normalize_text(node_payload.get("id"))
    node_type = _normalize_text(node_payload.get("type")).lower()
    label = _normalize_text(node_payload.get("label"))
    aliases = _node_aliases_from_payload(node_payload)
    search_text = _node_search_text_from_entry(entry)

    query_fingerprint = _phrase_fingerprint(query_text)
    label_fingerprint = _phrase_fingerprint(label)
    alias_fingerprints = tuple(_phrase_fingerprint(alias) for alias in aliases if _phrase_fingerprint(alias))
    query_term_set = set(query_terms)
    label_terms = set(_lookup_terms(label))
    alias_term_sets = [set(_lookup_terms(alias)) for alias in aliases]
    search_term_set = set(_lookup_terms(search_text))
    exact_label_phrase = bool(label_fingerprint and label_fingerprint in query_fingerprint)
    exact_alias_phrase = any(alias and alias in query_fingerprint for alias in alias_fingerprints)
    label_overlap = len(query_term_set & label_terms)
    alias_overlap = max((len(query_term_set & alias_terms) for alias_terms in alias_term_sets), default=0)
    search_overlap = len(query_term_set & search_term_set)
    contact_overlap = len((query_term_set & _GRAPH_CONTACT_QUERY_TERMS) & search_term_set)
    label_coverage = (label_overlap / len(label_terms)) if label_terms else 0.0
    alias_coverage = max(((len(query_term_set & alias_terms) / len(alias_terms)) for alias_terms in alias_term_sets if alias_terms), default=0.0)
    person_bias = 1.0 if node_type == "person" else 0.0
    exact_contact_value = bool(
        node_type in {"email", "phone"}
        and label_fingerprint
        and label_fingerprint in query_fingerprint
    )

    total_score = (
        (100.0 if exact_label_phrase else 0.0)
        + (85.0 if exact_alias_phrase else 0.0)
        + (40.0 * label_coverage)
        + (20.0 * alias_coverage)
        + (4.0 * search_overlap)
        + (8.0 * contact_overlap)
        + (3.0 * person_bias)
        + (30.0 if exact_contact_value else 0.0)
        - (0.05 * remote_rank)
    )
    return {
        "node_id": node_id,
        "label": label or node_id,
        "node_type": node_type or "unknown",
        "origins": list(origins),
        "remote_rank": remote_rank,
        "total_score": total_score,
        "score_components": {
            "exact_label_phrase": exact_label_phrase,
            "exact_alias_phrase": exact_alias_phrase,
            "exact_contact_value": exact_contact_value,
            "label_overlap": label_overlap,
            "alias_overlap": alias_overlap,
            "label_coverage": round(label_coverage, 4),
            "alias_coverage": round(alias_coverage, 4),
            "search_overlap": search_overlap,
            "contact_overlap": contact_overlap,
            "person_bias": person_bias,
        },
        "debug": {
            "aliases": list(aliases[:4]),
            "search_text_excerpt": truncate_text(search_text, limit=160),
        },
    }


def _rerank_graph_node_payloads(
    *,
    query_text: str,
    candidate_payloads: Sequence[Mapping[str, object]],
    current_entries_by_item_id: Mapping[str, object],
    candidate_limit: int,
    origins_by_item_id: Mapping[str, Sequence[str]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Rerank graph-node candidates locally so exact person/contact cues beat popularity noise."""

    query_terms = _lookup_terms(query_text)
    scored_candidates: list[tuple[dict[str, object], dict[str, object]]] = []
    for remote_rank, payload in enumerate(candidate_payloads):
        node_id = _normalize_text(payload.get("id"))
        if not node_id:
            continue
        scorecard = _graph_node_rerank_scorecard(
            query_text=query_text,
            query_terms=query_terms,
            node_payload=payload,
            entry=current_entries_by_item_id.get(node_id),
            remote_rank=remote_rank,
            origins=origins_by_item_id.get(node_id, ()),
        )
        scored_candidates.append((dict(payload), scorecard))

    scored_candidates.sort(
        key=lambda item: (
            -cast(float, item[1]["total_score"]),
            cast(int, item[1]["remote_rank"]),
            _normalize_text(item[1]["node_id"]),
        )
    )
    bounded_limit = max(1, int(candidate_limit))
    selected_payloads = tuple(payload for payload, _scorecard in scored_candidates[:bounded_limit])
    selected_ids = {
        _normalize_text(payload.get("id"))
        for payload in selected_payloads
        if _normalize_text(payload.get("id"))
    }
    debug_candidates: list[dict[str, object]] = []
    for payload, scorecard in scored_candidates[:_GRAPH_NODE_RERANK_DEBUG_LIMIT]:
        debug_entry = dict(scorecard)
        debug_entry["selected_seed"] = _normalize_text(payload.get("id")) in selected_ids
        debug_candidates.append(debug_entry)
    return selected_payloads, tuple(debug_candidates)


def _filter_edge_payloads_for_seed_nodes(
    *,
    edge_payloads: Sequence[Mapping[str, object]],
    seed_node_ids: Sequence[str],
    subject_node_id: str,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Keep only edge-search hits that reinforce the reranked node seeds."""

    normalized_seed_ids = {
        normalized
        for normalized in (_normalize_text(item) for item in seed_node_ids)
        if normalized
    }
    selected: list[dict[str, object]] = []
    deferred: list[dict[str, object]] = []
    for payload in edge_payloads:
        source_node_id = _normalize_text(payload.get("source"))
        target_node_id = _normalize_text(payload.get("target"))
        touches_focus = bool(
            source_node_id in normalized_seed_ids
            or target_node_id in normalized_seed_ids
            or source_node_id == subject_node_id
            or target_node_id == subject_node_id
        )
        target_bucket = selected if touches_focus else deferred
        target_bucket.append(dict(payload))
    if selected:
        return tuple(selected), tuple(deferred)
    if deferred:
        # Keep one bounded fallback edge when node seeds did not help, so graph-only
        # prompts can still bootstrap a path/neighbor expansion instead of going empty.
        return (deferred[0],), tuple(deferred[1:])
    return (), ()


def _catalog_entry_locator(entry: object) -> tuple[str | None, str | None]:
    """Extract optional document locator fields from a remote catalog entry."""

    document_id = _normalize_text(getattr(entry, "document_id", None))
    uri = _normalize_text(getattr(entry, "uri", None))
    return document_id or None, uri or None


def _edge_content(edge_payload: Mapping[str, object]) -> str:
    parts: list[str] = [
        _normalize_text(edge_payload.get("source")),
        _normalize_text(edge_payload.get("type")),
        _normalize_text(edge_payload.get("target")),
        _normalize_text(edge_payload.get("status")),
    ]
    attributes = edge_payload.get("attributes")
    if isinstance(attributes, Mapping):
        for key, value in attributes.items():
            key_text = _normalize_text(key)
            if key_text:
                parts.append(key_text)
            if isinstance(value, str):
                value_text = _normalize_text(value)
                if value_text:
                    parts.append(value_text)
    return " ".join(part for part in parts if part)


def _node_metadata(document: TwinrGraphDocumentV1, node_payload: Mapping[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {
        "kind": _normalize_text(node_payload.get("type")),
        "summary": _normalize_text(node_payload.get("label")),
        "search_text": _node_search_text_for_catalog(document, node_payload),
        "owner_person_node_ids": list(_node_owner_person_ids(document, node_payload)),
        "updated_at": _normalize_text(document.updated_at),
        "created_at": _normalize_text(document.created_at),
        "selection_projection": dict(node_payload),
    }
    return {key: value for key, value in metadata.items() if value}


def _edge_metadata(document: TwinrGraphDocumentV1, edge_payload: Mapping[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {
        "kind": _normalize_text(edge_payload.get("type")),
        "summary": " ".join(
            part
            for part in (
                _normalize_text(edge_payload.get("source")),
                _normalize_text(edge_payload.get("type")),
                _normalize_text(edge_payload.get("target")),
            )
            if part
        ),
        "updated_at": _normalize_text(document.updated_at),
        "created_at": _normalize_text(document.created_at),
        "selection_projection": dict(edge_payload),
    }
    return {key: value for key, value in metadata.items() if value}


def _document_generation_id(document: TwinrGraphDocumentV1) -> str:
    payload = document.to_payload()
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"gen_{hashlib.sha1(serialized).hexdigest()[:16]}"


def _graph_index_name(remote_state: "LongTermRemoteStateStore") -> str:
    namespace = _normalize_text(getattr(remote_state, "namespace", None)) or "twinr_longterm_v1"
    return f"twinr_graph_{slugify_identifier(namespace, fallback='namespace')}"


def _topology_ref(*, generation_id: str, node_id: str) -> str:
    return f"{generation_id}:{node_id}"


class TwinrRemoteGraphState:
    """Persist graph topology remotely without using whole-graph snapshot blobs."""

    def __init__(self, remote_state: "LongTermRemoteStateStore | None") -> None:
        self.remote_state = remote_state
        from twinr.memory.longterm.storage.remote_catalog import LongTermRemoteCatalogStore

        self._catalog = LongTermRemoteCatalogStore(remote_state)

    def enabled(self) -> bool:
        return bool(self.remote_state is not None and self.remote_state.enabled)

    def probe_current_view(self) -> dict[str, object] | None:
        if not self.enabled():
            return None
        return self._current_view_summary(use_probe=True)

    def probe_current_view_for_readiness(self) -> dict[str, object] | None:
        if not self.enabled():
            return None
        return self._current_view_summary(use_probe=True, readiness_mode=True)

    def current_view_summary(self) -> dict[str, object] | None:
        if not self.enabled():
            return None
        return self._current_view_summary(use_probe=False)

    def ensure_seeded(self, *, document: TwinrGraphDocumentV1) -> bool:
        if not self.enabled():
            return False
        try:
            if self.validate_current_view():
                return False
        except _remote_unavailable_error_type():
            pass
        self.persist_document(document=document)
        return True

    def validate_current_view(self) -> bool:
        """Return whether the advertised current view is readably hydratable."""

        if not self.enabled():
            return False
        summary = self.current_view_summary()
        if not isinstance(summary, Mapping):
            return False
        self._load_current_nodes(summary=summary)
        return True

    def persist_document(self, *, document: TwinrGraphDocumentV1) -> dict[str, object] | None:
        if not self.enabled():
            return None
        remote_state = self._require_remote_state()
        generation_id = _document_generation_id(document)
        index_name = _graph_index_name(remote_state)
        try:
            # Repair writes must tolerate a broken advertised current view; that
            # is the exact state this persist path is asked to heal.
            current_summary = self.current_view_summary()
        except _remote_unavailable_error_type():
            current_summary = None
        if not isinstance(current_summary, Mapping) or current_summary.get("generation_id") != generation_id:
            self._store_topology(
                document=document,
                generation_id=generation_id,
                index_name=index_name,
            )

        topology_refs = {
            node.node_id: _topology_ref(generation_id=generation_id, node_id=node.node_id)
            for node in document.nodes
        }
        node_head = self._catalog.build_catalog_payload(
            snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND,
            item_payloads=(node.to_payload() for node in document.nodes),
            item_id_getter=lambda payload: payload.get("id"),
            metadata_builder=lambda payload, document=document: _node_metadata(document, payload),
            content_builder=lambda payload, document=document: _node_content_for_document(document, payload),
            skip_async_document_id_wait=True,
        )
        node_head.update(
            {
                "subject_node_id": document.subject_node_id,
                "graph_id": document.graph_id,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
                "generation_id": generation_id,
                "topology_index_name": index_name,
                "topology_refs": topology_refs,
            }
        )
        self._catalog.persist_catalog_payload(
            snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND,
            payload=node_head,
            skip_async_document_id_wait=True,
        )

        edge_head = self._catalog.build_catalog_payload(
            snapshot_kind=_GRAPH_EDGE_SNAPSHOT_KIND,
            item_payloads=(edge.to_payload() for edge in document.edges),
            item_id_getter=_edge_item_id_from_payload,
            metadata_builder=lambda payload, document=document: _edge_metadata(document, payload),
            content_builder=_edge_content,
            skip_async_document_id_wait=True,
        )
        edge_head.update(
            {
                "subject_node_id": document.subject_node_id,
                "graph_id": document.graph_id,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
                "generation_id": generation_id,
                "topology_index_name": index_name,
            }
        )
        self._catalog.persist_catalog_payload(
            snapshot_kind=_GRAPH_EDGE_SNAPSHOT_KIND,
            payload=edge_head,
            skip_async_document_id_wait=True,
        )
        return dict(node_head)

    def load_document(self) -> TwinrGraphDocumentV1 | None:
        if not self.enabled():
            return None
        summary = self.current_view_summary()
        if not isinstance(summary, Mapping):
            return None
        nodes = self._load_current_nodes(summary=summary)
        current_node_ids = {node.node_id for node in nodes}
        edges = self._load_current_edges(current_node_ids=current_node_ids)
        return TwinrGraphDocumentV1(
            subject_node_id=_normalize_text(summary.get("subject_node_id")),
            graph_id=_normalize_text(summary.get("graph_id")) or None,
            created_at=_normalize_text(summary.get("created_at")),
            updated_at=_normalize_text(summary.get("updated_at")),
            nodes=tuple(sorted(nodes, key=lambda item: item.node_id)),
            edges=tuple(sorted(edges, key=lambda item: (item.source_node_id, item.edge_type, item.target_node_id))),
            metadata={
                "kind": "personal_graph",
                "generation_id": _normalize_text(summary.get("generation_id")),
                "topology_index_name": _normalize_text(summary.get("topology_index_name")),
            },
        )

    def _load_current_nodes(self, *, summary: Mapping[str, object]) -> tuple[TwinrGraphNodeV1, ...]:
        topology_refs = self._topology_refs(summary)
        subject_node_id = _normalize_text(summary.get("subject_node_id"))
        if subject_node_id not in topology_refs:
            raise _remote_unavailable_error(
                f"Remote graph current view is missing the subject node {subject_node_id!r} in topology_refs."
            )
        node_entries = self._current_entries_by_item_id(snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND)
        nodes: list[TwinrGraphNodeV1] = []
        for node_id in sorted(topology_refs):
            entry = node_entries.get(node_id)
            payload = self._catalog_entry_projection_payload(entry)
            if payload is None:
                document_id, uri = _catalog_entry_locator(entry)
                payload = self._catalog.load_item_payload(
                    snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND,
                    item_id=node_id,
                    document_id=document_id,
                    uri=uri,
                )
            if payload is None:
                raise _remote_unavailable_error(
                    f"Remote graph current view is missing node payload {node_id!r}."
                )
            try:
                node = TwinrGraphNodeV1.from_payload(payload)
            except Exception as exc:
                raise _remote_unavailable_error(
                    f"Remote graph current view has an invalid node payload for {node_id!r}."
                ) from exc
            if node.node_id != node_id:
                raise _remote_unavailable_error(
                    f"Remote graph current view node payload {node_id!r} resolved to {node.node_id!r}."
                )
            nodes.append(node)
        return tuple(nodes)

    def _load_current_edges(self, *, current_node_ids: set[str]) -> tuple[TwinrGraphEdgeV1, ...]:
        edge_entries = tuple(self._current_entries_by_item_id(snapshot_kind=_GRAPH_EDGE_SNAPSHOT_KIND).values())
        edges: list[TwinrGraphEdgeV1] = []
        for entry in edge_entries:
            payload = self._catalog_entry_projection_payload(entry)
            item_id = _normalize_text(getattr(entry, "item_id", None)) or ""
            if payload is None:
                document_id, uri = _catalog_entry_locator(entry)
                payload = self._catalog.load_item_payload(
                    snapshot_kind=_GRAPH_EDGE_SNAPSHOT_KIND,
                    item_id=item_id,
                    document_id=document_id,
                    uri=uri,
                )
            if payload is None:
                LOGGER.warning("Skipping remote graph edge with unreadable payload: %s", item_id)
                continue
            try:
                edge = TwinrGraphEdgeV1.from_payload(payload)
            except Exception:
                LOGGER.warning("Ignoring invalid remote graph edge payload.", exc_info=True)
                continue
            if edge.source_node_id not in current_node_ids or edge.target_node_id not in current_node_ids:
                LOGGER.warning(
                    "Skipping remote graph edge outside the current node set: %s",
                    _edge_item_id(edge),
                )
                continue
            edges.append(edge)
        return tuple(edges)

    @staticmethod
    def _catalog_entry_projection_payload(entry: object) -> dict[str, object] | None:
        return _catalog_entry_projection_payload(entry)

    def _current_entries_by_item_id(self, *, snapshot_kind: str) -> dict[str, object]:
        payload = self._resolve_current_head_payload(
            snapshot_kind=snapshot_kind,
            use_probe=False,
        )
        if not isinstance(payload, Mapping):
            return {}
        try:
            entries = self._catalog.load_catalog_entries(
                snapshot_kind=snapshot_kind,
                payload=payload,
                bypass_cache=True,
            )
        except _remote_unavailable_error_type():
            return {}
        return {entry.item_id: entry for entry in entries}

    def _graph_item_id_from_payload(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> str | None:
        """Return the canonical graph item id for one node/edge payload."""

        if snapshot_kind == _GRAPH_NODE_SNAPSHOT_KIND:
            item_id = _normalize_text(payload.get("id"))
            return item_id or None
        if snapshot_kind == _GRAPH_EDGE_SNAPSHOT_KIND:
            return _edge_item_id_from_payload(payload)
        return None

    def _load_graph_item_payloads(
        self,
        *,
        snapshot_kind: str,
        item_ids: Sequence[str],
        current_entries_by_item_id: Mapping[str, object] | None = None,
    ) -> tuple[dict[str, object], ...]:
        """Hydrate graph items from current-head projections before exact item reads.

        Fresh readers can observe the authoritative graph current-head and its
        catalog segments slightly earlier than every exact item URI/document id.
        Query-first selection must therefore reuse the selection projections
        already carried on the current entries instead of immediately dropping
        to per-item document loads.
        """

        ordered_item_ids: list[str] = []
        loaded_by_item_id: dict[str, dict[str, object]] = {}
        missing_item_ids: list[str] = []
        entries_by_item_id = (
            dict(current_entries_by_item_id)
            if isinstance(current_entries_by_item_id, Mapping)
            else self._current_entries_by_item_id(snapshot_kind=snapshot_kind)
        )
        for raw_item_id in item_ids:
            item_id = _normalize_text(raw_item_id)
            if not item_id or item_id in ordered_item_ids:
                continue
            ordered_item_ids.append(item_id)
            cached_payload = self._catalog._cached_item_payload(  # pylint: disable=protected-access
                snapshot_kind=snapshot_kind,
                item_id=item_id,
            )
            if isinstance(cached_payload, Mapping):
                loaded_by_item_id[item_id] = dict(cached_payload)
                continue
            projection_payload = self._catalog_entry_projection_payload(entries_by_item_id.get(item_id))
            if isinstance(projection_payload, Mapping):
                payload_dict = dict(projection_payload)
                loaded_by_item_id[item_id] = payload_dict
                self._catalog._store_item_payload(  # pylint: disable=protected-access
                    snapshot_kind=snapshot_kind,
                    item_id=item_id,
                    payload=payload_dict,
                )
                continue
            missing_item_ids.append(item_id)
        if missing_item_ids:
            loaded_payloads = self._catalog.load_item_payloads(
                snapshot_kind=snapshot_kind,
                item_ids=tuple(missing_item_ids),
            )
            for payload in loaded_payloads:
                if not isinstance(payload, Mapping):
                    continue
                resolved_item_id = self._graph_item_id_from_payload(snapshot_kind=snapshot_kind, payload=payload)
                if resolved_item_id:
                    loaded_by_item_id[resolved_item_id] = dict(payload)
        return tuple(loaded_by_item_id[item_id] for item_id in ordered_item_ids if item_id in loaded_by_item_id)

    def query_current_path(
        self,
        *,
        source_node_id: str,
        target_node_id: str,
        edge_types: Sequence[str] | None = None,
    ) -> dict[str, object] | None:
        summary = self.current_view_summary()
        if not isinstance(summary, Mapping):
            return None
        return self._query_current_path_with_summary(
            summary=summary,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_types=edge_types,
        )

    def query_current_neighbors(
        self,
        *,
        node_id: str,
        edge_types: Sequence[str] | None = None,
        limit: int = 10,
    ) -> dict[str, object] | None:
        summary = self.current_view_summary()
        if not isinstance(summary, Mapping):
            return None
        return self._query_current_neighbors_with_summary(
            summary=summary,
            node_id=node_id,
            edge_types=edge_types,
            limit=limit,
        )

    def select_current_subgraph(
        self,
        *,
        query_text: str | None,
        candidate_limit: int = 6,
        neighbor_limit: int = 4,
        fallback_limit: int = 3,
    ) -> TwinrRemoteGraphSelection | None:
        """Select one query-first remote graph subgraph and explain how it was chosen."""

        clean_query = _normalize_text(query_text)
        if not clean_query or not self.enabled():
            return None
        summary = self.current_view_summary()
        if not isinstance(summary, Mapping):
            return None
        subject_node_id = _normalize_text(summary.get("subject_node_id"))
        if not subject_node_id:
            return None

        matched_node_payloads, node_fallback_used = self._search_or_top_graph_payloads(
            snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND,
            query_text=clean_query,
            search_limit=max(1, int(candidate_limit)),
            fallback_limit=max(0, int(fallback_limit)),
        )
        matched_edge_payloads, edge_fallback_used = self._search_or_top_graph_payloads(
            snapshot_kind=_GRAPH_EDGE_SNAPSHOT_KIND,
            query_text=clean_query,
            search_limit=max(1, int(candidate_limit)),
            fallback_limit=max(0, int(fallback_limit)),
        )

        current_node_entries_by_item_id = self._current_entries_by_item_id(snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND)
        local_node_entries: tuple["LongTermRemoteCatalogEntry", ...] = ()
        if current_node_entries_by_item_id:
            local_node_entries = self._catalog._local_search_catalog_entries(  # pylint: disable=protected-access
                snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND,
                entries=cast(tuple["LongTermRemoteCatalogEntry", ...], tuple(current_node_entries_by_item_id.values())),
                query_text=clean_query,
                limit=max(1, int(candidate_limit)),
            )
        merged_node_payloads, node_candidate_origins = _merge_graph_node_candidate_payloads(
            initial_payloads=matched_node_payloads,
            local_entries=local_node_entries,
        )
        merged_node_payloads, node_candidate_origins, owner_seed_promotions = _promote_owner_person_candidate_payloads(
            candidate_payloads=merged_node_payloads,
            current_entries_by_item_id=current_node_entries_by_item_id,
            origins_by_item_id=node_candidate_origins,
        )
        reranked_node_payloads, node_rerank_debug = _rerank_graph_node_payloads(
            query_text=clean_query,
            candidate_payloads=merged_node_payloads,
            current_entries_by_item_id=current_node_entries_by_item_id,
            candidate_limit=max(1, int(candidate_limit)),
            origins_by_item_id=node_candidate_origins,
        )
        matched_node_ids: set[str] = {
            _normalize_text(payload.get("id"))
            for payload in reranked_node_payloads
            if isinstance(payload, Mapping) and _normalize_text(payload.get("id"))
        }
        workflow_decision(
            msg="twinr_remote_graph_node_rerank",
            question="Which graph nodes should seed the query-first remote subgraph?",
            selected={
                "id": "local_graph_node_rerank",
                "summary": "Use bounded Twinr-side reranking over remote and catalog graph-node candidates.",
                "selected_seed_node_ids": sorted(matched_node_ids),
            },
            options=[
                {
                    "id": _normalize_text(candidate.get("node_id")) or f"candidate:{index}",
                    "summary": _normalize_text(candidate.get("label")) or _normalize_text(candidate.get("node_id")),
                    "score_components": dict(
                        cast(Mapping[str, object], candidate.get("score_components") or {})
                    ),
                    "constraints_violated": [] if bool(candidate.get("selected_seed")) else ["not_top_seed"],
                }
                for index, candidate in enumerate(node_rerank_debug, start=1)
            ],
            context={
                "query_text": clean_query,
                "candidate_count_before_rerank": len(merged_node_payloads),
                "candidate_count_after_rerank": len(reranked_node_payloads),
                "node_fallback_used": bool(node_fallback_used),
                "owner_seed_promotion_count": len(owner_seed_promotions),
            },
            confidence="high",
            guardrails=[
                "Keep the remote query-first path authoritative, but correct same-surname ambiguity locally.",
                "Prefer exact person-label or alias matches over remote popularity order when contact cues agree.",
                "Direct contact-method hits may promote their owning person node into the same seed pool.",
            ],
            kpi_impact_estimate={
                "extra_local_catalog_candidates": len(local_node_entries),
                "selected_seed_count": len(matched_node_ids),
            },
        )
        matched_edge_ids: set[str] = set()
        filtered_edge_payloads, deferred_edge_payloads = _filter_edge_payloads_for_seed_nodes(
            edge_payloads=matched_edge_payloads,
            seed_node_ids=tuple(sorted(matched_node_ids)),
            subject_node_id=subject_node_id,
        )
        for payload in filtered_edge_payloads:
            if not isinstance(payload, Mapping):
                continue
            source_node_id = _normalize_text(payload.get("source"))
            target_node_id = _normalize_text(payload.get("target"))
            edge_type = _normalize_text(payload.get("type"))
            if source_node_id:
                matched_node_ids.add(source_node_id)
            if target_node_id:
                matched_node_ids.add(target_node_id)
            if source_node_id and target_node_id and edge_type:
                matched_edge_ids.add(f"{source_node_id}|{edge_type}|{target_node_id}")
        if not matched_node_ids and subject_node_id:
            matched_node_ids.add(subject_node_id)

        selected_node_ids: set[str] = {subject_node_id, *matched_node_ids}
        expanded_path_nodes: set[str] = set()
        path_hints: list[dict[str, object]] = []
        neighbor_hints: list[dict[str, object]] = []
        path_query_events: list[dict[str, object]] = []
        neighbor_query_events: list[dict[str, object]] = []
        for node_id in sorted(item for item in matched_node_ids if item and item != subject_node_id):
            path_payload, path_query_event = self._query_current_path_with_summary_resilient(
                summary=summary,
                source_node_id=subject_node_id,
                target_node_id=node_id,
            )
            if isinstance(path_query_event, Mapping):
                path_query_events.append(dict(path_query_event))
            logical_path = path_payload.get("logical_path") if isinstance(path_payload, Mapping) else None
            if isinstance(logical_path, list) and logical_path:
                normalized_path = [
                    normalized for normalized in (_normalize_text(item) for item in logical_path) if normalized
                ]
                if normalized_path:
                    expanded_path_nodes.update(normalized_path)
                    path_hints.append(
                        {
                            "target_node_id": node_id,
                            "logical_path": normalized_path,
                        }
                    )
            neighbor_payload, neighbor_query_event = self._query_current_neighbors_with_summary_resilient(
                summary=summary,
                node_id=node_id,
                limit=max(1, int(neighbor_limit)),
            )
            if isinstance(neighbor_query_event, Mapping):
                neighbor_query_events.append(dict(neighbor_query_event))
            neighbors = neighbor_payload.get("neighbors") if isinstance(neighbor_payload, Mapping) else None
            neighbor_count = 0
            if isinstance(neighbors, list):
                for item in neighbors:
                    if not isinstance(item, Mapping):
                        continue
                    logical_neighbor_id = _normalize_text(item.get("logical_node_id"))
                    edge_type = _normalize_text(item.get("edge_type"))
                    if logical_neighbor_id:
                        selected_node_ids.add(logical_neighbor_id)
                    if logical_neighbor_id and edge_type:
                        matched_edge_ids.add(f"{node_id}|{edge_type}|{logical_neighbor_id}")
                    neighbor_count += 1
            neighbor_hints.append({"node_id": node_id, "neighbor_count": neighbor_count})
        selected_node_ids.update(expanded_path_nodes)

        node_payloads = self._load_graph_item_payloads(
            snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND,
            item_ids=tuple(sorted(selected_node_ids)),
            current_entries_by_item_id=current_node_entries_by_item_id,
        )
        if not node_payloads:
            return None
        nodes: list[TwinrGraphNodeV1] = []
        for payload in node_payloads:
            try:
                nodes.append(TwinrGraphNodeV1.from_payload(payload))
            except Exception:
                LOGGER.warning("Ignoring invalid remote graph node payload during query-first selection.", exc_info=True)
        if not nodes:
            return None
        node_ids = {node.node_id for node in nodes}

        edge_payloads_by_id: dict[str, dict[str, object]] = {}
        for payload in filtered_edge_payloads:
            if not isinstance(payload, Mapping):
                continue
            edge_item_id = _edge_item_id_from_payload(payload)
            if edge_item_id:
                edge_payloads_by_id[edge_item_id] = dict(payload)
        current_edge_entries_by_item_id = self._current_entries_by_item_id(snapshot_kind=_GRAPH_EDGE_SNAPSHOT_KIND)
        missing_edge_ids = tuple(sorted(edge_id for edge_id in matched_edge_ids if edge_id not in edge_payloads_by_id))
        if missing_edge_ids:
            loaded_edge_payloads = self._load_graph_item_payloads(
                snapshot_kind=_GRAPH_EDGE_SNAPSHOT_KIND,
                item_ids=missing_edge_ids,
                current_entries_by_item_id=current_edge_entries_by_item_id,
            )
            for payload in loaded_edge_payloads:
                edge_item_id = _edge_item_id_from_payload(payload)
                if edge_item_id:
                    edge_payloads_by_id[edge_item_id] = dict(payload)

        edges: list[TwinrGraphEdgeV1] = []
        for edge_item_id, payload in edge_payloads_by_id.items():
            try:
                edge = TwinrGraphEdgeV1.from_payload(payload)
            except Exception:
                LOGGER.warning("Ignoring invalid remote graph edge payload during query-first selection.", exc_info=True)
                continue
            if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
                continue
            if _edge_item_id(edge) != edge_item_id:
                continue
            edges.append(edge)

        query_plan: dict[str, object] = {
            "schema": "twinr_graph_query_plan_v1",
            "mode": "remote_query_first_subgraph",
            "query_text": clean_query,
            "current_view": {
                "generation_id": _normalize_text(summary.get("generation_id")),
                "subject_node_id": subject_node_id,
                "graph_id": _normalize_text(summary.get("graph_id")),
            },
            "access_path": [
                "catalog_current_head",
                "topk_scope_query",
                "retrieve_batch",
                "graph_path_query",
                "graph_neighbors_query",
            ],
            "fallbacks": {
                "node_catalog_top": bool(node_fallback_used),
                "edge_catalog_top": bool(edge_fallback_used),
            },
            "graph_node_rerank": {
                "candidate_count_before_rerank": len(merged_node_payloads),
                "candidate_count_after_rerank": len(reranked_node_payloads),
                "candidates": list(node_rerank_debug),
                "owner_seed_promotions": list(owner_seed_promotions),
            },
            "edge_candidate_filter": {
                "selected_edge_input_count": len(filtered_edge_payloads),
                "deferred_edge_input_count": len(deferred_edge_payloads),
            },
            "matched_node_ids": sorted(matched_node_ids),
            "matched_edge_ids": sorted(matched_edge_ids),
            "selected_node_ids": sorted(node_ids),
            "selected_edge_ids": sorted(_edge_item_id(edge) for edge in edges),
            "path_hints": path_hints[: max(1, int(candidate_limit))],
            "neighbor_hints": neighbor_hints[: max(1, int(candidate_limit))],
            "path_query_events": path_query_events[: max(1, int(candidate_limit))],
            "neighbor_query_events": neighbor_query_events[: max(1, int(candidate_limit))],
        }
        workflow_event(
            kind="retrieval",
            msg="twinr_remote_graph_query_plan",
            details=query_plan,
        )
        return TwinrRemoteGraphSelection(
            document=TwinrGraphDocumentV1(
                subject_node_id=subject_node_id,
                graph_id=_normalize_text(summary.get("graph_id")) or None,
                created_at=_normalize_text(summary.get("created_at")),
                updated_at=_normalize_text(summary.get("updated_at")),
                nodes=tuple(sorted(nodes, key=lambda item: item.node_id)),
                edges=tuple(sorted(edges, key=lambda item: (item.source_node_id, item.edge_type, item.target_node_id))),
                metadata={
                    "kind": "personal_graph",
                    "selection_mode": "remote_query_first_subgraph",
                    "generation_id": _normalize_text(summary.get("generation_id")),
                },
            ),
            query_plan=query_plan,
        )

    def _search_or_top_graph_payloads(
        self,
        *,
        snapshot_kind: str,
        query_text: str,
        search_limit: int,
        fallback_limit: int,
    ) -> tuple[tuple[dict[str, object], ...], bool]:
        """Prefer current-scope search, then bounded current-catalog fallback payloads."""

        payloads = self._catalog.search_current_item_payloads(
            snapshot_kind=snapshot_kind,
            query_text=query_text,
            limit=max(1, int(search_limit)),
            allow_catalog_fallback=True,
        )
        if payloads:
            return tuple(dict(payload) for payload in payloads if isinstance(payload, Mapping)), False
        if fallback_limit <= 0:
            return (), False
        current_entries_by_item_id = self._current_entries_by_item_id(snapshot_kind=snapshot_kind)
        if current_entries_by_item_id:
            current_entries = cast(
                tuple["LongTermRemoteCatalogEntry", ...],
                tuple(current_entries_by_item_id.values()),
            )
            selected_entries = self._catalog._local_search_catalog_entries(  # pylint: disable=protected-access
                snapshot_kind=snapshot_kind,
                entries=current_entries,
                query_text=query_text,
                limit=max(1, int(fallback_limit)),
            )
            if not selected_entries:
                selected_entries = self._catalog.top_catalog_entries(
                    snapshot_kind=snapshot_kind,
                    limit=max(1, int(fallback_limit)),
                )
            if selected_entries:
                loaded = self._load_graph_item_payloads(
                    snapshot_kind=snapshot_kind,
                    item_ids=tuple(entry.item_id for entry in selected_entries),
                    current_entries_by_item_id=current_entries_by_item_id,
                )
                if loaded:
                    return loaded, True
        entries = self._catalog.top_catalog_entries(
            snapshot_kind=snapshot_kind,
            limit=max(1, int(fallback_limit)),
        )
        if not entries:
            return (), False
        loaded = self._load_graph_item_payloads(
            snapshot_kind=snapshot_kind,
            item_ids=tuple(entry.item_id for entry in entries),
            current_entries_by_item_id=current_entries_by_item_id,
        )
        return tuple(dict(payload) for payload in loaded if isinstance(payload, Mapping)), True

    def _query_current_path_with_summary(
        self,
        *,
        summary: Mapping[str, object],
        source_node_id: str,
        target_node_id: str,
        edge_types: Sequence[str] | None = None,
    ) -> dict[str, object] | None:
        payload, _event = self._run_graph_query_with_retry(
            summary=summary,
            operation="graph_path_query",
            request_path="/v1/external/graph/path",
            request_payload_kind="graph_path_query",
            item_id=f"{source_node_id}->{target_node_id}",
            invoke=self._graph_path_request(
                summary=summary,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_types=edge_types,
            ),
            annotate=lambda payload: self._annotate_graph_path_payload(summary=summary, payload=payload),
            swallow_retryable_failure=False,
        )
        return payload

    def _query_current_path_with_summary_resilient(
        self,
        *,
        summary: Mapping[str, object],
        source_node_id: str,
        target_node_id: str,
        edge_types: Sequence[str] | None = None,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        return self._run_graph_query_with_retry(
            summary=summary,
            operation="graph_path_query",
            request_path="/v1/external/graph/path",
            request_payload_kind="graph_path_query",
            item_id=f"{source_node_id}->{target_node_id}",
            invoke=self._graph_path_request(
                summary=summary,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_types=edge_types,
            ),
            annotate=lambda payload: self._annotate_graph_path_payload(summary=summary, payload=payload),
            swallow_retryable_failure=True,
        )

    def _graph_path_request(
        self,
        *,
        summary: Mapping[str, object],
        source_node_id: str,
        target_node_id: str,
        edge_types: Sequence[str] | None = None,
    ):
        topology_refs = self._topology_refs(summary)
        source_ref = topology_refs.get(source_node_id)
        target_ref = topology_refs.get(target_node_id)
        if not source_ref or not target_ref:
            return None
        remote_state = self._require_remote_state()
        read_client = getattr(remote_state, "read_client", None)
        if read_client is None:
            return None
        return lambda: getattr(read_client, "graph_path")(
            ChonkyDBGraphPathRequest(
                index_name=_normalize_text(summary.get("topology_index_name")) or None,
                source=source_ref,
                target=target_ref,
                edge_types=tuple(edge_types) if edge_types is not None else None,
            )
        )

    def _annotate_graph_path_payload(
        self,
        *,
        summary: Mapping[str, object],
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        topology_refs = self._topology_refs(summary)
        annotated = dict(payload)
        raw_path = payload.get("path")
        if isinstance(raw_path, list):
            inverse = {value: key for key, value in topology_refs.items()}
            annotated["logical_path"] = [inverse.get(_normalize_text(item), _normalize_text(item)) for item in raw_path]
        return annotated

    def _query_current_neighbors_with_summary(
        self,
        *,
        summary: Mapping[str, object],
        node_id: str,
        edge_types: Sequence[str] | None = None,
        limit: int = 10,
    ) -> dict[str, object] | None:
        payload, _event = self._run_graph_query_with_retry(
            summary=summary,
            operation="graph_neighbors_query",
            request_path="/v1/external/graph/neighbors",
            request_payload_kind="graph_neighbors_query",
            item_id=node_id,
            invoke=self._graph_neighbors_request(
                summary=summary,
                node_id=node_id,
                edge_types=edge_types,
                limit=limit,
            ),
            annotate=lambda payload: self._annotate_graph_neighbors_payload(summary=summary, payload=payload),
            swallow_retryable_failure=False,
        )
        return payload

    def _query_current_neighbors_with_summary_resilient(
        self,
        *,
        summary: Mapping[str, object],
        node_id: str,
        edge_types: Sequence[str] | None = None,
        limit: int = 10,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        return self._run_graph_query_with_retry(
            summary=summary,
            operation="graph_neighbors_query",
            request_path="/v1/external/graph/neighbors",
            request_payload_kind="graph_neighbors_query",
            item_id=node_id,
            invoke=self._graph_neighbors_request(
                summary=summary,
                node_id=node_id,
                edge_types=edge_types,
                limit=limit,
            ),
            annotate=lambda payload: self._annotate_graph_neighbors_payload(summary=summary, payload=payload),
            swallow_retryable_failure=True,
        )

    def _graph_neighbors_request(
        self,
        *,
        summary: Mapping[str, object],
        node_id: str,
        edge_types: Sequence[str] | None = None,
        limit: int = 10,
    ):
        topology_refs = self._topology_refs(summary)
        node_ref = topology_refs.get(node_id)
        if not node_ref:
            return None
        remote_state = self._require_remote_state()
        read_client = getattr(remote_state, "read_client", None)
        if read_client is None:
            return None
        return lambda: getattr(read_client, "graph_neighbors")(
            ChonkyDBGraphNeighborsRequest(
                index_name=_normalize_text(summary.get("topology_index_name")) or None,
                label_or_id=node_ref,
                edge_types=tuple(edge_types) if edge_types is not None else None,
                with_edges=True,
                limit=max(1, int(limit)),
            )
        )

    def _annotate_graph_neighbors_payload(
        self,
        *,
        summary: Mapping[str, object],
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        topology_refs = self._topology_refs(summary)
        inverse = {value: key for key, value in topology_refs.items()}
        annotated = dict(payload)
        neighbors = payload.get("neighbors")
        if isinstance(neighbors, list):
            normalized_neighbors = []
            for item in neighbors:
                if not isinstance(item, Mapping):
                    normalized_neighbors.append(item)
                    continue
                normalized = dict(item)
                label = _normalize_text(item.get("label"))
                if label:
                    normalized["logical_node_id"] = inverse.get(label, label)
                normalized_neighbors.append(normalized)
            annotated["neighbors"] = normalized_neighbors
        return annotated

    def _graph_query_retry_attempts(self) -> int:
        remote_state = self.remote_state
        config = getattr(remote_state, "config", None)
        try:
            configured = int(
                getattr(
                    config,
                    "long_term_memory_remote_retry_attempts",
                    _MAX_GRAPH_QUERY_RETRY_ATTEMPTS,
                )
            )
        except (TypeError, ValueError):
            configured = _MAX_GRAPH_QUERY_RETRY_ATTEMPTS
        return max(1, min(_MAX_GRAPH_QUERY_RETRY_ATTEMPTS, configured))

    def _graph_query_retry_backoff_s(self) -> float:
        remote_state = self.remote_state
        config = getattr(remote_state, "config", None)
        try:
            configured = float(
                getattr(
                    config,
                    "long_term_memory_remote_retry_backoff_s",
                    _MAX_GRAPH_QUERY_RETRY_BACKOFF_S,
                )
            )
        except (TypeError, ValueError):
            configured = _MAX_GRAPH_QUERY_RETRY_BACKOFF_S
        return min(_MAX_GRAPH_QUERY_RETRY_BACKOFF_S, max(0.0, configured))

    def _graph_store_retry_attempts(self) -> int:
        remote_state = self.remote_state
        config = getattr(remote_state, "config", None)
        try:
            configured = int(
                getattr(
                    config,
                    "long_term_memory_remote_retry_attempts",
                    _MAX_GRAPH_STORE_RETRY_ATTEMPTS,
                )
            )
        except (TypeError, ValueError):
            configured = _MAX_GRAPH_STORE_RETRY_ATTEMPTS
        return max(1, min(_MAX_GRAPH_STORE_RETRY_ATTEMPTS, configured))

    def _graph_store_retry_backoff_s(self) -> float:
        remote_state = self.remote_state
        config = getattr(remote_state, "config", None)
        try:
            configured = float(
                getattr(
                    config,
                    "long_term_memory_remote_retry_backoff_s",
                    1.0,
                )
            )
        except (TypeError, ValueError):
            configured = 1.0
        return min(30.0, max(0.0, configured))

    def _graph_query_context(
        self,
        *,
        remote_state: "LongTermRemoteStateStore",
        operation: str,
        request_path: str,
        request_payload_kind: str,
        item_id: str | None,
        attempt_index: int,
        attempt_count: int,
    ):
        from twinr.memory.longterm.storage.remote_read_diagnostics import (
            LongTermRemoteReadContext,
        )
        retry_attempts = self._graph_query_retry_attempts()
        return LongTermRemoteReadContext(
            snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND,
            operation=operation,
            request_method="POST",
            request_payload_kind=request_payload_kind,
            item_id=item_id,
            request_path=request_path,
            namespace=_normalize_text(getattr(remote_state, "namespace", None)),
            attempt_index=attempt_index,
            attempt_count=attempt_count,
            retry_attempts_configured=retry_attempts,
            retry_backoff_s=self._graph_query_retry_backoff_s(),
            retry_mode="bounded_graph_query_retry" if retry_attempts > 1 else "single_attempt",
            access_classification="topk_scope_query",
        )

    def _run_graph_query_with_retry(
        self,
        *,
        summary: Mapping[str, object],
        operation: str,
        request_path: str,
        request_payload_kind: str,
        item_id: str | None,
        invoke,
        annotate,
        swallow_retryable_failure: bool,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        from twinr.memory.longterm.storage.remote_read_diagnostics import (
            record_remote_read_diagnostic,
        )
        from twinr.memory.longterm.storage.remote_read_observability import record_remote_read_observation
        from twinr.memory.longterm.storage._remote_retry import (
            remote_read_retry_delay_s,
            should_retry_remote_read_error,
        )

        del summary
        if invoke is None:
            return None, None
        remote_state = self._require_remote_state()
        attempt_count = self._graph_query_retry_attempts()
        retry_backoff_s = self._graph_query_retry_backoff_s()
        for attempt_index in range(1, attempt_count + 1):
            context = self._graph_query_context(
                remote_state=remote_state,
                operation=operation,
                request_path=request_path,
                request_payload_kind=request_payload_kind,
                item_id=item_id,
                attempt_index=attempt_index,
                attempt_count=attempt_count,
            )
            started_monotonic = time.monotonic()
            try:
                payload = invoke()
            except Exception as exc:
                retryable = should_retry_remote_read_error(exc)
                if retryable and attempt_index < attempt_count:
                    delay_s = min(
                        _MAX_GRAPH_QUERY_RETRY_BACKOFF_S,
                        remote_read_retry_delay_s(
                            exc,
                            default_backoff_s=retry_backoff_s,
                            attempt_index=attempt_index - 1,
                        ),
                    )
                    workflow_event(
                        kind="retrieval",
                        msg="twinr_remote_graph_query_retry",
                        details={
                            "operation": operation,
                            "item_id": item_id,
                            "attempt_index": attempt_index,
                            "attempt_count": attempt_count,
                            "retry_delay_s": delay_s,
                            "error_type": type(exc).__name__,
                        },
                    )
                    if delay_s > 0.0:
                        time.sleep(delay_s)
                    continue
                if retryable and swallow_retryable_failure:
                    record_remote_read_diagnostic(
                        remote_state=remote_state,
                        context=context,
                        exc=exc,
                        started_monotonic=started_monotonic,
                        outcome="degraded",
                    )
                    LOGGER.warning(
                        "Remote graph %s degraded after bounded retry for %s.",
                        operation,
                        item_id or "selection",
                        exc_info=True,
                    )
                    return None, self._graph_query_event_note(
                        operation=operation,
                        item_id=item_id,
                        status="degraded",
                        retry_count=max(0, attempt_index - 1),
                        exc=exc,
                    )
                record_remote_read_diagnostic(
                    remote_state=remote_state,
                    context=context,
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="failed",
                )
                raise
            if not isinstance(payload, Mapping):
                raise TypeError(f"{operation} returned payload type {type(payload).__name__}.")
            record_remote_read_observation(
                remote_state=remote_state,
                context=context,
                latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
                outcome="ok",
                classification="ok",
            )
            annotated = annotate(payload)
            retry_note = None
            if attempt_index > 1:
                retry_note = self._graph_query_event_note(
                    operation=operation,
                    item_id=item_id,
                    status="retried_ok",
                    retry_count=attempt_index - 1,
                    exc=None,
                )
            return dict(annotated) if isinstance(annotated, Mapping) else None, retry_note
        return None, None

    @staticmethod
    def _graph_query_event_note(
        *,
        operation: str,
        item_id: str | None,
        status: str,
        retry_count: int,
        exc: Exception | None,
    ) -> dict[str, object]:
        from twinr.memory.chonkydb.client import ChonkyDBError

        status_code = None
        error_type = None
        if exc is not None:
            error_type = type(exc).__name__
            if isinstance(exc, ChonkyDBError):
                try:
                    status_code = int(exc.status_code) if exc.status_code is not None else None
                except (TypeError, ValueError):
                    status_code = None
        payload = {
            "operation": operation,
            "item_id": item_id,
            "status": status,
            "retry_count": max(0, int(retry_count)),
            "error_type": error_type,
            "status_code": status_code,
        }
        return {key: value for key, value in payload.items() if value is not None}

    def _store_topology(
        self,
        *,
        document: TwinrGraphDocumentV1,
        generation_id: str,
        index_name: str,
    ) -> None:
        from twinr.memory.longterm.storage.remote_read_diagnostics import (
            LongTermRemoteWriteContext,
            record_remote_write_diagnostic,
        )
        from twinr.memory.longterm.storage._remote_retry import (
            remote_write_retry_delay_s,
            raise_if_remote_operation_cancelled,
            retryable_remote_write_attempts,
            sleep_with_remote_operation_abort,
            should_retry_remote_write_error,
        )
        from twinr.memory.longterm.storage.remote_read_observability import record_remote_write_observation

        remote_state = self._require_remote_state()
        write_client = getattr(remote_state, "write_client", None)
        if write_client is None:
            raise _remote_unavailable_error("Remote graph topology write client is unavailable.")
        timeout_s = self._graph_store_timeout_seconds()
        retry_attempts = self._graph_store_retry_attempts()
        retry_backoff_s = self._graph_store_retry_backoff_s()
        nodes = tuple(
            ChonkyDBGraphStoreManyNode(label=_topology_ref(generation_id=generation_id, node_id=node.node_id))
            for node in sorted(document.nodes, key=lambda item: item.node_id)
        )
        edges = tuple(
            ChonkyDBGraphStoreManyEdge(
                source_label=_topology_ref(generation_id=generation_id, node_id=edge.source_node_id),
                target_label=_topology_ref(generation_id=generation_id, node_id=edge.target_node_id),
                edge_type=edge.edge_type,
            )
            for edge in sorted(document.edges, key=lambda item: (item.source_node_id, item.edge_type, item.target_node_id))
        )
        request = ChonkyDBGraphStoreManyRequest(
            index_name=index_name,
            nodes=nodes,
            edges=edges,
            assume_new=False,
            timeout_seconds=timeout_s,
        )
        request_payload = request.to_payload()
        request_bytes = len(
            json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        started_monotonic = time.monotonic()
        last_error: Exception | None = None
        attempt_index = 0
        while attempt_index < retry_attempts:
            context = LongTermRemoteWriteContext(
                snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND,
                operation="graph_store_many",
                request_method="POST",
                request_payload_kind="graph_store_many_request",
                request_path="/v1/external/graph/store_many",
                timeout_s=timeout_s,
                namespace=_normalize_text(getattr(remote_state, "namespace", None)),
                access_classification="graph_topology_write",
                attempt_count=attempt_index + 1,
                retry_attempts_configured=retry_attempts,
                retry_backoff_s=retry_backoff_s,
                retry_mode="bounded_graph_topology_retry" if retry_attempts > 1 else "single_attempt",
                request_item_count=len(nodes) + len(edges),
                request_bytes=request_bytes,
            )
            try:
                candidate = getattr(write_client, "graph_store_many")(request)
                if isinstance(candidate, Mapping) and candidate.get("success") is False:
                    detail = _normalize_text(candidate.get("detail")) or _normalize_text(candidate.get("error"))
                    error_type = _normalize_text(candidate.get("error_type"))
                    response_json: dict[str, object] = {
                        key: value
                        for key, value in (
                            ("detail", detail or None),
                            ("error", _normalize_text(candidate.get("error")) or None),
                            ("error_type", error_type or None),
                        )
                        if value is not None
                    }
                    retryable_status = 503 if error_type == "ServiceUnavailable" else None
                    transient_markers = ("graph_initializing", "graph_warmup_failed", "upstream unavailable")
                    if retryable_status is None and detail and any(marker in detail.lower() for marker in transient_markers):
                        retryable_status = 503
                    raise ChonkyDBError(
                        detail or "graph_store_many failed",
                        status_code=retryable_status,
                        response_json=response_json or None,
                    )
                last_error = None
                break
            except Exception as exc:
                last_error = exc if isinstance(exc, Exception) else Exception(str(exc))
                resolved_retry_attempts = retryable_remote_write_attempts(retry_attempts, exc=exc)
                if should_retry_remote_write_error(exc) and attempt_index + 1 < resolved_retry_attempts:
                    raise_if_remote_operation_cancelled(operation="Remote graph topology write")
                    delay_s = remote_write_retry_delay_s(
                        exc,
                        default_backoff_s=retry_backoff_s,
                        attempt_index=attempt_index,
                    )
                    if delay_s > 0.0:
                        sleep_with_remote_operation_abort(
                            delay_s,
                            operation="Remote graph topology write retry",
                        )
                    attempt_index += 1
                    continue
                record_remote_write_diagnostic(
                    remote_state=remote_state,
                    context=context,
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="failed",
                )
                raise _remote_unavailable_error(
                    f"Failed to persist graph topology generation {generation_id!r} to remote ChonkyDB."
                ) from exc
        if last_error is not None:
            raise _remote_unavailable_error(
                f"Failed to persist graph topology generation {generation_id!r} to remote ChonkyDB."
            ) from last_error
        record_remote_write_observation(
            remote_state=remote_state,
            context=LongTermRemoteWriteContext(
                snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND,
                operation="graph_store_many",
                request_method="POST",
                request_payload_kind="graph_store_many_request",
                request_path="/v1/external/graph/store_many",
                timeout_s=timeout_s,
                namespace=_normalize_text(getattr(remote_state, "namespace", None)),
                access_classification="graph_topology_write",
                attempt_count=attempt_index + 1,
                retry_attempts_configured=retry_attempts,
                retry_backoff_s=retry_backoff_s,
                retry_mode="bounded_graph_topology_retry" if retry_attempts > 1 else "single_attempt",
                request_item_count=len(nodes) + len(edges),
                request_bytes=request_bytes,
            ),
            latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
            outcome="ok",
            classification="ok",
        )

    def _current_view_summary(
        self,
        *,
        use_probe: bool,
        readiness_mode: bool = False,
    ) -> dict[str, object] | None:
        node_head = self._resolve_current_head_payload(
            snapshot_kind=_GRAPH_NODE_SNAPSHOT_KIND,
            use_probe=use_probe,
            readiness_mode=readiness_mode,
        )
        # The node head is the authoritative gate for whether a remote current
        # view exists at all. On a fresh namespace, reading the edge head after
        # an explicit node-head miss only adds one more remote round trip
        # without contributing any additional state.
        if not isinstance(node_head, Mapping):
            return None
        edge_head = self._resolve_current_head_payload(
            snapshot_kind=_GRAPH_EDGE_SNAPSHOT_KIND,
            use_probe=use_probe,
            readiness_mode=readiness_mode,
        )
        if not isinstance(edge_head, Mapping):
            return None
        generation_id = _normalize_text(node_head.get("generation_id"))
        edge_generation_id = _normalize_text(edge_head.get("generation_id"))
        if not generation_id or generation_id != edge_generation_id:
            raise _remote_unavailable_error("Remote graph current-view heads are missing or generation-mismatched.")
        topology_index_name = _normalize_text(node_head.get("topology_index_name")) or _normalize_text(
            edge_head.get("topology_index_name")
        )
        subject_node_id = _normalize_text(node_head.get("subject_node_id")) or _normalize_text(
            edge_head.get("subject_node_id")
        )
        graph_id = _normalize_text(node_head.get("graph_id")) or _normalize_text(edge_head.get("graph_id"))
        created_at = _normalize_text(node_head.get("created_at")) or _normalize_text(edge_head.get("created_at"))
        updated_at = _normalize_text(node_head.get("updated_at")) or _normalize_text(edge_head.get("updated_at"))
        topology_refs = self._topology_refs(node_head)
        if not topology_index_name or not subject_node_id or not graph_id or not created_at or not updated_at or not topology_refs:
            raise _remote_unavailable_error("Remote graph current-view heads are incomplete.")
        return {
            "generation_id": generation_id,
            "topology_index_name": topology_index_name,
            "subject_node_id": subject_node_id,
            "graph_id": graph_id,
            "created_at": created_at,
            "updated_at": updated_at,
            "topology_refs": topology_refs,
        }

    def _current_head_payload_complete(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> bool:
        generation_id = _normalize_text(payload.get("generation_id"))
        topology_index_name = _normalize_text(payload.get("topology_index_name"))
        subject_node_id = _normalize_text(payload.get("subject_node_id"))
        graph_id = _normalize_text(payload.get("graph_id"))
        created_at = _normalize_text(payload.get("created_at"))
        updated_at = _normalize_text(payload.get("updated_at"))
        if not generation_id or not topology_index_name or not subject_node_id or not graph_id or not created_at or not updated_at:
            return False
        if snapshot_kind == _GRAPH_NODE_SNAPSHOT_KIND and not self._topology_refs(payload):
            return False
        return True

    def _resolve_current_head_payload(
        self,
        *,
        snapshot_kind: str,
        use_probe: bool,
        readiness_mode: bool = False,
    ) -> dict[str, object] | None:
        """Resolve one graph head read-only from direct or compatible snapshot state.

        Graph readiness/bootstrap only needs the node/edge catalog head
        metadata. Falling back to the generic `load_catalog_payload()` helper
        would promote compatible snapshot heads into `catalog/current`, which
        turns a health check into a blocking write path. Reuse the direct head
        when it exists, otherwise read the snapshot head in a strictly read-
        only way and trust it when it already matches the graph catalog
        contract.
        """

        if not use_probe and self._direct_current_head_missing_fast_fail(snapshot_kind=snapshot_kind):
            compatible_head = self._probe_compatible_snapshot_head_payload(snapshot_kind=snapshot_kind)
            if isinstance(compatible_head, Mapping):
                compatible_payload = dict(compatible_head)
                if self._current_head_payload_complete(snapshot_kind=snapshot_kind, payload=compatible_payload):
                    return compatible_payload
            return None
        direct_head = self._read_direct_current_head_payload(
            snapshot_kind=snapshot_kind,
            use_probe=use_probe,
            readiness_mode=readiness_mode,
        )
        if isinstance(direct_head, Mapping):
            direct_payload = dict(direct_head)
            if self._current_head_payload_complete(snapshot_kind=snapshot_kind, payload=direct_payload):
                return direct_payload
        remote_state = self.remote_state
        if remote_state is None:
            return None
        compatible_head = self._probe_compatible_snapshot_head_payload(snapshot_kind=snapshot_kind)
        if isinstance(compatible_head, Mapping):
            compatible_payload = dict(compatible_head)
            if self._current_head_payload_complete(snapshot_kind=snapshot_kind, payload=compatible_payload):
                return compatible_payload
        probe_loader = getattr(remote_state, "probe_snapshot_load", None)
        if use_probe and callable(probe_loader):
            probe_kwargs = {
                "snapshot_kind": snapshot_kind,
                # Graph node/edge current heads mutate behind stable URIs.
                # Reusing cached exact document ids can pin fresh readers to a
                # superseded head and reintroduce the slow compatibility read
                # path that blocked `--openai-prompt` startup.
                "prefer_cached_document_id": False,
                "prefer_metadata_only": True,
            }
            parameters: Mapping[str, inspect.Parameter]
            try:
                parameters = inspect.signature(probe_loader).parameters
            except (TypeError, ValueError):
                parameters = cast("Mapping[str, inspect.Parameter]", {})
            if "fast_fail" in parameters:
                probe_kwargs["fast_fail"] = use_probe
            try:
                probe = probe_loader(**probe_kwargs)
            except Exception as exc:
                workflow_event(
                    kind="retrieval",
                    msg="twinr_remote_graph_current_head_probe_failed",
                    details={
                        "snapshot_kind": snapshot_kind,
                        "use_probe": bool(use_probe),
                        "readiness_mode": bool(readiness_mode),
                        "error_type": type(exc).__name__,
                        "error": _normalize_text(exc),
                    },
                )
                raise
            payload = getattr(probe, "payload", None)
        else:
            load_snapshot = getattr(remote_state, "load_snapshot")
            parameters = cast("Mapping[str, inspect.Parameter]", {})
            try:
                parameters = inspect.signature(load_snapshot).parameters
            except (TypeError, ValueError):
                parameters = cast("Mapping[str, inspect.Parameter]", {})
            load_kwargs: dict[str, object] = {"snapshot_kind": snapshot_kind}
            if "prefer_cached_document_id" in parameters:
                load_kwargs["prefer_cached_document_id"] = False
            try:
                payload = load_snapshot(**load_kwargs)
            except Exception as exc:
                workflow_event(
                    kind="retrieval",
                    msg="twinr_remote_graph_current_head_snapshot_load_failed",
                    details={
                        "snapshot_kind": snapshot_kind,
                        "use_probe": bool(use_probe),
                        "readiness_mode": bool(readiness_mode),
                        "error_type": type(exc).__name__,
                        "error": _normalize_text(exc),
                    },
                )
                raise
        if not isinstance(payload, Mapping):
            return None
        payload_dict = dict(payload)
        if not self._catalog.is_catalog_payload(snapshot_kind=snapshot_kind, payload=payload_dict):
            return None
        if not self._current_head_payload_complete(snapshot_kind=snapshot_kind, payload=payload_dict):
            return None
        return payload_dict

    def _direct_current_head_missing_fast_fail(self, *, snapshot_kind: str) -> bool:
        """Return whether the fixed current head is already an explicit fast-fail miss."""

        probe_catalog_payload_result = getattr(self._catalog, "probe_catalog_payload_result", None)
        if not callable(probe_catalog_payload_result):
            return False
        try:
            status, _payload = probe_catalog_payload_result(snapshot_kind=snapshot_kind, fast_fail=True)
        except Exception:
            return False
        return str(status or "").strip().lower() == "not_found"

    def _read_direct_current_head_payload(
        self,
        *,
        snapshot_kind: str,
        use_probe: bool,
        readiness_mode: bool = False,
    ) -> dict[str, object] | None:
        """Read the fixed-URI current head with probe-vs-load semantics preserved."""

        if use_probe:
            if readiness_mode:
                return self._probe_direct_current_head_payload_read_only(snapshot_kind=snapshot_kind)
            strict_payload = self._fetch_strict_direct_current_head_payload(
                snapshot_kind=snapshot_kind,
                metadata_only=True,
            )
            if isinstance(strict_payload, Mapping):
                return dict(strict_payload)
            payload = self._catalog.probe_catalog_payload(snapshot_kind=snapshot_kind)
            return dict(payload) if isinstance(payload, Mapping) else None
        strict_payload = self._fetch_strict_direct_current_head_payload(
            snapshot_kind=snapshot_kind,
            metadata_only=False,
        )
        if isinstance(strict_payload, Mapping):
            return dict(strict_payload)
        load_head_payload = getattr(self._catalog, "_load_catalog_head_payload", None)
        if callable(load_head_payload):
            payload = load_head_payload(snapshot_kind=snapshot_kind, metadata_only=False)
            return dict(payload) if isinstance(payload, Mapping) else None
        payload = self._catalog.load_catalog_payload(snapshot_kind=snapshot_kind)
        return dict(payload) if isinstance(payload, Mapping) else None

    def _fetch_strict_direct_current_head_payload(
        self,
        *,
        snapshot_kind: str,
        metadata_only: bool,
    ) -> dict[str, object] | None:
        """Read one graph current-head envelope and ignore metadata-only shadows.

        The fixed-URI `graph_nodes` / `graph_edges` heads must describe the
        current generation in their own payload/body/content. Accepting only a
        hidden `metadata.twinr_payload` would let a broken direct head masquerade
        as healthy and prevent `ensure_remote_snapshot()` from repairing the
        advertised current view from a valid local graph cache.
        """

        remote_state = self.remote_state
        if remote_state is None:
            return None
        require_client = getattr(self._catalog, "_require_client", None)
        fetch_catalog_head_envelope = getattr(self._catalog, "_fetch_catalog_head_envelope", None)
        if not callable(require_client) or not callable(fetch_catalog_head_envelope):
            return None
        read_client = require_client(getattr(remote_state, "read_client", None), operation="read")
        effective_read_client = read_client
        if not metadata_only:
            catalog_head_read_client = getattr(self._catalog, "_catalog_head_read_client", None)
            if callable(catalog_head_read_client):
                try:
                    effective_read_client = catalog_head_read_client(read_client=read_client)
                except Exception:
                    effective_read_client = read_client
        try:
            envelope = fetch_catalog_head_envelope(
                read_client=effective_read_client,
                snapshot_kind=snapshot_kind,
                metadata_only=metadata_only,
            )
        except Exception:
            return None
        return self._strict_catalog_payload_from_direct_head_envelope(
            snapshot_kind=snapshot_kind,
            envelope=envelope,
        )

    def _strict_catalog_payload_from_direct_head_envelope(
        self,
        *,
        snapshot_kind: str,
        envelope: object,
    ) -> dict[str, object] | None:
        if not isinstance(envelope, Mapping):
            return None
        is_catalog_payload = getattr(self._catalog, "is_catalog_payload", None)
        if not callable(is_catalog_payload):
            extract_catalog_payload = getattr(self._catalog, "_extract_catalog_payload_from_document", None)
            if callable(extract_catalog_payload):
                payload = extract_catalog_payload(snapshot_kind=snapshot_kind, payload=envelope)
                return dict(payload) if isinstance(payload, Mapping) else None
            return None
        for candidate in _iter_direct_graph_head_candidates(envelope):
            if not isinstance(candidate, Mapping):
                continue
            candidate_dict = dict(candidate)
            if is_catalog_payload(snapshot_kind=snapshot_kind, payload=candidate_dict) or _looks_like_graph_catalog_payload(
                snapshot_kind=snapshot_kind,
                payload=candidate_dict,
                catalog=self._catalog,
            ):
                return candidate_dict
        return None

    def _probe_direct_current_head_payload_read_only(
        self,
        *,
        snapshot_kind: str,
    ) -> dict[str, object] | None:
        """Probe one graph current head with a strict readiness-only timeout cap.

        Readiness must fail closed quickly. The generic catalog-head helper
        inflates fixed-URI reads to the cold bootstrap/origin-resolution
        timeout, which is appropriate for foreground recovery but too slow for
        the read-only readiness contract and can keep shutdown drains pinned on
        already unhealthy remote memory. Reuse the remote state's own status-
        probe timeout first so the readiness path stays aligned with the
        watchdog/readiness contract, and only fall back to the historical small
        hard cap when the remote-state adapter cannot provide that probe client.
        """

        remote_state = self.remote_state
        if remote_state is None:
            return None
        if self._direct_current_head_missing_fast_fail(snapshot_kind=snapshot_kind):
            return None
        read_client = getattr(remote_state, "read_client", None)
        require_client = getattr(self._catalog, "_require_client", None)
        if not callable(require_client):
            return None
        effective_read_client = require_client(read_client, operation="read")
        status_probe_client = getattr(remote_state, "_status_probe_client", None)
        if callable(status_probe_client):
            try:
                effective_read_client = status_probe_client(effective_read_client)
            except Exception:
                effective_read_client = require_client(read_client, operation="read")
        else:
            from twinr.memory.longterm.storage._remote_retry import clone_client_with_capped_timeout

            effective_read_client = clone_client_with_capped_timeout(
                effective_read_client,
                timeout_s=_READINESS_CURRENT_HEAD_TIMEOUT_S,
            )
        try:
            envelope = effective_read_client.fetch_full_document(
                origin_uri=self._catalog._catalog_head_uri(snapshot_kind=snapshot_kind),
                include_content=False,
                max_content_chars=self._catalog._metadata_only_max_content_chars(),
            )
        except ChonkyDBError as exc:
            if int(exc.status_code or 0) == 400:
                return None
            raise
        payload = self._strict_catalog_payload_from_direct_head_envelope(
            snapshot_kind=snapshot_kind,
            envelope=envelope,
        )
        return dict(payload) if isinstance(payload, Mapping) else None

    def _probe_compatible_snapshot_head_payload(self, *, snapshot_kind: str) -> dict[str, object] | None:
        """Read one legacy graph current head directly from its stable snapshot URI.

        The graph compatibility head is a small catalog payload written behind a
        stable legacy snapshot URI. Reading that URI directly keeps bootstrap
        on the read-only contract and avoids the generic pointer/document-id
        snapshot recovery path that can pin fresh readers to superseded heads.
        """

        remote_state = self.remote_state
        if remote_state is None:
            return None
        read_client = getattr(remote_state, "read_client", None)
        snapshot_uri = getattr(remote_state, "_snapshot_uri", None)
        extract_snapshot_body = getattr(remote_state, "_extract_snapshot_body", None)
        if read_client is None or not callable(snapshot_uri) or not callable(extract_snapshot_body):
            return None
        try:
            envelope = getattr(read_client, "fetch_full_document")(
                origin_uri=snapshot_uri(snapshot_kind),
                include_content=True,
                max_content_chars=self._catalog._max_content_chars(),
            )
        except Exception:
            return None
        if not isinstance(envelope, Mapping):
            return None
        payload = extract_snapshot_body(envelope, snapshot_kind=snapshot_kind)
        if not isinstance(payload, Mapping):
            return None
        payload_dict = dict(payload)
        if not self._catalog.is_catalog_payload(snapshot_kind=snapshot_kind, payload=payload_dict):
            return None
        return payload_dict

    def _graph_store_timeout_seconds(self) -> int:
        remote_state = self._require_remote_state()
        config = getattr(remote_state, "config", None)
        try:
            configured = int(float(getattr(config, "long_term_memory_remote_flush_timeout_s", 20.0)))
        except (TypeError, ValueError):
            configured = 20
        return max(_MIN_GRAPH_STORE_TIMEOUT_S, configured)

    def _topology_refs(self, payload: Mapping[str, object]) -> dict[str, str]:
        raw_mapping = payload.get("topology_refs")
        if not isinstance(raw_mapping, Mapping):
            return {}
        resolved: dict[str, str] = {}
        for key, value in raw_mapping.items():
            node_id = _normalize_text(key)
            topology_ref = _normalize_text(value)
            if node_id and topology_ref:
                resolved[node_id] = topology_ref
        return resolved

    def _require_remote_state(self) -> "LongTermRemoteStateStore":
        if self.remote_state is None:
            raise _remote_unavailable_error("Remote graph state is not configured.")
        return self.remote_state
