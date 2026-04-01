"""Build one internal unified retrieval plan across all long-term sources.

The planner keeps the public ``LongTermMemoryContext`` shape stable while
enforcing one query-first candidate contract across graph, episodic, durable,
conflict, midterm, and adaptive memory. The key invariant is that the query
plan reflects the minimal kept evidence set, not every candidate that happened
to be fetched upstream.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from twinr.memory.chonkydb.schema import TwinrGraphDocumentV1, TwinrGraphEdgeV1, TwinrGraphNodeV1
from twinr.memory.context_store import PersistentMemoryEntry
from twinr.memory.longterm.core.models import LongTermConflictQueueItemV1, LongTermMidtermPacketV1
from twinr.memory.longterm.reasoning.turn_continuity import turn_continuity_recall_hints
from twinr.text_utils import folded_lookup_text

_ANCHOR_ATTRIBUTE_KEYS: tuple[str, ...] = (
    "person_ref",
    "subject_ref",
    "place_ref",
    "location_ref",
    "plan_ref",
    "event_ref",
    "thread_ref",
    "graph_node_id",
    "environment_id",
    "memory_domain",
)
_GRAPH_NODE_TYPE_TO_ANCHORS: dict[str, tuple[str, ...]] = {
    "person": ("person_ref",),
    "user": ("subject_ref",),
    "place": ("place_ref", "location_ref"),
    "plan": ("plan_ref",),
    "event": ("event_ref",),
}
_EXPLAINABLE_JOIN_PREFIXES = frozenset(
    {
        "person_ref",
        "subject_ref",
        "place_ref",
        "location_ref",
        "plan_ref",
        "event_ref",
        "thread_ref",
        "environment_id",
        "memory_domain",
    }
)
_NON_CONNECTIVE_ANCHOR_PREFIXES = frozenset({"memory_domain"})
_STRUCTURED_PRACTICAL_SOURCES = frozenset({"durable", "conflict", "adaptive"})
_TEMPORAL_CONTINUITY_SOURCES = frozenset({"episodic", "midterm"})
_GRAPH_SELECTION_BONUS = 0.35
_SUPPORT_SOURCE_BONUS = 0.05
_PRACTICAL_FAMILY_MODE_WEIGHT = 2.0
_CONTINUITY_FAMILY_MODE_WEIGHT = 1.0
_DOMINANT_FAMILY_RATIO = 1.25
_DOMINANT_FAMILY_MARGIN = 0.2
_TOP_EPISODIC_SUPPORT_LIMIT = 1
_FOCAL_GRAPH_NODE_TYPES = frozenset({"person", "user", "place", "plan", "event"})


def _normalize_text(value: object | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_text_tuple(value: object | None) -> tuple[str, ...]:
    if isinstance(value, str):
        normalized = _normalize_text(value)
        return (normalized,) if normalized else ()
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        values: list[str] = []
        for item in value:
            normalized = _normalize_text(item)
            if normalized:
                values.append(normalized)
        return tuple(values)
    normalized = _normalize_text(value)
    return (normalized,) if normalized else ()


def _structured_object_anchors(item: object) -> tuple[str, ...]:
    anchors: list[str] = []
    memory_id = _normalize_text(getattr(item, "memory_id", None))
    slot_key = _normalize_text(getattr(item, "slot_key", None))
    value_key = _normalize_text(getattr(item, "value_key", None))
    if memory_id:
        anchors.append(f"memory:{memory_id}")
    if slot_key:
        anchors.append(f"slot:{slot_key}")
    if value_key:
        anchors.append(f"value:{value_key}")
    attributes = getattr(item, "attributes", None)
    if isinstance(attributes, Mapping):
        for key in _ANCHOR_ATTRIBUTE_KEYS:
            anchors.extend(f"{key}:{value}" for value in _as_text_tuple(attributes.get(key)))
    return tuple(dict.fromkeys(anchor for anchor in anchors if anchor))


def _structured_object_state_lookup_fragments(item: object) -> tuple[str, ...]:
    """Return state-bearing lookup fragments for one structured memory object.

    Unified retrieval must score the same confirmation/current-state cues that
    the structured store uses for meta-memory queries. Otherwise the later
    pruning stage can discard the correct confirmed fact and keep only a
    generic sibling that happens to mention the broad topic word.
    """

    fragments: list[str] = []
    status = _normalize_text(getattr(item, "status", None))
    if status:
        fragments.append(status)
    if status == "active":
        fragments.extend(("current", "stored", "available", "aktuell", "gespeichert"))
    elif status == "superseded":
        fragments.extend(("previous", "former", "superseded", "frueher", "vorher"))
    elif status in {"candidate", "uncertain"}:
        fragments.extend(("pending", "unconfirmed", "candidate", "unbestaetigt", "unklar"))
    elif status == "invalid":
        fragments.extend(("invalid", "discarded"))
    elif status == "expired":
        fragments.extend(("expired", "outdated"))
    confirmed_by_user = bool(getattr(item, "confirmed_by_user", False))
    attributes = getattr(item, "attributes", None)
    if isinstance(attributes, Mapping):
        confirmed_by_user = confirmed_by_user or attributes.get("review_confirmed_by_user") is True
    if confirmed_by_user:
        fragments.extend(("confirmed_by_user", "confirmed", "user_confirmed", "bestaetigt"))
    return tuple(dict.fromkeys(fragment for fragment in fragments if fragment))


def _structured_object_lookup_fragments(item: object) -> tuple[str, ...]:
    """Return lookup fragments that keep durable-state semantics queryable."""

    fragments = [
        _normalize_text(getattr(item, "summary", None)),
        _normalize_text(getattr(item, "details", None)),
        _normalize_text(getattr(item, "slot_key", None)),
        _normalize_text(getattr(item, "value_key", None)),
        *_structured_object_state_lookup_fragments(item),
    ]
    return tuple(fragment for fragment in fragments if fragment)


def _conflict_anchors(
    item: LongTermConflictQueueItemV1,
    *,
    structured_anchors_by_memory_id: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    anchors: list[str] = []
    slot_key = _normalize_text(item.slot_key)
    candidate_memory_id = _normalize_text(item.candidate_memory_id)
    if slot_key:
        anchors.append(f"slot:{slot_key}")
    if candidate_memory_id:
        anchors.append(f"memory:{candidate_memory_id}")
        anchors.extend(structured_anchors_by_memory_id.get(candidate_memory_id, ()))
    for option in item.options:
        option_memory_id = _normalize_text(option.memory_id)
        option_value_key = _normalize_text(option.value_key)
        if option_memory_id:
            anchors.append(f"memory:{option_memory_id}")
            anchors.extend(structured_anchors_by_memory_id.get(option_memory_id, ()))
        if option_value_key:
            anchors.append(f"value:{option_value_key}")
    return tuple(dict.fromkeys(anchor for anchor in anchors if anchor))


def _graph_node_anchors(node: TwinrGraphNodeV1) -> tuple[str, ...]:
    anchors = [
        f"graph_node:{node.node_id}",
        f"node:{node.node_id}",
        f"entity:{node.node_id}",
    ]
    for anchor_key in _GRAPH_NODE_TYPE_TO_ANCHORS.get(_normalize_text(node.node_type), ()):
        anchors.append(f"{anchor_key}:{node.node_id}")
    return tuple(dict.fromkeys(anchor for anchor in anchors if anchor))


def _graph_edge_anchors(edge: TwinrGraphEdgeV1) -> tuple[str, ...]:
    anchors = [
        f"graph_edge:{edge.source_node_id}|{edge.edge_type}|{edge.target_node_id}",
        f"node:{edge.source_node_id}",
        f"node:{edge.target_node_id}",
        f"entity:{edge.source_node_id}",
        f"entity:{edge.target_node_id}",
    ]
    return tuple(dict.fromkeys(anchor for anchor in anchors if anchor))


def _midterm_packet_anchors(
    item: LongTermMidtermPacketV1,
    *,
    structured_anchors_by_memory_id: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    anchors: list[str] = [f"midterm:{_normalize_text(item.packet_id)}"]
    for memory_id in item.source_memory_ids:
        normalized_memory_id = _normalize_text(memory_id)
        if not normalized_memory_id:
            continue
        anchors.append(f"memory:{normalized_memory_id}")
        anchors.extend(structured_anchors_by_memory_id.get(normalized_memory_id, ()))
    attributes = item.attributes
    if isinstance(attributes, Mapping):
        for key in _ANCHOR_ATTRIBUTE_KEYS:
            anchors.extend(f"{key}:{value}" for value in _as_text_tuple(attributes.get(key)))
    return tuple(dict.fromkeys(anchor for anchor in anchors if anchor))


def _lookup_terms(value: object | None) -> tuple[str, ...]:
    normalized = " ".join(folded_lookup_text(_normalize_text(value)).split())
    if not normalized:
        return ()
    terms: list[str] = []
    seen: set[str] = set()
    for item in normalized.split():
        if item and item not in seen:
            seen.add(item)
            terms.append(item)
    return tuple(terms)


def _anchor_prefix(anchor: str) -> str:
    return anchor.partition(":")[0].strip()


def _anchor_value_terms(anchor: str) -> tuple[str, ...]:
    value = anchor.partition(":")[2] or anchor
    return _lookup_terms(value)


def _is_explainable_join_anchor(anchor: str) -> bool:
    return _anchor_prefix(anchor) in _EXPLAINABLE_JOIN_PREFIXES


def _is_focus_semantic_anchor(anchor: str) -> bool:
    return _is_explainable_join_anchor(anchor) and _anchor_prefix(anchor) != "memory_domain"


def _candidate_family(source: str) -> str:
    if source in _STRUCTURED_PRACTICAL_SOURCES:
        return "practical"
    if source in _TEMPORAL_CONTINUITY_SOURCES:
        return "continuity"
    if source == "graph":
        return "graph"
    return "other"


def _connective_anchors(anchors: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        anchor
        for anchor in anchors
        if _anchor_prefix(anchor) not in _NON_CONNECTIVE_ANCHOR_PREFIXES
    )


def _has_explainable_anchor(anchors: Sequence[str]) -> bool:
    return any(_is_focus_semantic_anchor(anchor) for anchor in anchors)


@dataclass(frozen=True, slots=True)
class UnifiedEpisodicSelectionInput:
    """Carry one selected episodic entry plus its explicit anchor keys."""

    entry: PersistentMemoryEntry
    anchors: tuple[str, ...]


def build_episodic_selection_input(
    *,
    entry: PersistentMemoryEntry,
    source_object: object,
) -> UnifiedEpisodicSelectionInput:
    """Attach structured-object anchors to one rendered episodic entry."""

    anchors = list(_structured_object_anchors(source_object))
    normalized_entry_id = _normalize_text(entry.entry_id)
    if normalized_entry_id:
        anchors.append(f"memory:{normalized_entry_id}")
    return UnifiedEpisodicSelectionInput(
        entry=entry,
        anchors=tuple(dict.fromkeys(anchor for anchor in anchors if anchor)),
    )


@dataclass(frozen=True, slots=True)
class UnifiedGraphSelectionInput:
    """Hold the already selected graph document plus its query plan."""

    document: TwinrGraphDocumentV1
    query_plan: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class UnifiedRetrievalSelection:
    """Carry the materialized selections and the explainable unified plan."""

    episodic_entries: tuple[PersistentMemoryEntry, ...]
    durable_objects: tuple[object, ...]
    conflict_queue: tuple[LongTermConflictQueueItemV1, ...]
    midterm_packets: tuple[LongTermMidtermPacketV1, ...]
    adaptive_packets: tuple[LongTermMidtermPacketV1, ...]
    graph_selection: UnifiedGraphSelectionInput | None
    query_plan: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Candidate:
    source: str
    family: str
    candidate_type: str
    candidate_id: str
    label: str
    anchors: tuple[str, ...]
    original_rank: int
    lookup_terms: tuple[str, ...]
    matched_query_terms: tuple[str, ...]
    query_score: float
    support_sources: tuple[str, ...]
    selected: bool
    metadata: dict[str, object]
    drop_reason: str | None = None
    lookup_fragments: tuple[str, ...] = ()

    @property
    def direct_match(self) -> bool:
        return bool(self.metadata.get("direct_match"))

    def to_payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "family": self.family,
            "type": self.candidate_type,
            "id": self.candidate_id,
            "label": self.label,
            "anchors": list(self.anchors),
            "original_rank": self.original_rank,
            "matched_query_terms": list(self.matched_query_terms),
            "query_score": self.query_score,
            "support_sources": list(self.support_sources),
            "selected": self.selected,
            "metadata": dict(self.metadata),
            "drop_reason": self.drop_reason,
        }


def _candidate_lookup_terms(candidate: _Candidate) -> tuple[str, ...]:
    fragments: list[object] = list(candidate.lookup_fragments) or [candidate.label, candidate.candidate_id, *candidate.anchors]
    for value in candidate.metadata.values():
        if isinstance(value, (str, int, float)):
            fragments.append(value)
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
            fragments.extend(value)
    terms: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        for term in _lookup_terms(fragment):
            if term not in seen:
                seen.add(term)
                terms.append(term)
    return tuple(terms)


def _enrich_candidates(
    candidates: Sequence[_Candidate],
    *,
    query_terms: tuple[str, ...],
) -> tuple[_Candidate, ...]:
    anchor_sources: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        for anchor in _connective_anchors(candidate.anchors):
            anchor_sources[anchor].add(candidate.source)

    query_term_document_frequency: dict[str, int] = defaultdict(int)
    lookup_terms_by_key: dict[tuple[str, str], tuple[str, ...]] = {}
    for candidate in candidates:
        lookup_terms = _candidate_lookup_terms(candidate)
        lookup_terms_by_key[(candidate.source, candidate.candidate_id)] = lookup_terms
        lookup_term_set = set(lookup_terms)
        for query_term in query_terms:
            if query_term in lookup_term_set:
                query_term_document_frequency[query_term] += 1

    enriched: list[_Candidate] = []
    for candidate in candidates:
        lookup_terms = lookup_terms_by_key[(candidate.source, candidate.candidate_id)]
        matched_query_terms = tuple(
            query_term
            for query_term in query_terms
            if query_term in set(lookup_terms)
        )
        query_score = sum(
            1.0 / max(1, query_term_document_frequency[query_term])
            for query_term in matched_query_terms
        )
        support_sources = tuple(
            sorted(
                {
                    source
                    for anchor in _connective_anchors(candidate.anchors)
                    for source in anchor_sources.get(anchor, set())
                    if source != candidate.source
                }
            )
        )
        query_score += _SUPPORT_SOURCE_BONUS * len(support_sources)
        if candidate.direct_match:
            query_score += _GRAPH_SELECTION_BONUS
        enriched.append(
            replace(
                candidate,
                lookup_terms=lookup_terms,
                matched_query_terms=matched_query_terms,
                query_score=query_score,
                support_sources=support_sources,
            )
        )
    return tuple(enriched)


def _connected_components(candidates: Sequence[_Candidate]) -> dict[int, int]:
    anchor_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        for anchor in _connective_anchors(candidate.anchors):
            anchor_to_indices[anchor].append(index)

    neighbors: dict[int, set[int]] = defaultdict(set)
    for indices in anchor_to_indices.values():
        if len(indices) <= 1:
            continue
        for index in indices:
            neighbors[index].update(other for other in indices if other != index)

    component_by_index: dict[int, int] = {}
    next_component_id = 0
    for index in range(len(candidates)):
        if index in component_by_index:
            continue
        queue: deque[int] = deque((index,))
        component_by_index[index] = next_component_id
        while queue:
            current = queue.popleft()
            for neighbor in neighbors.get(current, ()):
                if neighbor in component_by_index:
                    continue
                component_by_index[neighbor] = next_component_id
                queue.append(neighbor)
        next_component_id += 1
    return component_by_index


def _component_score(candidates: Sequence[_Candidate], indices: Iterable[int]) -> float:
    selected = [candidates[index] for index in indices]
    if not selected:
        return 0.0
    sorted_scores = sorted((candidate.query_score for candidate in selected), reverse=True)
    source_diversity = len({candidate.source for candidate in selected})
    return (
        sum(sorted_scores[:3])
        # A default graph subject hit (for example ``user:main``) should not
        # outweigh disconnected practical recall that actually matches the
        # user's memory question. Keep the direct-match bonus for non-graph
        # components where a real structured/continuity anchor matched.
        + (1.0 if any(candidate.direct_match and candidate.family != "graph" for candidate in selected) else 0.0)
        + (0.1 * source_diversity)
    )


def _focus_component_ids(candidates: Sequence[_Candidate]) -> set[int]:
    if not candidates:
        return set()
    component_by_index = _connected_components(candidates)
    direct_match_components = {
        component_by_index[index]
        for index, candidate in enumerate(candidates)
        if candidate.direct_match and candidate.family != "graph"
    }
    if direct_match_components:
        return direct_match_components
    component_scores: dict[int, float] = defaultdict(float)
    for component_id in set(component_by_index.values()):
        component_scores[component_id] = _component_score(
            candidates,
            (index for index, cid in component_by_index.items() if cid == component_id),
        )
    if not component_scores:
        return set()
    top_score = max(component_scores.values())
    return {
        component_id
        for component_id, score in component_scores.items()
        if score >= top_score
    }


def _candidate_sort_key(candidate: _Candidate) -> tuple[float, float, int, str, str]:
    return (
        -candidate.query_score,
        -len(candidate.support_sources),
        candidate.original_rank,
        candidate.source,
        candidate.candidate_id,
    )


def _focus_semantic_anchors(
    candidates: Sequence[_Candidate],
    *,
    query_terms: tuple[str, ...],
) -> tuple[str, ...]:
    if not candidates:
        return ()
    seed_candidates = [candidate for candidate in candidates if candidate.direct_match]
    if not seed_candidates:
        seed_candidates = sorted(candidates, key=_candidate_sort_key)[:2]
    query_term_set = set(query_terms)
    anchors: list[str] = []
    seen: set[str] = set()
    for candidate in seed_candidates:
        for anchor in candidate.anchors:
            if not _is_explainable_join_anchor(anchor):
                continue
            if not _is_focus_semantic_anchor(anchor):
                continue
            anchor_terms = set(_anchor_value_terms(anchor))
            if candidate.direct_match or anchor_terms & query_term_set:
                if anchor not in seen:
                    seen.add(anchor)
                    anchors.append(anchor)
    if anchors:
        return tuple(anchors)
    for candidate in seed_candidates:
        for anchor in candidate.anchors:
            if not _is_explainable_join_anchor(anchor):
                continue
            if not _is_focus_semantic_anchor(anchor):
                continue
            if anchor not in seen:
                seen.add(anchor)
                anchors.append(anchor)
    return tuple(anchors)


def _focus_query_terms(focus_anchors: Sequence[str]) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for anchor in focus_anchors:
        for term in _anchor_value_terms(anchor):
            if term not in seen:
                seen.add(term)
                terms.append(term)
    return tuple(terms)


def _focus_is_generic_subject(focus_anchors: Sequence[str]) -> bool:
    return bool(focus_anchors) and all(anchor.startswith("subject_ref:user:") for anchor in focus_anchors)


def _candidate_specific_terms(
    candidate: _Candidate,
    *,
    focus_terms: Sequence[str],
) -> tuple[str, ...]:
    focus_term_set = set(focus_terms)
    return tuple(term for term in candidate.matched_query_terms if term not in focus_term_set)


def _candidate_specific_score(candidate: _Candidate, *, focus_terms: Sequence[str]) -> float:
    specific_terms = _candidate_specific_terms(candidate, focus_terms=focus_terms)
    if not specific_terms:
        return 0.0
    lookup_terms = set(candidate.lookup_terms)
    return float(sum(1 for term in specific_terms if term in lookup_terms))


def _candidate_in_focus(candidate: _Candidate, *, focus_anchors: Sequence[str]) -> bool:
    if not focus_anchors:
        return True
    focus_anchor_set = set(focus_anchors)
    return any(anchor in focus_anchor_set for anchor in candidate.anchors)


def _dominant_mode(
    candidates: Sequence[_Candidate],
    *,
    focus_terms: Sequence[str],
    generic_subject_focus: bool,
) -> tuple[str, dict[str, float]]:
    raw_family_scores = {"practical": 0.0, "continuity": 0.0, "graph": 0.0}
    for candidate in candidates:
        if candidate.family not in raw_family_scores:
            continue
        if candidate.family == "graph":
            score = candidate.query_score
        else:
            score = _candidate_specific_score(candidate, focus_terms=focus_terms)
        raw_family_scores[candidate.family] = max(raw_family_scores[candidate.family], score)

    family_scores = {
        "practical": raw_family_scores["practical"] * _PRACTICAL_FAMILY_MODE_WEIGHT,
        "continuity": raw_family_scores["continuity"] * _CONTINUITY_FAMILY_MODE_WEIGHT,
        "graph": raw_family_scores["graph"],
    }

    practical_score = family_scores["practical"]
    continuity_score = family_scores["continuity"]
    if (
        practical_score > 0.0
        and practical_score >= (continuity_score * _DOMINANT_FAMILY_RATIO)
        and (practical_score - continuity_score) >= _DOMINANT_FAMILY_MARGIN
    ):
        return "practical", family_scores
    if (
        continuity_score > 0.0
        and continuity_score >= (practical_score * _DOMINANT_FAMILY_RATIO)
        and (continuity_score - practical_score) >= _DOMINANT_FAMILY_MARGIN
    ):
        return "continuity", family_scores
    if practical_score > 0.0 or continuity_score > 0.0:
        return "mixed", family_scores
    if generic_subject_focus and any(candidate.family == "practical" for candidate in candidates):
        return "practical", family_scores
    return "graph_only", family_scores


def _select_graph_candidates(candidates: Sequence[_Candidate]) -> tuple[_Candidate, ...]:
    graph_candidates = [candidate for candidate in candidates if candidate.family == "graph"]
    if not graph_candidates:
        return ()
    direct_matches = [candidate for candidate in graph_candidates if candidate.direct_match]
    if direct_matches:
        focal_nodes = [
            candidate
            for candidate in direct_matches
            if candidate.candidate_type == "graph_node"
            and _normalize_text(candidate.metadata.get("node_type")) in _FOCAL_GRAPH_NODE_TYPES
        ]
        if focal_nodes:
            non_user_focal_nodes = [
                candidate
                for candidate in focal_nodes
                if _normalize_text(candidate.metadata.get("node_type")) != "user"
            ]
            return tuple(sorted(non_user_focal_nodes or focal_nodes, key=_candidate_sort_key))
        direct_nodes = [
            candidate
            for candidate in direct_matches
            if candidate.candidate_type == "graph_node"
        ]
        if direct_nodes:
            return tuple(sorted(direct_nodes, key=_candidate_sort_key)[:1])
        return tuple(sorted(direct_matches, key=_candidate_sort_key)[:1])
    focal_nodes = [
        candidate
        for candidate in graph_candidates
        if candidate.candidate_type == "graph_node"
        and _normalize_text(candidate.metadata.get("node_type")) in _FOCAL_GRAPH_NODE_TYPES
    ]
    if focal_nodes:
        return tuple(sorted(focal_nodes, key=_candidate_sort_key)[:1])
    return tuple(sorted(graph_candidates, key=_candidate_sort_key)[:1])


def _select_support_candidates(
    candidates: Sequence[_Candidate],
    *,
    mode: str,
    focus_terms: Sequence[str],
) -> tuple[_Candidate, ...]:
    if not candidates:
        return ()
    continuity_candidates = [
        candidate
        for candidate in candidates
        if candidate.family == "continuity"
    ]
    if not continuity_candidates:
        return ()
    continuity_candidates = sorted(continuity_candidates, key=_candidate_sort_key)
    if mode == "continuity":
        return tuple(
            candidate
            for candidate in continuity_candidates
            if candidate.direct_match or _candidate_specific_score(candidate, focus_terms=focus_terms) > 0.0
        )
    kept: list[_Candidate] = []
    episodic_candidates = [
        candidate
        for candidate in continuity_candidates
        if candidate.source == "episodic" and candidate.query_score > 0.0
    ]
    midterm_candidates = [
        candidate
        for candidate in continuity_candidates
        if candidate.source == "midterm"
        and (
            candidate.direct_match
            or _candidate_specific_score(candidate, focus_terms=focus_terms) > 0.0
            or candidate.query_score > 0.0
        )
    ]
    kept.extend(midterm_candidates)
    kept.extend(episodic_candidates[:_TOP_EPISODIC_SUPPORT_LIMIT])
    return tuple(kept)


def _select_practical_candidates(
    candidates: Sequence[_Candidate],
    *,
    focus_terms: Sequence[str],
    allow_upstream_rank_fallback: bool,
) -> tuple[_Candidate, ...]:
    practical_candidates = [
        candidate
        for candidate in candidates
        if candidate.family == "practical"
        and (
            candidate.direct_match
            or _candidate_specific_score(candidate, focus_terms=focus_terms) > 0.0
        )
    ]
    if practical_candidates:
        return tuple(sorted(practical_candidates, key=_candidate_sort_key))
    if allow_upstream_rank_fallback:
        fallback_candidates = [
            candidate
            for candidate in candidates
            if candidate.family == "practical"
        ]
        if fallback_candidates:
            return tuple(sorted(fallback_candidates, key=_candidate_sort_key)[:1])
    return ()


def _mark_selection(
    candidates: Sequence[_Candidate],
    *,
    selected_keys: set[tuple[str, str]],
    dropped_reason_by_key: Mapping[tuple[str, str], str],
) -> tuple[_Candidate, ...]:
    marked: list[_Candidate] = []
    for candidate in candidates:
        key = (candidate.source, candidate.candidate_id)
        marked.append(
            replace(
                candidate,
                selected=key in selected_keys,
                drop_reason=None if key in selected_keys else dropped_reason_by_key.get(key),
            )
        )
    return tuple(marked)


def _selected_join_anchors(
    candidates: Sequence[_Candidate],
    *,
    query_terms: Sequence[str],
    focus_anchors: Sequence[str],
) -> tuple[tuple[str, list[str]], ...]:
    anchor_sources: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        for anchor in candidate.anchors:
            anchor_sources[anchor].add(candidate.source)
    query_term_set = set(query_terms)
    focus_anchor_set = set(focus_anchors)
    payload: list[tuple[str, list[str]]] = []
    for anchor, sources in sorted(anchor_sources.items(), key=lambda item: item[0]):
        if len(sources) <= 1 or not _is_explainable_join_anchor(anchor):
            continue
        anchor_terms = set(_anchor_value_terms(anchor))
        if focus_anchor_set and anchor not in focus_anchor_set and not (anchor_terms & query_term_set):
            continue
        payload.append((anchor, sorted(sources)))
    return tuple(payload)


def build_unified_retrieval_selection(
    *,
    query_texts: tuple[str, ...],
    episodic_entries: Iterable[UnifiedEpisodicSelectionInput],
    durable_objects: Iterable[object],
    conflict_queue: Iterable[LongTermConflictQueueItemV1],
    conflict_supporting_objects: Iterable[object],
    midterm_packets: Iterable[LongTermMidtermPacketV1],
    graph_selection: UnifiedGraphSelectionInput | None,
) -> UnifiedRetrievalSelection:
    """Build one explainable internal plan and return the kept evidence set."""

    episodic_items = tuple(episodic_entries)
    durable_items = tuple(durable_objects)
    conflict_items = tuple(conflict_queue)
    supporting_items = tuple(conflict_supporting_objects)
    midterm_items = tuple(midterm_packets)

    structured_anchor_map: dict[str, tuple[str, ...]] = {}
    durable_candidates: list[tuple[object, tuple[str, ...], int]] = []
    for index, item in enumerate(durable_items):
        memory_id = _normalize_text(getattr(item, "memory_id", None))
        anchors = _structured_object_anchors(item)
        if memory_id:
            structured_anchor_map[memory_id] = anchors
        durable_candidates.append((item, anchors, index))
    for item in supporting_items:
        memory_id = _normalize_text(getattr(item, "memory_id", None))
        anchors = _structured_object_anchors(item)
        if memory_id and memory_id not in structured_anchor_map:
            structured_anchor_map[memory_id] = anchors

    conflict_candidates: list[tuple[LongTermConflictQueueItemV1, tuple[str, ...], int]] = []
    for index, item in enumerate(conflict_items):
        conflict_candidates.append(
            (
                item,
                _conflict_anchors(item, structured_anchors_by_memory_id=structured_anchor_map),
                index,
            )
        )
    midterm_candidates: list[tuple[LongTermMidtermPacketV1, tuple[str, ...], int]] = []
    for index, item in enumerate(midterm_items):
        midterm_candidates.append(
            (
                item,
                _midterm_packet_anchors(item, structured_anchors_by_memory_id=structured_anchor_map),
                index,
            )
        )

    episodic_candidates: list[tuple[UnifiedEpisodicSelectionInput, int]] = [
        (item, index)
        for index, item in enumerate(episodic_items)
    ]

    graph_candidates: list[_Candidate] = []
    access_path: list[str] = []
    graph_plan_fragment: dict[str, object] | None = None
    if episodic_items or durable_items or conflict_items:
        access_path.append("structured_query_first")
    if graph_selection is not None:
        graph_plan_fragment = (
            dict(graph_selection.query_plan)
            if isinstance(graph_selection.query_plan, Mapping)
            else None
        )
        if graph_plan_fragment is not None:
            raw_access_path = graph_plan_fragment.get("access_path")
            if isinstance(raw_access_path, list):
                access_path.extend(
                    normalized
                    for normalized in (_normalize_text(item) for item in raw_access_path)
                    if normalized
                )
            matched_node_ids = {
                normalized
                for normalized in (_normalize_text(item) for item in graph_plan_fragment.get("matched_node_ids", ()))
                if normalized
            }
            matched_edge_ids = {
                normalized
                for normalized in (_normalize_text(item) for item in graph_plan_fragment.get("matched_edge_ids", ()))
                if normalized
            }
        else:
            access_path.append("graph_document_load")
            matched_node_ids = set()
            matched_edge_ids = set()

        for index, node in enumerate(graph_selection.document.nodes):
            node_id = _normalize_text(node.node_id)
            graph_candidates.append(
                _Candidate(
                    source="graph",
                    family="graph",
                    candidate_type="graph_node",
                    candidate_id=node_id,
                    label=_normalize_text(node.label) or node_id,
                    anchors=_graph_node_anchors(node),
                    original_rank=index,
                    lookup_terms=(),
                    matched_query_terms=(),
                    query_score=0.0,
                    support_sources=(),
                    selected=False,
                    metadata={
                        "direct_match": node_id in matched_node_ids,
                        "node_type": _normalize_text(node.node_type),
                    },
                )
            )
        for index, edge in enumerate(graph_selection.document.edges):
            edge_id = f"{edge.source_node_id}|{edge.edge_type}|{edge.target_node_id}"
            graph_candidates.append(
                _Candidate(
                    source="graph",
                    family="graph",
                    candidate_type="graph_edge",
                    candidate_id=edge_id,
                    label=" ".join(
                        part
                        for part in (edge.source_node_id, edge.edge_type, edge.target_node_id)
                        if part
                    ),
                    anchors=_graph_edge_anchors(edge),
                    original_rank=index,
                    lookup_terms=(),
                    matched_query_terms=(),
                    query_score=0.0,
                    support_sources=(),
                    selected=False,
                    metadata={
                        "direct_match": edge_id in matched_edge_ids,
                        "edge_type": _normalize_text(edge.edge_type),
                    },
                )
            )

    pending_candidates: list[_Candidate] = []
    for item, index in episodic_candidates:
        entry_id = _normalize_text(item.entry.entry_id) or f"episodic:{index}"
        pending_candidates.append(
            _Candidate(
                source="episodic",
                family="continuity",
                candidate_type="episodic_entry",
                candidate_id=entry_id,
                label=_normalize_text(item.entry.summary) or entry_id,
                anchors=item.anchors,
                original_rank=index,
                lookup_terms=(),
                matched_query_terms=(),
                query_score=0.0,
                support_sources=(),
                selected=False,
                metadata={"kind": item.entry.kind},
                lookup_fragments=(
                    _normalize_text(getattr(item.entry, "summary", None)),
                    _normalize_text(getattr(item.entry, "details", None)),
                ),
            )
        )
    for item, anchors, index in durable_candidates:
        memory_id = _normalize_text(getattr(item, "memory_id", None)) or f"durable:{index}"
        pending_candidates.append(
            _Candidate(
                source="durable",
                family="practical",
                candidate_type="structured_object",
                candidate_id=memory_id,
                label=_normalize_text(getattr(item, "summary", None)) or memory_id,
                anchors=anchors,
                original_rank=index,
                lookup_terms=(),
                matched_query_terms=(),
                query_score=0.0,
                support_sources=(),
                selected=False,
                metadata={
                    "kind": _normalize_text(getattr(item, "kind", None)),
                    "status": _normalize_text(getattr(item, "status", None)),
                    "confirmed_by_user": bool(getattr(item, "confirmed_by_user", False)),
                    "slot_key": _normalize_text(getattr(item, "slot_key", None)),
                    "value_key": _normalize_text(getattr(item, "value_key", None)),
                },
                lookup_fragments=_structured_object_lookup_fragments(item),
            )
        )
    for item, anchors, index in conflict_candidates:
        slot_key = _normalize_text(item.slot_key) or f"conflict:{index}"
        pending_candidates.append(
            _Candidate(
                source="conflict",
                family="practical",
                candidate_type="conflict_queue_item",
                candidate_id=slot_key,
                label=_normalize_text(item.question) or slot_key,
                anchors=anchors,
                original_rank=index,
                lookup_terms=(),
                matched_query_terms=(),
                query_score=0.0,
                support_sources=(),
                selected=False,
                metadata={"candidate_memory_id": item.candidate_memory_id},
                lookup_fragments=(
                    _normalize_text(item.question),
                    _normalize_text(item.reason),
                    _normalize_text(item.slot_key),
                ),
            )
        )
    for item, anchors, index in midterm_candidates:
        packet_id = _normalize_text(item.packet_id) or f"midterm:{index}"
        pending_candidates.append(
            _Candidate(
                source="midterm",
                family="continuity",
                candidate_type="midterm_packet",
                candidate_id=packet_id,
                label=_normalize_text(item.summary) or packet_id,
                anchors=anchors,
                original_rank=index,
                lookup_terms=(),
                matched_query_terms=(),
                query_score=0.0,
                support_sources=(),
                selected=False,
                metadata={"kind": item.kind},
                lookup_fragments=(
                    _normalize_text(item.summary),
                    _normalize_text(item.details),
                    *_as_text_tuple(item.query_hints),
                    *turn_continuity_recall_hints(
                        kind=item.kind,
                        attributes=item.attributes if isinstance(item.attributes, Mapping) else None,
                    ),
                ),
            )
        )

    query_terms = tuple(
        dict.fromkeys(
            term
            for query_text in query_texts
            for term in _lookup_terms(query_text)
        )
    )
    all_candidates = tuple([*pending_candidates, *graph_candidates])
    enriched_candidates = _enrich_candidates(all_candidates, query_terms=query_terms)
    component_by_index = _connected_components(enriched_candidates)
    focus_component_ids = _focus_component_ids(enriched_candidates)

    focus_candidates = tuple(
        candidate
        for index, candidate in enumerate(enriched_candidates)
        if (
            component_by_index.get(index) in focus_component_ids
            or (
                not _has_explainable_anchor(candidate.anchors)
                and candidate.query_score > 0.0
            )
        )
    )
    focus_anchors = _focus_semantic_anchors(focus_candidates, query_terms=query_terms)
    generic_subject_focus = _focus_is_generic_subject(focus_anchors)
    if generic_subject_focus:
        rescued_practical = [
            candidate
            for candidate in enriched_candidates
            if candidate.family == "practical" and not _has_explainable_anchor(candidate.anchors)
        ]
        if rescued_practical:
            focus_candidates = tuple(
                {
                    (candidate.source, candidate.candidate_id): candidate
                    for candidate in (*focus_candidates, *rescued_practical)
                }.values()
            )
    focus_terms = _focus_query_terms(focus_anchors)
    selection_mode, family_scores = _dominant_mode(
        focus_candidates,
        focus_terms=focus_terms,
        generic_subject_focus=generic_subject_focus,
    )

    selected_candidates: list[_Candidate] = list(_select_graph_candidates(focus_candidates))
    if selection_mode in {"practical", "mixed"}:
        selected_candidates.extend(
            _select_practical_candidates(
                focus_candidates,
                focus_terms=focus_terms,
                allow_upstream_rank_fallback=generic_subject_focus,
            )
        )
        selected_candidates.extend(
            _select_support_candidates(
                focus_candidates,
                mode=selection_mode,
                focus_terms=focus_terms,
            )
        )
    elif selection_mode == "continuity":
        selected_candidates.extend(
            _select_support_candidates(
                focus_candidates,
                mode="continuity",
                focus_terms=focus_terms,
            )
        )

    if not selected_candidates and focus_candidates:
        selected_candidates.append(sorted(focus_candidates, key=_candidate_sort_key)[0])

    selected_by_key: dict[tuple[str, str], _Candidate] = {}
    for candidate in sorted(selected_candidates, key=_candidate_sort_key):
        selected_by_key[(candidate.source, candidate.candidate_id)] = candidate

    dropped_reason_by_key: dict[tuple[str, str], str] = {}
    for index, candidate in enumerate(enriched_candidates):
        key = (candidate.source, candidate.candidate_id)
        if key in selected_by_key:
            continue
        if component_by_index.get(index) not in focus_component_ids:
            dropped_reason_by_key[key] = "out_of_focus_component"
            continue
        if candidate.family == "graph":
            dropped_reason_by_key[key] = "graph_not_direct_focus"
            continue
        if selection_mode == "continuity" and candidate.family == "practical":
            dropped_reason_by_key[key] = "pruned_non_dominant_family"
            continue
        if selection_mode in {"practical", "mixed"} and candidate.family == "continuity":
            dropped_reason_by_key[key] = "support_redundancy"
            continue
        dropped_reason_by_key[key] = "low_specific_relevance"

    marked_candidates = _mark_selection(
        enriched_candidates,
        selected_keys=set(selected_by_key),
        dropped_reason_by_key=dropped_reason_by_key,
    )
    kept_candidates = tuple(
        candidate
        for candidate in marked_candidates
        if candidate.selected
    )
    dropped_candidates = tuple(
        candidate
        for candidate in marked_candidates
        if not candidate.selected
    )

    score_by_candidate_id = {
        (candidate.source, candidate.candidate_id): candidate.query_score
        for candidate in kept_candidates
    }

    reordered_episodic = tuple(
        item.entry
        for item, index in sorted(
            (
                row
                for row in episodic_candidates
                if ("episodic", _normalize_text(row[0].entry.entry_id) or f"episodic:{row[1]}") in selected_by_key
            ),
            key=lambda row: (
                -score_by_candidate_id.get(
                    (
                        "episodic",
                        _normalize_text(row[0].entry.entry_id) or f"episodic:{row[1]}",
                    ),
                    0,
                ),
                row[1],
            ),
        )
    )
    reordered_durable = tuple(
        item
        for item, _anchors, index in sorted(
            (
                row
                for row in durable_candidates
                if ("durable", _normalize_text(getattr(row[0], "memory_id", None)) or f"durable:{row[2]}") in selected_by_key
            ),
            key=lambda row: (
                -score_by_candidate_id.get(
                    (
                        "durable",
                        _normalize_text(getattr(row[0], "memory_id", None)) or f"durable:{row[2]}",
                    ),
                    0,
                ),
                row[2],
            ),
        )
    )
    reordered_conflicts = tuple(
        item
        for item, _anchors, index in sorted(
            (
                row
                for row in conflict_candidates
                if ("conflict", _normalize_text(row[0].slot_key) or f"conflict:{row[2]}") in selected_by_key
            ),
            key=lambda row: (
                -score_by_candidate_id.get(("conflict", _normalize_text(row[0].slot_key) or f"conflict:{row[2]}"), 0),
                row[2],
            ),
        )
    )
    reordered_midterm = tuple(
        item
        for item, _anchors, index in sorted(
            (
                row
                for row in midterm_candidates
                if ("midterm", _normalize_text(row[0].packet_id) or f"midterm:{row[2]}") in selected_by_key
            ),
            key=lambda row: (
                -score_by_candidate_id.get(("midterm", _normalize_text(row[0].packet_id) or f"midterm:{row[2]}"), 0),
                row[2],
            ),
        )
    )

    selected_graph_nodes = [
        candidate.candidate_id
        for candidate in sorted(kept_candidates, key=_candidate_sort_key)
        if candidate.candidate_type == "graph_node"
    ]
    selected_graph_edges = [
        candidate.candidate_id
        for candidate in sorted(kept_candidates, key=_candidate_sort_key)
        if candidate.candidate_type == "graph_edge"
    ]
    join_anchors = _selected_join_anchors(
        kept_candidates,
        query_terms=query_terms,
        focus_anchors=focus_anchors,
    )
    query_plan = {
        "schema": "twinr_unified_retrieval_plan_v1",
        "mode": "internal_unified_selection",
        "query_texts": list(query_texts),
        "sources": {
            "episodic_count": len(reordered_episodic),
            "durable_count": len(reordered_durable),
            "conflict_count": len(reordered_conflicts),
            "midterm_count": len(reordered_midterm),
            "adaptive_count": 0,
            "graph_selected": bool(selected_graph_nodes or selected_graph_edges),
        },
        "access_path": list(dict.fromkeys(access_path)),
        "join_anchors": [
            {"anchor": anchor, "sources": sources}
            for anchor, sources in join_anchors
        ],
        "candidates": [candidate.to_payload() for candidate in kept_candidates],
        "dropped_candidates": [candidate.to_payload() for candidate in dropped_candidates],
        "selected": {
            "episodic_entry_ids": [
                normalized
                for normalized in (_normalize_text(item.entry_id) for item in reordered_episodic)
                if normalized
            ],
            "durable_memory_ids": [
                normalized
                for normalized in (_normalize_text(getattr(item, "memory_id", None)) for item in reordered_durable)
                if normalized
            ],
            "conflict_slot_keys": [
                normalized
                for normalized in (_normalize_text(item.slot_key) for item in reordered_conflicts)
                if normalized
            ],
            "midterm_packet_ids": [
                normalized
                for normalized in (_normalize_text(item.packet_id) for item in reordered_midterm)
                if normalized
            ],
            "adaptive_packet_ids": [],
            "graph_node_ids": selected_graph_nodes,
            "graph_edge_ids": selected_graph_edges,
        },
        "adaptive": {
            "schema": "twinr_unified_adaptive_plan_v1",
            "selected_packet_ids": [],
            "candidate_count": 0,
            "source": "adaptive_policy_builder",
        },
        "pruning": {
            "schema": "twinr_unified_pruning_plan_v1",
            "mode": selection_mode,
            "query_terms": list(query_terms),
            "focus_anchors": list(focus_anchors),
            "focus_query_terms": list(focus_terms),
            "focus_component_ids": sorted(focus_component_ids),
            "family_scores": family_scores,
            "candidate_count_before_prune": len(enriched_candidates),
            "candidate_count_after_prune": len(kept_candidates),
            "dropped_candidate_count": len(dropped_candidates),
        },
        "graph_query_plan": graph_plan_fragment,
    }
    return UnifiedRetrievalSelection(
        episodic_entries=reordered_episodic,
        durable_objects=reordered_durable,
        conflict_queue=reordered_conflicts,
        midterm_packets=reordered_midterm,
        adaptive_packets=(),
        graph_selection=graph_selection,
        query_plan=query_plan,
    )


def attach_adaptive_packets_to_query_plan(
    *,
    query_plan: dict[str, object] | None,
    durable_objects: Iterable[object],
    adaptive_packets: Iterable[LongTermMidtermPacketV1],
) -> tuple[LongTermMidtermPacketV1, ...]:
    """Attach adaptive packets to an existing unified retrieval plan in-place."""

    adaptive_items = tuple(adaptive_packets)
    if not isinstance(query_plan, dict) or not adaptive_items:
        return adaptive_items

    sources = query_plan.get("sources")
    if not isinstance(sources, dict):
        sources = {}
        query_plan["sources"] = sources

    selected = query_plan.get("selected")
    if not isinstance(selected, dict):
        selected = {}
        query_plan["selected"] = selected

    existing_candidates = query_plan.get("candidates")
    if not isinstance(existing_candidates, list):
        existing_candidates = []
        query_plan["candidates"] = existing_candidates
    dropped_candidates = query_plan.get("dropped_candidates")
    if not isinstance(dropped_candidates, list):
        dropped_candidates = []
        query_plan["dropped_candidates"] = dropped_candidates

    structured_anchor_map = {
        memory_id: _structured_object_anchors(item)
        for item in durable_objects
        if (memory_id := _normalize_text(getattr(item, "memory_id", None)))
    }
    anchor_sources: dict[str, set[str]] = defaultdict(set)
    for candidate in existing_candidates:
        if not isinstance(candidate, Mapping):
            continue
        source = _normalize_text(candidate.get("source"))
        if not source:
            continue
        for anchor in _as_text_tuple(candidate.get("anchors")):
            if _anchor_prefix(anchor) not in _NON_CONNECTIVE_ANCHOR_PREFIXES:
                anchor_sources[anchor].add(source)

    adaptive_candidates: list[tuple[LongTermMidtermPacketV1, tuple[str, ...], int]] = []
    for index, item in enumerate(adaptive_items):
        anchors = _midterm_packet_anchors(item, structured_anchors_by_memory_id=structured_anchor_map)
        adaptive_candidates.append((item, anchors, index))

    pruning = query_plan.get("pruning")
    pruning_mode = ""
    focus_anchors: tuple[str, ...] = ()
    focus_terms: tuple[str, ...] = ()
    query_terms: tuple[str, ...] = ()
    if isinstance(pruning, Mapping):
        pruning_mode = _normalize_text(pruning.get("mode"))
        focus_anchors = _as_text_tuple(pruning.get("focus_anchors"))
        focus_terms = _as_text_tuple(pruning.get("focus_query_terms"))
        query_terms = _as_text_tuple(pruning.get("query_terms"))

    if pruning_mode not in {"practical", "mixed"}:
        for index, item in enumerate(adaptive_items):
            packet_id = _normalize_text(item.packet_id) or f"adaptive:{index}"
            dropped_candidates.append(
                _Candidate(
                    source="adaptive",
                    family="practical",
                    candidate_type="adaptive_midterm_packet",
                    candidate_id=packet_id,
                    label=_normalize_text(item.summary) or packet_id,
                    anchors=_midterm_packet_anchors(item, structured_anchors_by_memory_id=structured_anchor_map),
                    original_rank=index,
                    lookup_terms=(),
                    matched_query_terms=(),
                    query_score=0.0,
                    support_sources=(),
                    selected=False,
                    metadata={"kind": item.kind},
                    drop_reason="pruned_non_dominant_family",
                    lookup_fragments=(
                        _normalize_text(item.summary),
                        _normalize_text(item.details),
                    ),
                ).to_payload()
            )
        sources["adaptive_count"] = 0
        selected["adaptive_packet_ids"] = []
        query_plan["adaptive"] = {
            "schema": "twinr_unified_adaptive_plan_v1",
            "selected_packet_ids": [],
            "candidate_count": len(adaptive_items),
            "source": "adaptive_policy_builder",
        }
        return ()

    candidate_payloads: list[dict[str, object]] = []
    kept_adaptive: list[LongTermMidtermPacketV1] = []
    for item, anchors, index in adaptive_candidates:
        packet_id = _normalize_text(item.packet_id) or f"adaptive:{index}"
        support_sources = sorted(
            {
                source
                for anchor in _connective_anchors(anchors)
                for source in anchor_sources.get(anchor, set())
                if source != "adaptive"
            }
        )
        candidate = _Candidate(
            source="adaptive",
            family="practical",
            candidate_type="adaptive_midterm_packet",
            candidate_id=packet_id,
            label=_normalize_text(item.summary) or packet_id,
            anchors=anchors,
            original_rank=index,
            lookup_terms=(),
            matched_query_terms=tuple(
                query_term
                for query_term in query_terms
                if query_term in set(_candidate_lookup_terms(
                    _Candidate(
                        source="adaptive",
                        family="practical",
                        candidate_type="adaptive_midterm_packet",
                        candidate_id=packet_id,
                        label=_normalize_text(item.summary) or packet_id,
                        anchors=anchors,
                        original_rank=index,
                        lookup_terms=(),
                        matched_query_terms=(),
                        query_score=0.0,
                        support_sources=(),
                        selected=False,
                        metadata={"kind": item.kind},
                        lookup_fragments=(
                            _normalize_text(item.summary),
                            _normalize_text(item.details),
                        ),
                    )
                ))
            ),
            query_score=0.0,
            support_sources=tuple(support_sources),
            selected=False,
            metadata={
                "kind": item.kind,
                "policy_scope": _normalize_text(getattr(item, "attributes", {}).get("policy_scope"))
                if isinstance(getattr(item, "attributes", None), Mapping)
                else "",
            },
            lookup_fragments=(
                _normalize_text(item.summary),
                _normalize_text(item.details),
            ),
        )
        specific_score = _candidate_specific_score(candidate, focus_terms=focus_terms)
        anchored = _candidate_in_focus(candidate, focus_anchors=focus_anchors)
        if anchored and (specific_score > 0.0 or support_sources):
            kept_adaptive.append(item)
            kept_candidate = replace(
                candidate,
                lookup_terms=_candidate_lookup_terms(candidate),
                query_score=specific_score + (_SUPPORT_SOURCE_BONUS * len(support_sources)),
                selected=True,
            )
            candidate_payloads.append(kept_candidate.to_payload())
            for anchor in _connective_anchors(anchors):
                anchor_sources[anchor].add("adaptive")
        else:
            dropped_candidates.append(
                replace(
                    candidate,
                    lookup_terms=_candidate_lookup_terms(candidate),
                    query_score=specific_score,
                    selected=False,
                    drop_reason="low_specific_relevance",
                ).to_payload()
            )

    reordered_adaptive = tuple(kept_adaptive)

    existing_candidates.extend(candidate_payloads)
    sources["adaptive_count"] = len(reordered_adaptive)
    selected["adaptive_packet_ids"] = [
        normalized
        for normalized in (_normalize_text(item.packet_id) for item in reordered_adaptive)
        if normalized
    ]
    query_plan["adaptive"] = {
        "schema": "twinr_unified_adaptive_plan_v1",
        "selected_packet_ids": list(selected["adaptive_packet_ids"]),
        "candidate_count": len(adaptive_items),
        "source": "adaptive_policy_builder",
    }
    query_plan["join_anchors"] = [
        {"anchor": anchor, "sources": sorted(sources_for_anchor)}
        for anchor, sources_for_anchor in sorted(anchor_sources.items(), key=lambda item: item[0])
        if len(sources_for_anchor) > 1
        and _is_explainable_join_anchor(anchor)
        and (
            anchor in set(focus_anchors)
            or (set(_anchor_value_terms(anchor)) & set(query_terms))
        )
    ]
    if isinstance(pruning, dict):
        pruning["candidate_count_after_prune"] = len(existing_candidates)
        pruning["dropped_candidate_count"] = len(dropped_candidates)
    return reordered_adaptive


__all__ = [
    "UnifiedEpisodicSelectionInput",
    "UnifiedGraphSelectionInput",
    "UnifiedRetrievalSelection",
    "attach_adaptive_packets_to_query_plan",
    "build_episodic_selection_input",
    "build_unified_retrieval_selection",
]
