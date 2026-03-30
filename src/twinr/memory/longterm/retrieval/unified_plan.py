"""Build one internal unified retrieval plan across structured memory and graph context.

The first rollout keeps the public ``LongTermMemoryContext`` shape stable while
stopping the retriever from reasoning about graph and structured memory as
completely unrelated planes. This module normalizes selected graph, durable,
and conflict candidates into one explainable plan and returns the reordered
materialized results that the existing renderers should consume.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from twinr.memory.chonkydb.schema import TwinrGraphDocumentV1, TwinrGraphEdgeV1, TwinrGraphNodeV1
from twinr.memory.longterm.core.models import LongTermConflictQueueItemV1, LongTermMidtermPacketV1

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


@dataclass(frozen=True, slots=True)
class UnifiedGraphSelectionInput:
    """Hold the already selected graph document plus its query plan."""

    document: TwinrGraphDocumentV1
    query_plan: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class UnifiedRetrievalSelection:
    """Carry the materialized selections and the explainable unified plan."""

    durable_objects: tuple[object, ...]
    conflict_queue: tuple[LongTermConflictQueueItemV1, ...]
    midterm_packets: tuple[LongTermMidtermPacketV1, ...]
    graph_selection: UnifiedGraphSelectionInput | None
    query_plan: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Candidate:
    source: str
    candidate_type: str
    candidate_id: str
    label: str
    anchors: tuple[str, ...]
    original_rank: int
    support_sources: tuple[str, ...]
    selected: bool
    metadata: dict[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "type": self.candidate_type,
            "id": self.candidate_id,
            "label": self.label,
            "anchors": list(self.anchors),
            "original_rank": self.original_rank,
            "support_sources": list(self.support_sources),
            "selected": self.selected,
            "metadata": dict(self.metadata),
        }


def build_unified_retrieval_selection(
    *,
    query_texts: tuple[str, ...],
    durable_objects: Iterable[object],
    conflict_queue: Iterable[LongTermConflictQueueItemV1],
    conflict_supporting_objects: Iterable[object],
    midterm_packets: Iterable[LongTermMidtermPacketV1],
    graph_selection: UnifiedGraphSelectionInput | None,
) -> UnifiedRetrievalSelection:
    """Build one explainable internal plan and reorder materialized results.

    The planner is intentionally conservative in the first rollout:
    - it only uses explicit anchors already present in structured attributes,
      slot/value ids, and graph node ids
    - it keeps the current output shape stable
    - it only reorders structured sections by cross-source support; it does not
      widen recall or invent new candidates
    """

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

    graph_candidates: list[_Candidate] = []
    access_path: list[str] = []
    graph_plan_fragment: dict[str, object] | None = None
    if durable_items or conflict_items:
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
                    candidate_type="graph_node",
                    candidate_id=node_id,
                    label=_normalize_text(node.label) or node_id,
                    anchors=_graph_node_anchors(node),
                    original_rank=index,
                    support_sources=(),
                    selected=True,
                    metadata={"direct_match": node_id in matched_node_ids},
                )
            )
        for index, edge in enumerate(graph_selection.document.edges):
            edge_id = f"{edge.source_node_id}|{edge.edge_type}|{edge.target_node_id}"
            graph_candidates.append(
                _Candidate(
                    source="graph",
                    candidate_type="graph_edge",
                    candidate_id=edge_id,
                    label=" ".join(
                        part
                        for part in (edge.source_node_id, edge.edge_type, edge.target_node_id)
                        if part
                    ),
                    anchors=_graph_edge_anchors(edge),
                    original_rank=index,
                    support_sources=(),
                    selected=True,
                    metadata={"direct_match": edge_id in matched_edge_ids},
                )
            )

    pending_candidates: list[_Candidate] = []
    for item, anchors, index in durable_candidates:
        memory_id = _normalize_text(getattr(item, "memory_id", None)) or f"durable:{index}"
        pending_candidates.append(
            _Candidate(
                source="durable",
                candidate_type="structured_object",
                candidate_id=memory_id,
                label=_normalize_text(getattr(item, "summary", None)) or memory_id,
                anchors=anchors,
                original_rank=index,
                support_sources=(),
                selected=True,
                metadata={},
            )
        )
    for item, anchors, index in conflict_candidates:
        slot_key = _normalize_text(item.slot_key) or f"conflict:{index}"
        pending_candidates.append(
            _Candidate(
                source="conflict",
                candidate_type="conflict_queue_item",
                candidate_id=slot_key,
                label=_normalize_text(item.question) or slot_key,
                anchors=anchors,
                original_rank=index,
                support_sources=(),
                selected=True,
                metadata={"candidate_memory_id": item.candidate_memory_id},
            )
        )
    for item, anchors, index in midterm_candidates:
        packet_id = _normalize_text(item.packet_id) or f"midterm:{index}"
        pending_candidates.append(
            _Candidate(
                source="midterm",
                candidate_type="midterm_packet",
                candidate_id=packet_id,
                label=_normalize_text(item.summary) or packet_id,
                anchors=anchors,
                original_rank=index,
                support_sources=(),
                selected=True,
                metadata={"kind": item.kind},
            )
        )

    all_candidates = [*pending_candidates, *graph_candidates]
    anchor_sources: dict[str, set[str]] = defaultdict(set)
    for candidate in all_candidates:
        for anchor in candidate.anchors:
            anchor_sources[anchor].add(candidate.source)

    enriched_candidates: list[_Candidate] = []
    for candidate in all_candidates:
        support_sources = sorted(
            {
                source
                for anchor in candidate.anchors
                for source in anchor_sources.get(anchor, set())
                if source != candidate.source
            }
        )
        enriched_candidates.append(
            _Candidate(
                source=candidate.source,
                candidate_type=candidate.candidate_type,
                candidate_id=candidate.candidate_id,
                label=candidate.label,
                anchors=candidate.anchors,
                original_rank=candidate.original_rank,
                support_sources=tuple(support_sources),
                selected=candidate.selected,
                metadata=dict(candidate.metadata),
            )
        )

    support_by_candidate_id = {
        (candidate.source, candidate.candidate_id): len(candidate.support_sources)
        for candidate in enriched_candidates
    }

    reordered_durable = tuple(
        item
        for item, _anchors, index in sorted(
            durable_candidates,
            key=lambda row: (
                -support_by_candidate_id.get(
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
            conflict_candidates,
            key=lambda row: (
                -support_by_candidate_id.get(("conflict", _normalize_text(row[0].slot_key) or f"conflict:{row[2]}"), 0),
                row[2],
            ),
        )
    )
    reordered_midterm = tuple(
        item
        for item, _anchors, index in sorted(
            midterm_candidates,
            key=lambda row: (
                -support_by_candidate_id.get(("midterm", _normalize_text(row[0].packet_id) or f"midterm:{row[2]}"), 0),
                row[2],
            ),
        )
    )

    join_anchors = sorted(
        {
            anchor: sorted(sources)
            for anchor, sources in anchor_sources.items()
            if len(sources) > 1
        }.items(),
        key=lambda item: item[0],
    )
    query_plan = {
        "schema": "twinr_unified_retrieval_plan_v1",
        "mode": "internal_unified_selection",
        "query_texts": list(query_texts),
        "sources": {
            "durable_count": len(reordered_durable),
            "conflict_count": len(reordered_conflicts),
            "midterm_count": len(reordered_midterm),
            "graph_selected": graph_selection is not None,
        },
        "access_path": list(dict.fromkeys(access_path)),
        "join_anchors": [
            {"anchor": anchor, "sources": sources}
            for anchor, sources in join_anchors
        ],
        "candidates": [candidate.to_payload() for candidate in enriched_candidates],
        "selected": {
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
            "graph_node_ids": list(graph_plan_fragment.get("selected_node_ids", ()))
            if isinstance(graph_plan_fragment, Mapping)
            else [],
            "graph_edge_ids": list(graph_plan_fragment.get("selected_edge_ids", ()))
            if isinstance(graph_plan_fragment, Mapping)
            else [],
        },
        "graph_query_plan": graph_plan_fragment,
    }
    return UnifiedRetrievalSelection(
        durable_objects=reordered_durable,
        conflict_queue=reordered_conflicts,
        midterm_packets=reordered_midterm,
        graph_selection=graph_selection,
        query_plan=query_plan,
    )


__all__ = [
    "UnifiedGraphSelectionInput",
    "UnifiedRetrievalSelection",
    "build_unified_retrieval_selection",
]
