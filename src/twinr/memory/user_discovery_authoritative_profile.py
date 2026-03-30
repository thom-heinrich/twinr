"""Read authoritative long-term profile facts for discovery topic completion.

This module keeps the discovery state machine separate from the concrete
long-term store layouts. It answers one narrow question: does authoritative
memory already cover a discovery topic strongly enough that Twinr should stop
asking it again?
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.chonkydb.schema import TwinrGraphDocumentV1
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1
from twinr.memory.longterm.runtime.live_object_selectors import (
    select_discovery_basics_objects,
    select_discovery_companion_style_objects,
)
from twinr.memory.longterm.storage.store import LongTermStructuredStore

_ACTIVE_MEMORY_STATUSES = frozenset({"active"})
_BASICS_GRAPH_CATEGORIES = frozenset({"preferred_name"})
_COMPANION_STYLE_GRAPH_CATEGORIES = frozenset(
    {
        "address_preference",
        "address_style",
        "answer_style",
        "communication_style",
        "humor",
        "initiative",
        "tone",
        "verbosity",
    }
)
_COMPANION_STYLE_PREFERENCE_TYPES = frozenset({"initiative", "verbosity", "humor"})
_COMPANION_STYLE_PREDICATES = frozenset(
    {
        "user_prefers_answer_style",
        "user_prefers_small_follow_up_when_helpful",
    }
)
_COMPANION_STYLE_FEEDBACK_TARGETS = frozenset({"humor"})
def _normalize_token(value: object | None, *, limit: int = 80) -> str:
    text = " ".join(str(value or "").split()).strip().lower().replace("-", "_").replace(" ", "_")
    if len(text) <= limit:
        return text
    return text[:limit]


def _attributes(memory_object: LongTermMemoryObjectV1) -> Mapping[str, object]:
    attributes = memory_object.attributes
    if isinstance(attributes, Mapping):
        return attributes
    return {}


@dataclass(frozen=True, slots=True)
class UserDiscoveryAuthoritativeCoverage:
    """Describe which discovery topics are already backed by authoritative memory."""

    covered_topics: frozenset[str] = frozenset()

    def covers(self, topic_id: str) -> bool:
        return _normalize_token(topic_id) in self.covered_topics


@dataclass(slots=True)
class UserDiscoveryAuthoritativeProfileReader:
    """Project authoritative long-term facts into discovery-topic coverage."""

    graph_store: TwinrPersonalGraphStore | None = None
    object_store: LongTermStructuredStore | None = None
    user_node_id: str = "user:main"

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "UserDiscoveryAuthoritativeProfileReader":
        graph_store = TwinrPersonalGraphStore.from_config(config)
        return cls(
            graph_store=graph_store,
            object_store=LongTermStructuredStore.from_config(config),
            user_node_id=graph_store.user_node_id,
        )

    def load(self) -> UserDiscoveryAuthoritativeCoverage:
        covered_topics: set[str] = set()
        basics_objects = tuple(select_discovery_basics_objects(self.object_store))
        companion_style_objects = tuple(select_discovery_companion_style_objects(self.object_store))
        basics_covered_by_objects = self._objects_cover_basics(basics_objects)
        companion_style_covered_by_objects = self._objects_cover_companion_style(companion_style_objects)
        document = None
        if (not basics_covered_by_objects or not companion_style_covered_by_objects) and self.graph_store is not None:
            document = self.graph_store.load_document()

        if basics_covered_by_objects or self._graph_covers_basics(document):
            covered_topics.add("basics")
        if companion_style_covered_by_objects or self._graph_covers_companion_style(document):
            covered_topics.add("companion_style")
        return UserDiscoveryAuthoritativeCoverage(covered_topics=frozenset(covered_topics))

    def _objects_cover_basics(self, objects: tuple[LongTermMemoryObjectV1, ...]) -> bool:
        for memory_object in self._iter_authoritative_user_objects(objects):
            attributes = _attributes(memory_object)
            if _normalize_token(attributes.get("preference_type")) == "name":
                return True
            if _normalize_token(attributes.get("predicate")) in {"prefers_name", "preferred_name"}:
                return True
        return False

    def _objects_cover_companion_style(self, objects: tuple[LongTermMemoryObjectV1, ...]) -> bool:
        for memory_object in self._iter_authoritative_user_objects(objects):
            attributes = _attributes(memory_object)
            if _normalize_token(attributes.get("preference_type")) in _COMPANION_STYLE_PREFERENCE_TYPES:
                return True
            if _normalize_token(attributes.get("style_dimension")) in _COMPANION_STYLE_PREFERENCE_TYPES:
                return True
            if _normalize_token(attributes.get("feedback_target")) in _COMPANION_STYLE_FEEDBACK_TARGETS:
                return True
            if _normalize_token(attributes.get("predicate")) in _COMPANION_STYLE_PREDICATES:
                return True
        return False

    def _graph_covers_basics(self, document: TwinrGraphDocumentV1 | None) -> bool:
        return self._graph_has_preference_category(document, categories=_BASICS_GRAPH_CATEGORIES)

    def _graph_covers_companion_style(self, document: TwinrGraphDocumentV1 | None) -> bool:
        return self._graph_has_preference_category(document, categories=_COMPANION_STYLE_GRAPH_CATEGORIES)

    def _graph_has_preference_category(
        self,
        document: TwinrGraphDocumentV1 | None,
        *,
        categories: frozenset[str],
    ) -> bool:
        if document is None:
            return False
        nodes = {node.node_id: node for node in document.nodes}
        for edge in document.edges:
            if edge.source_node_id != self.user_node_id or edge.status != "active":
                continue
            if edge.edge_type not in {"user_prefers", "user_avoids", "user_engages_with"}:
                continue
            category = _normalize_token((edge.attributes or {}).get("category"))
            if category in categories:
                return True
            target = nodes.get(edge.target_node_id)
            if target is not None and _normalize_token(target.node_type) in categories:
                return True
        return False

    def _iter_authoritative_user_objects(
        self,
        objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        selected: list[LongTermMemoryObjectV1] = []
        for memory_object in objects:
            if memory_object.status not in _ACTIVE_MEMORY_STATUSES:
                continue
            if memory_object.kind == "episode":
                continue
            subject_ref = _normalize_token(_attributes(memory_object).get("subject_ref"))
            if subject_ref and subject_ref != self.user_node_id:
                continue
            selected.append(memory_object)
        return tuple(selected)


__all__ = [
    "UserDiscoveryAuthoritativeCoverage",
    "UserDiscoveryAuthoritativeProfileReader",
]
