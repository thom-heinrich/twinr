from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping

from twinr.memory.chonkydb.models import JsonDict

TWINR_GRAPH_SCHEMA_NAME = "twinr_graph"
TWINR_GRAPH_SCHEMA_VERSION = 1

TWINR_GRAPH_EDGE_TYPES_BY_NAMESPACE: dict[str, tuple[str, ...]] = {
    "social": (
        "social_family_of",
        "social_friend_of",
        "social_supports_user_as",
    ),
    "general": (
        "general_alias_of",
        "general_carries_brand",
        "general_has_contact_method",
        "general_related_to",
        "general_sells",
    ),
    "temporal": (
        "temporal_occurs_on",
        "temporal_usually_happens_in",
        "temporal_valid_from",
        "temporal_valid_to",
    ),
    "spatial": (
        "spatial_located_in",
        "spatial_near",
    ),
    "user": (
        "user_dislikes",
        "user_likes",
        "user_plans",
        "user_prefers_brand",
        "user_usually_buys_at",
    ),
}
TWINR_GRAPH_ALLOWED_EDGE_TYPES = frozenset(
    edge_type
    for edge_types in TWINR_GRAPH_EDGE_TYPES_BY_NAMESPACE.values()
    for edge_type in edge_types
)
TWINR_GRAPH_EDGE_STATUSES = frozenset(("active", "uncertain", "superseded", "invalid"))
TWINR_GRAPH_NODE_STATUSES = frozenset(("active", "inactive", "merged", "invalid"))

_NODE_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_NODE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*:[a-z0-9][a-z0-9._:-]*$")


def graph_edge_namespace(edge_type: str) -> str:
    prefix, _, _rest = edge_type.partition("_")
    return prefix


def is_allowed_graph_edge_type(edge_type: str) -> bool:
    return edge_type in TWINR_GRAPH_ALLOWED_EDGE_TYPES


def _drop_none(payload: JsonDict) -> JsonDict:
    return {key: value for key, value in payload.items() if value is not None}


def _validate_node_id(node_id: str, *, field_name: str) -> None:
    if not _NODE_ID_RE.fullmatch(node_id):
        raise ValueError(
            f"{field_name} must look like '<type>:<stable-id>' using lowercase letters, digits, '.', '_', '-', or ':'."
        )


def _validate_node_type(node_type: str) -> None:
    if not _NODE_TYPE_RE.fullmatch(node_type):
        raise ValueError("node_type must use lowercase letters, digits, and underscores only.")


def _validate_status(status: str, *, allowed: frozenset[str], field_name: str) -> None:
    if status not in allowed:
        raise ValueError(f"{field_name} must be one of: {', '.join(sorted(allowed))}.")


@dataclass(frozen=True, slots=True)
class TwinrGraphNodeV1:
    node_id: str
    node_type: str
    label: str
    aliases: tuple[str, ...] = ()
    attributes: Mapping[str, object] | None = None
    status: str = "active"
    graph_ref: str | None = None

    def __post_init__(self) -> None:
        _validate_node_id(self.node_id, field_name="node_id")
        _validate_node_type(self.node_type)
        if not self.label.strip():
            raise ValueError("label is required.")
        _validate_status(self.status, allowed=TWINR_GRAPH_NODE_STATUSES, field_name="status")
        if self.graph_ref is not None and not self.graph_ref.strip():
            raise ValueError("graph_ref cannot be blank when provided.")
        for alias in self.aliases:
            if not alias.strip():
                raise ValueError("aliases cannot contain blank values.")
        if not self.node_id.startswith(f"{self.node_type}:"):
            raise ValueError("node_id must start with '<node_type>:'.")

    def to_payload(self) -> JsonDict:
        return _drop_none(
            {
                "id": self.node_id,
                "type": self.node_type,
                "label": self.label,
                "aliases": list(self.aliases) if self.aliases else None,
                "attributes": dict(self.attributes) if self.attributes is not None else None,
                "status": self.status,
                "graph_ref": self.graph_ref,
            }
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TwinrGraphNodeV1":
        aliases = payload.get("aliases")
        attributes = payload.get("attributes")
        return cls(
            node_id=str(payload.get("id", "")),
            node_type=str(payload.get("type", "")),
            label=str(payload.get("label", "")),
            aliases=tuple(str(item) for item in aliases if isinstance(item, str)) if isinstance(aliases, list) else (),
            attributes=dict(attributes) if isinstance(attributes, Mapping) else None,
            status=str(payload.get("status", "active")),
            graph_ref=str(payload["graph_ref"]) if payload.get("graph_ref") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class TwinrGraphEdgeV1:
    source_node_id: str
    edge_type: str
    target_node_id: str
    status: str = "active"
    confidence: float | None = None
    confirmed_by_user: bool = False
    origin: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    attributes: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _validate_node_id(self.source_node_id, field_name="source_node_id")
        _validate_node_id(self.target_node_id, field_name="target_node_id")
        if not is_allowed_graph_edge_type(self.edge_type):
            raise ValueError(f"edge_type '{self.edge_type}' is not part of Twinr graph schema v1.")
        _validate_status(self.status, allowed=TWINR_GRAPH_EDGE_STATUSES, field_name="status")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        if self.origin is not None and not self.origin.strip():
            raise ValueError("origin cannot be blank when provided.")

    def to_payload(self) -> JsonDict:
        return _drop_none(
            {
                "source": self.source_node_id,
                "type": self.edge_type,
                "target": self.target_node_id,
                "status": self.status,
                "confidence": self.confidence,
                "confirmed_by_user": self.confirmed_by_user,
                "origin": self.origin,
                "valid_from": self.valid_from,
                "valid_to": self.valid_to,
                "attributes": dict(self.attributes) if self.attributes is not None else None,
            }
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TwinrGraphEdgeV1":
        attributes = payload.get("attributes")
        confidence = payload.get("confidence")
        return cls(
            source_node_id=str(payload.get("source", "")),
            edge_type=str(payload.get("type", "")),
            target_node_id=str(payload.get("target", "")),
            status=str(payload.get("status", "active")),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            confirmed_by_user=bool(payload.get("confirmed_by_user", False)),
            origin=str(payload["origin"]) if payload.get("origin") is not None else None,
            valid_from=str(payload["valid_from"]) if payload.get("valid_from") is not None else None,
            valid_to=str(payload["valid_to"]) if payload.get("valid_to") is not None else None,
            attributes=dict(attributes) if isinstance(attributes, Mapping) else None,
        )


@dataclass(frozen=True, slots=True)
class TwinrGraphDocumentV1:
    subject_node_id: str
    nodes: tuple[TwinrGraphNodeV1, ...]
    edges: tuple[TwinrGraphEdgeV1, ...]
    graph_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _validate_node_id(self.subject_node_id, field_name="subject_node_id")
        seen: set[str] = set()
        for node in self.nodes:
            if node.node_id in seen:
                raise ValueError(f"Duplicate graph node id: {node.node_id}")
            seen.add(node.node_id)
        if self.subject_node_id not in seen:
            raise ValueError("subject_node_id must exist in nodes.")
        for edge in self.edges:
            if edge.source_node_id not in seen:
                raise ValueError(f"Edge source is missing from nodes: {edge.source_node_id}")
            if edge.target_node_id not in seen:
                raise ValueError(f"Edge target is missing from nodes: {edge.target_node_id}")
        if self.graph_id is not None and not self.graph_id.strip():
            raise ValueError("graph_id cannot be blank when provided.")

    @property
    def schema_version(self) -> int:
        return TWINR_GRAPH_SCHEMA_VERSION

    @property
    def schema_name(self) -> str:
        return TWINR_GRAPH_SCHEMA_NAME

    def node(self, node_id: str) -> TwinrGraphNodeV1:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def to_payload(self) -> JsonDict:
        return _drop_none(
            {
                "schema": {
                    "name": TWINR_GRAPH_SCHEMA_NAME,
                    "version": TWINR_GRAPH_SCHEMA_VERSION,
                },
                "graph_id": self.graph_id,
                "subject_node_id": self.subject_node_id,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "nodes": [node.to_payload() for node in self.nodes],
                "edges": [edge.to_payload() for edge in self.edges],
                "metadata": dict(self.metadata) if self.metadata is not None else None,
            }
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TwinrGraphDocumentV1":
        nodes = payload.get("nodes", ())
        edges = payload.get("edges", ())
        metadata = payload.get("metadata")
        schema = payload.get("schema")
        version = None
        if isinstance(schema, Mapping):
            raw_version = schema.get("version")
            version = int(raw_version) if isinstance(raw_version, (int, float)) else None
        if version is not None and version != TWINR_GRAPH_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Twinr graph schema version: {version}. "
                f"Expected {TWINR_GRAPH_SCHEMA_VERSION}."
            )
        return cls(
            subject_node_id=str(payload.get("subject_node_id", "")),
            nodes=tuple(TwinrGraphNodeV1.from_payload(item) for item in nodes if isinstance(item, Mapping)),
            edges=tuple(TwinrGraphEdgeV1.from_payload(item) for item in edges if isinstance(item, Mapping)),
            graph_id=str(payload["graph_id"]) if payload.get("graph_id") is not None else None,
            created_at=str(payload["created_at"]) if payload.get("created_at") is not None else None,
            updated_at=str(payload["updated_at"]) if payload.get("updated_at") is not None else None,
            metadata=dict(metadata) if isinstance(metadata, Mapping) else None,
        )
