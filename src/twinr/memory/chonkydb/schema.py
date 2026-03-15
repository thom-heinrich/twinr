from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Mapping

from twinr.memory.chonkydb.models import JsonDict
from twinr.text_utils import is_valid_identifier_namespace, is_valid_namespaced_identifier

TWINR_GRAPH_SCHEMA_NAME = "twinr_graph"
TWINR_GRAPH_SCHEMA_VERSION = 2

TWINR_GRAPH_EDGE_TYPES_BY_NAMESPACE: dict[str, tuple[str, ...]] = {
    "social": (
        "social_related_to",
        "social_related_to_user",
    ),
    "general": (
        "general_alias_of",
        "general_has_contact_method",
        "general_related_to",
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
        "user_plans",
        "user_prefers",
        "user_avoids",
        "user_engages_with",
    ),
}
TWINR_GRAPH_ALLOWED_EDGE_TYPES = frozenset(
    edge_type
    for edge_types in TWINR_GRAPH_EDGE_TYPES_BY_NAMESPACE.values()
    for edge_type in edge_types
)
TWINR_GRAPH_EDGE_STATUSES = frozenset(("active", "uncertain", "superseded", "invalid"))
TWINR_GRAPH_NODE_STATUSES = frozenset(("active", "inactive", "merged", "invalid"))

_LEGACY_EDGE_TYPE_MAP: dict[str, tuple[str, dict[str, object]]] = {
    "social_family_of": ("social_related_to_user", {"relation": "family"}),
    "social_friend_of": ("social_related_to_user", {"relation": "friend"}),
    "social_supports_user_as": ("social_related_to_user", {}),
    "general_carries_brand": ("general_related_to", {"relation": "carries"}),
    "general_sells": ("general_related_to", {"relation": "sells"}),
    "user_prefers_brand": ("user_prefers", {"preference_mode": "preferred_brand"}),
    "user_usually_buys_at": ("user_prefers", {"preference_mode": "usual_source"}),
    "user_likes": ("user_prefers", {}),
    "user_dislikes": ("user_avoids", {}),
}


def normalize_graph_edge_type(
    edge_type: str,
    attributes: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    clean_edge_type = str(edge_type or "").strip()
    normalized_attributes = dict(attributes or {})
    if clean_edge_type in _LEGACY_EDGE_TYPE_MAP:
        canonical_edge_type, defaults = _LEGACY_EDGE_TYPE_MAP[clean_edge_type]
        for key, value in defaults.items():
            normalized_attributes.setdefault(key, value)
        return canonical_edge_type, normalized_attributes
    return clean_edge_type, normalized_attributes


def graph_edge_namespace(edge_type: str) -> str:
    normalized_edge_type, _attributes = normalize_graph_edge_type(edge_type, None)
    prefix, _, _rest = normalized_edge_type.partition("_")
    return prefix


def is_allowed_graph_edge_type(edge_type: str) -> bool:
    normalized_edge_type, _attributes = normalize_graph_edge_type(edge_type, None)
    return normalized_edge_type in TWINR_GRAPH_ALLOWED_EDGE_TYPES


# AUDIT-FIX(#3): Reject non-JSON-safe metadata/attributes early and freeze them to avoid later
# persistence failures plus post-validation mutation through shared dict/list references.
def _freeze_json_value(value: object, *, field_name: str) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain NaN or infinite float values.")
        return value
    if isinstance(value, Mapping):
        frozen_mapping: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} keys must be strings.")
            frozen_mapping[key] = _freeze_json_value(item, field_name=f"{field_name}.{key}")
        return MappingProxyType(frozen_mapping)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"{field_name} must contain only JSON-serializable values.")


def _json_value_to_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value_to_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value_to_payload(item) for item in value]
    return value


def _drop_none(payload: JsonDict) -> JsonDict:
    return {key: value for key, value in payload.items() if value is not None}


def _require_mapping_payload(payload: object, *, field_name: str = "payload") -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return payload


def _require_string(
    value: object,
    *,
    field_name: str,
    blank_message: str | None = None,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    clean_value = value.strip()
    if not clean_value:
        raise ValueError(blank_message or f"{field_name} cannot be blank.")
    return clean_value


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name=field_name)


# AUDIT-FIX(#1): Parse incoming sequences strictly so malformed aliases/nodes/edges are rejected
# instead of being silently discarded during deserialization.
def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple of strings.")
    return tuple(
        _require_string(
            item,
            field_name=f"{field_name}[{index}]",
            blank_message=f"{field_name} cannot contain blank values.",
        )
        for index, item in enumerate(value)
    )


def _mapping_or_none(value: object, *, field_name: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    frozen_value = _freeze_json_value(value, field_name=field_name)
    if not isinstance(frozen_value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return frozen_value


def _mapping_tuple(value: object, *, field_name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple.")
    normalized_items: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        normalized_items.append(_require_mapping_payload(item, field_name=f"{field_name}[{index}]"))
    return tuple(normalized_items)


def _object_tuple(value: object, *, field_name: str, expected_type: type[object]) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple.")
    normalized_items: list[object] = []
    for index, item in enumerate(value):
        if not isinstance(item, expected_type):
            raise ValueError(f"{field_name}[{index}] must be a {expected_type.__name__}.")
        normalized_items.append(item)
    return tuple(normalized_items)


# AUDIT-FIX(#4): Parse booleans strictly so payloads such as "false" or "0" never become True.
def _coerce_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"{field_name} must be a boolean or one of: true, false, 1, 0.")


def _coerce_probability(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.") from exc
    if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return numeric_value


def _coerce_schema_version(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("schema.version must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("schema.version must be an integer.")
        return int(value)
    if isinstance(value, str):
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("schema.version cannot be blank.")
        if clean_value.isdigit():
            return int(clean_value)
    raise ValueError("schema.version must be an integer.")


# AUDIT-FIX(#6): Enforce ISO-8601 temporal fields and explicit timezone offsets for datetime
# values so DST and ordering bugs do not leak into persistence or scheduling layers.
def _parse_temporal_value(
    value: str,
    *,
    field_name: str,
    allow_date: bool,
) -> date | datetime:
    if allow_date:
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    try:
        parsed_value = datetime.fromisoformat(value)
    except ValueError as exc:
        suffix = "date or timezone-aware datetime" if allow_date else "timezone-aware datetime"
        raise ValueError(f"{field_name} must be an ISO-8601 {suffix}.") from exc
    if parsed_value.tzinfo is None or parsed_value.utcoffset() is None:
        raise ValueError(f"{field_name} datetime values must include an explicit timezone offset.")
    return parsed_value


def _normalize_temporal_value(
    value: object,
    *,
    field_name: str,
    allow_date: bool,
) -> str | None:
    if value is None:
        return None
    clean_value = _require_string(value, field_name=field_name)
    parsed_value = _parse_temporal_value(clean_value, field_name=field_name, allow_date=allow_date)
    return parsed_value.isoformat()


def _validate_temporal_bounds(
    start_value: str | None,
    end_value: str | None,
    *,
    start_field: str,
    end_field: str,
    allow_date: bool,
) -> None:
    if start_value is None or end_value is None:
        return
    parsed_start = _parse_temporal_value(start_value, field_name=start_field, allow_date=allow_date)
    parsed_end = _parse_temporal_value(end_value, field_name=end_field, allow_date=allow_date)
    if type(parsed_start) is not type(parsed_end):
        raise ValueError(
            f"{start_field} and {end_field} must both be dates or both be timezone-aware datetimes."
        )
    if parsed_start > parsed_end:
        raise ValueError(f"{start_field} must be less than or equal to {end_field}.")


def _validate_node_id(node_id: str, *, field_name: str) -> None:
    if not is_valid_namespaced_identifier(node_id):
        raise ValueError(
            f"{field_name} must look like '<type>:<stable-id>' using lowercase letters, digits, '.', '_', '-', or ':'."
        )


def _validate_node_type(node_type: str) -> None:
    if not is_valid_identifier_namespace(node_type):
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
        # AUDIT-FIX(#2): Make the frozen dataclass actually stable by copying user-provided
        # mutable containers into immutable internal representations before validation.
        node_id = _require_string(self.node_id, field_name="node_id")
        node_type = _require_string(self.node_type, field_name="node_type")
        label = _require_string(self.label, field_name="label", blank_message="label is required.")
        aliases = _string_tuple(self.aliases, field_name="aliases")
        attributes = _mapping_or_none(self.attributes, field_name="attributes")
        status = _require_string(self.status, field_name="status")
        graph_ref = _optional_string(self.graph_ref, field_name="graph_ref")

        _validate_node_id(node_id, field_name="node_id")
        _validate_node_type(node_type)
        _validate_status(status, allowed=TWINR_GRAPH_NODE_STATUSES, field_name="status")
        if not node_id.startswith(f"{node_type}:"):
            raise ValueError("node_id must start with '<node_type>:'.")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "node_type", node_type)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "graph_ref", graph_ref)

    def to_payload(self) -> JsonDict:
        attributes_payload = _json_value_to_payload(self.attributes) if self.attributes is not None else None  # AUDIT-FIX(#3): Rehydrate immutable JSON values into standard payload containers.
        return _drop_none(
            {
                "id": self.node_id,
                "type": self.node_type,
                "label": self.label,
                "aliases": list(self.aliases) if self.aliases else None,
                "attributes": attributes_payload,
                "status": self.status,
                "graph_ref": self.graph_ref,
            }
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TwinrGraphNodeV1":
        payload = _require_mapping_payload(payload)  # AUDIT-FIX(#1): Reject malformed top-level payloads instead of letting AttributeError leak or silently skipping data.
        return cls(
            node_id=payload.get("id", ""),
            node_type=payload.get("type", ""),
            label=payload.get("label", ""),
            aliases=payload.get("aliases"),
            attributes=payload.get("attributes"),
            status=payload.get("status", "active"),
            graph_ref=payload.get("graph_ref"),
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
        source_node_id = _require_string(self.source_node_id, field_name="source_node_id")
        target_node_id = _require_string(self.target_node_id, field_name="target_node_id")
        edge_type = _require_string(self.edge_type, field_name="edge_type")
        status = _require_string(self.status, field_name="status")
        confidence = _coerce_probability(self.confidence, field_name="confidence")
        confirmed_by_user = _coerce_bool(self.confirmed_by_user, field_name="confirmed_by_user")  # AUDIT-FIX(#4): Strict boolean parsing prevents string "false" from becoming True.
        origin = _optional_string(self.origin, field_name="origin")
        valid_from = _normalize_temporal_value(self.valid_from, field_name="valid_from", allow_date=True)
        valid_to = _normalize_temporal_value(self.valid_to, field_name="valid_to", allow_date=True)
        raw_attributes = _mapping_or_none(self.attributes, field_name="attributes")

        # AUDIT-FIX(#7): Canonicalize legacy edge types at construction time so in-memory graph
        # objects are stable before serialization, equality checks, or caching.
        normalized_edge_type, normalized_attributes = normalize_graph_edge_type(edge_type, raw_attributes)
        attributes = _mapping_or_none(normalized_attributes or None, field_name="attributes")

        _validate_node_id(source_node_id, field_name="source_node_id")
        _validate_node_id(target_node_id, field_name="target_node_id")
        if not is_allowed_graph_edge_type(normalized_edge_type):
            raise ValueError(
                f"edge_type '{normalized_edge_type}' is not part of Twinr graph schema v{TWINR_GRAPH_SCHEMA_VERSION}."
            )
        _validate_status(status, allowed=TWINR_GRAPH_EDGE_STATUSES, field_name="status")
        _validate_temporal_bounds(
            valid_from,
            valid_to,
            start_field="valid_from",
            end_field="valid_to",
            allow_date=True,
        )

        object.__setattr__(self, "source_node_id", source_node_id)
        object.__setattr__(self, "edge_type", normalized_edge_type)
        object.__setattr__(self, "target_node_id", target_node_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "confirmed_by_user", confirmed_by_user)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "valid_from", valid_from)
        object.__setattr__(self, "valid_to", valid_to)
        object.__setattr__(self, "attributes", attributes)

    def to_payload(self) -> JsonDict:
        normalized_edge_type, normalized_attributes = normalize_graph_edge_type(self.edge_type, self.attributes)
        attributes_payload = _json_value_to_payload(normalized_attributes) if normalized_attributes else None  # AUDIT-FIX(#3): Rehydrate immutable JSON values into standard payload containers.
        return _drop_none(
            {
                "source": self.source_node_id,
                "type": normalized_edge_type,
                "target": self.target_node_id,
                "status": self.status,
                "confidence": self.confidence,
                "confirmed_by_user": self.confirmed_by_user,
                "origin": self.origin,
                "valid_from": self.valid_from,
                "valid_to": self.valid_to,
                "attributes": attributes_payload,
            }
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TwinrGraphEdgeV1":
        payload = _require_mapping_payload(payload)  # AUDIT-FIX(#1): Reject malformed top-level payloads instead of letting AttributeError leak or silently skipping data.
        return cls(
            source_node_id=payload.get("source", ""),
            edge_type=payload.get("type", ""),
            target_node_id=payload.get("target", ""),
            status=payload.get("status", "active"),
            confidence=payload.get("confidence"),
            confirmed_by_user=payload.get("confirmed_by_user", False),
            origin=payload.get("origin"),
            valid_from=payload.get("valid_from"),
            valid_to=payload.get("valid_to"),
            attributes=payload.get("attributes"),
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
        subject_node_id = _require_string(self.subject_node_id, field_name="subject_node_id")
        nodes = _object_tuple(self.nodes, field_name="nodes", expected_type=TwinrGraphNodeV1)
        edges = _object_tuple(self.edges, field_name="edges", expected_type=TwinrGraphEdgeV1)
        graph_id = _optional_string(self.graph_id, field_name="graph_id")
        created_at = _normalize_temporal_value(self.created_at, field_name="created_at", allow_date=False)
        updated_at = _normalize_temporal_value(self.updated_at, field_name="updated_at", allow_date=False)
        metadata = _mapping_or_none(self.metadata, field_name="metadata")

        _validate_node_id(subject_node_id, field_name="subject_node_id")
        _validate_temporal_bounds(
            created_at,
            updated_at,
            start_field="created_at",
            end_field="updated_at",
            allow_date=False,
        )

        seen: set[str] = set()
        for node in nodes:
            if node.node_id in seen:
                raise ValueError(f"Duplicate graph node id: {node.node_id}")
            seen.add(node.node_id)
        if subject_node_id not in seen:
            raise ValueError("subject_node_id must exist in nodes.")
        for edge in edges:
            if edge.source_node_id not in seen:
                raise ValueError(f"Edge source is missing from nodes: {edge.source_node_id}")
            if edge.target_node_id not in seen:
                raise ValueError(f"Edge target is missing from nodes: {edge.target_node_id}")

        object.__setattr__(self, "subject_node_id", subject_node_id)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "graph_id", graph_id)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "metadata", metadata)

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
        metadata_payload = _json_value_to_payload(self.metadata) if self.metadata is not None else None  # AUDIT-FIX(#3): Rehydrate immutable JSON values into standard payload containers.
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
                "metadata": metadata_payload,
            }
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TwinrGraphDocumentV1":
        payload = _require_mapping_payload(payload)  # AUDIT-FIX(#1): Reject malformed top-level payloads instead of letting AttributeError leak or silently skipping data.
        nodes = payload.get("nodes", ())
        edges = payload.get("edges", ())
        metadata = payload.get("metadata")
        schema = payload.get("schema")

        # AUDIT-FIX(#5): Reject foreign/corrupt schema envelopes instead of silently loading
        # them as Twinr graph documents and risking cross-document memory corruption.
        version = None
        if schema is not None:
            schema = _require_mapping_payload(schema, field_name="schema")
            raw_name = schema.get("name")
            if raw_name is not None:
                schema_name = _require_string(raw_name, field_name="schema.name")
                if schema_name != TWINR_GRAPH_SCHEMA_NAME:
                    raise ValueError(
                        f"Unsupported Twinr graph schema name: {schema_name}. "
                        f"Expected {TWINR_GRAPH_SCHEMA_NAME}."
                    )
            version = _coerce_schema_version(schema.get("version"))
        if version is not None and version not in {1, TWINR_GRAPH_SCHEMA_VERSION}:
            raise ValueError(
                f"Unsupported Twinr graph schema version: {version}. "
                f"Expected 1 or {TWINR_GRAPH_SCHEMA_VERSION}."
            )

        return cls(
            subject_node_id=payload.get("subject_node_id", ""),
            nodes=tuple(TwinrGraphNodeV1.from_payload(item) for item in _mapping_tuple(nodes, field_name="nodes")),
            edges=tuple(TwinrGraphEdgeV1.from_payload(item) for item in _mapping_tuple(edges, field_name="edges")),
            graph_id=payload.get("graph_id"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            metadata=metadata,
        )


__all__ = [
    "TWINR_GRAPH_ALLOWED_EDGE_TYPES",
    "TWINR_GRAPH_EDGE_STATUSES",
    "TWINR_GRAPH_NODE_STATUSES",
    "TWINR_GRAPH_SCHEMA_NAME",
    "TWINR_GRAPH_SCHEMA_VERSION",
    "TwinrGraphDocumentV1",
    "TwinrGraphEdgeV1",
    "TwinrGraphNodeV1",
    "graph_edge_namespace",
    "is_allowed_graph_edge_type",
    "normalize_graph_edge_type",
]