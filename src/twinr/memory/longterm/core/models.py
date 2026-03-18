"""Define the canonical long-term memory schemas used across Twinr.

The dataclasses in this module form the versioned payload contract shared by
ingestion, reasoning, retrieval, storage, and evaluation code. Constructors
validate and normalize input eagerly so persisted long-term memory state stays
canonical and timezone-safe.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from types import MappingProxyType
from typing import Callable, Mapping, TypeVar, cast

from twinr.memory.longterm.core.ontology import (
    LONGTERM_MEMORY_SENSITIVITIES,
    normalize_memory_kind,
    normalize_memory_sensitivity,
)


LONGTERM_MEMORY_OBJECT_SCHEMA = "twinr_memory_object"
LONGTERM_MEMORY_OBJECT_VERSION = 1
LONGTERM_MEMORY_CONFLICT_SCHEMA = "twinr_memory_conflict"
LONGTERM_MEMORY_CONFLICT_VERSION = 1
LONGTERM_TURN_EXTRACTION_SCHEMA = "twinr_turn_extraction"
LONGTERM_TURN_EXTRACTION_VERSION = 1
LONGTERM_CONSOLIDATION_SCHEMA = "twinr_memory_consolidation"
LONGTERM_CONSOLIDATION_VERSION = 1
LONGTERM_REFLECTION_SCHEMA = "twinr_memory_reflection"
LONGTERM_REFLECTION_VERSION = 1
LONGTERM_MIDTERM_PACKET_SCHEMA = "twinr_memory_midterm_packet"
LONGTERM_MIDTERM_PACKET_VERSION = 1
LONGTERM_PROACTIVE_PLAN_SCHEMA = "twinr_memory_proactive_plan"
LONGTERM_PROACTIVE_PLAN_VERSION = 1
LONGTERM_RETENTION_SCHEMA = "twinr_memory_retention"
LONGTERM_RETENTION_VERSION = 1
LONGTERM_MULTIMODAL_EVIDENCE_SCHEMA = "twinr_multimodal_evidence"
LONGTERM_MULTIMODAL_EVIDENCE_VERSION = 1
LONGTERM_CONFLICT_QUEUE_SCHEMA = "twinr_memory_conflict_queue"
LONGTERM_CONFLICT_QUEUE_VERSION = 1
LONGTERM_CONFLICT_RESOLUTION_SCHEMA = "twinr_memory_conflict_resolution"
LONGTERM_CONFLICT_RESOLUTION_VERSION = 1
LONGTERM_MEMORY_REVIEW_SCHEMA = "twinr_memory_review"
LONGTERM_MEMORY_REVIEW_VERSION = 1
LONGTERM_MEMORY_MUTATION_SCHEMA = "twinr_memory_mutation"
LONGTERM_MEMORY_MUTATION_VERSION = 1
LONGTERM_MEMORY_MUTATION_ACTIONS = frozenset({"confirm", "invalidate", "delete"})

LONGTERM_MEMORY_STATUSES = frozenset(
    {
        "candidate",
        "active",
        "uncertain",
        "superseded",
        "invalid",
        "expired",
    }
)
LONGTERM_MEMORY_SENSITIVITY = LONGTERM_MEMORY_SENSITIVITIES


_T = TypeVar("_T")
_MISSING = object()


def _utcnow() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _normalize_text(value: object | None) -> str:
    """Collapse arbitrary input into a normalized single-line string."""

    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _drop_none(payload: dict[str, object]) -> dict[str, object]:
    """Return a payload copy without keys whose value is ``None``."""

    return {key: value for key, value in payload.items() if value is not None}


# AUDIT-FIX(#3): Freeze nested JSON-like state so frozen dataclasses stay effectively immutable.
def _freeze_jsonish(value: object) -> object:
    """Freeze nested JSON-like structures into immutable equivalents."""

    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("mapping keys must be strings.")
            frozen[key] = _freeze_jsonish(item)
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze_jsonish(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_jsonish(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_jsonish(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_jsonish(item) for item in value)
    return deepcopy(value)


def _thaw_jsonish(value: object) -> object:
    """Convert frozen JSON-like structures back into plain Python containers."""

    if isinstance(value, Mapping):
        return {key: _thaw_jsonish(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_jsonish(item) for item in value]
    if isinstance(value, frozenset):
        return [_thaw_jsonish(item) for item in value]
    return deepcopy(value)


def _mapping_dict(value: Mapping[str, object] | None) -> dict[str, object] | None:
    """Return a mutable dictionary copy from an optional mapping payload."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("value must be a mapping.")
    return cast(dict[str, object], _thaw_jsonish(value))


def _freeze_mapping(value: Mapping[str, object] | None, *, field_name: str) -> Mapping[str, object] | None:
    """Validate and freeze an optional mapping used by a frozen dataclass."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return cast(Mapping[str, object], _freeze_jsonish(dict(value)))


# AUDIT-FIX(#4): Reject silent None/"None"/type coercions at the persistence boundary.
def _require_str(value: object, *, field_name: str) -> str:
    """Require a non-blank string field and return its original value."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    if not _normalize_text(value):
        raise ValueError(f"{field_name} is required.")
    return value


def _optional_str(
    value: object | None,
    *,
    field_name: str,
    blank_to_none: bool = False,
) -> str | None:
    """Validate an optional string field and optionally collapse blanks to ``None``."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string when provided.")
    if not _normalize_text(value):
        if blank_to_none:
            return None
        raise ValueError(f"{field_name} cannot be blank when provided.")
    return value


def _payload_value(payload: Mapping[str, object], key: str) -> object:
    """Return a payload value or the module-level missing sentinel."""

    if key in payload:
        return payload[key]
    return _MISSING


def _payload_required_str(payload: Mapping[str, object], key: str) -> str:
    """Read a required string field from a payload, allowing missing to surface later."""

    value = _payload_value(payload, key)
    if value is _MISSING or value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value


def _payload_optional_str(
    payload: Mapping[str, object],
    key: str,
    *,
    default: str | None = None,
    blank_to_none: bool = False,
) -> str | None:
    """Read and validate an optional string field from a payload mapping."""

    value = _payload_value(payload, key)
    if value is _MISSING or value is None:
        return default
    return _optional_str(value, field_name=key, blank_to_none=blank_to_none)


# AUDIT-FIX(#1): Avoid bool("false") == True when hydrating persisted payloads.
def _coerce_bool(value: object, *, field_name: str, default: bool = False) -> bool:
    """Coerce persisted boolean-like values into a canonical ``bool``."""

    if value is _MISSING or value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    raise ValueError(f"{field_name} must be a boolean.")


def _coerce_float(value: object, *, field_name: str, default: float | None = None) -> float:
    """Coerce persisted numeric input into a floating-point value."""

    if value is _MISSING or value is None:
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required.")
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a float.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            if default is not None:
                return default
            raise ValueError(f"{field_name} is required.")
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a float.") from exc
    raise ValueError(f"{field_name} must be a float.")


def _coerce_int(value: object, *, field_name: str, default: int | None = None) -> int:
    """Coerce persisted numeric input into an integer value."""

    if value is _MISSING or value is None:
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required.")
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            if default is not None:
                return default
            raise ValueError(f"{field_name} is required.")
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer.") from exc
    raise ValueError(f"{field_name} must be an integer.")


# AUDIT-FIX(#6): Guard negative or impossible count fields before they poison queue/review state.
def _coerce_non_negative_int(value: object, *, field_name: str) -> int:
    """Coerce an integer field and reject negative values."""

    coerced = _coerce_int(value, field_name=field_name)
    if coerced < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return coerced


# AUDIT-FIX(#2): Canonicalize naive/aware datetimes to timezone-aware UTC for DST-safe comparisons.
def _coerce_datetime(
    value: object,
    *,
    field_name: str,
    default: datetime | None = None,
) -> datetime:
    """Coerce a datetime field into a timezone-aware UTC timestamp."""

    if value is _MISSING or value is None:
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required.")
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            if default is not None:
                return default
            raise ValueError(f"{field_name} is required.")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 datetime.") from exc
    else:
        raise ValueError(f"{field_name} must be a datetime or ISO-8601 string.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# AUDIT-FIX(#7): Normalize tuple-like fields consistently and reject non-string members early.
def _coerce_str_tuple(
    value: object | None,
    *,
    field_name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    """Coerce a list-like string field into an immutable tuple."""

    if value is None:
        result: tuple[str, ...] = ()
    elif isinstance(value, (list, tuple)):
        items: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"{field_name}[{index}] must be a string.")
            if not _normalize_text(item):
                raise ValueError(f"{field_name}[{index}] cannot be blank.")
            items.append(item)
        result = tuple(items)
    else:
        raise ValueError(f"{field_name} must be a list or tuple of strings.")
    if not result and not allow_empty:
        raise ValueError(f"{field_name} cannot be empty.")
    return result


def _coerce_instance(value: object, field_name: str, cls: type[_T]) -> _T:
    """Require that a value already be an instance of the requested class."""

    if isinstance(value, cls):
        return value
    raise ValueError(f"{field_name} must be a {cls.__name__}.")


def _coerce_tuple(
    value: object | None,
    *,
    field_name: str,
    item_name: str,
    item_coercer: Callable[[object, str], _T],
    allow_empty: bool = True,
) -> tuple[_T, ...]:
    """Coerce a sequence of nested payload items into a validated tuple."""

    if value is None:
        result: tuple[_T, ...] = ()
    elif isinstance(value, (list, tuple)):
        result = tuple(item_coercer(item, f"{field_name}[{index}]") for index, item in enumerate(value))
    else:
        raise ValueError(f"{field_name} must be a list or tuple of {item_name}.")
    if not result and not allow_empty:
        raise ValueError(f"{field_name} cannot be empty.")
    return result


# AUDIT-FIX(#8): Enforce schema/version invariants on every versioned model path.
def _validate_schema_version(
    *,
    schema: object,
    expected_schema: str,
    version: object,
    expected_version: int,
) -> None:
    """Validate the schema and version markers on a versioned payload."""

    if not isinstance(schema, str):
        raise ValueError("schema must be a string.")
    if schema != expected_schema:
        raise ValueError(f"schema must be {expected_schema!r}.")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("version must be an integer.")
    if version != expected_version:
        raise ValueError(f"version must be {expected_version}.")


def _coerce_source_ref(value: object, field_name: str) -> "LongTermSourceRefV1":
    """Coerce a nested source reference from an object or payload mapping."""

    if isinstance(value, LongTermSourceRefV1):
        return value
    if isinstance(value, Mapping):
        return LongTermSourceRefV1.from_payload(value)
    raise ValueError(f"{field_name} must be a LongTermSourceRefV1 or mapping.")


def _coerce_memory_object(value: object, field_name: str) -> "LongTermMemoryObjectV1":
    """Coerce a nested memory object from an object or payload mapping."""

    if isinstance(value, LongTermMemoryObjectV1):
        return value
    if isinstance(value, Mapping):
        return LongTermMemoryObjectV1.from_payload(value)
    raise ValueError(f"{field_name} must be a LongTermMemoryObjectV1 or mapping.")


def _coerce_graph_edge(value: object, field_name: str) -> "LongTermGraphEdgeCandidateV1":
    """Require a graph-edge candidate instance."""

    return _coerce_instance(value, field_name, LongTermGraphEdgeCandidateV1)


def _coerce_conflict(value: object, field_name: str) -> "LongTermMemoryConflictV1":
    """Require a memory-conflict instance."""

    return _coerce_instance(value, field_name, LongTermMemoryConflictV1)


def _coerce_midterm_packet(value: object, field_name: str) -> "LongTermMidtermPacketV1":
    """Coerce a midterm packet from an object or payload mapping."""

    if isinstance(value, LongTermMidtermPacketV1):
        return value
    if isinstance(value, Mapping):
        return LongTermMidtermPacketV1.from_payload(value)
    raise ValueError(f"{field_name} must be a LongTermMidtermPacketV1 or mapping.")


def _coerce_proactive_candidate(value: object, field_name: str) -> "LongTermProactiveCandidateV1":
    """Require a proactive-candidate instance."""

    return _coerce_instance(value, field_name, LongTermProactiveCandidateV1)


def _coerce_conflict_option(value: object, field_name: str) -> "LongTermConflictOptionV1":
    """Require a conflict-option instance."""

    return _coerce_instance(value, field_name, LongTermConflictOptionV1)


@dataclass(frozen=True, slots=True)
class LongTermConversationTurn:
    """Represent one conversation turn queued for long-term ingestion."""

    transcript: str
    response: str
    source: str = "conversation"
    created_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        # AUDIT-FIX(#4): Refuse non-string conversation fields instead of silently stringifying them later.
        if not isinstance(self.transcript, str):
            raise ValueError("transcript must be a string.")
        if not isinstance(self.response, str):
            raise ValueError("response must be a string.")
        object.__setattr__(self, "source", _require_str(self.source, field_name="source"))
        # AUDIT-FIX(#2): Keep direct-construction timestamps comparable with persisted UTC timestamps.
        object.__setattr__(self, "created_at", _coerce_datetime(self.created_at, field_name="created_at"))


@dataclass(frozen=True, slots=True)
class LongTermEnqueueResult:
    """Report whether an async long-term write was accepted."""

    accepted: bool
    pending_count: int
    dropped_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", _coerce_bool(self.accepted, field_name="accepted"))
        # AUDIT-FIX(#6): Prevent impossible negative queue counters.
        object.__setattr__(self, "pending_count", _coerce_non_negative_int(self.pending_count, field_name="pending_count"))
        object.__setattr__(self, "dropped_count", _coerce_non_negative_int(self.dropped_count, field_name="dropped_count"))


@dataclass(frozen=True, slots=True)
class LongTermMultimodalEvidence:
    """Represent one multimodal/device event queued for long-term ingestion."""

    event_name: str
    modality: str
    source: str = "device_event"
    message: str | None = None
    data: Mapping[str, object] | None = None
    created_at: datetime = field(default_factory=_utcnow)
    schema: str = LONGTERM_MULTIMODAL_EVIDENCE_SCHEMA
    version: int = LONGTERM_MULTIMODAL_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_name", _require_str(self.event_name, field_name="event_name"))
        object.__setattr__(self, "modality", _require_str(self.modality, field_name="modality"))
        object.__setattr__(self, "source", _require_str(self.source, field_name="source"))
        object.__setattr__(self, "message", _optional_str(self.message, field_name="message", blank_to_none=True))
        # AUDIT-FIX(#3): Detach caller-owned evidence payloads from frozen dataclass instances.
        object.__setattr__(self, "data", _freeze_mapping(self.data, field_name="data"))
        # AUDIT-FIX(#2): Normalize evidence timestamps to aware UTC.
        object.__setattr__(self, "created_at", _coerce_datetime(self.created_at, field_name="created_at"))
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_MULTIMODAL_EVIDENCE_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_MULTIMODAL_EVIDENCE_VERSION,
        )


@dataclass(frozen=True, slots=True)
class LongTermMemoryContext:
    """Hold memory-derived context fragments for provider prompting."""

    subtext_context: str | None = None
    midterm_context: str | None = None
    durable_context: str | None = None
    episodic_context: str | None = None
    graph_context: str | None = None
    conflict_context: str | None = None

    def system_messages(self) -> tuple[str, ...]:
        """Return non-empty context fragments in provider injection order."""

        messages: list[str] = []
        if self.subtext_context:
            messages.append(self.subtext_context)
        if self.midterm_context:
            messages.append(self.midterm_context)
        if self.durable_context:
            messages.append(self.durable_context)
        if self.episodic_context:
            messages.append(self.episodic_context)
        if self.graph_context:
            messages.append(self.graph_context)
        if self.conflict_context:
            messages.append(self.conflict_context)
        return tuple(messages)


@dataclass(frozen=True, slots=True)
class LongTermSourceRefV1:
    """Describe the provenance metadata for a long-term memory object."""

    source_type: str
    event_ids: tuple[str, ...] = ()
    speaker: str | None = None
    modality: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_type", _require_str(self.source_type, field_name="source_type"))
        # AUDIT-FIX(#7): Accept tuple/list inputs consistently and preserve IDs instead of silently dropping tuple-form input.
        object.__setattr__(self, "event_ids", _coerce_str_tuple(self.event_ids, field_name="event_ids"))
        object.__setattr__(self, "speaker", _optional_str(self.speaker, field_name="speaker"))
        object.__setattr__(self, "modality", _optional_str(self.modality, field_name="modality"))

    def to_payload(self) -> dict[str, object]:
        """Serialize the source reference into a plain JSON-compatible payload."""

        return _drop_none(
            {
                "type": self.source_type,
                "event_ids": list(self.event_ids) if self.event_ids else None,
                "speaker": self.speaker,
                "modality": self.modality,
            }
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "LongTermSourceRefV1":
        """Build a source reference from a persisted payload mapping."""

        # AUDIT-FIX(#4): Treat nulls as missing instead of stringifying them into the literal "None".
        return cls(
            source_type=_payload_required_str(payload, "type"),
            event_ids=_coerce_str_tuple(payload.get("event_ids"), field_name="event_ids"),
            speaker=_payload_optional_str(payload, "speaker"),
            modality=_payload_optional_str(payload, "modality"),
        )


@dataclass(frozen=True, slots=True)
class LongTermGraphEdgeCandidateV1:
    """Represent a candidate graph edge derived from memory reasoning."""

    source_ref: str
    edge_type: str
    target_ref: str
    confidence: float = 0.5
    confirmed_by_user: bool = False
    valid_from: str | None = None
    valid_to: str | None = None
    attributes: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ref", _require_str(self.source_ref, field_name="source_ref"))
        object.__setattr__(self, "edge_type", _require_str(self.edge_type, field_name="edge_type"))
        object.__setattr__(self, "target_ref", _require_str(self.target_ref, field_name="target_ref"))
        object.__setattr__(self, "confidence", _coerce_float(self.confidence, field_name="confidence"))
        object.__setattr__(self, "confirmed_by_user", _coerce_bool(self.confirmed_by_user, field_name="confirmed_by_user"))
        object.__setattr__(self, "valid_from", _optional_str(self.valid_from, field_name="valid_from", blank_to_none=True))
        object.__setattr__(self, "valid_to", _optional_str(self.valid_to, field_name="valid_to", blank_to_none=True))
        # AUDIT-FIX(#3): Freeze graph-edge attributes to avoid caller-side mutation races.
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes, field_name="attributes"))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")

    def to_payload(self) -> dict[str, object]:
        """Serialize the graph-edge candidate into a plain payload mapping."""

        return _drop_none(
            {
                "source_ref": self.source_ref,
                "edge_type": self.edge_type,
                "target_ref": self.target_ref,
                "confidence": self.confidence,
                "confirmed_by_user": self.confirmed_by_user,
                "valid_from": self.valid_from,
                "valid_to": self.valid_to,
                "attributes": _mapping_dict(self.attributes),
            }
        )


@dataclass(frozen=True, slots=True)
class LongTermMemoryObjectV1:
    """Represent the canonical persisted long-term memory object."""

    memory_id: str
    kind: str
    summary: str
    source: LongTermSourceRefV1
    details: str | None = None
    status: str = "candidate"
    confidence: float = 0.5
    canonical_language: str = "en"
    confirmed_by_user: bool = False
    sensitivity: str = "normal"
    slot_key: str | None = None
    value_key: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    archived_at: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    attributes: Mapping[str, object] | None = None
    conflicts_with: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    schema: str = LONGTERM_MEMORY_OBJECT_SCHEMA
    version: int = LONGTERM_MEMORY_OBJECT_VERSION

    def __post_init__(self) -> None:
        # AUDIT-FIX(#3): Enforce constructor invariants, canonicalize ontology data, and detach nested mutable state.
        object.__setattr__(self, "memory_id", _require_str(self.memory_id, field_name="memory_id"))
        object.__setattr__(self, "summary", _require_str(self.summary, field_name="summary"))
        object.__setattr__(self, "source", _coerce_source_ref(self.source, field_name="source"))
        frozen_attributes = _freeze_mapping(self.attributes, field_name="attributes")
        normalized_kind, normalized_attributes = normalize_memory_kind(
            _require_str(self.kind, field_name="kind"),
            _mapping_dict(frozen_attributes),
        )
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "attributes", _freeze_mapping(normalized_attributes or None, field_name="attributes"))
        object.__setattr__(self, "details", _optional_str(self.details, field_name="details", blank_to_none=True))
        object.__setattr__(self, "status", _require_str(self.status, field_name="status"))
        object.__setattr__(self, "confidence", _coerce_float(self.confidence, field_name="confidence"))
        object.__setattr__(self, "canonical_language", _require_str(self.canonical_language, field_name="canonical_language"))
        object.__setattr__(self, "confirmed_by_user", _coerce_bool(self.confirmed_by_user, field_name="confirmed_by_user"))
        object.__setattr__(self, "sensitivity", normalize_memory_sensitivity(_require_str(self.sensitivity, field_name="sensitivity")))
        object.__setattr__(self, "slot_key", _optional_str(self.slot_key, field_name="slot_key", blank_to_none=True))
        object.__setattr__(self, "value_key", _optional_str(self.value_key, field_name="value_key", blank_to_none=True))
        object.__setattr__(self, "valid_from", _optional_str(self.valid_from, field_name="valid_from", blank_to_none=True))
        object.__setattr__(self, "valid_to", _optional_str(self.valid_to, field_name="valid_to", blank_to_none=True))
        object.__setattr__(self, "archived_at", _optional_str(self.archived_at, field_name="archived_at", blank_to_none=True))
        object.__setattr__(self, "conflicts_with", _coerce_str_tuple(self.conflicts_with, field_name="conflicts_with"))
        object.__setattr__(self, "supersedes", _coerce_str_tuple(self.supersedes, field_name="supersedes"))
        # AUDIT-FIX(#2): Force all memory timestamps onto aware UTC to avoid naive/aware comparison crashes.
        object.__setattr__(self, "created_at", _coerce_datetime(self.created_at, field_name="created_at"))
        object.__setattr__(self, "updated_at", _coerce_datetime(self.updated_at, field_name="updated_at"))
        if self.status not in LONGTERM_MEMORY_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(LONGTERM_MEMORY_STATUSES))}.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        if self.sensitivity not in LONGTERM_MEMORY_SENSITIVITY:
            raise ValueError(f"sensitivity must be one of: {', '.join(sorted(LONGTERM_MEMORY_SENSITIVITY))}.")
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_MEMORY_OBJECT_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_MEMORY_OBJECT_VERSION,
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the memory object into its versioned payload form."""

        return _drop_none(
            {
                "schema": self.schema,
                "version": self.version,
                "memory_id": self.memory_id,
                "kind": self.kind,
                "summary": self.summary,
                "details": self.details,
                "status": self.status,
                "confidence": self.confidence,
                "canonical_language": self.canonical_language,
                "confirmed_by_user": self.confirmed_by_user,
                "sensitivity": self.sensitivity,
                "slot_key": self.slot_key,
                "value_key": self.value_key,
                "valid_from": self.valid_from,
                "valid_to": self.valid_to,
                "archived_at": self.archived_at,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "source": self.source.to_payload(),
                "attributes": _mapping_dict(self.attributes),
                "conflicts_with": list(self.conflicts_with) if self.conflicts_with else None,
                "supersedes": list(self.supersedes) if self.supersedes else None,
            }
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "LongTermMemoryObjectV1":
        """Build a memory object from a persisted payload mapping."""

        source_payload = payload.get("source")
        if not isinstance(source_payload, Mapping):
            raise ValueError("source payload is required.")
        attributes = payload.get("attributes")
        if attributes is not None and not isinstance(attributes, Mapping):
            raise ValueError("attributes must be a mapping when provided.")
        normalized_kind, normalized_attributes = normalize_memory_kind(
            _payload_required_str(payload, "kind"),
            dict(attributes) if isinstance(attributes, Mapping) else None,
        )
        return cls(
            memory_id=_payload_required_str(payload, "memory_id"),
            kind=normalized_kind,
            summary=_payload_required_str(payload, "summary"),
            source=LongTermSourceRefV1.from_payload(source_payload),
            details=_payload_optional_str(payload, "details", blank_to_none=True),
            status=cast(str, _payload_optional_str(payload, "status", default="candidate")),
            confidence=_coerce_float(payload.get("confidence", 0.5), field_name="confidence", default=0.5),
            canonical_language=cast(str, _payload_optional_str(payload, "canonical_language", default="en")),
            # AUDIT-FIX(#1): Parse persisted booleans strictly so "false" does not become True.
            confirmed_by_user=_coerce_bool(payload.get("confirmed_by_user", False), field_name="confirmed_by_user", default=False),
            sensitivity=normalize_memory_sensitivity(cast(str, _payload_optional_str(payload, "sensitivity", default="normal"))),
            slot_key=_payload_optional_str(payload, "slot_key", blank_to_none=True),
            value_key=_payload_optional_str(payload, "value_key", blank_to_none=True),
            valid_from=_payload_optional_str(payload, "valid_from", blank_to_none=True),
            valid_to=_payload_optional_str(payload, "valid_to", blank_to_none=True),
            archived_at=_payload_optional_str(payload, "archived_at", blank_to_none=True),
            # AUDIT-FIX(#2): Salvage legacy naive timestamps by interpreting them as UTC and normalizing them.
            created_at=_coerce_datetime(payload.get("created_at", _MISSING), field_name="created_at", default=_utcnow()),
            updated_at=_coerce_datetime(payload.get("updated_at", _MISSING), field_name="updated_at", default=_utcnow()),
            attributes=normalized_attributes or None,
            conflicts_with=_coerce_str_tuple(payload.get("conflicts_with"), field_name="conflicts_with"),
            supersedes=_coerce_str_tuple(payload.get("supersedes"), field_name="supersedes"),
            schema=cast(str, _payload_optional_str(payload, "schema", default=LONGTERM_MEMORY_OBJECT_SCHEMA)),
            version=_coerce_int(payload.get("version", LONGTERM_MEMORY_OBJECT_VERSION), field_name="version", default=LONGTERM_MEMORY_OBJECT_VERSION),
        )

    def with_updates(self, **changes: object) -> "LongTermMemoryObjectV1":
        """Return a validated copy of the memory object with field overrides."""

        payload = self.to_payload()
        normalized_changes = dict(changes)
        # AUDIT-FIX(#5): Accept LongTermSourceRefV1 updates directly instead of silently discarding them.
        if isinstance(normalized_changes.get("source"), LongTermSourceRefV1):
            normalized_changes["source"] = cast(LongTermSourceRefV1, normalized_changes["source"]).to_payload()
        payload.update(normalized_changes)
        if "source" not in payload:
            payload["source"] = self.source.to_payload()
        elif isinstance(payload["source"], LongTermSourceRefV1):
            payload["source"] = cast(LongTermSourceRefV1, payload["source"]).to_payload()
        elif not isinstance(payload["source"], Mapping):
            raise ValueError("source must be a LongTermSourceRefV1 or mapping.")
        if "created_at" in payload and isinstance(payload["created_at"], datetime):
            payload["created_at"] = payload["created_at"].isoformat()
        if "updated_at" in payload and isinstance(payload["updated_at"], datetime):
            payload["updated_at"] = payload["updated_at"].isoformat()
        return LongTermMemoryObjectV1.from_payload(payload)

    def canonicalized(self) -> "LongTermMemoryObjectV1":
        """Return a copy with kind and attributes normalized canonically."""

        normalized_kind, normalized_attributes = normalize_memory_kind(self.kind, _mapping_dict(self.attributes))
        current_attributes = _mapping_dict(self.attributes) or {}
        if normalized_kind == self.kind and normalized_attributes == current_attributes:
            return self
        return self.with_updates(
            kind=normalized_kind,
            attributes=normalized_attributes,
        )


@dataclass(frozen=True, slots=True)
class LongTermMemoryConflictV1:
    """Represent a slot conflict that requires clarification or resolution."""

    slot_key: str
    candidate_memory_id: str
    existing_memory_ids: tuple[str, ...]
    question: str
    reason: str
    schema: str = LONGTERM_MEMORY_CONFLICT_SCHEMA
    version: int = LONGTERM_MEMORY_CONFLICT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_key", _require_str(self.slot_key, field_name="slot_key"))
        object.__setattr__(self, "candidate_memory_id", _require_str(self.candidate_memory_id, field_name="candidate_memory_id"))
        object.__setattr__(self, "existing_memory_ids", _coerce_str_tuple(self.existing_memory_ids, field_name="existing_memory_ids", allow_empty=False))
        object.__setattr__(self, "question", _require_str(self.question, field_name="question"))
        object.__setattr__(self, "reason", _require_str(self.reason, field_name="reason"))
        # AUDIT-FIX(#8): This versioned model now rejects schema/version mismatches like its peers.
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_MEMORY_CONFLICT_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_MEMORY_CONFLICT_VERSION,
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the conflict into its versioned payload form."""

        return {
            "schema": self.schema,
            "version": self.version,
            "slot_key": self.slot_key,
            "candidate_memory_id": self.candidate_memory_id,
            "existing_memory_ids": list(self.existing_memory_ids),
            "question": self.question,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "LongTermMemoryConflictV1":
        """Build a memory conflict from a persisted payload mapping."""

        return cls(
            slot_key=_payload_required_str(payload, "slot_key"),
            candidate_memory_id=_payload_required_str(payload, "candidate_memory_id"),
            existing_memory_ids=_coerce_str_tuple(payload.get("existing_memory_ids"), field_name="existing_memory_ids", allow_empty=False),
            question=_payload_required_str(payload, "question"),
            reason=_payload_required_str(payload, "reason"),
            schema=_payload_required_str(payload, "schema"),
            version=_coerce_int(payload.get("version", LONGTERM_MEMORY_CONFLICT_VERSION), field_name="version", default=LONGTERM_MEMORY_CONFLICT_VERSION),
        )

    def catalog_item_id(self) -> str:
        """Return the canonical remote/local document identifier for this conflict."""

        return json.dumps(
            [self.slot_key, self.candidate_memory_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class LongTermTurnExtractionV1:
    """Capture extraction output for one conversation turn."""

    turn_id: str
    occurred_at: datetime
    episode: LongTermMemoryObjectV1
    candidate_objects: tuple[LongTermMemoryObjectV1, ...] = ()
    graph_edges: tuple[LongTermGraphEdgeCandidateV1, ...] = ()
    warnings: tuple[str, ...] = ()
    schema: str = LONGTERM_TURN_EXTRACTION_SCHEMA
    version: int = LONGTERM_TURN_EXTRACTION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "turn_id", _require_str(self.turn_id, field_name="turn_id"))
        # AUDIT-FIX(#3): Fail early on wrong nested types instead of crashing later during serialization/use.
        object.__setattr__(self, "episode", _coerce_memory_object(self.episode, field_name="episode"))
        object.__setattr__(
            self,
            "candidate_objects",
            _coerce_tuple(
                self.candidate_objects,
                field_name="candidate_objects",
                item_name="LongTermMemoryObjectV1",
                item_coercer=_coerce_memory_object,
            ),
        )
        object.__setattr__(
            self,
            "graph_edges",
            _coerce_tuple(
                self.graph_edges,
                field_name="graph_edges",
                item_name="LongTermGraphEdgeCandidateV1",
                item_coercer=_coerce_graph_edge,
            ),
        )
        object.__setattr__(self, "warnings", _coerce_str_tuple(self.warnings, field_name="warnings"))
        # AUDIT-FIX(#2): Keep extraction timestamps UTC-normalized.
        object.__setattr__(self, "occurred_at", _coerce_datetime(self.occurred_at, field_name="occurred_at"))
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_TURN_EXTRACTION_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_TURN_EXTRACTION_VERSION,
        )

    def all_objects(self) -> tuple[LongTermMemoryObjectV1, ...]:
        """Return the episodic object plus all extracted candidate objects."""

        return (self.episode, *self.candidate_objects)


@dataclass(frozen=True, slots=True)
class LongTermConsolidationResultV1:
    """Capture the consolidator output for one processed turn."""

    turn_id: str
    occurred_at: datetime
    episodic_objects: tuple[LongTermMemoryObjectV1, ...]
    durable_objects: tuple[LongTermMemoryObjectV1, ...]
    deferred_objects: tuple[LongTermMemoryObjectV1, ...]
    conflicts: tuple[LongTermMemoryConflictV1, ...]
    graph_edges: tuple[LongTermGraphEdgeCandidateV1, ...]
    schema: str = LONGTERM_CONSOLIDATION_SCHEMA
    version: int = LONGTERM_CONSOLIDATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "turn_id", _require_str(self.turn_id, field_name="turn_id"))
        # AUDIT-FIX(#3): Normalize tuple/list inputs and validate nested result objects at construction time.
        object.__setattr__(
            self,
            "episodic_objects",
            _coerce_tuple(
                self.episodic_objects,
                field_name="episodic_objects",
                item_name="LongTermMemoryObjectV1",
                item_coercer=_coerce_memory_object,
            ),
        )
        object.__setattr__(
            self,
            "durable_objects",
            _coerce_tuple(
                self.durable_objects,
                field_name="durable_objects",
                item_name="LongTermMemoryObjectV1",
                item_coercer=_coerce_memory_object,
            ),
        )
        object.__setattr__(
            self,
            "deferred_objects",
            _coerce_tuple(
                self.deferred_objects,
                field_name="deferred_objects",
                item_name="LongTermMemoryObjectV1",
                item_coercer=_coerce_memory_object,
            ),
        )
        object.__setattr__(
            self,
            "conflicts",
            _coerce_tuple(
                self.conflicts,
                field_name="conflicts",
                item_name="LongTermMemoryConflictV1",
                item_coercer=_coerce_conflict,
            ),
        )
        object.__setattr__(
            self,
            "graph_edges",
            _coerce_tuple(
                self.graph_edges,
                field_name="graph_edges",
                item_name="LongTermGraphEdgeCandidateV1",
                item_coercer=_coerce_graph_edge,
            ),
        )
        # AUDIT-FIX(#2): Keep consolidation timestamps UTC-normalized.
        object.__setattr__(self, "occurred_at", _coerce_datetime(self.occurred_at, field_name="occurred_at"))
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_CONSOLIDATION_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_CONSOLIDATION_VERSION,
        )

    @property
    def clarification_needed(self) -> bool:
        """Return whether the consolidation result still contains conflicts."""

        return bool(self.conflicts)


@dataclass(frozen=True, slots=True)
class LongTermReflectionResultV1:
    """Collect reflection outputs produced across stored memories."""

    reflected_objects: tuple[LongTermMemoryObjectV1, ...]
    created_summaries: tuple[LongTermMemoryObjectV1, ...]
    midterm_packets: tuple["LongTermMidtermPacketV1", ...] = ()
    schema: str = LONGTERM_REFLECTION_SCHEMA
    version: int = LONGTERM_REFLECTION_VERSION

    def __post_init__(self) -> None:
        # AUDIT-FIX(#3): Validate nested reflection payloads immediately, not only when later consumed.
        object.__setattr__(
            self,
            "reflected_objects",
            _coerce_tuple(
                self.reflected_objects,
                field_name="reflected_objects",
                item_name="LongTermMemoryObjectV1",
                item_coercer=_coerce_memory_object,
            ),
        )
        object.__setattr__(
            self,
            "created_summaries",
            _coerce_tuple(
                self.created_summaries,
                field_name="created_summaries",
                item_name="LongTermMemoryObjectV1",
                item_coercer=_coerce_memory_object,
            ),
        )
        object.__setattr__(
            self,
            "midterm_packets",
            _coerce_tuple(
                self.midterm_packets,
                field_name="midterm_packets",
                item_name="LongTermMidtermPacketV1",
                item_coercer=_coerce_midterm_packet,
            ),
        )
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_REFLECTION_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_REFLECTION_VERSION,
        )


@dataclass(frozen=True, slots=True)
class LongTermMidtermPacketV1:
    """Represent a synthesized mid-term packet derived from memories."""

    packet_id: str
    kind: str
    summary: str
    details: str | None = None
    source_memory_ids: tuple[str, ...] = ()
    query_hints: tuple[str, ...] = ()
    canonical_language: str = "en"
    sensitivity: str = "normal"
    valid_from: str | None = None
    valid_to: str | None = None
    updated_at: datetime = field(default_factory=_utcnow)
    attributes: Mapping[str, object] | None = None
    schema: str = LONGTERM_MIDTERM_PACKET_SCHEMA
    version: int = LONGTERM_MIDTERM_PACKET_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "packet_id", _require_str(self.packet_id, field_name="packet_id"))
        object.__setattr__(self, "kind", _require_str(self.kind, field_name="kind"))
        object.__setattr__(self, "summary", _require_str(self.summary, field_name="summary"))
        object.__setattr__(self, "details", _optional_str(self.details, field_name="details", blank_to_none=True))
        object.__setattr__(self, "source_memory_ids", _coerce_str_tuple(self.source_memory_ids, field_name="source_memory_ids"))
        object.__setattr__(self, "query_hints", _coerce_str_tuple(self.query_hints, field_name="query_hints"))
        canonical_language = _require_str(self.canonical_language, field_name="canonical_language").strip().lower()
        object.__setattr__(self, "canonical_language", canonical_language)
        object.__setattr__(self, "sensitivity", normalize_memory_sensitivity(_require_str(self.sensitivity, field_name="sensitivity")))
        object.__setattr__(self, "valid_from", _optional_str(self.valid_from, field_name="valid_from", blank_to_none=True))
        object.__setattr__(self, "valid_to", _optional_str(self.valid_to, field_name="valid_to", blank_to_none=True))
        # AUDIT-FIX(#2): Keep packet timestamps comparable with memory object timestamps.
        object.__setattr__(self, "updated_at", _coerce_datetime(self.updated_at, field_name="updated_at"))
        # AUDIT-FIX(#3): Freeze nested packet attributes to avoid shared-state mutation.
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes, field_name="attributes"))
        if canonical_language != "en":
            raise ValueError("midterm packets must use canonical English.")
        if self.sensitivity not in LONGTERM_MEMORY_SENSITIVITY:
            raise ValueError(f"sensitivity must be one of: {', '.join(sorted(LONGTERM_MEMORY_SENSITIVITY))}.")
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_MIDTERM_PACKET_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_MIDTERM_PACKET_VERSION,
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the midterm packet into its versioned payload form."""

        return _drop_none(
            {
                "schema": self.schema,
                "version": self.version,
                "packet_id": self.packet_id,
                "kind": self.kind,
                "summary": self.summary,
                "details": self.details,
                "source_memory_ids": list(self.source_memory_ids) if self.source_memory_ids else None,
                "query_hints": list(self.query_hints) if self.query_hints else None,
                "canonical_language": self.canonical_language,
                "sensitivity": self.sensitivity,
                "valid_from": self.valid_from,
                "valid_to": self.valid_to,
                "updated_at": self.updated_at.isoformat(),
                "attributes": _mapping_dict(self.attributes),
            }
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "LongTermMidtermPacketV1":
        """Build a midterm packet from a persisted payload mapping."""

        attributes = payload.get("attributes")
        if attributes is not None and not isinstance(attributes, Mapping):
            raise ValueError("attributes must be a mapping when provided.")
        return cls(
            packet_id=_payload_required_str(payload, "packet_id"),
            kind=_payload_required_str(payload, "kind"),
            summary=_payload_required_str(payload, "summary"),
            details=_payload_optional_str(payload, "details", blank_to_none=True),
            source_memory_ids=_coerce_str_tuple(payload.get("source_memory_ids"), field_name="source_memory_ids"),
            query_hints=_coerce_str_tuple(payload.get("query_hints"), field_name="query_hints"),
            canonical_language=cast(str, _payload_optional_str(payload, "canonical_language", default="en")),
            sensitivity=cast(str, _payload_optional_str(payload, "sensitivity", default="normal")),
            valid_from=_payload_optional_str(payload, "valid_from", blank_to_none=True),
            valid_to=_payload_optional_str(payload, "valid_to", blank_to_none=True),
            updated_at=_coerce_datetime(payload.get("updated_at", _MISSING), field_name="updated_at", default=_utcnow()),
            attributes=_mapping_dict(attributes) if isinstance(attributes, Mapping) else None,
            # AUDIT-FIX(#8): Preserve stored schema/version so incompatible packets cannot masquerade as v1 objects.
            schema=cast(str, _payload_optional_str(payload, "schema", default=LONGTERM_MIDTERM_PACKET_SCHEMA)),
            version=_coerce_int(payload.get("version", LONGTERM_MIDTERM_PACKET_VERSION), field_name="version", default=LONGTERM_MIDTERM_PACKET_VERSION),
        )


@dataclass(frozen=True, slots=True)
class LongTermProactiveCandidateV1:
    """Represent a candidate proactive follow-up derived from memory."""

    candidate_id: str
    kind: str
    summary: str
    rationale: str
    due_date: str | None = None
    confidence: float = 0.5
    source_memory_ids: tuple[str, ...] = ()
    sensitivity: str = "normal"

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _require_str(self.candidate_id, field_name="candidate_id"))
        object.__setattr__(self, "kind", _require_str(self.kind, field_name="kind"))
        object.__setattr__(self, "summary", _require_str(self.summary, field_name="summary"))
        object.__setattr__(self, "rationale", _require_str(self.rationale, field_name="rationale"))
        object.__setattr__(self, "due_date", _optional_str(self.due_date, field_name="due_date", blank_to_none=True))
        object.__setattr__(self, "confidence", _coerce_float(self.confidence, field_name="confidence"))
        object.__setattr__(self, "source_memory_ids", _coerce_str_tuple(self.source_memory_ids, field_name="source_memory_ids"))
        object.__setattr__(self, "sensitivity", normalize_memory_sensitivity(_require_str(self.sensitivity, field_name="sensitivity")))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        if self.sensitivity not in LONGTERM_MEMORY_SENSITIVITY:
            raise ValueError(f"sensitivity must be one of: {', '.join(sorted(LONGTERM_MEMORY_SENSITIVITY))}.")

    def to_payload(self) -> dict[str, object]:
        """Serialize the proactive candidate into a plain payload mapping."""

        return _drop_none(
            {
                "candidate_id": self.candidate_id,
                "kind": self.kind,
                "summary": self.summary,
                "rationale": self.rationale,
                "due_date": self.due_date,
                "confidence": self.confidence,
                "source_memory_ids": list(self.source_memory_ids) if self.source_memory_ids else None,
                "sensitivity": self.sensitivity,
            }
        )


@dataclass(frozen=True, slots=True)
class LongTermProactivePlanV1:
    """Hold the set of proactive candidates proposed for a cycle."""

    candidates: tuple[LongTermProactiveCandidateV1, ...]
    schema: str = LONGTERM_PROACTIVE_PLAN_SCHEMA
    version: int = LONGTERM_PROACTIVE_PLAN_VERSION

    def __post_init__(self) -> None:
        # AUDIT-FIX(#3): Normalize proactive plan candidates to an immutable tuple and validate each element.
        object.__setattr__(
            self,
            "candidates",
            _coerce_tuple(
                self.candidates,
                field_name="candidates",
                item_name="LongTermProactiveCandidateV1",
                item_coercer=_coerce_proactive_candidate,
            ),
        )
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_PROACTIVE_PLAN_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_PROACTIVE_PLAN_VERSION,
        )


@dataclass(frozen=True, slots=True)
class LongTermRetentionResultV1:
    """Summarize retention outcomes after pruning or archiving memories."""

    kept_objects: tuple[LongTermMemoryObjectV1, ...]
    expired_objects: tuple[LongTermMemoryObjectV1, ...]
    pruned_memory_ids: tuple[str, ...]
    archived_objects: tuple[LongTermMemoryObjectV1, ...] = ()
    schema: str = LONGTERM_RETENTION_SCHEMA
    version: int = LONGTERM_RETENTION_VERSION

    def __post_init__(self) -> None:
        # AUDIT-FIX(#3): Normalize retention result collections and validate nested types immediately.
        object.__setattr__(
            self,
            "kept_objects",
            _coerce_tuple(
                self.kept_objects,
                field_name="kept_objects",
                item_name="LongTermMemoryObjectV1",
                item_coercer=_coerce_memory_object,
            ),
        )
        object.__setattr__(
            self,
            "expired_objects",
            _coerce_tuple(
                self.expired_objects,
                field_name="expired_objects",
                item_name="LongTermMemoryObjectV1",
                item_coercer=_coerce_memory_object,
            ),
        )
        object.__setattr__(self, "pruned_memory_ids", _coerce_str_tuple(self.pruned_memory_ids, field_name="pruned_memory_ids"))
        object.__setattr__(
            self,
            "archived_objects",
            _coerce_tuple(
                self.archived_objects,
                field_name="archived_objects",
                item_name="LongTermMemoryObjectV1",
                item_coercer=_coerce_memory_object,
            ),
        )
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_RETENTION_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_RETENTION_VERSION,
        )


@dataclass(frozen=True, slots=True)
class LongTermConflictOptionV1:
    """Represent one selectable option in a conflict-resolution prompt."""

    memory_id: str
    summary: str
    status: str
    details: str | None = None
    value_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _require_str(self.memory_id, field_name="memory_id"))
        object.__setattr__(self, "summary", _require_str(self.summary, field_name="summary"))
        object.__setattr__(self, "status", _require_str(self.status, field_name="status"))
        object.__setattr__(self, "details", _optional_str(self.details, field_name="details", blank_to_none=True))
        object.__setattr__(self, "value_key", _optional_str(self.value_key, field_name="value_key", blank_to_none=True))
        if self.status not in LONGTERM_MEMORY_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(LONGTERM_MEMORY_STATUSES))}.")

    def to_payload(self) -> dict[str, object]:
        """Serialize the conflict option into a plain payload mapping."""

        return _drop_none(
            {
                "memory_id": self.memory_id,
                "summary": self.summary,
                "details": self.details,
                "status": self.status,
                "value_key": self.value_key,
            }
        )


@dataclass(frozen=True, slots=True)
class LongTermConflictQueueItemV1:
    """Represent a queued conflict surfaced to retrieval or UI flows."""

    slot_key: str
    question: str
    reason: str
    candidate_memory_id: str
    options: tuple[LongTermConflictOptionV1, ...]
    schema: str = LONGTERM_CONFLICT_QUEUE_SCHEMA
    version: int = LONGTERM_CONFLICT_QUEUE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_key", _require_str(self.slot_key, field_name="slot_key"))
        object.__setattr__(self, "question", _require_str(self.question, field_name="question"))
        object.__setattr__(self, "reason", _require_str(self.reason, field_name="reason"))
        object.__setattr__(self, "candidate_memory_id", _require_str(self.candidate_memory_id, field_name="candidate_memory_id"))
        # AUDIT-FIX(#3): Normalize option collections and validate element types eagerly.
        object.__setattr__(
            self,
            "options",
            _coerce_tuple(
                self.options,
                field_name="options",
                item_name="LongTermConflictOptionV1",
                item_coercer=_coerce_conflict_option,
                allow_empty=False,
            ),
        )
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_CONFLICT_QUEUE_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_CONFLICT_QUEUE_VERSION,
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the queued conflict into its versioned payload form."""

        return {
            "schema": self.schema,
            "version": self.version,
            "slot_key": self.slot_key,
            "question": self.question,
            "reason": self.reason,
            "candidate_memory_id": self.candidate_memory_id,
            "options": [item.to_payload() for item in self.options],
        }


@dataclass(frozen=True, slots=True)
class LongTermConflictResolutionV1:
    """Capture the outcome of resolving one queued memory conflict."""

    slot_key: str
    selected_memory_id: str
    updated_objects: tuple[LongTermMemoryObjectV1, ...]
    remaining_conflicts: tuple[LongTermMemoryConflictV1, ...]
    schema: str = LONGTERM_CONFLICT_RESOLUTION_SCHEMA
    version: int = LONGTERM_CONFLICT_RESOLUTION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_key", _require_str(self.slot_key, field_name="slot_key"))
        object.__setattr__(self, "selected_memory_id", _require_str(self.selected_memory_id, field_name="selected_memory_id"))
        # AUDIT-FIX(#3): Normalize updated/conflict object collections to immutable tuples and validate elements.
        object.__setattr__(
            self,
            "updated_objects",
            _coerce_tuple(
                self.updated_objects,
                field_name="updated_objects",
                item_name="LongTermMemoryObjectV1",
                item_coercer=_coerce_memory_object,
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "remaining_conflicts",
            _coerce_tuple(
                self.remaining_conflicts,
                field_name="remaining_conflicts",
                item_name="LongTermMemoryConflictV1",
                item_coercer=_coerce_conflict,
            ),
        )
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_CONFLICT_RESOLUTION_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_CONFLICT_RESOLUTION_VERSION,
        )


@dataclass(frozen=True, slots=True)
class LongTermMemoryReviewItemV1:
    """Represent one memory item returned by review flows."""

    memory_id: str
    kind: str
    summary: str
    status: str
    confidence: float
    updated_at: datetime
    details: str | None = None
    confirmed_by_user: bool = False
    sensitivity: str = "normal"
    slot_key: str | None = None
    value_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _require_str(self.memory_id, field_name="memory_id"))
        object.__setattr__(self, "kind", _require_str(self.kind, field_name="kind"))
        object.__setattr__(self, "summary", _require_str(self.summary, field_name="summary"))
        object.__setattr__(self, "status", _require_str(self.status, field_name="status"))
        object.__setattr__(self, "confidence", _coerce_float(self.confidence, field_name="confidence"))
        # AUDIT-FIX(#2): Normalize review timestamps to aware UTC.
        object.__setattr__(self, "updated_at", _coerce_datetime(self.updated_at, field_name="updated_at"))
        object.__setattr__(self, "details", _optional_str(self.details, field_name="details", blank_to_none=True))
        object.__setattr__(self, "confirmed_by_user", _coerce_bool(self.confirmed_by_user, field_name="confirmed_by_user"))
        object.__setattr__(self, "sensitivity", normalize_memory_sensitivity(_require_str(self.sensitivity, field_name="sensitivity")))
        object.__setattr__(self, "slot_key", _optional_str(self.slot_key, field_name="slot_key", blank_to_none=True))
        object.__setattr__(self, "value_key", _optional_str(self.value_key, field_name="value_key", blank_to_none=True))
        if self.status not in LONGTERM_MEMORY_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(LONGTERM_MEMORY_STATUSES))}.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        if self.sensitivity not in LONGTERM_MEMORY_SENSITIVITY:
            raise ValueError(f"sensitivity must be one of: {', '.join(sorted(LONGTERM_MEMORY_SENSITIVITY))}.")

    def to_payload(self) -> dict[str, object]:
        """Serialize the review item into a plain payload mapping."""

        return _drop_none(
            {
                "memory_id": self.memory_id,
                "kind": self.kind,
                "summary": self.summary,
                "details": self.details,
                "status": self.status,
                "confidence": self.confidence,
                "updated_at": self.updated_at.isoformat(),
                "confirmed_by_user": self.confirmed_by_user,
                "sensitivity": self.sensitivity,
                "slot_key": self.slot_key,
                "value_key": self.value_key,
            }
        )


@dataclass(frozen=True, slots=True)
class LongTermMemoryReviewResultV1:
    """Represent a filtered review result set over long-term memories."""

    items: tuple[LongTermMemoryReviewItemV1, ...]
    total_count: int
    query_text: str | None = None
    status_filter: str | None = None
    kind_filter: str | None = None
    include_episodes: bool = False
    schema: str = LONGTERM_MEMORY_REVIEW_SCHEMA
    version: int = LONGTERM_MEMORY_REVIEW_VERSION

    def __post_init__(self) -> None:
        # AUDIT-FIX(#3): Normalize the item collection to an immutable tuple and validate its elements.
        object.__setattr__(
            self,
            "items",
            _coerce_tuple(
                self.items,
                field_name="items",
                item_name="LongTermMemoryReviewItemV1",
                item_coercer=lambda value, field_name: _coerce_instance(value, field_name, LongTermMemoryReviewItemV1),
            ),
        )
        # AUDIT-FIX(#6): Reject impossible review counters like negative totals or totals smaller than the returned page.
        object.__setattr__(self, "total_count", _coerce_non_negative_int(self.total_count, field_name="total_count"))
        object.__setattr__(self, "query_text", _optional_str(self.query_text, field_name="query_text", blank_to_none=True))
        object.__setattr__(self, "status_filter", _optional_str(self.status_filter, field_name="status_filter", blank_to_none=True))
        object.__setattr__(self, "kind_filter", _optional_str(self.kind_filter, field_name="kind_filter", blank_to_none=True))
        object.__setattr__(self, "include_episodes", _coerce_bool(self.include_episodes, field_name="include_episodes"))
        if self.total_count < len(self.items):
            raise ValueError("total_count cannot be smaller than the number of returned items.")
        if self.status_filter is not None and self.status_filter not in LONGTERM_MEMORY_STATUSES:
            raise ValueError(f"status_filter must be one of: {', '.join(sorted(LONGTERM_MEMORY_STATUSES))}.")
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_MEMORY_REVIEW_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_MEMORY_REVIEW_VERSION,
        )


@dataclass(frozen=True, slots=True)
class LongTermMemoryMutationResultV1:
    """Capture the result of confirming, invalidating, or deleting a memory."""

    action: str
    target_memory_id: str
    updated_objects: tuple[LongTermMemoryObjectV1, ...] = ()
    deleted_memory_ids: tuple[str, ...] = ()
    remaining_conflicts: tuple[LongTermMemoryConflictV1, ...] = ()
    schema: str = LONGTERM_MEMORY_MUTATION_SCHEMA
    version: int = LONGTERM_MEMORY_MUTATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _require_str(self.action, field_name="action"))
        object.__setattr__(self, "target_memory_id", _require_str(self.target_memory_id, field_name="target_memory_id"))
        # AUDIT-FIX(#3): Normalize mutation result collections before downstream consumers iterate them.
        object.__setattr__(
            self,
            "updated_objects",
            _coerce_tuple(
                self.updated_objects,
                field_name="updated_objects",
                item_name="LongTermMemoryObjectV1",
                item_coercer=_coerce_memory_object,
            ),
        )
        object.__setattr__(self, "deleted_memory_ids", _coerce_str_tuple(self.deleted_memory_ids, field_name="deleted_memory_ids"))
        object.__setattr__(
            self,
            "remaining_conflicts",
            _coerce_tuple(
                self.remaining_conflicts,
                field_name="remaining_conflicts",
                item_name="LongTermMemoryConflictV1",
                item_coercer=_coerce_conflict,
            ),
        )
        if self.action not in LONGTERM_MEMORY_MUTATION_ACTIONS:
            raise ValueError(f"action must be one of: {', '.join(sorted(LONGTERM_MEMORY_MUTATION_ACTIONS))}.")
        if not self.updated_objects and not self.deleted_memory_ids:
            raise ValueError("mutation result must contain updated objects or deleted ids.")
        _validate_schema_version(
            schema=self.schema,
            expected_schema=LONGTERM_MEMORY_MUTATION_SCHEMA,
            version=self.version,
            expected_version=LONGTERM_MEMORY_MUTATION_VERSION,
        )


__all__ = [
    "LONGTERM_CONSOLIDATION_SCHEMA",
    "LONGTERM_CONSOLIDATION_VERSION",
    "LONGTERM_CONFLICT_QUEUE_SCHEMA",
    "LONGTERM_CONFLICT_QUEUE_VERSION",
    "LONGTERM_CONFLICT_RESOLUTION_SCHEMA",
    "LONGTERM_CONFLICT_RESOLUTION_VERSION",
    "LONGTERM_MEMORY_CONFLICT_SCHEMA",
    "LONGTERM_MEMORY_CONFLICT_VERSION",
    "LONGTERM_MIDTERM_PACKET_SCHEMA",
    "LONGTERM_MIDTERM_PACKET_VERSION",
    "LONGTERM_MULTIMODAL_EVIDENCE_SCHEMA",
    "LONGTERM_MULTIMODAL_EVIDENCE_VERSION",
    "LONGTERM_MEMORY_OBJECT_SCHEMA",
    "LONGTERM_MEMORY_OBJECT_VERSION",
    "LONGTERM_MEMORY_MUTATION_ACTIONS",
    "LONGTERM_MEMORY_MUTATION_SCHEMA",
    "LONGTERM_MEMORY_MUTATION_VERSION",
    "LONGTERM_MEMORY_REVIEW_SCHEMA",
    "LONGTERM_MEMORY_REVIEW_VERSION",
    "LONGTERM_MEMORY_SENSITIVITY",
    "LONGTERM_MEMORY_STATUSES",
    "LONGTERM_PROACTIVE_PLAN_SCHEMA",
    "LONGTERM_PROACTIVE_PLAN_VERSION",
    "LONGTERM_RETENTION_SCHEMA",
    "LONGTERM_RETENTION_VERSION",
    "LONGTERM_REFLECTION_SCHEMA",
    "LONGTERM_REFLECTION_VERSION",
    "LONGTERM_TURN_EXTRACTION_SCHEMA",
    "LONGTERM_TURN_EXTRACTION_VERSION",
    "LongTermConsolidationResultV1",
    "LongTermConflictOptionV1",
    "LongTermConflictQueueItemV1",
    "LongTermConflictResolutionV1",
    "LongTermConversationTurn",
    "LongTermEnqueueResult",
    "LongTermGraphEdgeCandidateV1",
    "LongTermMemoryConflictV1",
    "LongTermMemoryContext",
    "LongTermMemoryObjectV1",
    "LongTermMemoryMutationResultV1",
    "LongTermMemoryReviewItemV1",
    "LongTermMemoryReviewResultV1",
    "LongTermMidtermPacketV1",
    "LongTermMultimodalEvidence",
    "LongTermProactiveCandidateV1",
    "LongTermProactivePlanV1",
    "LongTermRetentionResultV1",
    "LongTermReflectionResultV1",
    "LongTermSourceRefV1",
    "LongTermTurnExtractionV1",
]
