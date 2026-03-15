from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

from twinr.memory.longterm.ontology import (
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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _drop_none(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if value is not None}


def _mapping_dict(value: Mapping[str, object] | None) -> dict[str, object] | None:
    if value is None:
        return None
    return dict(value)


def _tuple_str(value: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(item) for item in value if str(item).strip())


@dataclass(frozen=True, slots=True)
class LongTermConversationTurn:
    transcript: str
    response: str
    source: str = "conversation"
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True, slots=True)
class LongTermEnqueueResult:
    accepted: bool
    pending_count: int
    dropped_count: int = 0


@dataclass(frozen=True, slots=True)
class LongTermMultimodalEvidence:
    event_name: str
    modality: str
    source: str = "device_event"
    message: str | None = None
    data: Mapping[str, object] | None = None
    created_at: datetime = field(default_factory=_utcnow)
    schema: str = LONGTERM_MULTIMODAL_EVIDENCE_SCHEMA
    version: int = LONGTERM_MULTIMODAL_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if not _normalize_text(self.event_name):
            raise ValueError("event_name is required.")
        if not _normalize_text(self.modality):
            raise ValueError("modality is required.")
        if not _normalize_text(self.source):
            raise ValueError("source is required.")
        if self.schema != LONGTERM_MULTIMODAL_EVIDENCE_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_MULTIMODAL_EVIDENCE_SCHEMA!r}.")
        if self.version != LONGTERM_MULTIMODAL_EVIDENCE_VERSION:
            raise ValueError(f"version must be {LONGTERM_MULTIMODAL_EVIDENCE_VERSION}.")


@dataclass(frozen=True, slots=True)
class LongTermMemoryContext:
    subtext_context: str | None = None
    midterm_context: str | None = None
    durable_context: str | None = None
    episodic_context: str | None = None
    graph_context: str | None = None
    conflict_context: str | None = None

    def system_messages(self) -> tuple[str, ...]:
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
    source_type: str
    event_ids: tuple[str, ...] = ()
    speaker: str | None = None
    modality: str | None = None

    def __post_init__(self) -> None:
        if not _normalize_text(self.source_type):
            raise ValueError("source_type is required.")
        if self.speaker is not None and not _normalize_text(self.speaker):
            raise ValueError("speaker cannot be blank when provided.")
        if self.modality is not None and not _normalize_text(self.modality):
            raise ValueError("modality cannot be blank when provided.")

    def to_payload(self) -> dict[str, object]:
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
        event_ids = payload.get("event_ids")
        return cls(
            source_type=str(payload.get("type", "")),
            event_ids=tuple(str(item) for item in event_ids if isinstance(item, str)) if isinstance(event_ids, list) else (),
            speaker=str(payload["speaker"]) if payload.get("speaker") is not None else None,
            modality=str(payload["modality"]) if payload.get("modality") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class LongTermGraphEdgeCandidateV1:
    source_ref: str
    edge_type: str
    target_ref: str
    confidence: float = 0.5
    confirmed_by_user: bool = False
    valid_from: str | None = None
    valid_to: str | None = None
    attributes: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not _normalize_text(self.source_ref):
            raise ValueError("source_ref is required.")
        if not _normalize_text(self.edge_type):
            raise ValueError("edge_type is required.")
        if not _normalize_text(self.target_ref):
            raise ValueError("target_ref is required.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")

    def to_payload(self) -> dict[str, object]:
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
        object.__setattr__(self, "sensitivity", normalize_memory_sensitivity(self.sensitivity))
        if not _normalize_text(self.memory_id):
            raise ValueError("memory_id is required.")
        if not _normalize_text(self.kind):
            raise ValueError("kind is required.")
        if not _normalize_text(self.summary):
            raise ValueError("summary is required.")
        if self.status not in LONGTERM_MEMORY_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(LONGTERM_MEMORY_STATUSES))}.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        if not _normalize_text(self.canonical_language):
            raise ValueError("canonical_language is required.")
        if self.sensitivity not in LONGTERM_MEMORY_SENSITIVITY:
            raise ValueError(f"sensitivity must be one of: {', '.join(sorted(LONGTERM_MEMORY_SENSITIVITY))}.")
        if self.schema != LONGTERM_MEMORY_OBJECT_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_MEMORY_OBJECT_SCHEMA!r}.")
        if self.version != LONGTERM_MEMORY_OBJECT_VERSION:
            raise ValueError(f"version must be {LONGTERM_MEMORY_OBJECT_VERSION}.")

    def to_payload(self) -> dict[str, object]:
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
        source_payload = payload.get("source")
        if not isinstance(source_payload, Mapping):
            raise ValueError("source payload is required.")
        attributes = payload.get("attributes")
        normalized_kind, normalized_attributes = normalize_memory_kind(
            str(payload.get("kind", "")),
            dict(attributes) if isinstance(attributes, Mapping) else None,
        )
        return cls(
            memory_id=str(payload.get("memory_id", "")),
            kind=normalized_kind,
            summary=str(payload.get("summary", "")),
            source=LongTermSourceRefV1.from_payload(source_payload),
            details=str(payload["details"]) if payload.get("details") is not None else None,
            status=str(payload.get("status", "candidate")),
            confidence=float(payload.get("confidence", 0.5)),
            canonical_language=str(payload.get("canonical_language", "en")),
            confirmed_by_user=bool(payload.get("confirmed_by_user", False)),
            sensitivity=normalize_memory_sensitivity(str(payload.get("sensitivity", "normal"))),
            slot_key=str(payload["slot_key"]) if payload.get("slot_key") is not None else None,
            value_key=str(payload["value_key"]) if payload.get("value_key") is not None else None,
            valid_from=str(payload["valid_from"]) if payload.get("valid_from") is not None else None,
            valid_to=str(payload["valid_to"]) if payload.get("valid_to") is not None else None,
            archived_at=str(payload["archived_at"]) if payload.get("archived_at") is not None else None,
            created_at=datetime.fromisoformat(str(payload["created_at"])) if payload.get("created_at") else _utcnow(),
            updated_at=datetime.fromisoformat(str(payload["updated_at"])) if payload.get("updated_at") else _utcnow(),
            attributes=normalized_attributes or None,
            conflicts_with=_tuple_str(
                payload.get("conflicts_with")
                if isinstance(payload.get("conflicts_with"), (list, tuple))
                else None
            ),
            supersedes=_tuple_str(
                payload.get("supersedes")
                if isinstance(payload.get("supersedes"), (list, tuple))
                else None
            ),
            schema=str(payload.get("schema", LONGTERM_MEMORY_OBJECT_SCHEMA)),
            version=int(payload.get("version", LONGTERM_MEMORY_OBJECT_VERSION)),
        )

    def with_updates(self, **changes: object) -> "LongTermMemoryObjectV1":
        payload = self.to_payload()
        payload.update(changes)
        if "source" not in payload or not isinstance(payload["source"], Mapping):
            payload["source"] = self.source.to_payload()
        if "created_at" in payload and isinstance(payload["created_at"], datetime):
            payload["created_at"] = payload["created_at"].isoformat()
        if "updated_at" in payload and isinstance(payload["updated_at"], datetime):
            payload["updated_at"] = payload["updated_at"].isoformat()
        return LongTermMemoryObjectV1.from_payload(payload)

    def canonicalized(self) -> "LongTermMemoryObjectV1":
        normalized_kind, normalized_attributes = normalize_memory_kind(self.kind, self.attributes)
        current_attributes = dict(self.attributes or {})
        if normalized_kind == self.kind and normalized_attributes == current_attributes:
            return self
        return self.with_updates(
            kind=normalized_kind,
            attributes=normalized_attributes,
        )


@dataclass(frozen=True, slots=True)
class LongTermMemoryConflictV1:
    slot_key: str
    candidate_memory_id: str
    existing_memory_ids: tuple[str, ...]
    question: str
    reason: str
    schema: str = LONGTERM_MEMORY_CONFLICT_SCHEMA
    version: int = LONGTERM_MEMORY_CONFLICT_VERSION

    def __post_init__(self) -> None:
        if not _normalize_text(self.slot_key):
            raise ValueError("slot_key is required.")
        if not _normalize_text(self.candidate_memory_id):
            raise ValueError("candidate_memory_id is required.")
        if not self.existing_memory_ids:
            raise ValueError("existing_memory_ids cannot be empty.")
        if not _normalize_text(self.question):
            raise ValueError("question is required.")
        if not _normalize_text(self.reason):
            raise ValueError("reason is required.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "slot_key": self.slot_key,
            "candidate_memory_id": self.candidate_memory_id,
            "existing_memory_ids": list(self.existing_memory_ids),
            "question": self.question,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class LongTermTurnExtractionV1:
    turn_id: str
    occurred_at: datetime
    episode: LongTermMemoryObjectV1
    candidate_objects: tuple[LongTermMemoryObjectV1, ...] = ()
    graph_edges: tuple[LongTermGraphEdgeCandidateV1, ...] = ()
    warnings: tuple[str, ...] = ()
    schema: str = LONGTERM_TURN_EXTRACTION_SCHEMA
    version: int = LONGTERM_TURN_EXTRACTION_VERSION

    def __post_init__(self) -> None:
        if not _normalize_text(self.turn_id):
            raise ValueError("turn_id is required.")
        if self.schema != LONGTERM_TURN_EXTRACTION_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_TURN_EXTRACTION_SCHEMA!r}.")
        if self.version != LONGTERM_TURN_EXTRACTION_VERSION:
            raise ValueError(f"version must be {LONGTERM_TURN_EXTRACTION_VERSION}.")

    def all_objects(self) -> tuple[LongTermMemoryObjectV1, ...]:
        return (self.episode, *self.candidate_objects)


@dataclass(frozen=True, slots=True)
class LongTermConsolidationResultV1:
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
        if not _normalize_text(self.turn_id):
            raise ValueError("turn_id is required.")
        if self.schema != LONGTERM_CONSOLIDATION_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_CONSOLIDATION_SCHEMA!r}.")
        if self.version != LONGTERM_CONSOLIDATION_VERSION:
            raise ValueError(f"version must be {LONGTERM_CONSOLIDATION_VERSION}.")

    @property
    def clarification_needed(self) -> bool:
        return bool(self.conflicts)


@dataclass(frozen=True, slots=True)
class LongTermReflectionResultV1:
    reflected_objects: tuple[LongTermMemoryObjectV1, ...]
    created_summaries: tuple[LongTermMemoryObjectV1, ...]
    midterm_packets: tuple["LongTermMidtermPacketV1", ...] = ()
    schema: str = LONGTERM_REFLECTION_SCHEMA
    version: int = LONGTERM_REFLECTION_VERSION

    def __post_init__(self) -> None:
        if self.schema != LONGTERM_REFLECTION_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_REFLECTION_SCHEMA!r}.")
        if self.version != LONGTERM_REFLECTION_VERSION:
            raise ValueError(f"version must be {LONGTERM_REFLECTION_VERSION}.")


@dataclass(frozen=True, slots=True)
class LongTermMidtermPacketV1:
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
        object.__setattr__(self, "sensitivity", normalize_memory_sensitivity(self.sensitivity))
        if not _normalize_text(self.packet_id):
            raise ValueError("packet_id is required.")
        if not _normalize_text(self.kind):
            raise ValueError("kind is required.")
        if not _normalize_text(self.summary):
            raise ValueError("summary is required.")
        if (self.canonical_language or "en").strip().lower() != "en":
            raise ValueError("midterm packets must use canonical English.")
        if self.schema != LONGTERM_MIDTERM_PACKET_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_MIDTERM_PACKET_SCHEMA!r}.")
        if self.version != LONGTERM_MIDTERM_PACKET_VERSION:
            raise ValueError(f"version must be {LONGTERM_MIDTERM_PACKET_VERSION}.")

    def to_payload(self) -> dict[str, object]:
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
        return cls(
            packet_id=str(payload.get("packet_id", "")),
            kind=str(payload.get("kind", "")),
            summary=str(payload.get("summary", "")),
            details=str(payload["details"]) if payload.get("details") is not None else None,
            source_memory_ids=_tuple_str(payload.get("source_memory_ids")),
            query_hints=_tuple_str(payload.get("query_hints")),
            canonical_language=str(payload.get("canonical_language", "en") or "en"),
            sensitivity=str(payload.get("sensitivity", "normal") or "normal"),
            valid_from=str(payload["valid_from"]) if payload.get("valid_from") is not None else None,
            valid_to=str(payload["valid_to"]) if payload.get("valid_to") is not None else None,
            updated_at=datetime.fromisoformat(str(payload["updated_at"])) if payload.get("updated_at") else _utcnow(),
            attributes=_mapping_dict(payload.get("attributes")) if isinstance(payload.get("attributes"), Mapping) else None,
        )


@dataclass(frozen=True, slots=True)
class LongTermProactiveCandidateV1:
    candidate_id: str
    kind: str
    summary: str
    rationale: str
    due_date: str | None = None
    confidence: float = 0.5
    source_memory_ids: tuple[str, ...] = ()
    sensitivity: str = "normal"

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensitivity", normalize_memory_sensitivity(self.sensitivity))
        if not _normalize_text(self.candidate_id):
            raise ValueError("candidate_id is required.")
        if not _normalize_text(self.kind):
            raise ValueError("kind is required.")
        if not _normalize_text(self.summary):
            raise ValueError("summary is required.")
        if not _normalize_text(self.rationale):
            raise ValueError("rationale is required.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        if self.sensitivity not in LONGTERM_MEMORY_SENSITIVITY:
            raise ValueError(f"sensitivity must be one of: {', '.join(sorted(LONGTERM_MEMORY_SENSITIVITY))}.")

    def to_payload(self) -> dict[str, object]:
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
    candidates: tuple[LongTermProactiveCandidateV1, ...]
    schema: str = LONGTERM_PROACTIVE_PLAN_SCHEMA
    version: int = LONGTERM_PROACTIVE_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.schema != LONGTERM_PROACTIVE_PLAN_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_PROACTIVE_PLAN_SCHEMA!r}.")
        if self.version != LONGTERM_PROACTIVE_PLAN_VERSION:
            raise ValueError(f"version must be {LONGTERM_PROACTIVE_PLAN_VERSION}.")


@dataclass(frozen=True, slots=True)
class LongTermRetentionResultV1:
    kept_objects: tuple[LongTermMemoryObjectV1, ...]
    expired_objects: tuple[LongTermMemoryObjectV1, ...]
    pruned_memory_ids: tuple[str, ...]
    archived_objects: tuple[LongTermMemoryObjectV1, ...] = ()
    schema: str = LONGTERM_RETENTION_SCHEMA
    version: int = LONGTERM_RETENTION_VERSION

    def __post_init__(self) -> None:
        if self.schema != LONGTERM_RETENTION_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_RETENTION_SCHEMA!r}.")
        if self.version != LONGTERM_RETENTION_VERSION:
            raise ValueError(f"version must be {LONGTERM_RETENTION_VERSION}.")


@dataclass(frozen=True, slots=True)
class LongTermConflictOptionV1:
    memory_id: str
    summary: str
    status: str
    details: str | None = None
    value_key: str | None = None

    def __post_init__(self) -> None:
        if not _normalize_text(self.memory_id):
            raise ValueError("memory_id is required.")
        if not _normalize_text(self.summary):
            raise ValueError("summary is required.")
        if self.status not in LONGTERM_MEMORY_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(LONGTERM_MEMORY_STATUSES))}.")

    def to_payload(self) -> dict[str, object]:
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
    slot_key: str
    question: str
    reason: str
    candidate_memory_id: str
    options: tuple[LongTermConflictOptionV1, ...]
    schema: str = LONGTERM_CONFLICT_QUEUE_SCHEMA
    version: int = LONGTERM_CONFLICT_QUEUE_VERSION

    def __post_init__(self) -> None:
        if not _normalize_text(self.slot_key):
            raise ValueError("slot_key is required.")
        if not _normalize_text(self.question):
            raise ValueError("question is required.")
        if not _normalize_text(self.reason):
            raise ValueError("reason is required.")
        if not _normalize_text(self.candidate_memory_id):
            raise ValueError("candidate_memory_id is required.")
        if not self.options:
            raise ValueError("options cannot be empty.")
        if self.schema != LONGTERM_CONFLICT_QUEUE_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_CONFLICT_QUEUE_SCHEMA!r}.")
        if self.version != LONGTERM_CONFLICT_QUEUE_VERSION:
            raise ValueError(f"version must be {LONGTERM_CONFLICT_QUEUE_VERSION}.")

    def to_payload(self) -> dict[str, object]:
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
    slot_key: str
    selected_memory_id: str
    updated_objects: tuple[LongTermMemoryObjectV1, ...]
    remaining_conflicts: tuple[LongTermMemoryConflictV1, ...]
    schema: str = LONGTERM_CONFLICT_RESOLUTION_SCHEMA
    version: int = LONGTERM_CONFLICT_RESOLUTION_VERSION

    def __post_init__(self) -> None:
        if not _normalize_text(self.slot_key):
            raise ValueError("slot_key is required.")
        if not _normalize_text(self.selected_memory_id):
            raise ValueError("selected_memory_id is required.")
        if not self.updated_objects:
            raise ValueError("updated_objects cannot be empty.")
        if self.schema != LONGTERM_CONFLICT_RESOLUTION_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_CONFLICT_RESOLUTION_SCHEMA!r}.")
        if self.version != LONGTERM_CONFLICT_RESOLUTION_VERSION:
            raise ValueError(f"version must be {LONGTERM_CONFLICT_RESOLUTION_VERSION}.")


@dataclass(frozen=True, slots=True)
class LongTermMemoryReviewItemV1:
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
        object.__setattr__(self, "sensitivity", normalize_memory_sensitivity(self.sensitivity))
        if not _normalize_text(self.memory_id):
            raise ValueError("memory_id is required.")
        if not _normalize_text(self.kind):
            raise ValueError("kind is required.")
        if not _normalize_text(self.summary):
            raise ValueError("summary is required.")
        if self.status not in LONGTERM_MEMORY_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(LONGTERM_MEMORY_STATUSES))}.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")
        if self.sensitivity not in LONGTERM_MEMORY_SENSITIVITY:
            raise ValueError(f"sensitivity must be one of: {', '.join(sorted(LONGTERM_MEMORY_SENSITIVITY))}.")

    def to_payload(self) -> dict[str, object]:
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
    items: tuple[LongTermMemoryReviewItemV1, ...]
    total_count: int
    query_text: str | None = None
    status_filter: str | None = None
    kind_filter: str | None = None
    include_episodes: bool = False
    schema: str = LONGTERM_MEMORY_REVIEW_SCHEMA
    version: int = LONGTERM_MEMORY_REVIEW_VERSION

    def __post_init__(self) -> None:
        if self.total_count < 0:
            raise ValueError("total_count cannot be negative.")
        if self.status_filter is not None and self.status_filter not in LONGTERM_MEMORY_STATUSES:
            raise ValueError(f"status_filter must be one of: {', '.join(sorted(LONGTERM_MEMORY_STATUSES))}.")
        if self.schema != LONGTERM_MEMORY_REVIEW_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_MEMORY_REVIEW_SCHEMA!r}.")
        if self.version != LONGTERM_MEMORY_REVIEW_VERSION:
            raise ValueError(f"version must be {LONGTERM_MEMORY_REVIEW_VERSION}.")


@dataclass(frozen=True, slots=True)
class LongTermMemoryMutationResultV1:
    action: str
    target_memory_id: str
    updated_objects: tuple[LongTermMemoryObjectV1, ...] = ()
    deleted_memory_ids: tuple[str, ...] = ()
    remaining_conflicts: tuple[LongTermMemoryConflictV1, ...] = ()
    schema: str = LONGTERM_MEMORY_MUTATION_SCHEMA
    version: int = LONGTERM_MEMORY_MUTATION_VERSION

    def __post_init__(self) -> None:
        if self.action not in LONGTERM_MEMORY_MUTATION_ACTIONS:
            raise ValueError(f"action must be one of: {', '.join(sorted(LONGTERM_MEMORY_MUTATION_ACTIONS))}.")
        if not _normalize_text(self.target_memory_id):
            raise ValueError("target_memory_id is required.")
        if not self.updated_objects and not self.deleted_memory_ids:
            raise ValueError("mutation result must contain updated objects or deleted ids.")
        if self.schema != LONGTERM_MEMORY_MUTATION_SCHEMA:
            raise ValueError(f"schema must be {LONGTERM_MEMORY_MUTATION_SCHEMA!r}.")
        if self.version != LONGTERM_MEMORY_MUTATION_VERSION:
            raise ValueError(f"version must be {LONGTERM_MEMORY_MUTATION_VERSION}.")


__all__ = [
    "LONGTERM_CONSOLIDATION_SCHEMA",
    "LONGTERM_CONSOLIDATION_VERSION",
    "LONGTERM_CONFLICT_QUEUE_SCHEMA",
    "LONGTERM_CONFLICT_QUEUE_VERSION",
    "LONGTERM_CONFLICT_RESOLUTION_SCHEMA",
    "LONGTERM_CONFLICT_RESOLUTION_VERSION",
    "LONGTERM_MEMORY_CONFLICT_SCHEMA",
    "LONGTERM_MEMORY_CONFLICT_VERSION",
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
    "LongTermMultimodalEvidence",
    "LongTermProactiveCandidateV1",
    "LongTermProactivePlanV1",
    "LongTermRetentionResultV1",
    "LongTermReflectionResultV1",
    "LongTermSourceRefV1",
    "LongTermTurnExtractionV1",
]
