"""Define structured personality state, learning signals, and deltas.

The personality package separates prompt-facing companion identity from the raw
evidence that should shape it over time. These models cover three levels:

- stable promptable state such as traits, humor, place/world context, and
  reflection notes
- persistent learning evidence such as interaction, place, and world signals
- small policy-gated personality deltas derived from repeated evidence

All models remain storage-agnostic and serialize into plain payload mappings so
the same types can be persisted via remote snapshots today and a more direct
ChonkyDB object layer later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from twinr.agent.personality._payload_utils import (
    clean_text as _clean_text,
    mapping_items as _mapping_items,
    normalize_float as _normalize_float,
    normalize_int as _normalize_int,
    normalize_mapping as _normalize_mapping,
    normalize_string_tuple as _normalize_string_tuple,
    optional_text as _optional_text,
    required_mapping_text as _required_mapping_text,
)

DEFAULT_PERSONALITY_SNAPSHOT_KIND = "agent_personality_context_v1"
INTERACTION_SIGNAL_SNAPSHOT_KIND = "agent_personality_interaction_signals_v1"
PLACE_SIGNAL_SNAPSHOT_KIND = "agent_personality_place_signals_v1"
WORLD_SIGNAL_SNAPSHOT_KIND = "agent_personality_world_signals_v1"
PERSONALITY_DELTA_SNAPSHOT_KIND = "agent_personality_deltas_v1"


def _normalize_signed_float(
    value: object | None,
    *,
    field_name: str,
    default: float = 0.0,
) -> float:
    """Normalize a value onto the inclusive -1..1 band."""

    if value is None:
        return default
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if parsed < -1.0:
        return -1.0
    if parsed > 1.0:
        return 1.0
    return parsed


@dataclass(frozen=True, slots=True)
class PersonalityTrait:
    """Describe one stable or slowly evolving Twinr character trait."""

    name: str
    summary: str
    weight: float = 0.5
    stable: bool = True
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize trait fields for prompt-safe downstream rendering."""

        object.__setattr__(self, "name", _required_mapping_text({"name": self.name}, field_name="name"))
        object.__setattr__(self, "summary", _required_mapping_text({"summary": self.summary}, field_name="summary"))
        object.__setattr__(self, "weight", _normalize_float(self.weight, field_name="weight", default=0.5))
        object.__setattr__(self, "stable", bool(self.stable))
        object.__setattr__(
            self,
            "evidence",
            _normalize_string_tuple(self.evidence, field_name="evidence"),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the trait into a JSON-safe mapping."""

        return {
            "name": self.name,
            "summary": self.summary,
            "weight": self.weight,
            "stable": self.stable,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PersonalityTrait:
        """Build a trait from a snapshot payload item."""

        return cls(
            name=_required_mapping_text(payload, field_name="name", aliases=("trait",)),
            summary=_required_mapping_text(payload, field_name="summary", aliases=("description",)),
            weight=payload.get("weight"),
            stable=payload.get("stable", True),
            evidence=_normalize_string_tuple(payload.get("evidence"), field_name="evidence"),
        )


@dataclass(frozen=True, slots=True)
class HumorProfile:
    """Describe Twinr's currently learned humor style and boundaries."""

    style: str
    summary: str
    intensity: float = 0.25
    boundaries: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize humor metadata for deterministic rendering."""

        object.__setattr__(self, "style", _required_mapping_text({"style": self.style}, field_name="style"))
        object.__setattr__(self, "summary", _required_mapping_text({"summary": self.summary}, field_name="summary"))
        object.__setattr__(
            self,
            "intensity",
            _normalize_float(self.intensity, field_name="intensity", default=0.25),
        )
        object.__setattr__(
            self,
            "boundaries",
            _normalize_string_tuple(self.boundaries, field_name="boundaries"),
        )
        object.__setattr__(
            self,
            "evidence",
            _normalize_string_tuple(self.evidence, field_name="evidence"),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the humor profile into a JSON-safe mapping."""

        return {
            "style": self.style,
            "summary": self.summary,
            "intensity": self.intensity,
            "boundaries": list(self.boundaries),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> HumorProfile:
        """Build a humor profile from a snapshot payload item."""

        return cls(
            style=_required_mapping_text(payload, field_name="style"),
            summary=_required_mapping_text(payload, field_name="summary", aliases=("description",)),
            intensity=payload.get("intensity"),
            boundaries=payload.get("boundaries"),
            evidence=payload.get("evidence"),
        )


@dataclass(frozen=True, slots=True)
class ConversationStyleProfile:
    """Describe Twinr's learned default verbosity and initiative bands."""

    verbosity: float = 0.5
    initiative: float = 0.45
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize style-profile values into deterministic bounded fields."""

        object.__setattr__(
            self,
            "verbosity",
            _normalize_float(self.verbosity, field_name="verbosity", default=0.5),
        )
        object.__setattr__(
            self,
            "initiative",
            _normalize_float(self.initiative, field_name="initiative", default=0.45),
        )
        object.__setattr__(
            self,
            "evidence",
            _normalize_string_tuple(self.evidence, field_name="evidence"),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the conversation style profile into a JSON-safe mapping."""

        return {
            "verbosity": self.verbosity,
            "initiative": self.initiative,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ConversationStyleProfile":
        """Build a style profile from a snapshot payload item."""

        return cls(
            verbosity=payload.get("verbosity"),
            initiative=payload.get("initiative"),
            evidence=payload.get("evidence"),
        )


@dataclass(frozen=True, slots=True)
class RelationshipSignal:
    """Capture one durable relationship or user-interest learning."""

    topic: str
    summary: str
    salience: float = 0.5
    source: str = "conversation"
    stance: str = "affinity"
    updated_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize one relationship signal."""

        object.__setattr__(self, "topic", _required_mapping_text({"topic": self.topic}, field_name="topic"))
        object.__setattr__(self, "summary", _required_mapping_text({"summary": self.summary}, field_name="summary"))
        object.__setattr__(self, "salience", _normalize_float(self.salience, field_name="salience", default=0.5))
        object.__setattr__(self, "source", _required_mapping_text({"source": self.source}, field_name="source"))
        normalized_stance = _clean_text(self.stance).casefold() or "affinity"
        if normalized_stance not in {"affinity", "aversion"}:
            raise ValueError("stance must be either 'affinity' or 'aversion'.")
        object.__setattr__(self, "stance", normalized_stance)
        object.__setattr__(self, "updated_at", _optional_text(self.updated_at))

    def to_payload(self) -> dict[str, object]:
        """Serialize the relationship signal into a JSON-safe mapping."""

        payload = {
            "topic": self.topic,
            "summary": self.summary,
            "salience": self.salience,
            "source": self.source,
            "stance": self.stance,
        }
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RelationshipSignal:
        """Build a relationship signal from a snapshot payload item."""

        return cls(
            topic=_required_mapping_text(payload, field_name="topic", aliases=("name",)),
            summary=_required_mapping_text(payload, field_name="summary", aliases=("description",)),
            salience=payload.get("salience"),
            source=_clean_text(payload.get("source")) or "conversation",
            stance=_clean_text(payload.get("stance")) or "affinity",
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True, slots=True)
class ContinuityThread:
    """Describe one topic or life-thread Twinr should keep warm."""

    title: str
    summary: str
    salience: float = 0.5
    updated_at: str | None = None
    expires_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize continuity-thread fields."""

        object.__setattr__(self, "title", _required_mapping_text({"title": self.title}, field_name="title"))
        object.__setattr__(self, "summary", _required_mapping_text({"summary": self.summary}, field_name="summary"))
        object.__setattr__(self, "salience", _normalize_float(self.salience, field_name="salience", default=0.5))
        object.__setattr__(self, "updated_at", _optional_text(self.updated_at))
        object.__setattr__(self, "expires_at", _optional_text(self.expires_at))

    def to_payload(self) -> dict[str, object]:
        """Serialize the continuity thread into a JSON-safe mapping."""

        payload = {
            "title": self.title,
            "summary": self.summary,
            "salience": self.salience,
        }
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.expires_at is not None:
            payload["expires_at"] = self.expires_at
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ContinuityThread:
        """Build a continuity thread from a snapshot payload item."""

        return cls(
            title=_required_mapping_text(payload, field_name="title", aliases=("topic", "name")),
            summary=_required_mapping_text(payload, field_name="summary", aliases=("description",)),
            salience=payload.get("salience"),
            updated_at=payload.get("updated_at"),
            expires_at=payload.get("expires_at"),
        )


@dataclass(frozen=True, slots=True)
class PlaceFocus:
    """Describe one geographic area Twinr should treat as relevant context."""

    name: str
    summary: str
    geography: str | None = None
    salience: float = 0.5
    updated_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize a place-focus payload."""

        object.__setattr__(self, "name", _required_mapping_text({"name": self.name}, field_name="name"))
        object.__setattr__(self, "summary", _required_mapping_text({"summary": self.summary}, field_name="summary"))
        object.__setattr__(self, "geography", _optional_text(self.geography))
        object.__setattr__(self, "salience", _normalize_float(self.salience, field_name="salience", default=0.5))
        object.__setattr__(self, "updated_at", _optional_text(self.updated_at))

    def to_payload(self) -> dict[str, object]:
        """Serialize the place focus into a JSON-safe mapping."""

        payload = {
            "name": self.name,
            "summary": self.summary,
            "salience": self.salience,
        }
        if self.geography is not None:
            payload["geography"] = self.geography
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PlaceFocus:
        """Build a place focus from a snapshot payload item."""

        return cls(
            name=_required_mapping_text(payload, field_name="name", aliases=("place", "title")),
            summary=_required_mapping_text(payload, field_name="summary", aliases=("description",)),
            geography=_optional_text(payload.get("geography") or payload.get("scope")),
            salience=payload.get("salience"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True, slots=True)
class InteractionSignal:
    """Capture one interaction-derived learning signal before gating."""

    signal_id: str
    signal_kind: str
    target: str
    summary: str
    confidence: float = 0.5
    impact: float = 0.0
    evidence_count: int = 1
    source_event_ids: tuple[str, ...] = ()
    delta_target: str | None = None
    delta_value: float | None = None
    delta_summary: str | None = None
    explicit_user_requested: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Normalize one interaction signal."""

        object.__setattr__(self, "signal_id", _required_mapping_text({"signal_id": self.signal_id}, field_name="signal_id"))
        object.__setattr__(self, "signal_kind", _required_mapping_text({"signal_kind": self.signal_kind}, field_name="signal_kind"))
        object.__setattr__(self, "target", _required_mapping_text({"target": self.target}, field_name="target"))
        object.__setattr__(self, "summary", _required_mapping_text({"summary": self.summary}, field_name="summary"))
        object.__setattr__(self, "confidence", _normalize_float(self.confidence, field_name="confidence", default=0.5))
        object.__setattr__(self, "impact", _normalize_signed_float(self.impact, field_name="impact"))
        object.__setattr__(
            self,
            "evidence_count",
            _normalize_int(self.evidence_count, field_name="evidence_count", default=1, minimum=1),
        )
        object.__setattr__(
            self,
            "source_event_ids",
            _normalize_string_tuple(self.source_event_ids, field_name="source_event_ids"),
        )
        object.__setattr__(self, "delta_target", _optional_text(self.delta_target))
        if self.delta_value is None:
            normalized_delta_value = None
        else:
            normalized_delta_value = _normalize_signed_float(
                self.delta_value,
                field_name="delta_value",
            )
        object.__setattr__(self, "delta_value", normalized_delta_value)
        object.__setattr__(self, "delta_summary", _optional_text(self.delta_summary))
        object.__setattr__(self, "explicit_user_requested", bool(self.explicit_user_requested))
        object.__setattr__(self, "created_at", _optional_text(self.created_at))
        object.__setattr__(self, "updated_at", _optional_text(self.updated_at))
        object.__setattr__(self, "metadata", _normalize_mapping(self.metadata, field_name="metadata"))

    def to_payload(self) -> dict[str, object]:
        """Serialize the interaction signal into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "signal_id": self.signal_id,
            "signal_kind": self.signal_kind,
            "target": self.target,
            "summary": self.summary,
            "confidence": self.confidence,
            "impact": self.impact,
            "evidence_count": self.evidence_count,
            "source_event_ids": list(self.source_event_ids),
            "explicit_user_requested": self.explicit_user_requested,
        }
        if self.delta_target is not None:
            payload["delta_target"] = self.delta_target
        if self.delta_value is not None:
            payload["delta_value"] = self.delta_value
        if self.delta_summary is not None:
            payload["delta_summary"] = self.delta_summary
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.metadata is not None:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> InteractionSignal:
        """Build an interaction signal from a persisted payload item."""

        return cls(
            signal_id=_required_mapping_text(payload, field_name="signal_id", aliases=("id",)),
            signal_kind=_required_mapping_text(payload, field_name="signal_kind", aliases=("kind",)),
            target=_required_mapping_text(payload, field_name="target"),
            summary=_required_mapping_text(payload, field_name="summary", aliases=("description",)),
            confidence=payload.get("confidence"),
            impact=payload.get("impact"),
            evidence_count=payload.get("evidence_count"),
            source_event_ids=payload.get("source_event_ids"),
            delta_target=payload.get("delta_target"),
            delta_value=payload.get("delta_value"),
            delta_summary=payload.get("delta_summary"),
            explicit_user_requested=payload.get("explicit_user_requested", False),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            metadata=payload.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class PlaceSignal:
    """Capture one place-specific learning or relevance signal."""

    signal_id: str
    place_name: str
    summary: str
    geography: str | None = None
    salience: float = 0.5
    confidence: float = 0.5
    evidence_count: int = 1
    source_event_ids: tuple[str, ...] = ()
    updated_at: str | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Normalize one place signal."""

        object.__setattr__(self, "signal_id", _required_mapping_text({"signal_id": self.signal_id}, field_name="signal_id"))
        object.__setattr__(self, "place_name", _required_mapping_text({"place_name": self.place_name}, field_name="place_name"))
        object.__setattr__(self, "summary", _required_mapping_text({"summary": self.summary}, field_name="summary"))
        object.__setattr__(self, "geography", _optional_text(self.geography))
        object.__setattr__(self, "salience", _normalize_float(self.salience, field_name="salience", default=0.5))
        object.__setattr__(self, "confidence", _normalize_float(self.confidence, field_name="confidence", default=0.5))
        object.__setattr__(
            self,
            "evidence_count",
            _normalize_int(self.evidence_count, field_name="evidence_count", default=1, minimum=1),
        )
        object.__setattr__(
            self,
            "source_event_ids",
            _normalize_string_tuple(self.source_event_ids, field_name="source_event_ids"),
        )
        object.__setattr__(self, "updated_at", _optional_text(self.updated_at))
        object.__setattr__(self, "metadata", _normalize_mapping(self.metadata, field_name="metadata"))

    def to_payload(self) -> dict[str, object]:
        """Serialize the place signal into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "signal_id": self.signal_id,
            "place_name": self.place_name,
            "summary": self.summary,
            "salience": self.salience,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "source_event_ids": list(self.source_event_ids),
        }
        if self.geography is not None:
            payload["geography"] = self.geography
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.metadata is not None:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_place_focus(self) -> PlaceFocus:
        """Convert the signal into prompt-facing place context."""

        return PlaceFocus(
            name=self.place_name,
            summary=self.summary,
            geography=self.geography,
            salience=self.salience,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PlaceSignal:
        """Build a place signal from a persisted payload item."""

        return cls(
            signal_id=_required_mapping_text(payload, field_name="signal_id", aliases=("id",)),
            place_name=_required_mapping_text(payload, field_name="place_name", aliases=("name", "place")),
            summary=_required_mapping_text(payload, field_name="summary", aliases=("description",)),
            geography=payload.get("geography"),
            salience=payload.get("salience"),
            confidence=payload.get("confidence"),
            evidence_count=payload.get("evidence_count"),
            source_event_ids=payload.get("source_event_ids"),
            updated_at=payload.get("updated_at"),
            metadata=payload.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class WorldSignal:
    """Capture one relevant world or news development for prompt context."""

    topic: str
    summary: str
    region: str | None = None
    source: str = "world"
    salience: float = 0.5
    fresh_until: str | None = None
    evidence_count: int = 1
    source_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize a world-signal payload."""

        object.__setattr__(self, "topic", _required_mapping_text({"topic": self.topic}, field_name="topic"))
        object.__setattr__(self, "summary", _required_mapping_text({"summary": self.summary}, field_name="summary"))
        object.__setattr__(self, "region", _optional_text(self.region))
        object.__setattr__(self, "source", _required_mapping_text({"source": self.source}, field_name="source"))
        object.__setattr__(self, "salience", _normalize_float(self.salience, field_name="salience", default=0.5))
        object.__setattr__(self, "fresh_until", _optional_text(self.fresh_until))
        object.__setattr__(
            self,
            "evidence_count",
            _normalize_int(self.evidence_count, field_name="evidence_count", default=1, minimum=1),
        )
        object.__setattr__(
            self,
            "source_event_ids",
            _normalize_string_tuple(self.source_event_ids, field_name="source_event_ids"),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the world signal into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "topic": self.topic,
            "summary": self.summary,
            "source": self.source,
            "salience": self.salience,
            "evidence_count": self.evidence_count,
            "source_event_ids": list(self.source_event_ids),
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.fresh_until is not None:
            payload["fresh_until"] = self.fresh_until
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> WorldSignal:
        """Build a world signal from a snapshot payload item."""

        return cls(
            topic=_required_mapping_text(payload, field_name="topic", aliases=("title", "name")),
            summary=_required_mapping_text(payload, field_name="summary", aliases=("description",)),
            region=_optional_text(payload.get("region") or payload.get("scope")),
            source=_clean_text(payload.get("source")) or "world",
            salience=payload.get("salience"),
            fresh_until=payload.get("fresh_until") or payload.get("valid_until"),
            evidence_count=payload.get("evidence_count"),
            source_event_ids=payload.get("source_event_ids"),
        )


@dataclass(frozen=True, slots=True)
class ReflectionDelta:
    """Describe one small personality adjustment proposal or learning delta."""

    target: str
    change: str
    reason: str
    confidence: float = 0.5
    review_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize one reflection-delta payload."""

        object.__setattr__(self, "target", _required_mapping_text({"target": self.target}, field_name="target"))
        object.__setattr__(self, "change", _required_mapping_text({"change": self.change}, field_name="change"))
        object.__setattr__(self, "reason", _required_mapping_text({"reason": self.reason}, field_name="reason"))
        object.__setattr__(
            self,
            "confidence",
            _normalize_float(self.confidence, field_name="confidence", default=0.5),
        )
        object.__setattr__(self, "review_at", _optional_text(self.review_at))

    def to_payload(self) -> dict[str, object]:
        """Serialize the reflection delta into a JSON-safe mapping."""

        payload = {
            "target": self.target,
            "change": self.change,
            "reason": self.reason,
            "confidence": self.confidence,
        }
        if self.review_at is not None:
            payload["review_at"] = self.review_at
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ReflectionDelta:
        """Build a reflection delta from a snapshot payload item."""

        return cls(
            target=_required_mapping_text(payload, field_name="target"),
            change=_required_mapping_text(payload, field_name="change"),
            reason=_required_mapping_text(payload, field_name="reason"),
            confidence=payload.get("confidence"),
            review_at=payload.get("review_at"),
        )


@dataclass(frozen=True, slots=True)
class PersonalityDelta:
    """Describe one policy-gated personality change derived from signals."""

    delta_id: str
    target: str
    summary: str
    rationale: str
    delta_value: float
    confidence: float = 0.5
    support_count: int = 1
    source_signal_ids: tuple[str, ...] = ()
    status: str = "candidate"
    review_at: str | None = None
    explicit_user_requested: bool = False

    def __post_init__(self) -> None:
        """Normalize one persistent personality delta."""

        object.__setattr__(self, "delta_id", _required_mapping_text({"delta_id": self.delta_id}, field_name="delta_id"))
        object.__setattr__(self, "target", _required_mapping_text({"target": self.target}, field_name="target"))
        object.__setattr__(self, "summary", _required_mapping_text({"summary": self.summary}, field_name="summary"))
        object.__setattr__(self, "rationale", _required_mapping_text({"rationale": self.rationale}, field_name="rationale"))
        object.__setattr__(self, "delta_value", _normalize_signed_float(self.delta_value, field_name="delta_value"))
        object.__setattr__(
            self,
            "confidence",
            _normalize_float(self.confidence, field_name="confidence", default=0.5),
        )
        object.__setattr__(
            self,
            "support_count",
            _normalize_int(self.support_count, field_name="support_count", default=1, minimum=1),
        )
        object.__setattr__(
            self,
            "source_signal_ids",
            _normalize_string_tuple(self.source_signal_ids, field_name="source_signal_ids"),
        )
        object.__setattr__(self, "status", _required_mapping_text({"status": self.status}, field_name="status"))
        object.__setattr__(self, "review_at", _optional_text(self.review_at))
        object.__setattr__(self, "explicit_user_requested", bool(self.explicit_user_requested))

    def to_payload(self) -> dict[str, object]:
        """Serialize the persistent delta into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "delta_id": self.delta_id,
            "target": self.target,
            "summary": self.summary,
            "rationale": self.rationale,
            "delta_value": self.delta_value,
            "confidence": self.confidence,
            "support_count": self.support_count,
            "source_signal_ids": list(self.source_signal_ids),
            "status": self.status,
            "explicit_user_requested": self.explicit_user_requested,
        }
        if self.review_at is not None:
            payload["review_at"] = self.review_at
        return payload

    def to_reflection_delta(self) -> ReflectionDelta:
        """Convert the persistent delta into prompt-facing reflection context."""

        return ReflectionDelta(
            target=self.target,
            change=self.summary,
            reason=self.rationale,
            confidence=self.confidence,
            review_at=self.review_at,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PersonalityDelta:
        """Build a personality delta from a persisted payload item."""

        return cls(
            delta_id=_required_mapping_text(payload, field_name="delta_id", aliases=("id",)),
            target=_required_mapping_text(payload, field_name="target"),
            summary=_required_mapping_text(payload, field_name="summary"),
            rationale=_required_mapping_text(payload, field_name="rationale", aliases=("reason",)),
            delta_value=payload.get("delta_value"),
            confidence=payload.get("confidence"),
            support_count=payload.get("support_count"),
            source_signal_ids=payload.get("source_signal_ids"),
            status=_clean_text(payload.get("status")) or "candidate",
            review_at=payload.get("review_at"),
            explicit_user_requested=payload.get("explicit_user_requested", False),
        )


@dataclass(frozen=True, slots=True)
class PersonalitySnapshot:
    """Store the structured evolving personality state for one prompt build."""

    schema_version: int = 1
    generated_at: str | None = None
    core_traits: tuple[PersonalityTrait, ...] = ()
    style_profile: ConversationStyleProfile | None = None
    humor_profile: HumorProfile | None = None
    relationship_signals: tuple[RelationshipSignal, ...] = ()
    continuity_threads: tuple[ContinuityThread, ...] = ()
    place_focuses: tuple[PlaceFocus, ...] = ()
    world_signals: tuple[WorldSignal, ...] = ()
    reflection_deltas: tuple[ReflectionDelta, ...] = ()
    personality_deltas: tuple[PersonalityDelta, ...] = ()

    def __post_init__(self) -> None:
        """Normalize top-level snapshot fields."""

        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "generated_at", _optional_text(self.generated_at))
        object.__setattr__(self, "core_traits", tuple(self.core_traits))
        object.__setattr__(self, "relationship_signals", tuple(self.relationship_signals))
        object.__setattr__(self, "continuity_threads", tuple(self.continuity_threads))
        object.__setattr__(self, "place_focuses", tuple(self.place_focuses))
        object.__setattr__(self, "world_signals", tuple(self.world_signals))
        object.__setattr__(self, "reflection_deltas", tuple(self.reflection_deltas))
        object.__setattr__(self, "personality_deltas", tuple(self.personality_deltas))

    def to_payload(self) -> dict[str, object]:
        """Serialize the snapshot into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "core_traits": [item.to_payload() for item in self.core_traits],
            "relationship_signals": [item.to_payload() for item in self.relationship_signals],
            "continuity_threads": [item.to_payload() for item in self.continuity_threads],
            "place_focuses": [item.to_payload() for item in self.place_focuses],
            "world_signals": [item.to_payload() for item in self.world_signals],
            "reflection_deltas": [item.to_payload() for item in self.reflection_deltas],
            "personality_deltas": [item.to_payload() for item in self.personality_deltas],
        }
        if self.generated_at is not None:
            payload["generated_at"] = self.generated_at
        if self.style_profile is not None:
            payload["style_profile"] = self.style_profile.to_payload()
        if self.humor_profile is not None:
            payload["humor_profile"] = self.humor_profile.to_payload()
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PersonalitySnapshot:
        """Parse a stored mapping into a typed personality snapshot."""

        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping.")
        raw_humor = payload.get("humor_profile", payload.get("humor"))
        humor_profile = None
        if raw_humor is not None:
            if not isinstance(raw_humor, Mapping):
                raise ValueError("humor_profile must be a mapping.")
            humor_profile = HumorProfile.from_payload(raw_humor)
        raw_style = payload.get("style_profile", payload.get("conversation_style"))
        style_profile = None
        if raw_style is not None:
            if not isinstance(raw_style, Mapping):
                raise ValueError("style_profile must be a mapping.")
            style_profile = ConversationStyleProfile.from_payload(raw_style)
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            generated_at=_optional_text(payload.get("generated_at")),
            core_traits=tuple(
                PersonalityTrait.from_payload(item)
                for item in _mapping_items(payload.get("core_traits"), field_name="core_traits")
            ),
            style_profile=style_profile,
            humor_profile=humor_profile,
            relationship_signals=tuple(
                RelationshipSignal.from_payload(item)
                for item in _mapping_items(payload.get("relationship_signals"), field_name="relationship_signals")
            ),
            continuity_threads=tuple(
                ContinuityThread.from_payload(item)
                for item in _mapping_items(payload.get("continuity_threads"), field_name="continuity_threads")
            ),
            place_focuses=tuple(
                PlaceFocus.from_payload(item)
                for item in _mapping_items(payload.get("place_focuses"), field_name="place_focuses")
            ),
            world_signals=tuple(
                WorldSignal.from_payload(item)
                for item in _mapping_items(payload.get("world_signals"), field_name="world_signals")
            ),
            reflection_deltas=tuple(
                ReflectionDelta.from_payload(item)
                for item in _mapping_items(payload.get("reflection_deltas"), field_name="reflection_deltas")
            ),
            personality_deltas=tuple(
                PersonalityDelta.from_payload(item)
                for item in _mapping_items(payload.get("personality_deltas"), field_name="personality_deltas")
            ),
        )


@dataclass(frozen=True, slots=True)
class PersonalityPromptLayer:
    """Represent one rendered prompt layer before legacy assembly."""

    layer_id: str
    title: str
    content: str
    source: str
    instruction_authority: bool = False

    def __post_init__(self) -> None:
        """Normalize prompt-layer fields for deterministic rendering."""

        object.__setattr__(self, "layer_id", _required_mapping_text({"layer_id": self.layer_id}, field_name="layer_id"))
        object.__setattr__(self, "title", _required_mapping_text({"title": self.title}, field_name="title"))
        object.__setattr__(self, "content", _required_mapping_text({"content": self.content}, field_name="content"))
        object.__setattr__(self, "source", _required_mapping_text({"source": self.source}, field_name="source"))
        object.__setattr__(self, "instruction_authority", bool(self.instruction_authority))

    def to_section(self) -> tuple[str, str]:
        """Return the legacy ``(title, content)`` section tuple."""

        return (self.title, self.content)


@dataclass(frozen=True, slots=True)
class PersonalityPromptPlan:
    """Store the ordered prompt layers emitted by the context builder."""

    layers: tuple[PersonalityPromptLayer, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the stored layer tuple."""

        object.__setattr__(self, "layers", tuple(self.layers))

    def as_sections(self) -> tuple[tuple[str, str], ...]:
        """Convert the ordered layers into legacy prompt sections."""

        return tuple(layer.to_section() for layer in self.layers)
