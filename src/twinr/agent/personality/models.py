# CHANGELOG: 2026-03-27
# BUG-1: Reject NaN/inf for signed and unit floats; the old code could persist non-finite values and emit non-portable/invalid JSON payloads.
# BUG-2: Coerce/validate nested mappings in PersonalitySnapshot and PersonalityPromptPlan eagerly; the old code accepted raw dicts and failed later at to_payload()/as_sections().
# BUG-3: Validate PersonalityDelta.status against an explicit state machine; the old code allowed arbitrary strings that can silently break downstream gating logic.
# SEC-1: Bound and sanitize prompt-facing text, tuples, and metadata to reduce practical memory/context poisoning and Pi 4 prompt-bloat DoS risk from untrusted persisted content.
# SEC-2: Guard instruction_authority so only trusted prompt-layer sources can claim authority; this blocks accidental privilege escalation from user/memory-sourced layers.
# IMP-1: Add canonical ISO-8601 timestamp normalization plus freshness/expiry helpers for time-aware memory management.
# IMP-2: Add provenance/trust_score fields to learned signals and deltas, enabling trust-aware retrieval and memory sanitization patterns used by 2026 memory-agent stacks.
# IMP-3: Add compact_for_prompt(), canonical JSON helpers, payload digests, and optional JSON/MessagePack helpers for low-latency, schema-aware on-device operation.
# IMP-4: Keep backward parsing compatibility for v1 payloads while upgrading schema_version to 2.
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

2026 upgrade notes:
- schema_version defaults to 2 but v1 payloads remain readable
- prompt-facing fields are now bounded and canonicalized
- trusted-memory hooks (provenance/trust_score) are built in
- optional msgspec helpers are exposed when msgspec is installed
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, TypeVar, cast

try:  # pragma: no cover - optional dependency
    import msgspec as _msgspec  # pylint: disable=import-error
except Exception:  # pragma: no cover - optional dependency
    _msgspec = None

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

CURRENT_PERSONALITY_SCHEMA_VERSION = 2

DEFAULT_PERSONALITY_SNAPSHOT_KIND = "agent_personality_context_v1"
INTERACTION_SIGNAL_SNAPSHOT_KIND = "agent_personality_interaction_signals_v1"
PLACE_SIGNAL_SNAPSHOT_KIND = "agent_personality_place_signals_v1"
WORLD_SIGNAL_SNAPSHOT_KIND = "agent_personality_world_signals_v1"
PERSONALITY_DELTA_SNAPSHOT_KIND = "agent_personality_deltas_v1"

_MAX_NAME_LEN = 128
_MAX_TITLE_LEN = 160
_MAX_TEXT_LEN = 2048
_MAX_CONTENT_LEN = 8192
_MAX_SHORT_TEXT_LEN = 256
_MAX_ID_LEN = 128
_MAX_STRING_ITEMS = 64
_MAX_SIGNAL_IDS = 64
_MAX_METADATA_ITEMS = 64
_MAX_METADATA_DEPTH = 4
_MAX_TOP_LEVEL_ITEMS = 64

_ALLOWED_STANCES = frozenset({"affinity", "aversion", "neutral"})
_ALLOWED_DELTA_STATUSES = frozenset(
    {"candidate", "accepted", "applied", "rejected", "expired", "revoked"}
)
_DELTA_STATUS_ALIASES = {
    "approved": "accepted",
    "active": "applied",
    "done": "applied",
    "dismissed": "rejected",
}
_TRUSTED_PROMPT_LAYER_PREFIXES = (
    "system",
    "policy",
    "builder",
    "runtime-policy",
    "trusted",
)

_TModel = TypeVar("_TModel")


def _sanitize_text(text: str, *, field_name: str, max_length: int) -> str:
    """Remove control characters and enforce a bounded prompt-safe text field."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        ch if ch in {"\n", "\t"} or ord(ch) >= 32 else " "
        for ch in normalized
    ).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if len(normalized) > max_length:
        normalized = normalized[: max_length - 1].rstrip() + "…"
    return normalized


def _normalize_required_text_value(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str:
    raw = _required_mapping_text({field_name: value}, field_name=field_name)
    return _sanitize_text(raw, field_name=field_name, max_length=max_length)


def _normalize_required_text_from_mapping(
    payload: Mapping[str, object],
    *,
    field_name: str,
    max_length: int,
    aliases: tuple[str, ...] = (),
) -> str:
    raw = _required_mapping_text(payload, field_name=field_name, aliases=aliases)
    return _sanitize_text(raw, field_name=field_name, max_length=max_length)


def _normalize_optional_text_value(
    value: object | None,
    *,
    field_name: str,
    max_length: int,
) -> str | None:
    raw = _optional_text(value)
    if raw is None:
        return None
    return _sanitize_text(raw, field_name=field_name, max_length=max_length)


def _normalize_unit_float(
    value: object | None,
    *,
    field_name: str,
    default: float = 0.0,
) -> float:
    """Normalize a finite float onto the inclusive 0..1 band."""

    parsed = _normalize_float(value, field_name=field_name, default=default)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _normalize_signed_float(
    value: object | None,
    *,
    field_name: str,
    default: float = 0.0,
) -> float:
    """Normalize a finite float onto the inclusive -1..1 band."""

    if value is None:
        return default
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    if parsed < -1.0:
        return -1.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _normalize_trust_score(
    value: object | None,
    *,
    field_name: str = "trust_score",
) -> float:
    return _normalize_unit_float(value, field_name=field_name, default=1.0)


def _normalize_string_tuple_bounded(
    value: object | None,
    *,
    field_name: str,
    max_items: int = _MAX_STRING_ITEMS,
    item_max_length: int = _MAX_SHORT_TEXT_LEN,
) -> tuple[str, ...]:
    normalized = _normalize_string_tuple(value, field_name=field_name)
    items: list[str] = []
    seen: set[str] = set()
    for raw_item in normalized:
        item = _normalize_optional_text_value(
            raw_item,
            field_name=field_name,
            max_length=item_max_length,
        )
        if item is None or item in seen:
            continue
        items.append(item)
        seen.add(item)
        if len(items) >= max_items:
            break
    return tuple(items)


def _normalize_enum_text(
    value: object | None,
    *,
    field_name: str,
    allowed: frozenset[str],
    default: str,
    aliases: Mapping[str, str] | None = None,
) -> str:
    normalized = _clean_text(value).casefold() or default
    if aliases is not None:
        normalized = aliases.get(normalized, normalized)
    if normalized not in allowed:
        allowed_display = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {allowed_display}.")
    return normalized


def _normalize_timestamp(value: object | None, *, field_name: str) -> str | None:
    """Validate and canonicalize a date/datetime into ISO-8601 text."""

    text = _normalize_optional_text_value(
        value,
        field_name=field_name,
        max_length=_MAX_SHORT_TEXT_LEN,
    )
    if text is None:
        return None
    if len(text) == 10:
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            pass
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        parsed_dt = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 date or datetime.") from exc
    if parsed_dt.tzinfo is not None:
        rendered = parsed_dt.astimezone(timezone.utc).isoformat()
        if rendered.endswith("+00:00"):
            rendered = rendered[:-6] + "Z"
        return rendered
    return parsed_dt.isoformat()


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse canonical timestamp text for comparisons."""

    if value is None:
        return None
    candidate = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        parsed_dt = datetime.fromisoformat(candidate)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError:
            return None
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=timezone.utc)
    if parsed_dt.tzinfo is None:
        return parsed_dt.replace(tzinfo=timezone.utc)
    return parsed_dt.astimezone(timezone.utc)


def _timestamp_rank(value: str | None) -> float:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return float("-inf")
    return parsed.timestamp()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _freeze_jsonish(value: object, *, field_name: str, depth: int = 0) -> object:
    """Deep-freeze metadata into a bounded, JSON-safe, immutable structure."""

    if depth > _MAX_METADATA_DEPTH:
        raise ValueError(
            f"{field_name} exceeds the maximum nesting depth of {_MAX_METADATA_DEPTH}."
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain NaN or infinity.")
        return value
    if isinstance(value, str):
        return _sanitize_text(value, field_name=field_name, max_length=_MAX_TEXT_LEN)
    if isinstance(value, datetime):
        rendered = value.astimezone(timezone.utc).isoformat() if value.tzinfo is not None else value.isoformat()
        return rendered.replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        frozen_dict: dict[str, object] = {}
        for idx, (raw_key, raw_value) in enumerate(value.items()):
            if idx >= _MAX_METADATA_ITEMS:
                break
            key = _sanitize_text(
                str(raw_key),
                field_name=f"{field_name} key",
                max_length=_MAX_SHORT_TEXT_LEN,
            )
            frozen_dict[key] = _freeze_jsonish(
                raw_value,
                field_name=f"{field_name}.{key}",
                depth=depth + 1,
            )
        return MappingProxyType(frozen_dict)
    if isinstance(value, (list, tuple, set, frozenset)):
        frozen_list: list[object] = []
        for idx, item in enumerate(value):
            if idx >= _MAX_METADATA_ITEMS:
                break
            frozen_list.append(
                _freeze_jsonish(
                    item,
                    field_name=f"{field_name}[{idx}]",
                    depth=depth + 1,
                )
            )
        return tuple(frozen_list)
    raise ValueError(
        f"{field_name} contains unsupported metadata type {type(value).__name__}."
    )


def _normalize_metadata(
    value: object | None,
    *,
    field_name: str,
) -> Mapping[str, object] | None:
    normalized = _normalize_mapping(value, field_name=field_name)
    if normalized is None:
        return None
    frozen = _freeze_jsonish(normalized, field_name=field_name)
    return cast(Mapping[str, object], frozen)


def _thaw_jsonish(value: object) -> object:
    """Convert frozen metadata back into plain JSON-serializable containers."""

    if isinstance(value, Mapping):
        return {str(key): _thaw_jsonish(inner_value) for key, inner_value in value.items()}
    if isinstance(value, tuple):
        return [_thaw_jsonish(item) for item in value]
    return value


def _ensure_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _coerce_item(
    value: object,
    *,
    field_name: str,
    model_type: type[_TModel],
) -> _TModel:
    if isinstance(value, model_type):
        return value
    if isinstance(value, Mapping):
        return cast(_TModel, model_type.from_payload(value))  # type: ignore[attr-defined]
    raise ValueError(
        f"{field_name} items must be {model_type.__name__} instances or mappings."
    )


def _coerce_optional_item(
    value: object | None,
    *,
    field_name: str,
    model_type: type[_TModel],
) -> _TModel | None:
    if value is None:
        return None
    return _coerce_item(value, field_name=field_name, model_type=model_type)


def _coerce_tuple(
    values: object | None,
    *,
    field_name: str,
    model_type: type[_TModel],
    max_items: int = _MAX_TOP_LEVEL_ITEMS,
) -> tuple[_TModel, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{field_name} must be an iterable of {model_type.__name__} items.")
    iterable = cast(Iterable[object], values)
    items: list[_TModel] = []
    for raw_item in iterable:
        items.append(_coerce_item(raw_item, field_name=field_name, model_type=model_type))
        if len(items) >= max_items:
            break
    return tuple(items)


def _top_k(
    items: tuple[_TModel, ...],
    *,
    limit: int,
    score: Callable[[_TModel], tuple[object, ...]],
) -> tuple[_TModel, ...]:
    if len(items) <= limit:
        return items
    ranked = sorted(items, key=score, reverse=True)
    return tuple(ranked[:limit])


class _PayloadModel:
    """Common helpers for canonical serialization and hashing."""

    def to_payload(self) -> dict[str, object]:
        raise NotImplementedError

    def canonical_payload(self) -> dict[str, object]:
        return cast(dict[str, object], _thaw_jsonish(self.to_payload()))

    def to_json_str(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_json_bytes(self) -> bytes:
        return self.to_json_str().encode("utf-8")

    def payload_digest(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    def to_msgpack_bytes(self) -> bytes:
        """Encode the payload as MessagePack when msgspec is available."""

        if _msgspec is None:  # pragma: no cover - optional dependency
            raise RuntimeError("msgspec is required for MessagePack support.")
        return cast(bytes, _msgspec.msgpack.encode(self.canonical_payload()))

    @classmethod
    def from_json_bytes(cls, data: bytes | bytearray | str):
        """Decode JSON payload bytes or text through the class' from_payload parser."""

        if isinstance(data, (bytes, bytearray)):
            text = data.decode("utf-8")
        else:
            text = data
        raw_payload = json.loads(text)
        payload = _ensure_mapping(raw_payload, field_name="payload")
        return cls.from_payload(payload)  # type: ignore[attr-defined]  # pylint: disable=no-member

    @classmethod
    def from_msgpack_bytes(cls, data: bytes | bytearray):
        """Decode a MessagePack payload when msgspec is available."""

        if _msgspec is None:  # pragma: no cover - optional dependency
            raise RuntimeError("msgspec is required for MessagePack support.")
        raw_payload = _msgspec.msgpack.decode(bytes(data))
        payload = _ensure_mapping(raw_payload, field_name="payload")
        return cls.from_payload(payload)  # type: ignore[attr-defined]  # pylint: disable=no-member

    @classmethod
    def json_schema(cls) -> dict[str, object]:
        """Generate JSON Schema when msgspec is installed."""

        if _msgspec is None:  # pragma: no cover - optional dependency
            raise RuntimeError("msgspec is required for JSON Schema generation.")
        return cast(dict[str, object], _msgspec.json.schema(cls))


@dataclass(frozen=True, slots=True)
class PersonalityTrait(_PayloadModel):
    """Describe one stable or slowly evolving Twinr character trait."""

    name: str
    summary: str
    weight: float = 0.5
    stable: bool = True
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_required_text_value(self.name, field_name="name", max_length=_MAX_NAME_LEN),
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_required_text_value(self.summary, field_name="summary", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "weight",
            _normalize_unit_float(self.weight, field_name="weight", default=0.5),
        )
        object.__setattr__(self, "stable", bool(self.stable))
        object.__setattr__(
            self,
            "evidence",
            _normalize_string_tuple_bounded(self.evidence, field_name="evidence"),
        )

    def prompt_priority(self) -> tuple[float, float, int]:
        return (1.0 if self.stable else 0.0, self.weight, len(self.evidence))

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "summary": self.summary,
            "weight": self.weight,
            "stable": self.stable,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PersonalityTrait:
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            name=_normalize_required_text_from_mapping(
                payload,
                field_name="name",
                max_length=_MAX_NAME_LEN,
                aliases=("trait",),
            ),
            summary=_normalize_required_text_from_mapping(
                payload,
                field_name="summary",
                max_length=_MAX_TEXT_LEN,
                aliases=("description",),
            ),
            weight=payload.get("weight"),
            stable=payload.get("stable", True),
            evidence=payload.get("evidence"),
        )


@dataclass(frozen=True, slots=True)
class HumorProfile(_PayloadModel):
    """Describe Twinr's currently learned humor style and boundaries."""

    style: str
    summary: str
    intensity: float = 0.25
    boundaries: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "style",
            _normalize_required_text_value(self.style, field_name="style", max_length=_MAX_NAME_LEN),
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_required_text_value(self.summary, field_name="summary", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "intensity",
            _normalize_unit_float(self.intensity, field_name="intensity", default=0.25),
        )
        object.__setattr__(
            self,
            "boundaries",
            _normalize_string_tuple_bounded(self.boundaries, field_name="boundaries"),
        )
        object.__setattr__(
            self,
            "evidence",
            _normalize_string_tuple_bounded(self.evidence, field_name="evidence"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "style": self.style,
            "summary": self.summary,
            "intensity": self.intensity,
            "boundaries": list(self.boundaries),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> HumorProfile:
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            style=_normalize_required_text_from_mapping(
                payload,
                field_name="style",
                max_length=_MAX_NAME_LEN,
            ),
            summary=_normalize_required_text_from_mapping(
                payload,
                field_name="summary",
                max_length=_MAX_TEXT_LEN,
                aliases=("description",),
            ),
            intensity=payload.get("intensity"),
            boundaries=payload.get("boundaries"),
            evidence=payload.get("evidence"),
        )


@dataclass(frozen=True, slots=True)
class ConversationStyleProfile(_PayloadModel):
    """Describe Twinr's learned default verbosity and initiative bands."""

    verbosity: float = 0.5
    initiative: float = 0.45
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verbosity",
            _normalize_unit_float(self.verbosity, field_name="verbosity", default=0.5),
        )
        object.__setattr__(
            self,
            "initiative",
            _normalize_unit_float(self.initiative, field_name="initiative", default=0.45),
        )
        object.__setattr__(
            self,
            "evidence",
            _normalize_string_tuple_bounded(self.evidence, field_name="evidence"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "verbosity": self.verbosity,
            "initiative": self.initiative,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ConversationStyleProfile":
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            verbosity=payload.get("verbosity"),
            initiative=payload.get("initiative"),
            evidence=payload.get("evidence"),
        )


@dataclass(frozen=True, slots=True)
class RelationshipSignal(_PayloadModel):
    """Capture one durable relationship or user-interest learning."""

    topic: str
    summary: str
    salience: float = 0.5
    source: str = "conversation"
    stance: str = "affinity"
    updated_at: str | None = None
    provenance: str | None = None
    trust_score: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "topic",
            _normalize_required_text_value(self.topic, field_name="topic", max_length=_MAX_TITLE_LEN),
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_required_text_value(self.summary, field_name="summary", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "salience",
            _normalize_unit_float(self.salience, field_name="salience", default=0.5),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_required_text_value(self.source, field_name="source", max_length=_MAX_NAME_LEN),
        )
        object.__setattr__(
            self,
            "stance",
            _normalize_enum_text(
                self.stance,
                field_name="stance",
                allowed=_ALLOWED_STANCES,
                default="affinity",
            ),
        )
        object.__setattr__(self, "updated_at", _normalize_timestamp(self.updated_at, field_name="updated_at"))
        object.__setattr__(
            self,
            "provenance",
            _normalize_optional_text_value(
                self.provenance,
                field_name="provenance",
                max_length=_MAX_SHORT_TEXT_LEN,
            ),
        )
        object.__setattr__(
            self,
            "trust_score",
            _normalize_trust_score(self.trust_score),
        )

    def prompt_priority(self) -> tuple[float, float, float]:
        return (self.salience, self.trust_score, _timestamp_rank(self.updated_at))

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "topic": self.topic,
            "summary": self.summary,
            "salience": self.salience,
            "source": self.source,
            "stance": self.stance,
            "trust_score": self.trust_score,
        }
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.provenance is not None:
            payload["provenance"] = self.provenance
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RelationshipSignal:
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            topic=_normalize_required_text_from_mapping(
                payload,
                field_name="topic",
                max_length=_MAX_TITLE_LEN,
                aliases=("name",),
            ),
            summary=_normalize_required_text_from_mapping(
                payload,
                field_name="summary",
                max_length=_MAX_TEXT_LEN,
                aliases=("description",),
            ),
            salience=payload.get("salience"),
            source=_clean_text(payload.get("source")) or "conversation",
            stance=_clean_text(payload.get("stance")) or "affinity",
            updated_at=payload.get("updated_at"),
            provenance=payload.get("provenance", payload.get("origin")),
            trust_score=payload.get("trust_score", payload.get("trust")),
        )


@dataclass(frozen=True, slots=True)
class ContinuityThread(_PayloadModel):
    """Describe one topic or life-thread Twinr should keep warm."""

    title: str
    summary: str
    salience: float = 0.5
    updated_at: str | None = None
    expires_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "title",
            _normalize_required_text_value(self.title, field_name="title", max_length=_MAX_TITLE_LEN),
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_required_text_value(self.summary, field_name="summary", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "salience",
            _normalize_unit_float(self.salience, field_name="salience", default=0.5),
        )
        object.__setattr__(self, "updated_at", _normalize_timestamp(self.updated_at, field_name="updated_at"))
        object.__setattr__(self, "expires_at", _normalize_timestamp(self.expires_at, field_name="expires_at"))

    def is_expired(self, reference_at: str | None = None) -> bool:
        if self.expires_at is None:
            return False
        expires = _parse_timestamp(self.expires_at)
        if expires is None:
            return False
        reference = _parse_timestamp(reference_at) if reference_at is not None else _now_utc()
        if reference is None:
            reference = _now_utc()
        return expires < reference

    def prompt_priority(self) -> tuple[float, float, float]:
        freshness_penalty = -1.0 if self.is_expired() else 0.0
        expires_rank = _timestamp_rank(self.expires_at)
        expiry_score = 0.0 if expires_rank == float("-inf") else -expires_rank
        return (freshness_penalty + self.salience, _timestamp_rank(self.updated_at), expiry_score)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
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
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            title=_normalize_required_text_from_mapping(
                payload,
                field_name="title",
                max_length=_MAX_TITLE_LEN,
                aliases=("topic", "name"),
            ),
            summary=_normalize_required_text_from_mapping(
                payload,
                field_name="summary",
                max_length=_MAX_TEXT_LEN,
                aliases=("description",),
            ),
            salience=payload.get("salience"),
            updated_at=payload.get("updated_at"),
            expires_at=payload.get("expires_at"),
        )


@dataclass(frozen=True, slots=True)
class PlaceFocus(_PayloadModel):
    """Describe one geographic area Twinr should treat as relevant context."""

    name: str
    summary: str
    geography: str | None = None
    salience: float = 0.5
    updated_at: str | None = None
    provenance: str | None = None
    trust_score: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_required_text_value(self.name, field_name="name", max_length=_MAX_TITLE_LEN),
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_required_text_value(self.summary, field_name="summary", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "geography",
            _normalize_optional_text_value(
                self.geography,
                field_name="geography",
                max_length=_MAX_TITLE_LEN,
            ),
        )
        object.__setattr__(
            self,
            "salience",
            _normalize_unit_float(self.salience, field_name="salience", default=0.5),
        )
        object.__setattr__(self, "updated_at", _normalize_timestamp(self.updated_at, field_name="updated_at"))
        object.__setattr__(
            self,
            "provenance",
            _normalize_optional_text_value(
                self.provenance,
                field_name="provenance",
                max_length=_MAX_SHORT_TEXT_LEN,
            ),
        )
        object.__setattr__(self, "trust_score", _normalize_trust_score(self.trust_score))

    def prompt_priority(self) -> tuple[float, float, float]:
        return (self.salience, self.trust_score, _timestamp_rank(self.updated_at))

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "summary": self.summary,
            "salience": self.salience,
            "trust_score": self.trust_score,
        }
        if self.geography is not None:
            payload["geography"] = self.geography
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.provenance is not None:
            payload["provenance"] = self.provenance
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PlaceFocus:
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            name=_normalize_required_text_from_mapping(
                payload,
                field_name="name",
                max_length=_MAX_TITLE_LEN,
                aliases=("place", "title"),
            ),
            summary=_normalize_required_text_from_mapping(
                payload,
                field_name="summary",
                max_length=_MAX_TEXT_LEN,
                aliases=("description",),
            ),
            geography=_optional_text(payload.get("geography") or payload.get("scope")),
            salience=payload.get("salience"),
            updated_at=payload.get("updated_at"),
            provenance=payload.get("provenance", payload.get("origin")),
            trust_score=payload.get("trust_score", payload.get("trust")),
        )


@dataclass(frozen=True, slots=True)
class InteractionSignal(_PayloadModel):
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
    provenance: str | None = None
    trust_score: float = 1.0
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "signal_id",
            _normalize_required_text_value(self.signal_id, field_name="signal_id", max_length=_MAX_ID_LEN),
        )
        object.__setattr__(
            self,
            "signal_kind",
            _normalize_required_text_value(self.signal_kind, field_name="signal_kind", max_length=_MAX_NAME_LEN),
        )
        object.__setattr__(
            self,
            "target",
            _normalize_required_text_value(self.target, field_name="target", max_length=_MAX_TITLE_LEN),
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_required_text_value(self.summary, field_name="summary", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_unit_float(self.confidence, field_name="confidence", default=0.5),
        )
        object.__setattr__(
            self,
            "impact",
            _normalize_signed_float(self.impact, field_name="impact"),
        )
        object.__setattr__(
            self,
            "evidence_count",
            _normalize_int(self.evidence_count, field_name="evidence_count", default=1, minimum=1),
        )
        object.__setattr__(
            self,
            "source_event_ids",
            _normalize_string_tuple_bounded(
                self.source_event_ids,
                field_name="source_event_ids",
                max_items=_MAX_SIGNAL_IDS,
                item_max_length=_MAX_ID_LEN,
            ),
        )
        object.__setattr__(
            self,
            "delta_target",
            _normalize_optional_text_value(
                self.delta_target,
                field_name="delta_target",
                max_length=_MAX_TITLE_LEN,
            ),
        )
        normalized_delta_value = (
            None if self.delta_value is None else _normalize_signed_float(self.delta_value, field_name="delta_value")
        )
        object.__setattr__(self, "delta_value", normalized_delta_value)
        object.__setattr__(
            self,
            "delta_summary",
            _normalize_optional_text_value(
                self.delta_summary,
                field_name="delta_summary",
                max_length=_MAX_TEXT_LEN,
            ),
        )
        object.__setattr__(self, "explicit_user_requested", bool(self.explicit_user_requested))
        object.__setattr__(self, "created_at", _normalize_timestamp(self.created_at, field_name="created_at"))
        object.__setattr__(self, "updated_at", _normalize_timestamp(self.updated_at, field_name="updated_at"))
        object.__setattr__(
            self,
            "provenance",
            _normalize_optional_text_value(
                self.provenance,
                field_name="provenance",
                max_length=_MAX_SHORT_TEXT_LEN,
            ),
        )
        object.__setattr__(self, "trust_score", _normalize_trust_score(self.trust_score))
        object.__setattr__(
            self,
            "metadata",
            _normalize_metadata(self.metadata, field_name="metadata"),
        )

    def priority(self) -> tuple[float, float, int, float]:
        return (self.confidence, self.trust_score, self.evidence_count, abs(self.impact))

    def to_payload(self) -> dict[str, object]:
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
            "trust_score": self.trust_score,
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
        if self.provenance is not None:
            payload["provenance"] = self.provenance
        if self.metadata is not None:
            payload["metadata"] = cast(dict[str, object], _thaw_jsonish(self.metadata))
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> InteractionSignal:
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            signal_id=_normalize_required_text_from_mapping(
                payload,
                field_name="signal_id",
                max_length=_MAX_ID_LEN,
                aliases=("id",),
            ),
            signal_kind=_normalize_required_text_from_mapping(
                payload,
                field_name="signal_kind",
                max_length=_MAX_NAME_LEN,
                aliases=("kind",),
            ),
            target=_normalize_required_text_from_mapping(
                payload,
                field_name="target",
                max_length=_MAX_TITLE_LEN,
            ),
            summary=_normalize_required_text_from_mapping(
                payload,
                field_name="summary",
                max_length=_MAX_TEXT_LEN,
                aliases=("description",),
            ),
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
            provenance=payload.get("provenance", payload.get("origin")),
            trust_score=payload.get("trust_score", payload.get("trust")),
            metadata=payload.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class PlaceSignal(_PayloadModel):
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
    provenance: str | None = None
    trust_score: float = 1.0
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "signal_id",
            _normalize_required_text_value(self.signal_id, field_name="signal_id", max_length=_MAX_ID_LEN),
        )
        object.__setattr__(
            self,
            "place_name",
            _normalize_required_text_value(self.place_name, field_name="place_name", max_length=_MAX_TITLE_LEN),
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_required_text_value(self.summary, field_name="summary", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "geography",
            _normalize_optional_text_value(
                self.geography,
                field_name="geography",
                max_length=_MAX_TITLE_LEN,
            ),
        )
        object.__setattr__(
            self,
            "salience",
            _normalize_unit_float(self.salience, field_name="salience", default=0.5),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_unit_float(self.confidence, field_name="confidence", default=0.5),
        )
        object.__setattr__(
            self,
            "evidence_count",
            _normalize_int(self.evidence_count, field_name="evidence_count", default=1, minimum=1),
        )
        object.__setattr__(
            self,
            "source_event_ids",
            _normalize_string_tuple_bounded(
                self.source_event_ids,
                field_name="source_event_ids",
                max_items=_MAX_SIGNAL_IDS,
                item_max_length=_MAX_ID_LEN,
            ),
        )
        object.__setattr__(self, "updated_at", _normalize_timestamp(self.updated_at, field_name="updated_at"))
        object.__setattr__(
            self,
            "provenance",
            _normalize_optional_text_value(
                self.provenance,
                field_name="provenance",
                max_length=_MAX_SHORT_TEXT_LEN,
            ),
        )
        object.__setattr__(self, "trust_score", _normalize_trust_score(self.trust_score))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata, field_name="metadata"))

    def priority(self) -> tuple[float, float, float, int]:
        return (self.salience, self.trust_score, self.confidence, self.evidence_count)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "signal_id": self.signal_id,
            "place_name": self.place_name,
            "summary": self.summary,
            "salience": self.salience,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "source_event_ids": list(self.source_event_ids),
            "trust_score": self.trust_score,
        }
        if self.geography is not None:
            payload["geography"] = self.geography
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.provenance is not None:
            payload["provenance"] = self.provenance
        if self.metadata is not None:
            payload["metadata"] = cast(dict[str, object], _thaw_jsonish(self.metadata))
        return payload

    def to_place_focus(self) -> PlaceFocus:
        return PlaceFocus(
            name=self.place_name,
            summary=self.summary,
            geography=self.geography,
            salience=self.salience,
            updated_at=self.updated_at,
            provenance=self.provenance,
            trust_score=self.trust_score,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PlaceSignal:
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            signal_id=_normalize_required_text_from_mapping(
                payload,
                field_name="signal_id",
                max_length=_MAX_ID_LEN,
                aliases=("id",),
            ),
            place_name=_normalize_required_text_from_mapping(
                payload,
                field_name="place_name",
                max_length=_MAX_TITLE_LEN,
                aliases=("name", "place"),
            ),
            summary=_normalize_required_text_from_mapping(
                payload,
                field_name="summary",
                max_length=_MAX_TEXT_LEN,
                aliases=("description",),
            ),
            geography=payload.get("geography"),
            salience=payload.get("salience"),
            confidence=payload.get("confidence"),
            evidence_count=payload.get("evidence_count"),
            source_event_ids=payload.get("source_event_ids"),
            updated_at=payload.get("updated_at"),
            provenance=payload.get("provenance", payload.get("origin")),
            trust_score=payload.get("trust_score", payload.get("trust")),
            metadata=payload.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class WorldSignal(_PayloadModel):
    """Capture one relevant world or news development for prompt context."""

    topic: str
    summary: str
    region: str | None = None
    source: str = "world"
    salience: float = 0.5
    fresh_until: str | None = None
    evidence_count: int = 1
    source_event_ids: tuple[str, ...] = ()
    provenance: str | None = None
    trust_score: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "topic",
            _normalize_required_text_value(self.topic, field_name="topic", max_length=_MAX_TITLE_LEN),
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_required_text_value(self.summary, field_name="summary", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "region",
            _normalize_optional_text_value(
                self.region,
                field_name="region",
                max_length=_MAX_TITLE_LEN,
            ),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_required_text_value(self.source, field_name="source", max_length=_MAX_NAME_LEN),
        )
        object.__setattr__(
            self,
            "salience",
            _normalize_unit_float(self.salience, field_name="salience", default=0.5),
        )
        object.__setattr__(self, "fresh_until", _normalize_timestamp(self.fresh_until, field_name="fresh_until"))
        object.__setattr__(
            self,
            "evidence_count",
            _normalize_int(self.evidence_count, field_name="evidence_count", default=1, minimum=1),
        )
        object.__setattr__(
            self,
            "source_event_ids",
            _normalize_string_tuple_bounded(
                self.source_event_ids,
                field_name="source_event_ids",
                max_items=_MAX_SIGNAL_IDS,
                item_max_length=_MAX_ID_LEN,
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            _normalize_optional_text_value(
                self.provenance,
                field_name="provenance",
                max_length=_MAX_SHORT_TEXT_LEN,
            ),
        )
        object.__setattr__(self, "trust_score", _normalize_trust_score(self.trust_score))

    def is_fresh(self, reference_at: str | None = None) -> bool:
        if self.fresh_until is None:
            return True
        fresh_until = _parse_timestamp(self.fresh_until)
        if fresh_until is None:
            return True
        reference = _parse_timestamp(reference_at) if reference_at is not None else _now_utc()
        if reference is None:
            reference = _now_utc()
        return fresh_until >= reference

    def prompt_priority(self) -> tuple[float, float, int, float]:
        freshness_bonus = 1.0 if self.is_fresh() else -1.0
        return (freshness_bonus + self.salience, self.trust_score, self.evidence_count, _timestamp_rank(self.fresh_until))

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "topic": self.topic,
            "summary": self.summary,
            "source": self.source,
            "salience": self.salience,
            "evidence_count": self.evidence_count,
            "source_event_ids": list(self.source_event_ids),
            "trust_score": self.trust_score,
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.fresh_until is not None:
            payload["fresh_until"] = self.fresh_until
        if self.provenance is not None:
            payload["provenance"] = self.provenance
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> WorldSignal:
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            topic=_normalize_required_text_from_mapping(
                payload,
                field_name="topic",
                max_length=_MAX_TITLE_LEN,
                aliases=("title", "name"),
            ),
            summary=_normalize_required_text_from_mapping(
                payload,
                field_name="summary",
                max_length=_MAX_TEXT_LEN,
                aliases=("description",),
            ),
            region=_optional_text(payload.get("region") or payload.get("scope")),
            source=_clean_text(payload.get("source")) or "world",
            salience=payload.get("salience"),
            fresh_until=payload.get("fresh_until") or payload.get("valid_until"),
            evidence_count=payload.get("evidence_count"),
            source_event_ids=payload.get("source_event_ids"),
            provenance=payload.get("provenance", payload.get("origin")),
            trust_score=payload.get("trust_score", payload.get("trust")),
        )


@dataclass(frozen=True, slots=True)
class ReflectionDelta(_PayloadModel):
    """Describe one small personality adjustment proposal or learning delta."""

    target: str
    change: str
    reason: str
    confidence: float = 0.5
    review_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target",
            _normalize_required_text_value(self.target, field_name="target", max_length=_MAX_TITLE_LEN),
        )
        object.__setattr__(
            self,
            "change",
            _normalize_required_text_value(self.change, field_name="change", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "reason",
            _normalize_required_text_value(self.reason, field_name="reason", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_unit_float(self.confidence, field_name="confidence", default=0.5),
        )
        object.__setattr__(self, "review_at", _normalize_timestamp(self.review_at, field_name="review_at"))

    def prompt_priority(self) -> tuple[float, float]:
        return (self.confidence, _timestamp_rank(self.review_at))

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
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
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            target=_normalize_required_text_from_mapping(
                payload,
                field_name="target",
                max_length=_MAX_TITLE_LEN,
            ),
            change=_normalize_required_text_from_mapping(
                payload,
                field_name="change",
                max_length=_MAX_TEXT_LEN,
            ),
            reason=_normalize_required_text_from_mapping(
                payload,
                field_name="reason",
                max_length=_MAX_TEXT_LEN,
            ),
            confidence=payload.get("confidence"),
            review_at=payload.get("review_at"),
        )


@dataclass(frozen=True, slots=True)
class PersonalityDelta(_PayloadModel):
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
    provenance: str | None = None
    trust_score: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delta_id",
            _normalize_required_text_value(self.delta_id, field_name="delta_id", max_length=_MAX_ID_LEN),
        )
        object.__setattr__(
            self,
            "target",
            _normalize_required_text_value(self.target, field_name="target", max_length=_MAX_TITLE_LEN),
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_required_text_value(self.summary, field_name="summary", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "rationale",
            _normalize_required_text_value(self.rationale, field_name="rationale", max_length=_MAX_TEXT_LEN),
        )
        object.__setattr__(
            self,
            "delta_value",
            _normalize_signed_float(self.delta_value, field_name="delta_value"),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_unit_float(self.confidence, field_name="confidence", default=0.5),
        )
        object.__setattr__(
            self,
            "support_count",
            _normalize_int(self.support_count, field_name="support_count", default=1, minimum=1),
        )
        object.__setattr__(
            self,
            "source_signal_ids",
            _normalize_string_tuple_bounded(
                self.source_signal_ids,
                field_name="source_signal_ids",
                max_items=_MAX_SIGNAL_IDS,
                item_max_length=_MAX_ID_LEN,
            ),
        )
        object.__setattr__(
            self,
            "status",
            _normalize_enum_text(
                self.status,
                field_name="status",
                allowed=_ALLOWED_DELTA_STATUSES,
                default="candidate",
                aliases=_DELTA_STATUS_ALIASES,
            ),
        )
        object.__setattr__(self, "review_at", _normalize_timestamp(self.review_at, field_name="review_at"))
        object.__setattr__(self, "explicit_user_requested", bool(self.explicit_user_requested))
        object.__setattr__(
            self,
            "provenance",
            _normalize_optional_text_value(
                self.provenance,
                field_name="provenance",
                max_length=_MAX_SHORT_TEXT_LEN,
            ),
        )
        object.__setattr__(self, "trust_score", _normalize_trust_score(self.trust_score))

    def is_active(self) -> bool:
        return self.status in {"candidate", "accepted", "applied"}

    def prompt_priority(self) -> tuple[float, float, int, float]:
        active_bonus = 1.0 if self.is_active() else 0.0
        return (
            active_bonus + self.confidence,
            self.trust_score,
            self.support_count,
            abs(self.delta_value),
        )

    def to_payload(self) -> dict[str, object]:
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
            "trust_score": self.trust_score,
        }
        if self.review_at is not None:
            payload["review_at"] = self.review_at
        if self.provenance is not None:
            payload["provenance"] = self.provenance
        return payload

    def to_reflection_delta(self) -> ReflectionDelta:
        return ReflectionDelta(
            target=self.target,
            change=self.summary,
            reason=self.rationale,
            confidence=self.confidence,
            review_at=self.review_at,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PersonalityDelta:
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            delta_id=_normalize_required_text_from_mapping(
                payload,
                field_name="delta_id",
                max_length=_MAX_ID_LEN,
                aliases=("id",),
            ),
            target=_normalize_required_text_from_mapping(
                payload,
                field_name="target",
                max_length=_MAX_TITLE_LEN,
            ),
            summary=_normalize_required_text_from_mapping(
                payload,
                field_name="summary",
                max_length=_MAX_TEXT_LEN,
            ),
            rationale=_normalize_required_text_from_mapping(
                payload,
                field_name="rationale",
                max_length=_MAX_TEXT_LEN,
                aliases=("reason",),
            ),
            delta_value=payload.get("delta_value"),
            confidence=payload.get("confidence"),
            support_count=payload.get("support_count"),
            source_signal_ids=payload.get("source_signal_ids"),
            status=_clean_text(payload.get("status")) or "candidate",
            review_at=payload.get("review_at"),
            explicit_user_requested=payload.get("explicit_user_requested", False),
            provenance=payload.get("provenance", payload.get("origin")),
            trust_score=payload.get("trust_score", payload.get("trust")),
        )


@dataclass(frozen=True, slots=True)
class PersonalitySnapshot(_PayloadModel):
    """Store the structured evolving personality state for one prompt build."""

    # BREAKING: schema_version now defaults to 2; from_payload still reads legacy v1 payloads.
    schema_version: int = CURRENT_PERSONALITY_SCHEMA_VERSION
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
        schema_version = int(self.schema_version)
        if schema_version < 1:
            raise ValueError("schema_version must be >= 1.")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "generated_at", _normalize_timestamp(self.generated_at, field_name="generated_at"))
        object.__setattr__(
            self,
            "style_profile",
            _coerce_optional_item(
                self.style_profile,
                field_name="style_profile",
                model_type=ConversationStyleProfile,
            ),
        )
        object.__setattr__(
            self,
            "humor_profile",
            _coerce_optional_item(
                self.humor_profile,
                field_name="humor_profile",
                model_type=HumorProfile,
            ),
        )
        # BREAKING: prompt snapshots are now top-k budgeted at construction time to keep Pi 4 prompt builds bounded.
        object.__setattr__(
            self,
            "core_traits",
            _top_k(
                _coerce_tuple(self.core_traits, field_name="core_traits", model_type=PersonalityTrait),
                limit=16,
                score=lambda item: item.prompt_priority(),
            ),
        )
        object.__setattr__(
            self,
            "relationship_signals",
            _top_k(
                _coerce_tuple(
                    self.relationship_signals,
                    field_name="relationship_signals",
                    model_type=RelationshipSignal,
                ),
                limit=24,
                score=lambda item: item.prompt_priority(),
            ),
        )
        object.__setattr__(
            self,
            "continuity_threads",
            _top_k(
                _coerce_tuple(
                    self.continuity_threads,
                    field_name="continuity_threads",
                    model_type=ContinuityThread,
                ),
                limit=16,
                score=lambda item: item.prompt_priority(),
            ),
        )
        object.__setattr__(
            self,
            "place_focuses",
            _top_k(
                _coerce_tuple(
                    self.place_focuses,
                    field_name="place_focuses",
                    model_type=PlaceFocus,
                ),
                limit=16,
                score=lambda item: item.prompt_priority(),
            ),
        )
        object.__setattr__(
            self,
            "world_signals",
            _top_k(
                tuple(
                    item
                    for item in _coerce_tuple(
                        self.world_signals,
                        field_name="world_signals",
                        model_type=WorldSignal,
                    )
                    if item.is_fresh(self.generated_at)
                ),
                limit=16,
                score=lambda item: item.prompt_priority(),
            ),
        )
        object.__setattr__(
            self,
            "reflection_deltas",
            _top_k(
                _coerce_tuple(
                    self.reflection_deltas,
                    field_name="reflection_deltas",
                    model_type=ReflectionDelta,
                ),
                limit=16,
                score=lambda item: item.prompt_priority(),
            ),
        )
        object.__setattr__(
            self,
            "personality_deltas",
            _top_k(
                tuple(
                    item
                    for item in _coerce_tuple(
                        self.personality_deltas,
                        field_name="personality_deltas",
                        model_type=PersonalityDelta,
                    )
                    if item.is_active()
                ),
                limit=16,
                score=lambda item: item.prompt_priority(),
            ),
        )

    def compact_for_prompt(
        self,
        *,
        max_core_traits: int = 12,
        max_relationship_signals: int = 16,
        max_continuity_threads: int = 12,
        max_place_focuses: int = 12,
        max_world_signals: int = 12,
        max_reflection_deltas: int = 12,
        max_personality_deltas: int = 12,
        min_trust_score: float = 0.25,
    ) -> PersonalitySnapshot:
        """Return a tighter, trust-aware prompt snapshot for on-device assembly."""

        return PersonalitySnapshot(
            schema_version=self.schema_version,
            generated_at=self.generated_at,
            core_traits=_top_k(self.core_traits, limit=max_core_traits, score=lambda item: item.prompt_priority()),
            style_profile=self.style_profile,
            humor_profile=self.humor_profile,
            relationship_signals=_top_k(
                tuple(item for item in self.relationship_signals if item.trust_score >= min_trust_score),
                limit=max_relationship_signals,
                score=lambda item: item.prompt_priority(),
            ),
            continuity_threads=_top_k(
                tuple(item for item in self.continuity_threads if not item.is_expired(self.generated_at)),
                limit=max_continuity_threads,
                score=lambda item: item.prompt_priority(),
            ),
            place_focuses=_top_k(
                tuple(item for item in self.place_focuses if item.trust_score >= min_trust_score),
                limit=max_place_focuses,
                score=lambda item: item.prompt_priority(),
            ),
            world_signals=_top_k(
                tuple(
                    item
                    for item in self.world_signals
                    if item.trust_score >= min_trust_score and item.is_fresh(self.generated_at)
                ),
                limit=max_world_signals,
                score=lambda item: item.prompt_priority(),
            ),
            reflection_deltas=_top_k(
                self.reflection_deltas,
                limit=max_reflection_deltas,
                score=lambda item: item.prompt_priority(),
            ),
            personality_deltas=_top_k(
                tuple(item for item in self.personality_deltas if item.trust_score >= min_trust_score and item.is_active()),
                limit=max_personality_deltas,
                score=lambda item: item.prompt_priority(),
            ),
        )

    def to_payload(self) -> dict[str, object]:
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
        payload = _ensure_mapping(payload, field_name="payload")
        raw_humor = payload.get("humor_profile", payload.get("humor"))
        humor_profile = None
        if raw_humor is not None:
            raw_humor_mapping = _ensure_mapping(raw_humor, field_name="humor_profile")
            humor_profile = HumorProfile.from_payload(raw_humor_mapping)
        raw_style = payload.get("style_profile", payload.get("conversation_style"))
        style_profile = None
        if raw_style is not None:
            raw_style_mapping = _ensure_mapping(raw_style, field_name="style_profile")
            style_profile = ConversationStyleProfile.from_payload(raw_style_mapping)
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            generated_at=payload.get("generated_at"),
            core_traits=tuple(
                PersonalityTrait.from_payload(item)
                for item in _mapping_items(payload.get("core_traits", payload.get("traits")), field_name="core_traits")
            ),
            style_profile=style_profile,
            humor_profile=humor_profile,
            relationship_signals=tuple(
                RelationshipSignal.from_payload(item)
                for item in _mapping_items(
                    payload.get("relationship_signals", payload.get("relationships")),
                    field_name="relationship_signals",
                )
            ),
            continuity_threads=tuple(
                ContinuityThread.from_payload(item)
                for item in _mapping_items(
                    payload.get("continuity_threads", payload.get("threads")),
                    field_name="continuity_threads",
                )
            ),
            place_focuses=tuple(
                PlaceFocus.from_payload(item)
                for item in _mapping_items(
                    payload.get("place_focuses", payload.get("places")),
                    field_name="place_focuses",
                )
            ),
            world_signals=tuple(
                WorldSignal.from_payload(item)
                for item in _mapping_items(
                    payload.get("world_signals", payload.get("world")),
                    field_name="world_signals",
                )
            ),
            reflection_deltas=tuple(
                ReflectionDelta.from_payload(item)
                for item in _mapping_items(
                    payload.get("reflection_deltas"),
                    field_name="reflection_deltas",
                )
            ),
            personality_deltas=tuple(
                PersonalityDelta.from_payload(item)
                for item in _mapping_items(
                    payload.get("personality_deltas"),
                    field_name="personality_deltas",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class PersonalityPromptLayer(_PayloadModel):
    """Represent one rendered prompt layer before legacy assembly."""

    layer_id: str
    title: str
    content: str
    source: str
    instruction_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_id",
            _normalize_required_text_value(self.layer_id, field_name="layer_id", max_length=_MAX_ID_LEN),
        )
        object.__setattr__(
            self,
            "title",
            _normalize_required_text_value(self.title, field_name="title", max_length=_MAX_TITLE_LEN),
        )
        object.__setattr__(
            self,
            "content",
            _normalize_required_text_value(self.content, field_name="content", max_length=_MAX_CONTENT_LEN),
        )
        normalized_source = _normalize_required_text_value(
            self.source,
            field_name="source",
            max_length=_MAX_NAME_LEN,
        )
        object.__setattr__(self, "source", normalized_source)
        authority = bool(self.instruction_authority)
        if authority:
            lowered_source = normalized_source.casefold()
            if not any(
                lowered_source == prefix or lowered_source.startswith(f"{prefix}:")
                for prefix in _TRUSTED_PROMPT_LAYER_PREFIXES
            ):
                # BREAKING: only trusted builder/system/policy layers may claim instruction authority.
                raise ValueError(
                    "instruction_authority may only be enabled for trusted system/policy/builder sources."
                )
        object.__setattr__(self, "instruction_authority", authority)

    def to_section(self) -> tuple[str, str]:
        return (self.title, self.content)

    def to_payload(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "instruction_authority": self.instruction_authority,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PersonalityPromptLayer:
        payload = _ensure_mapping(payload, field_name="payload")
        return cls(
            layer_id=_normalize_required_text_from_mapping(
                payload,
                field_name="layer_id",
                max_length=_MAX_ID_LEN,
                aliases=("id",),
            ),
            title=_normalize_required_text_from_mapping(
                payload,
                field_name="title",
                max_length=_MAX_TITLE_LEN,
            ),
            content=_normalize_required_text_from_mapping(
                payload,
                field_name="content",
                max_length=_MAX_CONTENT_LEN,
            ),
            source=_normalize_required_text_from_mapping(
                payload,
                field_name="source",
                max_length=_MAX_NAME_LEN,
            ),
            instruction_authority=bool(payload.get("instruction_authority", False)),
        )


@dataclass(frozen=True, slots=True)
class PersonalityPromptPlan(_PayloadModel):
    """Store the ordered prompt layers emitted by the context builder."""

    layers: tuple[PersonalityPromptLayer, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layers",
            _coerce_tuple(self.layers, field_name="layers", model_type=PersonalityPromptLayer, max_items=32),
        )

    def as_sections(self) -> tuple[tuple[str, str], ...]:
        return tuple(layer.to_section() for layer in self.layers)

    def to_payload(self) -> dict[str, object]:
        return {"layers": [layer.to_payload() for layer in self.layers]}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PersonalityPromptPlan:
        payload = _ensure_mapping(payload, field_name="payload")
        raw_layers = payload.get("layers")
        if raw_layers is None and "sections" in payload:
            raw_sections = payload["sections"]
            layers: list[dict[str, object]] = []
            if isinstance(raw_sections, Iterable) and not isinstance(raw_sections, (str, bytes, bytearray, Mapping)):
                for idx, item in enumerate(raw_sections):
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        title, content = item
                        layers.append(
                            {
                                "layer_id": f"legacy_section_{idx}",
                                "title": title,
                                "content": content,
                                "source": "builder:legacy",
                                "instruction_authority": False,
                            }
                        )
                    elif isinstance(item, Mapping):
                        layers.append(dict(item))
                    else:
                        raise ValueError("sections items must be 2-tuples or mappings.")
                raw_layers = layers
        return cls(
            layers=tuple(
                PersonalityPromptLayer.from_payload(item)
                for item in _mapping_items(raw_layers, field_name="layers")
            )
        )
