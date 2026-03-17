"""Define versioned contract objects for the Adaptive Skill Engine core.

The dataclasses in this module are the storage and handoff boundary between
future dialogue, feasibility, compile-worker, activation, and web layers.
Each object validates itself eagerly and can round-trip through a strict
JSON-safe payload shape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from math import isfinite
from pathlib import PurePosixPath
import re
from typing import Any, ClassVar

from twinr.text_utils import is_valid_stable_identifier, slugify_identifier, truncate_text

from .status import (
    ArtifactKind,
    CapabilityRiskClass,
    CapabilityStatus,
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    LearnedSkillStatus,
    RequirementsDialogueStatus,
)

_SKILL_TRIGGER_SCHEMA = "twinr_self_coding_skill_trigger_v1"
_SKILL_SPEC_SCHEMA = "twinr_self_coding_skill_spec_v1"
_CAPABILITY_DEFINITION_SCHEMA = "twinr_self_coding_capability_definition_v1"
_CAPABILITY_AVAILABILITY_SCHEMA = "twinr_self_coding_capability_availability_v1"
_FEASIBILITY_RESULT_SCHEMA = "twinr_self_coding_feasibility_result_v1"
_REQUIREMENTS_DIALOGUE_SESSION_SCHEMA = "twinr_self_coding_requirements_dialogue_session_v1"
_COMPILE_JOB_SCHEMA = "twinr_self_coding_compile_job_v1"
_COMPILE_ARTIFACT_SCHEMA = "twinr_self_coding_compile_artifact_v1"
_COMPILE_RUN_STATUS_SCHEMA = "twinr_self_coding_compile_run_status_v1"
_ACTIVATION_RECORD_SCHEMA = "twinr_self_coding_activation_record_v1"
_SKILL_HEALTH_RECORD_SCHEMA = "twinr_self_coding_skill_health_record_v1"
_EXECUTION_RUN_STATUS_RECORD_SCHEMA = "twinr_self_coding_execution_run_status_record_v1"
_LIVE_E2E_STATUS_RECORD_SCHEMA = "twinr_self_coding_live_e2e_status_record_v1"

_MAX_STABLE_IDENTIFIER_LENGTH = 128
_MAX_PATH_LENGTH = 240
_MAX_JSON_DEPTH = 8
_MAX_JSON_CONTAINER_ITEMS = 256
_MAX_JSON_KEY_LENGTH = 240
_MAX_JSON_STRING_LENGTH = 4096
_ALLOWED_REQUIREMENTS_QUESTION_IDS = frozenset({"when", "what", "how"})
_ALLOWED_REQUIREMENTS_DIALOGUE_IDS = frozenset({"when", "what", "how", "confirm"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FrozenJsonDict(dict):
    """Read-only JSON mapping used inside frozen contract objects."""

    __slots__ = ()

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("JSON mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, other: object):  # pragma: no cover - defensive API surface
        self._immutable(other)
        return self

    def copy(self) -> dict[str, Any]:
        return dict(self)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")  # AUDIT-FIX(#2): Reject non-string boundary values instead of silently coercing them with str(...).
    return value


def _optional_string(value: object | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name=field_name)


def _normalize_datetime(value: datetime | str | None, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    else:
        raise TypeError(f"{field_name} must be a datetime or ISO timestamp")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return parsed.astimezone(UTC)


def _datetime_payload(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_text(value: object, *, field_name: str, limit: int) -> str:
    text = _require_string(value, field_name=field_name).strip()  # AUDIT-FIX(#4): Reject oversized text instead of silently truncating persisted contract values.
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if len(text) > limit:
        raise ValueError(f"{field_name} must be <= {limit} characters")
    return text


def _optional_text(value: object | None, *, field_name: str, limit: int) -> str | None:
    text = _optional_string(value, field_name=field_name)
    if text is None:
        return None
    normalized = text.strip()  # AUDIT-FIX(#4): Preserve exact failure semantics on long values rather than mutating them with truncate_text(...).
    if not normalized:
        return None
    if len(normalized) > limit:
        raise ValueError(f"{field_name} must be <= {limit} characters")
    return normalized


def _require_stable_identifier(value: object, *, field_name: str) -> str:
    text = _require_string(value, field_name=field_name).strip().lower()
    if len(text) > _MAX_STABLE_IDENTIFIER_LENGTH:
        raise ValueError(f"{field_name} must be <= {_MAX_STABLE_IDENTIFIER_LENGTH} characters")
    if not is_valid_stable_identifier(text):
        raise ValueError(f"{field_name} must be a stable identifier")
    return text


def _coerce_skill_id(value: object, *, name: str) -> str:
    if value is None:
        raw = ""
    else:
        raw = _require_string(value, field_name="skill_id").strip().lower()  # AUDIT-FIX(#2): Only accept explicit string skill IDs; None stays as the "generate one" sentinel.
    if not raw:
        generated = slugify_identifier(name, fallback="skill")
        if len(generated) > _MAX_STABLE_IDENTIFIER_LENGTH or not is_valid_stable_identifier(generated):
            raise ValueError("skill_id must be a stable identifier")
        return generated
    if len(raw) > _MAX_STABLE_IDENTIFIER_LENGTH:
        raise ValueError(f"skill_id must be <= {_MAX_STABLE_IDENTIFIER_LENGTH} characters")
    if not is_valid_stable_identifier(raw):
        raise ValueError("skill_id must be a stable identifier")
    return raw


def _optional_stable_identifier(value: object | None, *, field_name: str) -> str | None:
    text = _optional_string(value, field_name=field_name)
    if text is None:
        return None
    normalized = text.strip().lower()
    if not normalized:
        return None
    if len(normalized) > _MAX_STABLE_IDENTIFIER_LENGTH:
        raise ValueError(f"{field_name} must be <= {_MAX_STABLE_IDENTIFIER_LENGTH} characters")
    if not is_valid_stable_identifier(normalized):
        raise ValueError(f"{field_name} must be a stable identifier")
    return normalized


def _require_sequence(values: object, *, field_name: str) -> Sequence[object]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence, not a string")  # AUDIT-FIX(#3): Enforce tuple/list payload shape for repeated fields instead of accepting a bare string.
    if not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return values


def _stable_identifier_tuple(values: object, *, field_name: str) -> tuple[str, ...]:
    raw_items = _require_sequence(values, field_name=field_name)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = _require_stable_identifier(raw_item, field_name=field_name)
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _text_tuple(values: object, *, field_name: str, limit: int) -> tuple[str, ...]:
    raw_items = _require_sequence(values, field_name=field_name)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = _optional_text(raw_item, field_name=field_name, limit=limit)
        if item is None or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _require_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a boolean")


def _require_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return value


def _require_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def _optional_non_negative_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_non_negative_int(value, field_name=field_name)


def _optional_non_negative_float(value: object | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be >= 0")
    return normalized


def _optional_safe_relative_path(value: object | None, *, field_name: str, limit: int = _MAX_PATH_LENGTH) -> str | None:
    text = _optional_string(value, field_name=field_name)
    if text is None:
        return None
    normalized = text.strip()
    if not normalized:
        return None
    if len(normalized) > limit:
        raise ValueError(f"{field_name} must be <= {limit} characters")
    if "\x00" in normalized or "\\" in normalized or any(ord(char) < 32 for char in normalized):
        raise ValueError(f"{field_name} must not contain control characters or backslashes")
    path = PurePosixPath(normalized)
    parts = path.parts
    if path.is_absolute() or str(path) in {".", ".."} or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{field_name} must be a safe relative path")
    return str(path)


def _require_sha256(value: object, *, field_name: str) -> str:
    text = _optional_sha256(value, field_name=field_name)
    if text is None:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _optional_sha256(value: object | None, *, field_name: str) -> str | None:
    text = _optional_string(value, field_name=field_name)
    if text is None:
        return None
    normalized = text.strip().lower()
    if not normalized:
        return None
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 hex digest")
    return normalized


def _normalize_json_value(value: object, *, field_name: str, path: str = "$", depth: int = 0) -> Any:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"{field_name} exceeds max depth {_MAX_JSON_DEPTH} at {path}")  # AUDIT-FIX(#5): Bound generic JSON depth so malformed web/storage payloads cannot exhaust recursion on the RPi.
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise TypeError(f"{field_name} must not contain NaN or infinity at {path}")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_JSON_STRING_LENGTH:
            raise ValueError(f"{field_name} string values must be <= {_MAX_JSON_STRING_LENGTH} characters at {path}")  # AUDIT-FIX(#5): Cap embedded JSON string size to prevent oversized state blobs.
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_JSON_CONTAINER_ITEMS:
            raise ValueError(f"{field_name} mappings must contain <= {_MAX_JSON_CONTAINER_ITEMS} items at {path}")  # AUDIT-FIX(#5): Limit mapping fan-out for metadata/scope payloads.
        normalized: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{field_name} keys must be strings at {path}")
            if len(raw_key) > _MAX_JSON_KEY_LENGTH:
                raise ValueError(f"{field_name} keys must be <= {_MAX_JSON_KEY_LENGTH} characters at {path}")  # AUDIT-FIX(#5): Cap JSON key size to avoid pathological state payloads.
            child_path = f"{path}.{raw_key}"
            normalized[raw_key] = _normalize_json_value(raw_item, field_name=field_name, path=child_path, depth=depth + 1)
        return FrozenJsonDict(normalized)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > _MAX_JSON_CONTAINER_ITEMS:
            raise ValueError(f"{field_name} arrays must contain <= {_MAX_JSON_CONTAINER_ITEMS} items at {path}")  # AUDIT-FIX(#5): Limit sequence fan-out for generic JSON payloads.
        return tuple(
            _normalize_json_value(item, field_name=field_name, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise TypeError(f"{field_name} must be JSON serializable at {path}")


def _json_mapping(value: object | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return FrozenJsonDict()
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = _normalize_json_value(dict(value), field_name=field_name)
    if not isinstance(normalized, FrozenJsonDict):
        raise TypeError(f"{field_name} must serialize to a JSON object")
    return normalized


def _payload_json_mapping(value: object | None, *, field_name: str) -> dict[str, Any]:
    normalized = _json_mapping(value, field_name=field_name)
    payload = json.loads(json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False))
    if not isinstance(payload, dict):
        raise TypeError(f"{field_name} must serialize to a JSON object")
    return payload


def _require_mapping(payload: object, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} payload must be a mapping")
    return dict(payload)


def _require_key(payload: Mapping[str, Any], key: str, *, context: str) -> Any:
    if key not in payload:
        raise ValueError(f"{context} payload missing {key}")  # AUDIT-FIX(#1): Fail closed on partially written/corrupt payloads instead of inventing defaults during deserialization.
    return payload[key]


def _validate_schema(payload: Mapping[str, Any], *, expected: str, context: str) -> None:
    schema = payload.get("schema")
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"{context} payload missing schema")
    if schema != expected:
        raise ValueError(f"{context} payload schema mismatch: {schema!r} != {expected!r}")


def _ensure_not_before(later: datetime, earlier: datetime, *, later_name: str, earlier_name: str) -> None:
    if later < earlier:
        raise ValueError(f"{later_name} must be >= {earlier_name}")


def _coerce_enum(enum_type, value: object, *, field_name: str):
    if isinstance(value, enum_type):
        return value
    text = _require_string(value, field_name=field_name).strip().lower()  # AUDIT-FIX(#2): Enforce string-or-enum shape and stop truncating malformed enum payloads.
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if len(text) > 64:
        raise ValueError(f"{field_name} must be <= 64 characters")
    try:
        return enum_type(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be one of {[item.value for item in enum_type]}") from exc


@dataclass(frozen=True, slots=True)
class SkillTriggerSpec:
    """Store the trigger mode and conditions for one learned skill."""

    schema_name: ClassVar[str] = _SKILL_TRIGGER_SCHEMA

    mode: str
    conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_mode = _require_stable_identifier(self.mode, field_name="mode")
        if normalized_mode not in {"push", "pull"}:
            raise ValueError("mode must be 'push' or 'pull'")
        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "conditions", _stable_identifier_tuple(self.conditions, field_name="conditions"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "mode": self.mode,
            "conditions": list(self.conditions),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "SkillTriggerSpec":
        record = _require_mapping(payload, context="skill trigger")
        _validate_schema(record, expected=cls.schema_name, context="skill trigger")
        return cls(
            mode=_require_key(record, "mode", context="skill trigger"),
            conditions=_require_key(record, "conditions", context="skill trigger"),
        )


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """Describe a user-approved skill request ready for feasibility or compile."""

    schema_name: ClassVar[str] = _SKILL_SPEC_SCHEMA

    name: str
    action: str
    trigger: SkillTriggerSpec
    skill_id: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=_utc_now)
    version: int = 1

    def __post_init__(self) -> None:
        name = _require_text(self.name, field_name="name", limit=160)
        action = _require_text(self.action, field_name="action", limit=240)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "skill_id", _coerce_skill_id(self.skill_id, name=name))
        if not isinstance(self.trigger, SkillTriggerSpec):
            raise TypeError("trigger must be a SkillTriggerSpec")
        object.__setattr__(self, "scope", _json_mapping(self.scope, field_name="scope"))
        object.__setattr__(self, "constraints", _text_tuple(self.constraints, field_name="constraints", limit=220))
        object.__setattr__(self, "capabilities", _stable_identifier_tuple(self.capabilities, field_name="capabilities"))
        object.__setattr__(self, "created_at", _normalize_datetime(self.created_at, field_name="created_at"))
        object.__setattr__(self, "version", _require_positive_int(self.version, field_name="version"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "skill_id": self.skill_id,
            "name": self.name,
            "action": self.action,
            "trigger": self.trigger.to_payload(),
            "scope": _payload_json_mapping(self.scope, field_name="scope"),
            "constraints": list(self.constraints),
            "capabilities": list(self.capabilities),
            "created_at": _datetime_payload(self.created_at),
            "version": self.version,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "SkillSpec":
        record = _require_mapping(payload, context="skill spec")
        _validate_schema(record, expected=cls.schema_name, context="skill spec")
        return cls(
            skill_id=_require_key(record, "skill_id", context="skill spec"),
            name=_require_key(record, "name", context="skill spec"),
            action=_require_key(record, "action", context="skill spec"),
            trigger=SkillTriggerSpec.from_payload(_require_key(record, "trigger", context="skill spec")),
            scope=_require_key(record, "scope", context="skill spec"),
            constraints=_require_key(record, "constraints", context="skill spec"),
            capabilities=_require_key(record, "capabilities", context="skill spec"),
            created_at=_require_key(record, "created_at", context="skill spec"),
            version=_require_key(record, "version", context="skill spec"),
        )


@dataclass(frozen=True, slots=True)
class CapabilityDefinition:
    """Describe one curated capability exposed to the coding subsystem."""

    schema_name: ClassVar[str] = _CAPABILITY_DEFINITION_SCHEMA

    capability_id: str
    module_name: str
    summary: str
    risk_class: CapabilityRiskClass
    requires_configuration: bool = False
    integration_id: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability_id", _require_stable_identifier(self.capability_id, field_name="capability_id"))
        object.__setattr__(self, "module_name", _require_stable_identifier(self.module_name, field_name="module_name"))
        object.__setattr__(self, "summary", _require_text(self.summary, field_name="summary", limit=220))
        object.__setattr__(self, "risk_class", _coerce_enum(CapabilityRiskClass, self.risk_class, field_name="risk_class"))
        object.__setattr__(self, "requires_configuration", _require_bool(self.requires_configuration, field_name="requires_configuration"))
        object.__setattr__(
            self,
            "integration_id",
            None if self.integration_id is None else _require_stable_identifier(self.integration_id, field_name="integration_id"),
        )
        object.__setattr__(self, "tags", _stable_identifier_tuple(self.tags, field_name="tags"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "capability_id": self.capability_id,
            "module_name": self.module_name,
            "summary": self.summary,
            "risk_class": self.risk_class.value,
            "requires_configuration": self.requires_configuration,
            "integration_id": self.integration_id,
            "tags": list(self.tags),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CapabilityDefinition":
        record = _require_mapping(payload, context="capability definition")
        _validate_schema(record, expected=cls.schema_name, context="capability definition")
        return cls(
            capability_id=_require_key(record, "capability_id", context="capability definition"),
            module_name=_require_key(record, "module_name", context="capability definition"),
            summary=_require_key(record, "summary", context="capability definition"),
            risk_class=_require_key(record, "risk_class", context="capability definition"),
            requires_configuration=_require_key(record, "requires_configuration", context="capability definition"),
            integration_id=_require_key(record, "integration_id", context="capability definition"),
            tags=_require_key(record, "tags", context="capability definition"),
        )


@dataclass(frozen=True, slots=True)
class CapabilityAvailability:
    """Store the current runtime readiness of one capability."""

    schema_name: ClassVar[str] = _CAPABILITY_AVAILABILITY_SCHEMA

    capability_id: str
    status: CapabilityStatus
    detail: str = ""
    checked_at: datetime = field(default_factory=_utc_now)
    integration_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability_id", _require_stable_identifier(self.capability_id, field_name="capability_id"))
        object.__setattr__(self, "status", _coerce_enum(CapabilityStatus, self.status, field_name="status"))
        object.__setattr__(self, "detail", _optional_text(self.detail, field_name="detail", limit=240) or "")
        object.__setattr__(self, "checked_at", _normalize_datetime(self.checked_at, field_name="checked_at"))
        object.__setattr__(
            self,
            "integration_id",
            None if self.integration_id is None else _require_stable_identifier(self.integration_id, field_name="integration_id"),
        )
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))

    @property
    def available(self) -> bool:
        return self.status in {CapabilityStatus.READY, CapabilityStatus.UNCONFIGURED}

    @property
    def configured(self) -> bool:
        return self.status == CapabilityStatus.READY

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "capability_id": self.capability_id,
            "status": self.status.value,
            "detail": self.detail,
            "checked_at": _datetime_payload(self.checked_at),
            "integration_id": self.integration_id,
            "metadata": _payload_json_mapping(self.metadata, field_name="metadata"),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CapabilityAvailability":
        record = _require_mapping(payload, context="capability availability")
        _validate_schema(record, expected=cls.schema_name, context="capability availability")
        return cls(
            capability_id=_require_key(record, "capability_id", context="capability availability"),
            status=_require_key(record, "status", context="capability availability"),
            detail=_require_key(record, "detail", context="capability availability"),
            checked_at=_require_key(record, "checked_at", context="capability availability"),
            integration_id=_require_key(record, "integration_id", context="capability availability"),
            metadata=_require_key(record, "metadata", context="capability availability"),
        )


@dataclass(frozen=True, slots=True)
class FeasibilityResult:
    """Represent the deterministic outcome of a feasibility check."""

    schema_name: ClassVar[str] = _FEASIBILITY_RESULT_SCHEMA

    outcome: FeasibilityOutcome
    summary: str
    reasons: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    suggested_target: CompileTarget | None = None
    checked_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", _coerce_enum(FeasibilityOutcome, self.outcome, field_name="outcome"))
        object.__setattr__(self, "summary", _require_text(self.summary, field_name="summary", limit=220))
        object.__setattr__(self, "reasons", _text_tuple(self.reasons, field_name="reasons", limit=220))
        object.__setattr__(
            self,
            "missing_capabilities",
            _stable_identifier_tuple(self.missing_capabilities, field_name="missing_capabilities"),
        )
        object.__setattr__(
            self,
            "suggested_target",
            None if self.suggested_target is None else _coerce_enum(CompileTarget, self.suggested_target, field_name="suggested_target"),
        )
        object.__setattr__(self, "checked_at", _normalize_datetime(self.checked_at, field_name="checked_at"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "outcome": self.outcome.value,
            "summary": self.summary,
            "reasons": list(self.reasons),
            "missing_capabilities": list(self.missing_capabilities),
            "suggested_target": None if self.suggested_target is None else self.suggested_target.value,
            "checked_at": _datetime_payload(self.checked_at),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "FeasibilityResult":
        record = _require_mapping(payload, context="feasibility result")
        _validate_schema(record, expected=cls.schema_name, context="feasibility result")
        return cls(
            outcome=_require_key(record, "outcome", context="feasibility result"),
            summary=_require_key(record, "summary", context="feasibility result"),
            reasons=_require_key(record, "reasons", context="feasibility result"),
            missing_capabilities=_require_key(record, "missing_capabilities", context="feasibility result"),
            suggested_target=_require_key(record, "suggested_target", context="feasibility result"),
            checked_at=_require_key(record, "checked_at", context="feasibility result"),
        )


@dataclass(frozen=True, slots=True)
class RequirementsDialogueSession:
    """Persist one active self-coding requirements-gathering session."""

    schema_name: ClassVar[str] = _REQUIREMENTS_DIALOGUE_SESSION_SCHEMA

    session_id: str
    request_summary: str
    skill_name: str
    action: str
    capabilities: tuple[str, ...]
    feasibility: FeasibilityResult
    skill_id: str = ""
    status: RequirementsDialogueStatus = RequirementsDialogueStatus.QUESTIONING
    trigger_mode: str = "push"
    trigger_conditions: tuple[str, ...] = ()
    scope: dict[str, Any] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    current_question_id: str | None = "when"
    answered_question_ids: tuple[str, ...] = ()
    answer_summaries: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_stable_identifier(self.session_id, field_name="session_id"))
        object.__setattr__(self, "request_summary", _require_text(self.request_summary, field_name="request_summary", limit=220))
        object.__setattr__(self, "skill_name", _require_text(self.skill_name, field_name="skill_name", limit=160))
        object.__setattr__(self, "action", _require_text(self.action, field_name="action", limit=220))
        object.__setattr__(self, "capabilities", _stable_identifier_tuple(self.capabilities, field_name="capabilities"))
        if not self.capabilities:
            raise ValueError("capabilities must not be empty")
        if not isinstance(self.feasibility, FeasibilityResult):
            raise TypeError("feasibility must be a FeasibilityResult")
        object.__setattr__(self, "skill_id", _coerce_skill_id(self.skill_id, name=self.skill_name))
        object.__setattr__(self, "status", _coerce_enum(RequirementsDialogueStatus, self.status, field_name="status"))
        normalized_trigger_mode = _require_stable_identifier(self.trigger_mode, field_name="trigger_mode")
        if normalized_trigger_mode not in {"push", "pull"}:
            raise ValueError("trigger_mode must be 'push' or 'pull'")
        object.__setattr__(self, "trigger_mode", normalized_trigger_mode)
        object.__setattr__(
            self,
            "trigger_conditions",
            _stable_identifier_tuple(self.trigger_conditions, field_name="trigger_conditions"),
        )
        object.__setattr__(self, "scope", _json_mapping(self.scope, field_name="scope"))
        object.__setattr__(self, "constraints", _text_tuple(self.constraints, field_name="constraints", limit=220))
        current_question_id = _optional_stable_identifier(self.current_question_id, field_name="current_question_id")
        if self.status == RequirementsDialogueStatus.QUESTIONING and current_question_id not in _ALLOWED_REQUIREMENTS_QUESTION_IDS:
            raise ValueError("questioning sessions require current_question_id in {'when', 'what', 'how'}")
        if self.status == RequirementsDialogueStatus.CONFIRMING:
            current_question_id = "confirm"
        if self.status in {
            RequirementsDialogueStatus.READY_FOR_COMPILE,
            RequirementsDialogueStatus.CANCELLED,
        }:
            current_question_id = None
        object.__setattr__(self, "current_question_id", current_question_id)
        answered_question_ids = _stable_identifier_tuple(self.answered_question_ids, field_name="answered_question_ids")
        if any(question_id not in _ALLOWED_REQUIREMENTS_QUESTION_IDS for question_id in answered_question_ids):
            raise ValueError("answered_question_ids must be limited to {'when', 'what', 'how'}")  # AUDIT-FIX(#10): Keep dialogue state-machine identifiers bounded to the known question set.
        if current_question_id is not None and current_question_id in answered_question_ids:
            raise ValueError("current_question_id must not already be answered")  # AUDIT-FIX(#10): Reject impossible dialogue states that point at an already-completed question.
        object.__setattr__(self, "answered_question_ids", answered_question_ids)
        answer_summaries = _json_mapping(self.answer_summaries, field_name="answer_summaries")
        if any(key not in _ALLOWED_REQUIREMENTS_DIALOGUE_IDS for key in answer_summaries):
            raise ValueError("answer_summaries keys must be limited to {'when', 'what', 'how', 'confirm'}")  # AUDIT-FIX(#10): Prevent arbitrary answer-summary keys from corrupting dialogue recovery.
        object.__setattr__(self, "answer_summaries", answer_summaries)
        object.__setattr__(self, "created_at", _normalize_datetime(self.created_at, field_name="created_at"))
        object.__setattr__(self, "updated_at", _normalize_datetime(self.updated_at, field_name="updated_at"))
        _ensure_not_before(self.updated_at, self.created_at, later_name="updated_at", earlier_name="created_at")
        object.__setattr__(self, "version", _require_positive_int(self.version, field_name="version"))

    def to_skill_spec(self) -> SkillSpec:
        """Render the current dialogue draft as a full skill specification."""

        return SkillSpec(
            name=self.skill_name,
            action=self.action,
            trigger=SkillTriggerSpec(mode=self.trigger_mode, conditions=self.trigger_conditions),
            skill_id=self.skill_id,
            scope=self.scope,
            constraints=self.constraints,
            capabilities=self.capabilities,
            created_at=self.created_at,
            version=self.version,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "session_id": self.session_id,
            "request_summary": self.request_summary,
            "skill_name": self.skill_name,
            "action": self.action,
            "capabilities": list(self.capabilities),
            "feasibility": self.feasibility.to_payload(),
            "skill_id": self.skill_id,
            "status": self.status.value,
            "trigger_mode": self.trigger_mode,
            "trigger_conditions": list(self.trigger_conditions),
            "scope": _payload_json_mapping(self.scope, field_name="scope"),
            "constraints": list(self.constraints),
            "current_question_id": self.current_question_id,
            "answered_question_ids": list(self.answered_question_ids),
            "answer_summaries": _payload_json_mapping(self.answer_summaries, field_name="answer_summaries"),
            "created_at": _datetime_payload(self.created_at),
            "updated_at": _datetime_payload(self.updated_at),
            "version": self.version,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RequirementsDialogueSession":
        record = _require_mapping(payload, context="requirements dialogue session")
        _validate_schema(record, expected=cls.schema_name, context="requirements dialogue session")
        return cls(
            session_id=_require_key(record, "session_id", context="requirements dialogue session"),
            request_summary=_require_key(record, "request_summary", context="requirements dialogue session"),
            skill_name=_require_key(record, "skill_name", context="requirements dialogue session"),
            action=_require_key(record, "action", context="requirements dialogue session"),
            capabilities=_require_key(record, "capabilities", context="requirements dialogue session"),
            feasibility=FeasibilityResult.from_payload(_require_key(record, "feasibility", context="requirements dialogue session")),
            skill_id=_require_key(record, "skill_id", context="requirements dialogue session"),
            status=_require_key(record, "status", context="requirements dialogue session"),
            trigger_mode=_require_key(record, "trigger_mode", context="requirements dialogue session"),
            trigger_conditions=_require_key(record, "trigger_conditions", context="requirements dialogue session"),
            scope=_require_key(record, "scope", context="requirements dialogue session"),
            constraints=_require_key(record, "constraints", context="requirements dialogue session"),
            current_question_id=_require_key(record, "current_question_id", context="requirements dialogue session"),
            answered_question_ids=_require_key(record, "answered_question_ids", context="requirements dialogue session"),
            answer_summaries=_require_key(record, "answer_summaries", context="requirements dialogue session"),
            created_at=_require_key(record, "created_at", context="requirements dialogue session"),
            updated_at=_require_key(record, "updated_at", context="requirements dialogue session"),
            version=_require_key(record, "version", context="requirements dialogue session"),
        )


@dataclass(frozen=True, slots=True)
class CompileJobRecord:
    """Store the durable metadata for one compile job."""

    schema_name: ClassVar[str] = _COMPILE_JOB_SCHEMA

    job_id: str
    skill_id: str
    skill_name: str
    status: CompileJobStatus
    requested_target: CompileTarget
    spec_hash: str
    required_capabilities: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    attempt_count: int = 0
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _require_stable_identifier(self.job_id, field_name="job_id"))
        object.__setattr__(self, "skill_id", _require_stable_identifier(self.skill_id, field_name="skill_id"))
        object.__setattr__(self, "skill_name", _require_text(self.skill_name, field_name="skill_name", limit=160))
        object.__setattr__(self, "status", _coerce_enum(CompileJobStatus, self.status, field_name="status"))
        object.__setattr__(self, "requested_target", _coerce_enum(CompileTarget, self.requested_target, field_name="requested_target"))
        object.__setattr__(self, "spec_hash", _require_sha256(self.spec_hash, field_name="spec_hash"))  # AUDIT-FIX(#6): Treat spec_hash as an integrity digest, not arbitrary free text.
        object.__setattr__(
            self,
            "required_capabilities",
            _stable_identifier_tuple(self.required_capabilities, field_name="required_capabilities"),
        )
        object.__setattr__(self, "artifact_ids", _stable_identifier_tuple(self.artifact_ids, field_name="artifact_ids"))
        object.__setattr__(self, "created_at", _normalize_datetime(self.created_at, field_name="created_at"))
        object.__setattr__(self, "updated_at", _normalize_datetime(self.updated_at, field_name="updated_at"))
        _ensure_not_before(self.updated_at, self.created_at, later_name="updated_at", earlier_name="created_at")
        object.__setattr__(self, "attempt_count", _require_non_negative_int(self.attempt_count, field_name="attempt_count"))
        object.__setattr__(self, "last_error", _optional_text(self.last_error, field_name="last_error", limit=240))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "job_id": self.job_id,
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "status": self.status.value,
            "requested_target": self.requested_target.value,
            "spec_hash": self.spec_hash,
            "required_capabilities": list(self.required_capabilities),
            "artifact_ids": list(self.artifact_ids),
            "created_at": _datetime_payload(self.created_at),
            "updated_at": _datetime_payload(self.updated_at),
            "attempt_count": self.attempt_count,
            "last_error": self.last_error,
            "metadata": _payload_json_mapping(self.metadata, field_name="metadata"),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CompileJobRecord":
        record = _require_mapping(payload, context="compile job")
        _validate_schema(record, expected=cls.schema_name, context="compile job")
        return cls(
            job_id=_require_key(record, "job_id", context="compile job"),
            skill_id=_require_key(record, "skill_id", context="compile job"),
            skill_name=_require_key(record, "skill_name", context="compile job"),
            status=_require_key(record, "status", context="compile job"),
            requested_target=_require_key(record, "requested_target", context="compile job"),
            spec_hash=_require_key(record, "spec_hash", context="compile job"),
            required_capabilities=_require_key(record, "required_capabilities", context="compile job"),
            artifact_ids=_require_key(record, "artifact_ids", context="compile job"),
            created_at=_require_key(record, "created_at", context="compile job"),
            updated_at=_require_key(record, "updated_at", context="compile job"),
            attempt_count=_require_key(record, "attempt_count", context="compile job"),
            last_error=_require_key(record, "last_error", context="compile job"),
            metadata=_require_key(record, "metadata", context="compile job"),
        )


@dataclass(frozen=True, slots=True)
class CompileArtifactRecord:
    """Store the durable metadata for one compile artifact."""

    schema_name: ClassVar[str] = _COMPILE_ARTIFACT_SCHEMA

    artifact_id: str
    job_id: str
    kind: ArtifactKind
    media_type: str
    content_path: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    summary: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _require_stable_identifier(self.artifact_id, field_name="artifact_id"))
        object.__setattr__(self, "job_id", _require_stable_identifier(self.job_id, field_name="job_id"))
        object.__setattr__(self, "kind", _coerce_enum(ArtifactKind, self.kind, field_name="kind"))
        object.__setattr__(self, "media_type", _require_text(self.media_type, field_name="media_type", limit=120))
        object.__setattr__(self, "content_path", _optional_safe_relative_path(self.content_path, field_name="content_path"))
        object.__setattr__(self, "sha256", _optional_sha256(self.sha256, field_name="sha256"))
        object.__setattr__(self, "size_bytes", _optional_non_negative_int(self.size_bytes, field_name="size_bytes"))
        object.__setattr__(self, "summary", _optional_text(self.summary, field_name="summary", limit=220))
        object.__setattr__(self, "created_at", _normalize_datetime(self.created_at, field_name="created_at"))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "artifact_id": self.artifact_id,
            "job_id": self.job_id,
            "kind": self.kind.value,
            "media_type": self.media_type,
            "content_path": self.content_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "summary": self.summary,
            "created_at": _datetime_payload(self.created_at),
            "metadata": _payload_json_mapping(self.metadata, field_name="metadata"),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CompileArtifactRecord":
        record = _require_mapping(payload, context="compile artifact")
        _validate_schema(record, expected=cls.schema_name, context="compile artifact")
        return cls(
            artifact_id=_require_key(record, "artifact_id", context="compile artifact"),
            job_id=_require_key(record, "job_id", context="compile artifact"),
            kind=_require_key(record, "kind", context="compile artifact"),
            media_type=_require_key(record, "media_type", context="compile artifact"),
            content_path=_require_key(record, "content_path", context="compile artifact"),
            sha256=_require_key(record, "sha256", context="compile artifact"),
            size_bytes=_require_key(record, "size_bytes", context="compile artifact"),
            summary=_require_key(record, "summary", context="compile artifact"),
            created_at=_require_key(record, "created_at", context="compile artifact"),
            metadata=_require_key(record, "metadata", context="compile artifact"),
        )


@dataclass(frozen=True, slots=True)
class CompileRunStatusRecord:
    """Store operator-visible runtime status for one compile attempt."""

    schema_name: ClassVar[str] = _COMPILE_RUN_STATUS_SCHEMA

    job_id: str
    phase: str
    driver_name: str | None = None
    driver_attempts: tuple[str, ...] = ()
    event_count: int = 0
    last_event_kind: str | None = None
    last_event_message: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    final_message_seen: bool = False
    turn_completed: bool = False
    started_at: datetime | None = None
    updated_at: datetime = field(default_factory=_utc_now)
    completed_at: datetime | None = None
    error_message: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _require_stable_identifier(self.job_id, field_name="job_id"))
        object.__setattr__(self, "phase", _require_stable_identifier(self.phase, field_name="phase"))
        object.__setattr__(self, "driver_name", _optional_text(self.driver_name, field_name="driver_name", limit=120))
        object.__setattr__(self, "driver_attempts", _text_tuple(self.driver_attempts, field_name="driver_attempts", limit=120))
        object.__setattr__(self, "event_count", _require_non_negative_int(self.event_count, field_name="event_count"))
        object.__setattr__(self, "last_event_kind", _optional_text(self.last_event_kind, field_name="last_event_kind", limit=120))
        object.__setattr__(self, "last_event_message", _optional_text(self.last_event_message, field_name="last_event_message", limit=240))
        object.__setattr__(self, "thread_id", _optional_text(self.thread_id, field_name="thread_id", limit=120))
        object.__setattr__(self, "turn_id", _optional_text(self.turn_id, field_name="turn_id", limit=120))
        object.__setattr__(self, "final_message_seen", _require_bool(self.final_message_seen, field_name="final_message_seen"))
        object.__setattr__(self, "turn_completed", _require_bool(self.turn_completed, field_name="turn_completed"))
        if self.started_at is not None:
            object.__setattr__(self, "started_at", _normalize_datetime(self.started_at, field_name="started_at"))
        object.__setattr__(self, "updated_at", _normalize_datetime(self.updated_at, field_name="updated_at"))
        if self.completed_at is not None:
            object.__setattr__(self, "completed_at", _normalize_datetime(self.completed_at, field_name="completed_at"))
        if self.completed_at is not None and self.started_at is None:
            raise ValueError("completed_at requires started_at")  # AUDIT-FIX(#7): Reject impossible compile lifecycle snapshots that claim completion without a start time.
        if self.started_at is not None:
            _ensure_not_before(self.updated_at, self.started_at, later_name="updated_at", earlier_name="started_at")
        if self.completed_at is not None and self.started_at is not None:
            _ensure_not_before(self.completed_at, self.started_at, later_name="completed_at", earlier_name="started_at")
        if self.completed_at is not None:
            _ensure_not_before(self.updated_at, self.completed_at, later_name="updated_at", earlier_name="completed_at")
        object.__setattr__(self, "error_message", _optional_text(self.error_message, field_name="error_message", limit=240))
        object.__setattr__(self, "diagnostics", _json_mapping(self.diagnostics, field_name="diagnostics"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "job_id": self.job_id,
            "phase": self.phase,
            "driver_name": self.driver_name,
            "driver_attempts": list(self.driver_attempts),
            "event_count": self.event_count,
            "last_event_kind": self.last_event_kind,
            "last_event_message": self.last_event_message,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "final_message_seen": self.final_message_seen,
            "turn_completed": self.turn_completed,
            "started_at": None if self.started_at is None else _datetime_payload(self.started_at),
            "updated_at": _datetime_payload(self.updated_at),
            "completed_at": None if self.completed_at is None else _datetime_payload(self.completed_at),
            "error_message": self.error_message,
            "diagnostics": _payload_json_mapping(self.diagnostics, field_name="diagnostics"),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CompileRunStatusRecord":
        record = _require_mapping(payload, context="compile run status")
        _validate_schema(record, expected=cls.schema_name, context="compile run status")
        return cls(
            job_id=_require_key(record, "job_id", context="compile run status"),
            phase=_require_key(record, "phase", context="compile run status"),
            driver_name=_require_key(record, "driver_name", context="compile run status"),
            driver_attempts=_require_key(record, "driver_attempts", context="compile run status"),
            event_count=_require_key(record, "event_count", context="compile run status"),
            last_event_kind=_require_key(record, "last_event_kind", context="compile run status"),
            last_event_message=_require_key(record, "last_event_message", context="compile run status"),
            thread_id=_require_key(record, "thread_id", context="compile run status"),
            turn_id=_require_key(record, "turn_id", context="compile run status"),
            final_message_seen=_require_key(record, "final_message_seen", context="compile run status"),
            turn_completed=_require_key(record, "turn_completed", context="compile run status"),
            started_at=_require_key(record, "started_at", context="compile run status"),
            updated_at=_require_key(record, "updated_at", context="compile run status"),
            completed_at=_require_key(record, "completed_at", context="compile run status"),
            error_message=_require_key(record, "error_message", context="compile run status"),
            diagnostics=_require_key(record, "diagnostics", context="compile run status"),
        )


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    """Store the activation state for one learned skill version."""

    schema_name: ClassVar[str] = _ACTIVATION_RECORD_SCHEMA

    skill_id: str
    skill_name: str
    version: int
    status: LearnedSkillStatus
    job_id: str
    artifact_id: str
    updated_at: datetime = field(default_factory=_utc_now)
    activated_at: datetime | None = None
    feedback_due_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", _require_stable_identifier(self.skill_id, field_name="skill_id"))
        object.__setattr__(self, "skill_name", _require_text(self.skill_name, field_name="skill_name", limit=160))
        object.__setattr__(self, "version", _require_positive_int(self.version, field_name="version"))
        object.__setattr__(self, "status", _coerce_enum(LearnedSkillStatus, self.status, field_name="status"))
        object.__setattr__(self, "job_id", _require_stable_identifier(self.job_id, field_name="job_id"))
        object.__setattr__(self, "artifact_id", _require_stable_identifier(self.artifact_id, field_name="artifact_id"))
        object.__setattr__(self, "updated_at", _normalize_datetime(self.updated_at, field_name="updated_at"))
        object.__setattr__(
            self,
            "activated_at",
            None if self.activated_at is None else _normalize_datetime(self.activated_at, field_name="activated_at"),
        )
        object.__setattr__(
            self,
            "feedback_due_at",
            None if self.feedback_due_at is None else _normalize_datetime(self.feedback_due_at, field_name="feedback_due_at"),
        )
        if self.feedback_due_at is not None and self.activated_at is None:
            raise ValueError("feedback_due_at requires activated_at")  # AUDIT-FIX(#9): Do not persist post-activation follow-up deadlines without an activation timestamp.
        if self.activated_at is not None:
            _ensure_not_before(self.updated_at, self.activated_at, later_name="updated_at", earlier_name="activated_at")
        if self.feedback_due_at is not None and self.activated_at is not None:
            _ensure_not_before(self.feedback_due_at, self.activated_at, later_name="feedback_due_at", earlier_name="activated_at")
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "version": self.version,
            "status": self.status.value,
            "job_id": self.job_id,
            "artifact_id": self.artifact_id,
            "updated_at": _datetime_payload(self.updated_at),
            "activated_at": None if self.activated_at is None else _datetime_payload(self.activated_at),
            "feedback_due_at": None if self.feedback_due_at is None else _datetime_payload(self.feedback_due_at),
            "metadata": _payload_json_mapping(self.metadata, field_name="metadata"),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ActivationRecord":
        record = _require_mapping(payload, context="activation record")
        _validate_schema(record, expected=cls.schema_name, context="activation record")
        return cls(
            skill_id=_require_key(record, "skill_id", context="activation record"),
            skill_name=_require_key(record, "skill_name", context="activation record"),
            version=_require_key(record, "version", context="activation record"),
            status=_require_key(record, "status", context="activation record"),
            job_id=_require_key(record, "job_id", context="activation record"),
            artifact_id=_require_key(record, "artifact_id", context="activation record"),
            updated_at=_require_key(record, "updated_at", context="activation record"),
            activated_at=_require_key(record, "activated_at", context="activation record"),
            feedback_due_at=_require_key(record, "feedback_due_at", context="activation record"),
            metadata=_require_key(record, "metadata", context="activation record"),
        )


@dataclass(frozen=True, slots=True)
class SkillHealthRecord:
    """Persist bounded health counters for one learned skill version."""

    schema_name: ClassVar[str] = _SKILL_HEALTH_RECORD_SCHEMA

    skill_id: str
    version: int
    status: str = "unknown"
    trigger_count: int = 0
    delivered_count: int = 0
    error_count: int = 0
    consecutive_error_count: int = 0
    auto_pause_count: int = 0
    last_triggered_at: datetime | None = None
    last_delivered_at: datetime | None = None
    last_error_at: datetime | None = None
    last_error_message: str | None = None
    updated_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", _require_stable_identifier(self.skill_id, field_name="skill_id"))
        object.__setattr__(self, "version", _require_positive_int(self.version, field_name="version"))
        object.__setattr__(self, "status", _require_stable_identifier(self.status, field_name="status"))
        object.__setattr__(self, "trigger_count", _require_non_negative_int(self.trigger_count, field_name="trigger_count"))
        object.__setattr__(self, "delivered_count", _require_non_negative_int(self.delivered_count, field_name="delivered_count"))
        object.__setattr__(self, "error_count", _require_non_negative_int(self.error_count, field_name="error_count"))
        object.__setattr__(
            self,
            "consecutive_error_count",
            _require_non_negative_int(self.consecutive_error_count, field_name="consecutive_error_count"),
        )
        object.__setattr__(self, "auto_pause_count", _require_non_negative_int(self.auto_pause_count, field_name="auto_pause_count"))
        object.__setattr__(
            self,
            "last_triggered_at",
            None if self.last_triggered_at is None else _normalize_datetime(self.last_triggered_at, field_name="last_triggered_at"),
        )
        object.__setattr__(
            self,
            "last_delivered_at",
            None if self.last_delivered_at is None else _normalize_datetime(self.last_delivered_at, field_name="last_delivered_at"),
        )
        object.__setattr__(
            self,
            "last_error_at",
            None if self.last_error_at is None else _normalize_datetime(self.last_error_at, field_name="last_error_at"),
        )
        object.__setattr__(self, "last_error_message", _optional_text(self.last_error_message, field_name="last_error_message", limit=240))
        object.__setattr__(self, "updated_at", _normalize_datetime(self.updated_at, field_name="updated_at"))
        if self.delivered_count > self.trigger_count:
            raise ValueError("delivered_count must be <= trigger_count")  # AUDIT-FIX(#8): Enforce sane monotonic counters for skill health state.
        if self.consecutive_error_count > self.error_count:
            raise ValueError("consecutive_error_count must be <= error_count")  # AUDIT-FIX(#8): Prevent impossible error streak counters from hitting disk.
        if self.last_triggered_at is not None and self.trigger_count == 0:
            raise ValueError("last_triggered_at requires trigger_count >= 1")  # AUDIT-FIX(#8): Timestamp presence must match counter presence for recovery logic.
        if self.last_delivered_at is not None and self.delivered_count == 0:
            raise ValueError("last_delivered_at requires delivered_count >= 1")  # AUDIT-FIX(#8): Do not persist delivery timestamps for zero deliveries.
        if self.last_error_at is not None and self.error_count == 0:
            raise ValueError("last_error_at requires error_count >= 1")  # AUDIT-FIX(#8): Keep error telemetry internally consistent.
        if self.last_error_message is not None and self.error_count == 0:
            raise ValueError("last_error_message requires error_count >= 1")  # AUDIT-FIX(#8): Error detail without any errors indicates corrupted state.
        for timestamp_name in ("last_triggered_at", "last_delivered_at", "last_error_at"):
            timestamp = getattr(self, timestamp_name)
            if timestamp is not None:
                _ensure_not_before(self.updated_at, timestamp, later_name="updated_at", earlier_name=timestamp_name)  # AUDIT-FIX(#8): Health snapshots cannot predate their latest recorded event.
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "skill_id": self.skill_id,
            "version": self.version,
            "status": self.status,
            "trigger_count": self.trigger_count,
            "delivered_count": self.delivered_count,
            "error_count": self.error_count,
            "consecutive_error_count": self.consecutive_error_count,
            "auto_pause_count": self.auto_pause_count,
            "last_triggered_at": None if self.last_triggered_at is None else _datetime_payload(self.last_triggered_at),
            "last_delivered_at": None if self.last_delivered_at is None else _datetime_payload(self.last_delivered_at),
            "last_error_at": None if self.last_error_at is None else _datetime_payload(self.last_error_at),
            "last_error_message": self.last_error_message,
            "updated_at": _datetime_payload(self.updated_at),
            "metadata": _payload_json_mapping(self.metadata, field_name="metadata"),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "SkillHealthRecord":
        record = _require_mapping(payload, context="skill health record")
        _validate_schema(record, expected=cls.schema_name, context="skill health record")
        return cls(
            skill_id=_require_key(record, "skill_id", context="skill health record"),
            version=_require_key(record, "version", context="skill health record"),
            status=_require_key(record, "status", context="skill health record"),
            trigger_count=_require_key(record, "trigger_count", context="skill health record"),
            delivered_count=_require_key(record, "delivered_count", context="skill health record"),
            error_count=_require_key(record, "error_count", context="skill health record"),
            consecutive_error_count=_require_key(record, "consecutive_error_count", context="skill health record"),
            auto_pause_count=_require_key(record, "auto_pause_count", context="skill health record"),
            last_triggered_at=_require_key(record, "last_triggered_at", context="skill health record"),
            last_delivered_at=_require_key(record, "last_delivered_at", context="skill health record"),
            last_error_at=_require_key(record, "last_error_at", context="skill health record"),
            last_error_message=_require_key(record, "last_error_message", context="skill health record"),
            updated_at=_require_key(record, "updated_at", context="skill health record"),
            metadata=_require_key(record, "metadata", context="skill health record"),
        )


@dataclass(frozen=True, slots=True)
class ExecutionRunStatusRecord:
    """Persist one sandboxed execution or retest lifecycle record."""

    schema_name: ClassVar[str] = _EXECUTION_RUN_STATUS_RECORD_SCHEMA

    run_id: str
    run_kind: str
    skill_id: str
    version: int
    status: str
    reason: str | None = None
    timeout_seconds: float | None = None
    started_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_stable_identifier(self.run_id, field_name="run_id"))
        object.__setattr__(self, "run_kind", _require_stable_identifier(self.run_kind, field_name="run_kind"))
        object.__setattr__(self, "skill_id", _require_stable_identifier(self.skill_id, field_name="skill_id"))
        object.__setattr__(self, "version", _require_positive_int(self.version, field_name="version"))
        object.__setattr__(self, "status", _require_stable_identifier(self.status, field_name="status"))
        object.__setattr__(self, "reason", _optional_text(self.reason, field_name="reason", limit=240))
        object.__setattr__(
            self,
            "timeout_seconds",
            _optional_non_negative_float(self.timeout_seconds, field_name="timeout_seconds"),
        )
        object.__setattr__(self, "started_at", _normalize_datetime(self.started_at, field_name="started_at"))
        object.__setattr__(self, "updated_at", _normalize_datetime(self.updated_at, field_name="updated_at"))
        if self.completed_at is not None:
            object.__setattr__(self, "completed_at", _normalize_datetime(self.completed_at, field_name="completed_at"))
            _ensure_not_before(self.completed_at, self.started_at, later_name="completed_at", earlier_name="started_at")
            _ensure_not_before(self.updated_at, self.completed_at, later_name="updated_at", earlier_name="completed_at")
        else:
            _ensure_not_before(self.updated_at, self.started_at, later_name="updated_at", earlier_name="started_at")
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "skill_id": self.skill_id,
            "version": self.version,
            "status": self.status,
            "reason": self.reason,
            "timeout_seconds": self.timeout_seconds,
            "started_at": _datetime_payload(self.started_at),
            "updated_at": _datetime_payload(self.updated_at),
            "completed_at": None if self.completed_at is None else _datetime_payload(self.completed_at),
            "metadata": _payload_json_mapping(self.metadata, field_name="metadata"),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ExecutionRunStatusRecord":
        record = _require_mapping(payload, context="execution run status record")
        _validate_schema(record, expected=cls.schema_name, context="execution run status record")
        return cls(
            run_id=_require_key(record, "run_id", context="execution run status record"),
            run_kind=_require_key(record, "run_kind", context="execution run status record"),
            skill_id=_require_key(record, "skill_id", context="execution run status record"),
            version=_require_key(record, "version", context="execution run status record"),
            status=_require_key(record, "status", context="execution run status record"),
            reason=_require_key(record, "reason", context="execution run status record"),
            timeout_seconds=_require_key(record, "timeout_seconds", context="execution run status record"),
            started_at=_require_key(record, "started_at", context="execution run status record"),
            updated_at=_require_key(record, "updated_at", context="execution run status record"),
            completed_at=_require_key(record, "completed_at", context="execution run status record"),
            metadata=_require_key(record, "metadata", context="execution run status record"),
        )


@dataclass(frozen=True, slots=True)
class LiveE2EStatusRecord:
    """Persist the latest explicit live end-to-end proof status for one suite."""

    schema_name: ClassVar[str] = _LIVE_E2E_STATUS_RECORD_SCHEMA

    suite_id: str
    environment: str
    status: str
    duration_seconds: float | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    details: str | None = None
    updated_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "suite_id", _require_stable_identifier(self.suite_id, field_name="suite_id"))
        object.__setattr__(self, "environment", _require_stable_identifier(self.environment, field_name="environment"))
        object.__setattr__(self, "status", _require_stable_identifier(self.status, field_name="status"))
        object.__setattr__(
            self,
            "duration_seconds",
            _optional_non_negative_float(self.duration_seconds, field_name="duration_seconds"),
        )
        object.__setattr__(self, "model", _optional_text(self.model, field_name="model", limit=120))
        object.__setattr__(self, "reasoning_effort", _optional_text(self.reasoning_effort, field_name="reasoning_effort", limit=32))
        object.__setattr__(self, "details", _optional_text(self.details, field_name="details", limit=240))
        object.__setattr__(self, "updated_at", _normalize_datetime(self.updated_at, field_name="updated_at"))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "suite_id": self.suite_id,
            "environment": self.environment,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "details": self.details,
            "updated_at": _datetime_payload(self.updated_at),
            "metadata": _payload_json_mapping(self.metadata, field_name="metadata"),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "LiveE2EStatusRecord":
        record = _require_mapping(payload, context="live e2e status record")
        _validate_schema(record, expected=cls.schema_name, context="live e2e status record")
        return cls(
            suite_id=_require_key(record, "suite_id", context="live e2e status record"),
            environment=_require_key(record, "environment", context="live e2e status record"),
            status=_require_key(record, "status", context="live e2e status record"),
            duration_seconds=_require_key(record, "duration_seconds", context="live e2e status record"),
            model=_require_key(record, "model", context="live e2e status record"),
            reasoning_effort=_require_key(record, "reasoning_effort", context="live e2e status record"),
            details=_require_key(record, "details", context="live e2e status record"),
            updated_at=_require_key(record, "updated_at", context="live e2e status record"),
            metadata=_require_key(record, "metadata", context="live e2e status record"),
        )