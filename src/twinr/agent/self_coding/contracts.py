
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

_MAX_STABLE_IDENTIFIER_LENGTH = 128
_MAX_PATH_LENGTH = 240
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FrozenJsonDict(dict):
    """Read-only JSON mapping used inside frozen contract objects."""

    __slots__ = ()

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("JSON mappings are immutable")  # AUDIT-FIX(#9): Prevent mutation through nested dict fields inside frozen dataclasses.

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


def _stringify(value: object | None) -> str:
    return "" if value is None else str(value)


def _normalize_datetime(value: datetime | str | None, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"  # AUDIT-FIX(#2): Only normalize an explicit UTC suffix instead of replacing every 'Z' in the string.
        parsed = datetime.fromisoformat(text)
    else:
        raise TypeError(f"{field_name} must be a datetime or ISO timestamp")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")  # AUDIT-FIX(#2): Reject naive datetimes instead of silently assuming UTC.
    return parsed.astimezone(UTC)


def _datetime_payload(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_text(value: object, *, field_name: str, limit: int) -> str:
    text = truncate_text(_stringify(value), limit=limit).strip()  # AUDIT-FIX(#8): Strip whitespace so empty-looking values do not pass validation.
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _optional_text(value: object | None, *, limit: int) -> str | None:
    text = truncate_text(_stringify(value), limit=limit).strip()  # AUDIT-FIX(#8): Normalize surrounding whitespace for optional text fields too.
    return text or None


def _require_stable_identifier(value: object, *, field_name: str) -> str:
    text = _stringify(value).strip().lower()
    if len(text) > _MAX_STABLE_IDENTIFIER_LENGTH:
        raise ValueError(f"{field_name} must be <= {_MAX_STABLE_IDENTIFIER_LENGTH} characters")  # AUDIT-FIX(#3): Reject overlong IDs instead of truncating them into a different identifier.
    if not is_valid_stable_identifier(text):
        raise ValueError(f"{field_name} must be a stable identifier")
    return text


def _coerce_skill_id(value: object, *, name: str) -> str:
    raw = _stringify(value).strip().lower()
    if not raw:
        generated = slugify_identifier(name, fallback="skill")
        if not is_valid_stable_identifier(generated):
            raise ValueError("skill_id must be a stable identifier")
        return generated
    if len(raw) > _MAX_STABLE_IDENTIFIER_LENGTH:
        raise ValueError(f"skill_id must be <= {_MAX_STABLE_IDENTIFIER_LENGTH} characters")  # AUDIT-FIX(#3): Reject explicit IDs that are too long instead of truncating them.
    if not is_valid_stable_identifier(raw):
        raise ValueError("skill_id must be a stable identifier")  # AUDIT-FIX(#3): Only auto-generate a skill_id when it is blank, never when the caller supplied an invalid one.
    return raw


def _optional_stable_identifier(value: object | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    text = _stringify(value).strip().lower()
    if not text:
        return None
    if len(text) > _MAX_STABLE_IDENTIFIER_LENGTH:
        raise ValueError(f"{field_name} must be <= {_MAX_STABLE_IDENTIFIER_LENGTH} characters")  # AUDIT-FIX(#3): Reject overlong optional IDs instead of silently truncating them.
    if not is_valid_stable_identifier(text):
        raise ValueError(f"{field_name} must be a stable identifier")
    return text


def _stable_identifier_tuple(values: object, *, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raw_items: Sequence[object] = (values,)
    elif isinstance(values, Sequence):
        raw_items = values
    else:
        raise TypeError(f"{field_name} must be a sequence of identifiers")

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
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raw_items: Sequence[object] = (values,)
    elif isinstance(values, Sequence):
        raw_items = values
    else:
        raise TypeError(f"{field_name} must be a sequence of text values")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = _optional_text(raw_item, limit=limit)  # AUDIT-FIX(#8): Reuse stripped text normalization so whitespace-only entries are dropped.
        if item is None or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _require_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a boolean")  # AUDIT-FIX(#7): Reject truthy strings and integers instead of coercing them with bool(...).


def _require_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")  # AUDIT-FIX(#7): Reject bool/float/string inputs instead of truncating or reinterpreting them.
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return value


def _require_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")  # AUDIT-FIX(#7): Enforce exact integer shape for counters and byte sizes.
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def _optional_non_negative_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_non_negative_int(value, field_name=field_name)


def _optional_safe_relative_path(value: object | None, *, field_name: str, limit: int = _MAX_PATH_LENGTH) -> str | None:
    text = _stringify(value).strip()
    if not text:
        return None
    if len(text) > limit:
        raise ValueError(f"{field_name} must be <= {limit} characters")
    if "\x00" in text or "\\" in text or any(ord(char) < 32 for char in text):
        raise ValueError(f"{field_name} must not contain control characters or backslashes")  # AUDIT-FIX(#1): Reject path payloads that can confuse downstream file operations.
    path = PurePosixPath(text)
    parts = path.parts
    if path.is_absolute() or str(path) in {".", ".."} or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{field_name} must be a safe relative path")  # AUDIT-FIX(#1): Block absolute paths and traversal segments at the contract boundary.
    return str(path)


def _optional_sha256(value: object | None, *, field_name: str) -> str | None:
    text = _stringify(value).strip().lower()
    if not text:
        return None
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 hex digest")  # AUDIT-FIX(#1): Validate integrity digests instead of storing arbitrary text.
    return text


def _normalize_json_value(value: object, *, field_name: str, path: str = "$") -> Any:
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
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{field_name} keys must be strings at {path}")  # AUDIT-FIX(#9): Enforce deterministic JSON object keys instead of silently stringifying them.
            child_path = f"{path}.{raw_key}"
            normalized[raw_key] = _normalize_json_value(raw_item, field_name=field_name, path=child_path)
        return FrozenJsonDict(normalized)  # AUDIT-FIX(#9): Store frozen JSON mappings so nested state cannot be mutated after validation.
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _normalize_json_value(item, field_name=field_name, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{field_name} must be JSON serializable at {path}")


def _json_mapping(value: object | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return FrozenJsonDict()  # AUDIT-FIX(#9): Always hand out a dedicated immutable mapping, never a shared mutable default.
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
    return payload  # AUDIT-FIX(#9): Deep-copy mapping payloads so callers cannot mutate live contract state through to_payload().


def _require_mapping(payload: object, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} payload must be a mapping")
    return dict(payload)


def _validate_schema(payload: Mapping[str, Any], *, expected: str, context: str) -> None:
    schema = payload.get("schema")
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"{context} payload missing schema")  # AUDIT-FIX(#4): Require an explicit schema marker on every versioned payload.
    if schema != expected:
        raise ValueError(f"{context} payload schema mismatch: {schema!r} != {expected!r}")


def _ensure_not_before(later: datetime, earlier: datetime, *, later_name: str, earlier_name: str) -> None:
    if later < earlier:
        raise ValueError(f"{later_name} must be >= {earlier_name}")  # AUDIT-FIX(#2): Reject impossible temporal ordering before the state hits disk.


def _coerce_enum(enum_type, value: object, *, field_name: str):
    if isinstance(value, enum_type):
        return value
    text = truncate_text(_stringify(value), limit=64).strip().lower()
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
            mode=record.get("mode", ""),
            conditions=record.get("conditions", ()),  # AUDIT-FIX(#5): Preserve the raw payload shape so validators can reject strings/mappings instead of tuple-mangling them.
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
        object.__setattr__(self, "scope", _json_mapping(self.scope, field_name="scope"))  # AUDIT-FIX(#9): Freeze validated JSON mappings stored on the contract object.
        object.__setattr__(self, "constraints", _text_tuple(self.constraints, field_name="constraints", limit=220))  # AUDIT-FIX(#6): Keep constraints as free text so dialogue -> skill conversion does not fail.
        object.__setattr__(self, "capabilities", _stable_identifier_tuple(self.capabilities, field_name="capabilities"))
        object.__setattr__(self, "created_at", _normalize_datetime(self.created_at, field_name="created_at"))
        object.__setattr__(self, "version", _require_positive_int(self.version, field_name="version"))  # AUDIT-FIX(#7): Reject bool/float/string version payloads instead of coercing them.

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "skill_id": self.skill_id,
            "name": self.name,
            "action": self.action,
            "trigger": self.trigger.to_payload(),
            "scope": _payload_json_mapping(self.scope, field_name="scope"),  # AUDIT-FIX(#9): Serialize a detached copy of scope so callers cannot mutate the live object.
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
            skill_id=record.get("skill_id", ""),
            name=record.get("name", ""),
            action=record.get("action", ""),
            trigger=SkillTriggerSpec.from_payload(record.get("trigger", {})),
            scope=record.get("scope", {}),
            constraints=record.get("constraints", ()),  # AUDIT-FIX(#5): Do not wrap sequences with tuple(...); let validators inspect the original type.
            capabilities=record.get("capabilities", ()),  # AUDIT-FIX(#5): Prevent accidental string -> character tuple corruption on deserialize.
            created_at=record.get("created_at", _utc_now()),
            version=record.get("version", 1),
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
        object.__setattr__(self, "requires_configuration", _require_bool(self.requires_configuration, field_name="requires_configuration"))  # AUDIT-FIX(#7): Preserve boolean semantics instead of treating any truthy value as True.
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
            capability_id=record.get("capability_id", ""),
            module_name=record.get("module_name", ""),
            summary=record.get("summary", ""),
            risk_class=record.get("risk_class", CapabilityRiskClass.LOW.value),
            requires_configuration=record.get("requires_configuration", False),  # AUDIT-FIX(#7): Keep the original boolean payload so strict validation can reject bad shapes.
            integration_id=record.get("integration_id"),
            tags=record.get("tags", ()),  # AUDIT-FIX(#5): Preserve raw sequence shape for tag validation.
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
        object.__setattr__(self, "detail", _optional_text(self.detail, limit=240) or "")
        object.__setattr__(self, "checked_at", _normalize_datetime(self.checked_at, field_name="checked_at"))
        object.__setattr__(
            self,
            "integration_id",
            None if self.integration_id is None else _require_stable_identifier(self.integration_id, field_name="integration_id"),
        )
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))  # AUDIT-FIX(#9): Freeze metadata after JSON validation.

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
            "metadata": _payload_json_mapping(self.metadata, field_name="metadata"),  # AUDIT-FIX(#9): Return a detached metadata payload.
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CapabilityAvailability":
        record = _require_mapping(payload, context="capability availability")
        _validate_schema(record, expected=cls.schema_name, context="capability availability")
        return cls(
            capability_id=record.get("capability_id", ""),
            status=record.get("status", CapabilityStatus.MISSING.value),
            detail=record.get("detail", ""),
            checked_at=record.get("checked_at", _utc_now()),
            integration_id=record.get("integration_id"),
            metadata=record.get("metadata", {}),
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
            outcome=record.get("outcome", FeasibilityOutcome.RED.value),
            summary=record.get("summary", ""),
            reasons=record.get("reasons", ()),  # AUDIT-FIX(#5): Keep raw list/string input so reason validation can reject malformed payloads.
            missing_capabilities=record.get("missing_capabilities", ()),  # AUDIT-FIX(#5): Avoid tuple(...) coercion that can split strings into characters.
            suggested_target=record.get("suggested_target"),
            checked_at=record.get("checked_at", _utc_now()),
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
        object.__setattr__(self, "scope", _json_mapping(self.scope, field_name="scope"))  # AUDIT-FIX(#9): Freeze validated scope so shared async state cannot mutate it by reference.
        object.__setattr__(self, "constraints", _text_tuple(self.constraints, field_name="constraints", limit=220))
        current_question_id = _optional_stable_identifier(self.current_question_id, field_name="current_question_id")
        if self.status == RequirementsDialogueStatus.QUESTIONING and current_question_id not in {"when", "what", "how"}:
            raise ValueError("questioning sessions require current_question_id in {'when', 'what', 'how'}")
        if self.status == RequirementsDialogueStatus.CONFIRMING:
            current_question_id = "confirm"
        if self.status in {
            RequirementsDialogueStatus.READY_FOR_COMPILE,
            RequirementsDialogueStatus.CANCELLED,
        }:
            current_question_id = None
        object.__setattr__(self, "current_question_id", current_question_id)
        object.__setattr__(
            self,
            "answered_question_ids",
            _stable_identifier_tuple(self.answered_question_ids, field_name="answered_question_ids"),
        )
        object.__setattr__(self, "answer_summaries", _json_mapping(self.answer_summaries, field_name="answer_summaries"))  # AUDIT-FIX(#9): Freeze answer summaries after normalization.
        object.__setattr__(self, "created_at", _normalize_datetime(self.created_at, field_name="created_at"))
        object.__setattr__(self, "updated_at", _normalize_datetime(self.updated_at, field_name="updated_at"))
        _ensure_not_before(self.updated_at, self.created_at, later_name="updated_at", earlier_name="created_at")  # AUDIT-FIX(#2): Reject dialogue records whose update timestamp predates creation.
        object.__setattr__(self, "version", _require_positive_int(self.version, field_name="version"))  # AUDIT-FIX(#7): Enforce exact integer versions.

    def to_skill_spec(self) -> SkillSpec:
        """Render the current dialogue draft as a full skill specification."""

        return SkillSpec(
            name=self.skill_name,
            action=self.action,
            trigger=SkillTriggerSpec(mode=self.trigger_mode, conditions=self.trigger_conditions),
            skill_id=self.skill_id,
            scope=self.scope,
            constraints=self.constraints,  # AUDIT-FIX(#6): This now round-trips because SkillSpec.constraints accepts free text.
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
            "scope": _payload_json_mapping(self.scope, field_name="scope"),  # AUDIT-FIX(#9): Return detached JSON payloads for mutable mapping fields.
            "constraints": list(self.constraints),
            "current_question_id": self.current_question_id,
            "answered_question_ids": list(self.answered_question_ids),
            "answer_summaries": _payload_json_mapping(self.answer_summaries, field_name="answer_summaries"),  # AUDIT-FIX(#9): Detach answer summaries from the live object.
            "created_at": _datetime_payload(self.created_at),
            "updated_at": _datetime_payload(self.updated_at),
            "version": self.version,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RequirementsDialogueSession":
        record = _require_mapping(payload, context="requirements dialogue session")
        _validate_schema(record, expected=cls.schema_name, context="requirements dialogue session")
        return cls(
            session_id=record.get("session_id", ""),
            request_summary=record.get("request_summary", ""),
            skill_name=record.get("skill_name", ""),
            action=record.get("action", ""),
            capabilities=record.get("capabilities", ()),  # AUDIT-FIX(#5): Avoid tuple(...) coercion on deserialize for capability lists.
            feasibility=FeasibilityResult.from_payload(record.get("feasibility", {})),
            skill_id=record.get("skill_id", ""),
            status=record.get("status", RequirementsDialogueStatus.QUESTIONING.value),
            trigger_mode=record.get("trigger_mode", "push"),
            trigger_conditions=record.get("trigger_conditions", ()),  # AUDIT-FIX(#5): Preserve the original trigger_conditions shape for validation.
            scope=record.get("scope", {}),
            constraints=record.get("constraints", ()),  # AUDIT-FIX(#5): Keep raw constraint lists or strings so validation can make the right decision.
            current_question_id=record.get("current_question_id"),
            answered_question_ids=record.get("answered_question_ids", ()),  # AUDIT-FIX(#5): Prevent character-splitting on bad payloads.
            answer_summaries=record.get("answer_summaries", {}),
            created_at=record.get("created_at", _utc_now()),
            updated_at=record.get("updated_at", _utc_now()),
            version=record.get("version", 1),
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
        object.__setattr__(self, "spec_hash", _require_text(self.spec_hash, field_name="spec_hash", limit=128))
        object.__setattr__(
            self,
            "required_capabilities",
            _stable_identifier_tuple(self.required_capabilities, field_name="required_capabilities"),
        )
        object.__setattr__(self, "artifact_ids", _stable_identifier_tuple(self.artifact_ids, field_name="artifact_ids"))
        object.__setattr__(self, "created_at", _normalize_datetime(self.created_at, field_name="created_at"))
        object.__setattr__(self, "updated_at", _normalize_datetime(self.updated_at, field_name="updated_at"))
        _ensure_not_before(self.updated_at, self.created_at, later_name="updated_at", earlier_name="created_at")  # AUDIT-FIX(#2): Ensure compile job clocks move forward.
        object.__setattr__(self, "attempt_count", _require_non_negative_int(self.attempt_count, field_name="attempt_count"))  # AUDIT-FIX(#7): Reject coercion-prone counter payloads.
        object.__setattr__(self, "last_error", _optional_text(self.last_error, limit=240))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))  # AUDIT-FIX(#9): Freeze metadata to stop accidental post-validation mutation.

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
            "metadata": _payload_json_mapping(self.metadata, field_name="metadata"),  # AUDIT-FIX(#9): Return a detached metadata dict from to_payload().
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CompileJobRecord":
        record = _require_mapping(payload, context="compile job")
        _validate_schema(record, expected=cls.schema_name, context="compile job")
        return cls(
            job_id=record.get("job_id", ""),
            skill_id=record.get("skill_id", ""),
            skill_name=record.get("skill_name", ""),
            status=record.get("status", CompileJobStatus.DRAFT.value),
            requested_target=record.get("requested_target", CompileTarget.AUTOMATION_MANIFEST.value),
            spec_hash=record.get("spec_hash", ""),
            required_capabilities=record.get("required_capabilities", ()),  # AUDIT-FIX(#5): Preserve raw capability sequence shape.
            artifact_ids=record.get("artifact_ids", ()),  # AUDIT-FIX(#5): Preserve raw artifact sequence shape.
            created_at=record.get("created_at", _utc_now()),
            updated_at=record.get("updated_at", _utc_now()),
            attempt_count=record.get("attempt_count", 0),
            last_error=record.get("last_error"),
            metadata=record.get("metadata", {}),
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
        object.__setattr__(self, "content_path", _optional_safe_relative_path(self.content_path, field_name="content_path"))  # AUDIT-FIX(#1): Validate artifact paths at the contract boundary.
        object.__setattr__(self, "sha256", _optional_sha256(self.sha256, field_name="sha256"))  # AUDIT-FIX(#1): Store only valid SHA-256 digests.
        object.__setattr__(self, "size_bytes", _optional_non_negative_int(self.size_bytes, field_name="size_bytes"))  # AUDIT-FIX(#7): Enforce exact integer byte counts.
        object.__setattr__(self, "summary", _optional_text(self.summary, limit=220))
        object.__setattr__(self, "created_at", _normalize_datetime(self.created_at, field_name="created_at"))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))  # AUDIT-FIX(#9): Freeze artifact metadata after validation.

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
            "metadata": _payload_json_mapping(self.metadata, field_name="metadata"),  # AUDIT-FIX(#9): Detach artifact metadata from the live object.
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CompileArtifactRecord":
        record = _require_mapping(payload, context="compile artifact")
        _validate_schema(record, expected=cls.schema_name, context="compile artifact")
        return cls(
            artifact_id=record.get("artifact_id", ""),
            job_id=record.get("job_id", ""),
            kind=record.get("kind", ArtifactKind.LOG.value),
            media_type=record.get("media_type", "text/plain"),
            content_path=record.get("content_path"),
            sha256=record.get("sha256"),
            size_bytes=record.get("size_bytes"),
            summary=record.get("summary"),
            created_at=record.get("created_at", _utc_now()),
            metadata=record.get("metadata", {}),
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
        object.__setattr__(self, "driver_name", _optional_text(self.driver_name, limit=120))
        object.__setattr__(self, "driver_attempts", _text_tuple(self.driver_attempts, field_name="driver_attempts", limit=120))
        object.__setattr__(self, "event_count", _require_non_negative_int(self.event_count, field_name="event_count"))  # AUDIT-FIX(#7): Reject coercion-prone event counters.
        object.__setattr__(self, "last_event_kind", _optional_text(self.last_event_kind, limit=120))
        object.__setattr__(self, "last_event_message", _optional_text(self.last_event_message, limit=240))
        object.__setattr__(self, "thread_id", _optional_text(self.thread_id, limit=120))
        object.__setattr__(self, "turn_id", _optional_text(self.turn_id, limit=120))
        object.__setattr__(self, "final_message_seen", _require_bool(self.final_message_seen, field_name="final_message_seen"))  # AUDIT-FIX(#7): Keep compile state booleans strict.
        object.__setattr__(self, "turn_completed", _require_bool(self.turn_completed, field_name="turn_completed"))  # AUDIT-FIX(#7): Reject truthy strings and ints for completion flags.
        if self.started_at is not None:
            object.__setattr__(self, "started_at", _normalize_datetime(self.started_at, field_name="started_at"))
        object.__setattr__(self, "updated_at", _normalize_datetime(self.updated_at, field_name="updated_at"))
        if self.completed_at is not None:
            object.__setattr__(self, "completed_at", _normalize_datetime(self.completed_at, field_name="completed_at"))
        if self.started_at is not None:
            _ensure_not_before(self.updated_at, self.started_at, later_name="updated_at", earlier_name="started_at")  # AUDIT-FIX(#2): Runtime status updates cannot predate start time.
        if self.completed_at is not None and self.started_at is not None:
            _ensure_not_before(self.completed_at, self.started_at, later_name="completed_at", earlier_name="started_at")  # AUDIT-FIX(#2): Compile completion cannot predate compile start.
        if self.completed_at is not None:
            _ensure_not_before(self.updated_at, self.completed_at, later_name="updated_at", earlier_name="completed_at")  # AUDIT-FIX(#2): A completed record must be updated at or after completion time.
        object.__setattr__(self, "error_message", _optional_text(self.error_message, limit=240))
        object.__setattr__(self, "diagnostics", _json_mapping(self.diagnostics, field_name="diagnostics"))  # AUDIT-FIX(#9): Freeze runtime diagnostics after validation.

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
            "diagnostics": _payload_json_mapping(self.diagnostics, field_name="diagnostics"),  # AUDIT-FIX(#9): Return detached diagnostics payloads.
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CompileRunStatusRecord":
        record = _require_mapping(payload, context="compile run status")
        _validate_schema(record, expected=cls.schema_name, context="compile run status")
        return cls(
            job_id=record.get("job_id", ""),
            phase=record.get("phase", ""),
            driver_name=record.get("driver_name"),
            driver_attempts=record.get("driver_attempts", ()),  # AUDIT-FIX(#5): Preserve raw attempt sequence shape from storage/web payloads.
            event_count=record.get("event_count", 0),
            last_event_kind=record.get("last_event_kind"),
            last_event_message=record.get("last_event_message"),
            thread_id=record.get("thread_id"),
            turn_id=record.get("turn_id"),
            final_message_seen=record.get("final_message_seen", False),  # AUDIT-FIX(#7): Preserve raw boolean payloads for strict validation.
            turn_completed=record.get("turn_completed", False),  # AUDIT-FIX(#7): Do not bool(...) coerce incoming state flags.
            started_at=record.get("started_at"),
            updated_at=record.get("updated_at", _utc_now()),
            completed_at=record.get("completed_at"),
            error_message=record.get("error_message"),
            diagnostics=record.get("diagnostics", {}),
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
        object.__setattr__(self, "version", _require_positive_int(self.version, field_name="version"))  # AUDIT-FIX(#7): Enforce exact integer versions for activation records.
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
        if self.activated_at is not None:
            _ensure_not_before(self.updated_at, self.activated_at, later_name="updated_at", earlier_name="activated_at")  # AUDIT-FIX(#2): Activation snapshots cannot predate the activation moment.
        if self.feedback_due_at is not None and self.activated_at is not None:
            _ensure_not_before(self.feedback_due_at, self.activated_at, later_name="feedback_due_at", earlier_name="activated_at")  # AUDIT-FIX(#2): Feedback due dates cannot predate activation.
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))  # AUDIT-FIX(#9): Freeze activation metadata after validation.

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
            "metadata": _payload_json_mapping(self.metadata, field_name="metadata"),  # AUDIT-FIX(#9): Return detached activation metadata.
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ActivationRecord":
        record = _require_mapping(payload, context="activation record")
        _validate_schema(record, expected=cls.schema_name, context="activation record")
        return cls(
            skill_id=record.get("skill_id", ""),
            skill_name=record.get("skill_name", ""),
            version=record.get("version", 1),
            status=record.get("status", LearnedSkillStatus.DRAFT.value),
            job_id=record.get("job_id", ""),
            artifact_id=record.get("artifact_id", ""),
            updated_at=record.get("updated_at", _utc_now()),
            activated_at=record.get("activated_at"),
            feedback_due_at=record.get("feedback_due_at"),
            metadata=record.get("metadata", {}),
        )