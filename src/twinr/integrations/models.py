# CHANGELOG: 2026-03-30
# BUG-1: IntegrationRequest assignment is now validated continuously and payload containers are deep-frozen,
#        preventing post-construction mutation from bypassing invariants and causing late crashes/log injection.
# BUG-2: JSON normalization now enforces hard size/node/int/key limits, preventing practical CPU/RAM DoS on Raspberry Pi 4
#        before per-operation max_payload_bytes checks run.
# BUG-3: SecretReference now validates references per storage backend and rejects invalid env-var names and unsafe file paths early.
# BUG-4: Manifest operation lookups are now case-insensitive and duplicate IDs that differ only by case are rejected.
# SEC-1: redacted_parameters() now produces audit-safe output by escaping control characters and truncating oversized strings.
# SEC-2: Added native support for systemd credentials, a safer Linux secret-delivery path than environment variables/files.
# IMP-1: Added optional orjson-backed canonical JSON serialization for fast, deterministic payload sizing on Raspberry Pi ARM.
# IMP-2: Added manifest.evaluate_request() so policy/runtime/registry/provider packages share one enforcement path.
# BREAKING: redacted_parameters() now sanitizes audit text instead of returning original non-sensitive strings verbatim.
# BREAKING: Integers outside the IEEE-754 safe JSON range are rejected to avoid silent corruption in dashboards/web clients.

"""Define the canonical data contracts for Twinr integrations.

These models normalize identifiers, payloads, safety metadata, redaction rules,
and request-policy matching so policy, runtime, registry, and provider packages
share one contract.

The module intentionally keeps its hard runtime dependencies at zero. It adopts
frontier 2026 patterns for constrained ARM devices by combining:

* deep-immutable normalized payloads,
* strict bounded normalization for hostile/untrusted inputs,
* deterministic canonical JSON encoding with optional ``orjson`` acceleration,
* storage-specific secret-reference validation, and
* first-class manifest/request evaluation.

Optional dependency:
    - ``orjson``: faster canonical JSON serialization on Raspberry Pi ARM.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
import json
import math
from pathlib import PurePath
import re
from typing import TypeVar
from uuid import UUID

try:  # Optional acceleration path for ARM deployments.
    import orjson as _orjson
except ImportError:  # pragma: no cover - optional dependency
    _orjson = None


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_SENSITIVE_KEY_NORMALIZATION_RE = re.compile(r"[^a-z0-9]+")

# Bounded-normalization limits chosen for real Raspberry Pi 4 deployments:
# permissive enough for ordinary requests/results, but tight enough to stop
# accidental or malicious memory/CPU blow-ups in a single process.
_MAX_NESTING_DEPTH = 32
_MAX_TOTAL_NODES = 16_384
_MAX_CONTAINER_ITEMS = 4_096
_MAX_STRING_BYTES = 262_144
_MAX_AUDIT_STRING_BYTES = 8_192
_MAX_KEY_BYTES = 256
_MAX_ESTIMATED_JSON_BYTES = 1_048_576
_MAX_JSON_SAFE_INTEGER = 9_007_199_254_740_991  # 2**53 - 1

_REDACTED_VALUE = "<redacted>"
_TRUNCATED_VALUE_SUFFIX = "…<truncated>"

_StrEnumT = TypeVar("_StrEnumT", bound=StrEnum)


class FrozenJsonDict(dict[str, object]):
    """An immutable ``dict`` subclass for normalized JSON payloads."""

    __slots__ = ()

    def __init__(
        self,
        items: Mapping[str, object] | Iterable[tuple[str, object]] | None = None,
    ) -> None:
        super().__init__()
        if items is None:
            return
        source = items.items() if isinstance(items, Mapping) else items
        for key, value in source:
            dict.__setitem__(self, key, value)

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("Normalized payload mappings are immutable.")

    __setitem__ = _immutable  # type: ignore[assignment]
    __delitem__ = _immutable  # type: ignore[assignment]
    clear = _immutable  # type: ignore[assignment]
    pop = _immutable  # type: ignore[assignment]
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable  # type: ignore[assignment]
    update = _immutable  # type: ignore[assignment]

    def __ior__(self, _other: object) -> "FrozenJsonDict":
        self._immutable()

    def __copy__(self) -> "FrozenJsonDict":
        """Return a shallow copy compatible with the stdlib copy protocol."""

        return type(self)(dict.items(self))

    def __deepcopy__(self, memo: dict[int, object]) -> "FrozenJsonDict":
        """Return a deepcopy without routing through mutable dict reconstruction."""

        clone = type(self)((key, deepcopy(value, memo)) for key, value in dict.items(self))
        memo[id(self)] = clone
        return clone


class FrozenJsonList(list[object]):
    """An immutable ``list`` subclass for normalized JSON payload arrays."""

    __slots__ = ()

    def __init__(self, items: Iterable[object] | None = None) -> None:
        super().__init__(items or ())

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("Normalized payload arrays are immutable.")

    __setitem__ = _immutable  # type: ignore[assignment]
    __delitem__ = _immutable  # type: ignore[assignment]
    append = _immutable  # type: ignore[assignment]
    clear = _immutable  # type: ignore[assignment]
    extend = _immutable  # type: ignore[assignment]
    insert = _immutable  # type: ignore[assignment]
    pop = _immutable  # type: ignore[assignment]
    remove = _immutable  # type: ignore[assignment]
    reverse = _immutable  # type: ignore[assignment]
    sort = _immutable  # type: ignore[assignment]

    def __iadd__(self, _other: object) -> "FrozenJsonList":
        self._immutable()

    def __imul__(self, _other: object) -> "FrozenJsonList":
        self._immutable()

    def __copy__(self) -> "FrozenJsonList":
        """Return a shallow copy compatible with the stdlib copy protocol."""

        return type(self)(self)

    def __deepcopy__(self, memo: dict[int, object]) -> "FrozenJsonList":
        """Return a deepcopy without routing through mutable list reconstruction."""

        clone = type(self)(deepcopy(item, memo) for item in self)
        memo[id(self)] = clone
        return clone


@dataclass(slots=True)
class _NormalizationState:
    """Track bounded JSON normalization to fail fast on hostile inputs."""

    nodes: int = 0
    estimated_json_bytes: int = 0

    def note_node(self, path: str) -> None:
        self.nodes += 1
        if self.nodes > _MAX_TOTAL_NODES:
            raise ValueError(
                f"{path} exceeds the maximum supported JSON node count of {_MAX_TOTAL_NODES}."
            )

    def add_bytes(self, path: str, amount: int) -> None:
        self.estimated_json_bytes += amount
        if self.estimated_json_bytes > _MAX_ESTIMATED_JSON_BYTES:
            raise ValueError(
                f"{path} exceeds the maximum estimated JSON size of {_MAX_ESTIMATED_JSON_BYTES} bytes."
            )


def _normalize_sensitive_key(key: str) -> str:
    """Normalize a key for case- and punctuation-insensitive matching."""

    return _SENSITIVE_KEY_NORMALIZATION_RE.sub("", key.casefold())


def _ensure_safe_text(value: object, field_name: str) -> str:
    """Validate non-empty text without control characters."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if _CONTROL_CHAR_RE.search(normalized):
        raise ValueError(f"{field_name} must not contain control characters.")
    return normalized


def _ensure_identifier(value: object, field_name: str) -> str:
    """Validate an audit-safe identifier string."""

    normalized = _ensure_safe_text(value, field_name)
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must match {_IDENTIFIER_RE.pattern!r} to stay audit-safe."
        )
    return normalized


def _normalize_identifier_key(value: str) -> str:
    """Return the canonical lookup key for identifiers."""

    return value.casefold()


def _ensure_bool(value: object, field_name: str) -> bool:
    """Require a real boolean instead of a truthy proxy value."""

    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool.")
    return value


def _ensure_positive_int(value: object, field_name: str) -> int:
    """Require an integer greater than zero."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _ensure_json_safe_int(value: object, field_name: str) -> int:
    """Require an integer that round-trips safely through common JSON clients."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if abs(value) > _MAX_JSON_SAFE_INTEGER:
        raise ValueError(
            f"{field_name} exceeds the safe JSON integer range of ±{_MAX_JSON_SAFE_INTEGER}."
        )
    return value


def _coerce_str_enum(value: object, enum_type: type[_StrEnumT], field_name: str) -> _StrEnumT:
    """Coerce a raw value into a ``StrEnum`` member."""

    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        for member in enum_type:
            if member.value.casefold() == normalized.casefold():
                return member
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}.")
    raise TypeError(f"{field_name} must be a {enum_type.__name__} or matching string value.")


def _coerce_tuple(value: object, field_name: str) -> tuple[object, ...]:
    """Freeze tuple-like input into an immutable tuple."""

    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError(f"{field_name} must be a tuple, list, or None.")


def _normalize_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    """Normalize an iterable of user-visible text into a tuple."""

    raw_items = _coerce_tuple(value, field_name)
    return tuple(
        _ensure_safe_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(raw_items)
    )


def _estimate_string_json_bytes(value: str) -> int:
    """Return a conservative UTF-8 byte estimate for a JSON string."""

    return len(value.encode("utf-8")) + 2


def _validate_key(key: str, *, path: str, state: _NormalizationState) -> str:
    """Validate one JSON mapping key."""

    if _CONTROL_CHAR_RE.search(key):
        raise ValueError(f"{path} contains control characters in key {key!r}.")
    key_bytes = len(key.encode("utf-8"))
    if key_bytes > _MAX_KEY_BYTES:
        raise ValueError(f"{path} contains a key longer than {_MAX_KEY_BYTES} bytes: {key!r}.")
    state.add_bytes(path, key_bytes + 3)  # quotes + colon
    return key


def _ensure_secret_reference(reference: object, storage: "SecretStorage") -> str:
    """Validate secret references using storage-specific rules."""

    normalized = _ensure_safe_text(reference, "reference")
    if storage is SecretStorage.ENV_VAR:
        if not _ENV_VAR_RE.fullmatch(normalized):
            raise ValueError(
                "reference must be a valid environment variable name for ENV_VAR storage."
            )
        return normalized

    if storage is SecretStorage.SYSTEMD_CREDENTIAL:
        return _ensure_identifier(normalized, "reference")

    if storage is SecretStorage.FILE:
        if normalized.startswith("~"):
            raise ValueError("reference must not use shell-style home expansion.")
        if "$" in normalized:
            raise ValueError("reference must not use shell-style environment expansion.")
        path = PurePath(normalized)
        if any(part == ".." for part in path.parts):
            raise ValueError("reference must not contain parent-directory traversal.")
        if path.name in {"", ".", ".."}:
            raise ValueError("reference must point to a concrete file path.")
        return normalized

    # KEYRING and VAULT backends are intentionally text-only here; backend
    # packages can layer additional semantic checks on top.
    return normalized


def _escape_control_char(match: re.Match[str]) -> str:
    """Escape one control character for audit-safe rendering."""

    return f"\\x{ord(match.group(0)):02x}"


def _truncate_utf8(value: str, max_bytes: int) -> str:
    """Truncate a string to ``max_bytes`` UTF-8 bytes without breaking encoding."""

    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    suffix = _TRUNCATED_VALUE_SUFFIX
    suffix_bytes = suffix.encode("utf-8")
    budget = max_bytes - len(suffix_bytes)
    if budget <= 0:
        return suffix
    truncated = encoded[:budget].decode("utf-8", errors="ignore")
    return f"{truncated}{suffix}"


def _sanitize_audit_value(value: object) -> object:
    """Make JSON-like values safe for logs and audit trails."""

    if isinstance(value, str):
        escaped = _CONTROL_CHAR_RE.sub(_escape_control_char, value)
        return _truncate_utf8(escaped, _MAX_AUDIT_STRING_BYTES)

    if isinstance(value, Mapping):
        return {key: _sanitize_audit_value(nested_value) for key, nested_value in value.items()}

    if isinstance(value, list):
        return [_sanitize_audit_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_sanitize_audit_value(item) for item in value)

    return value


def _normalize_json_mapping(value: object, *, field_name: str) -> FrozenJsonDict:
    """Normalize a top-level mapping into a JSON-safe immutable mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping with string keys.")
    normalized = _normalize_json_value(
        value,
        path=field_name,
        depth=0,
        seen=set(),
        state=_NormalizationState(),
    )
    if not isinstance(normalized, FrozenJsonDict):  # pragma: no cover - defensive
        raise TypeError(f"{field_name} must normalize to a mapping.")
    return normalized


def _normalize_json_value(
    value: object,
    *,
    path: str,
    depth: int,
    seen: set[int],
    state: _NormalizationState,
) -> object:
    """Normalize nested payload data into deterministic JSON-safe values."""

    state.note_node(path)
    if depth > _MAX_NESTING_DEPTH:
        raise ValueError(f"{path} exceeds the maximum supported nesting depth of {_MAX_NESTING_DEPTH}.")

    if value is None:
        state.add_bytes(path, 4)
        return None

    if isinstance(value, str):
        value_bytes = len(value.encode("utf-8"))
        if value_bytes > _MAX_STRING_BYTES:
            raise ValueError(f"{path} exceeds the maximum string size of {_MAX_STRING_BYTES} bytes.")
        state.add_bytes(path, _estimate_string_json_bytes(value))
        return value

    if isinstance(value, bool):
        state.add_bytes(path, 4 if value else 5)
        return value

    if isinstance(value, int):
        safe_int = _ensure_json_safe_int(value, path)
        state.add_bytes(path, len(str(safe_int)))
        return safe_int

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float value.")
        state.add_bytes(path, len(repr(value)))
        return value

    if isinstance(value, StrEnum):
        state.add_bytes(path, _estimate_string_json_bytes(value.value))
        return value.value

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{path} contains a non-finite Decimal value.")
        serialized = str(value)
        state.add_bytes(path, _estimate_string_json_bytes(serialized))
        return serialized

    if isinstance(value, UUID):
        serialized = str(value)
        state.add_bytes(path, _estimate_string_json_bytes(serialized))
        return serialized

    if isinstance(value, PurePath):
        serialized = str(value)
        state.add_bytes(path, _estimate_string_json_bytes(serialized))
        return serialized

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{path} contains a timezone-naive datetime.")
        serialized = value.isoformat()
        state.add_bytes(path, _estimate_string_json_bytes(serialized))
        return serialized

    if isinstance(value, date):
        serialized = value.isoformat()
        state.add_bytes(path, _estimate_string_json_bytes(serialized))
        return serialized

    if isinstance(value, time):
        if value.tzinfo is not None and value.utcoffset() is None:
            raise ValueError(f"{path} contains a time with an invalid timezone offset.")
        serialized = value.isoformat()
        state.add_bytes(path, _estimate_string_json_bytes(serialized))
        return serialized

    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{path} contains binary data; pass a file reference instead.")

    if isinstance(value, Mapping):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError(f"{path} exceeds the maximum container size of {_MAX_CONTAINER_ITEMS} items.")
        container_id = id(value)
        if container_id in seen:
            raise ValueError(f"{path} contains a circular reference.")
        seen.add(container_id)
        try:
            normalized_items: dict[str, object] = {}
            state.add_bytes(path, 2)  # {}
            for raw_key, raw_value in value.items():
                if not isinstance(raw_key, str):
                    raise TypeError(f"{path} contains a non-string key: {raw_key!r}.")
                key = _validate_key(raw_key, path=path, state=state)
                normalized_items[key] = _normalize_json_value(
                    raw_value,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                    seen=seen,
                    state=state,
                )
            return FrozenJsonDict(normalized_items)
        finally:
            seen.remove(container_id)

    if isinstance(value, list):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError(f"{path} exceeds the maximum container size of {_MAX_CONTAINER_ITEMS} items.")
        container_id = id(value)
        if container_id in seen:
            raise ValueError(f"{path} contains a circular reference.")
        seen.add(container_id)
        try:
            state.add_bytes(path, 2)  # []
            return FrozenJsonList(
                [
                    _normalize_json_value(item, path=f"{path}[{index}]", depth=depth + 1, seen=seen, state=state)
                    for index, item in enumerate(value)
                ]
            )
        finally:
            seen.remove(container_id)

    if isinstance(value, tuple):
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError(f"{path} exceeds the maximum container size of {_MAX_CONTAINER_ITEMS} items.")
        container_id = id(value)
        if container_id in seen:
            raise ValueError(f"{path} contains a circular reference.")
        seen.add(container_id)
        try:
            state.add_bytes(path, 2)  # tuple becomes JSON array
            return tuple(
                _normalize_json_value(item, path=f"{path}[{index}]", depth=depth + 1, seen=seen, state=state)
                for index, item in enumerate(value)
            )
        finally:
            seen.remove(container_id)

    if isinstance(value, (set, frozenset)):
        raise TypeError(f"{path} contains an unordered set; use a list or tuple for deterministic payloads.")

    raise TypeError(f"{path} contains an unsupported value of type {type(value).__name__}.")


def _redact_value(value: object, *, sensitive_keys: frozenset[str]) -> object:
    """Redact sensitive keys recursively inside JSON-like values."""

    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for key, nested_value in value.items():
            normalized_key = _normalize_sensitive_key(key)
            if normalized_key in sensitive_keys:
                redacted[key] = _REDACTED_VALUE
            else:
                redacted[key] = _redact_value(nested_value, sensitive_keys=sensitive_keys)
        return redacted

    if isinstance(value, list):
        return [_redact_value(item, sensitive_keys=sensitive_keys) for item in value]

    if isinstance(value, tuple):
        return tuple(_redact_value(item, sensitive_keys=sensitive_keys) for item in value)

    return value


def _to_json_primitive(value: object) -> object:
    """Convert contract objects into plain JSON-safe Python primitives."""

    if isinstance(value, StrEnum):
        return value.value

    if is_dataclass(value):
        return {
            field_info.name: _to_json_primitive(getattr(value, field_info.name))
            for field_info in fields(value)
            if not field_info.name.startswith("_")
        }

    if isinstance(value, Mapping):
        return {str(key): _to_json_primitive(nested_value) for key, nested_value in value.items()}

    if isinstance(value, list):
        return [_to_json_primitive(item) for item in value]

    if isinstance(value, tuple):
        return [_to_json_primitive(item) for item in value]

    return value


def _canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-safe value into canonical UTF-8 bytes."""

    if _orjson is not None:
        return _orjson.dumps(value, option=_orjson.OPT_SORT_KEYS)
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


class JsonContractMixin:
    """Provide deterministic JSON serialization for contract objects."""

    def as_json_dict(self) -> dict[str, object]:
        """Return a plain JSON-safe mapping for this contract object."""

        primitive = _to_json_primitive(self)
        if not isinstance(primitive, dict):  # pragma: no cover - defensive
            raise TypeError(f"{type(self).__name__} must serialize to a JSON object.")
        return primitive

    def to_json_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON bytes for this contract object."""

        return _canonical_json_bytes(self.as_json_dict())

    def to_json(self) -> str:
        """Return canonical JSON text for this contract object."""

        return self.to_json_bytes().decode("utf-8")


class IntegrationDomain(StrEnum):
    """Enumerate the high-level domains Twinr integrations can belong to."""

    CALENDAR = "calendar"
    EMAIL = "email"
    MESSENGER = "messenger"
    SMART_HOME = "smart_home"
    SECURITY = "security"
    HEALTH = "health"


class IntegrationAction(StrEnum):
    """Enumerate the actions an integration operation can perform."""

    READ = "read"
    WRITE = "write"
    SEND = "send"
    QUERY = "query"
    CONTROL = "control"
    ALERT = "alert"


class RequestOrigin(StrEnum):
    """Enumerate the trusted surfaces that can issue integration requests."""

    LOCAL_DEVICE = "local_device"
    LOCAL_DASHBOARD = "local_dashboard"
    REMOTE_SERVICE = "remote_service"


class RiskLevel(StrEnum):
    """Describe the risk tier of an integration operation."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ConfirmationMode(StrEnum):
    """Describe which human confirmation level an operation requires."""

    NONE = "none"
    USER = "user"
    CAREGIVER = "caregiver"


class DataSensitivity(StrEnum):
    """Describe the sensitivity of data touched by an integration."""

    NORMAL = "normal"
    PERSONAL = "personal"
    SECURITY = "security"
    HEALTH = "health"


class SecretStorage(StrEnum):
    """Enumerate supported secret-storage backends referenced by manifests."""

    ENV_VAR = "env_var"
    FILE = "file"
    KEYRING = "keyring"
    VAULT = "vault"
    SYSTEMD_CREDENTIAL = "systemd_credential"


DEFAULT_SENSITIVE_PARAMETER_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "attachment",
        "authorization",
        "body",
        "client_secret",
        "cookie",
        "diagnosis",
        "id_token",
        "medical_notes",
        "message",
        "notes",
        "otp",
        "passcode",
        "password",
        "pin",
        "private_key",
        "refresh_token",
        "secret",
        "session_id",
        "token",
        "verification_code",
    }
)
_NORMALIZED_DEFAULT_SENSITIVE_PARAMETER_KEYS = frozenset(
    _normalize_sensitive_key(key) for key in DEFAULT_SENSITIVE_PARAMETER_KEYS
)


@dataclass(frozen=True, slots=True)
class SecretReference(JsonContractMixin):
    """Describe one secret dependency required by an integration."""

    name: str
    reference: str
    storage: SecretStorage = SecretStorage.ENV_VAR
    required: bool = True

    def __post_init__(self) -> None:
        """Normalize and validate secret-reference fields at construction time."""

        object.__setattr__(self, "name", _ensure_identifier(self.name, "name"))
        storage = _coerce_str_enum(self.storage, SecretStorage, "storage")
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "reference", _ensure_secret_reference(self.reference, storage))
        object.__setattr__(self, "required", _ensure_bool(self.required, "required"))


@dataclass(frozen=True, slots=True)
class SafetyProfile(JsonContractMixin):
    """Describe the safety requirements for one integration operation."""

    risk: RiskLevel
    confirmation: ConfirmationMode = ConfirmationMode.NONE
    sensitivity: DataSensitivity = DataSensitivity.NORMAL
    allow_background_polling: bool = False
    allow_remote_trigger: bool = False
    allow_free_text: bool = False
    max_payload_bytes: int = 4096

    def __post_init__(self) -> None:
        """Normalize safety fields and reject unsafe high-risk defaults."""

        object.__setattr__(self, "risk", _coerce_str_enum(self.risk, RiskLevel, "risk"))
        object.__setattr__(
            self,
            "confirmation",
            _coerce_str_enum(self.confirmation, ConfirmationMode, "confirmation"),
        )
        object.__setattr__(
            self,
            "sensitivity",
            _coerce_str_enum(self.sensitivity, DataSensitivity, "sensitivity"),
        )
        object.__setattr__(
            self,
            "allow_background_polling",
            _ensure_bool(self.allow_background_polling, "allow_background_polling"),
        )
        object.__setattr__(
            self,
            "allow_remote_trigger",
            _ensure_bool(self.allow_remote_trigger, "allow_remote_trigger"),
        )
        object.__setattr__(self, "allow_free_text", _ensure_bool(self.allow_free_text, "allow_free_text"))
        max_payload_bytes = _ensure_positive_int(self.max_payload_bytes, "max_payload_bytes")
        if max_payload_bytes > _MAX_ESTIMATED_JSON_BYTES:
            raise ValueError(
                f"max_payload_bytes must not exceed the hard normalization cap of {_MAX_ESTIMATED_JSON_BYTES}."
            )
        object.__setattr__(self, "max_payload_bytes", max_payload_bytes)

        if self.risk in {RiskLevel.HIGH, RiskLevel.CRITICAL} and self.confirmation is ConfirmationMode.NONE:
            raise ValueError(
                "High-risk and critical integrations must require user or caregiver confirmation."
            )


@dataclass(frozen=True, slots=True)
class IntegrationOperation(JsonContractMixin):
    """Describe one operation exposed by an integration manifest."""

    operation_id: str
    label: str
    action: IntegrationAction
    summary: str
    safety: SafetyProfile

    def __post_init__(self) -> None:
        """Validate operation identifiers, text, and safety metadata."""

        object.__setattr__(self, "operation_id", _ensure_identifier(self.operation_id, "operation_id"))
        object.__setattr__(self, "label", _ensure_safe_text(self.label, "label"))
        object.__setattr__(self, "action", _coerce_str_enum(self.action, IntegrationAction, "action"))
        object.__setattr__(self, "summary", _ensure_safe_text(self.summary, "summary"))
        if not isinstance(self.safety, SafetyProfile):
            raise TypeError("safety must be a SafetyProfile instance.")


@dataclass(frozen=True, slots=True)
class IntegrationManifest(JsonContractMixin):
    """Describe one integration and the operations it exposes."""

    integration_id: str
    domain: IntegrationDomain
    title: str
    summary: str
    operations: tuple[IntegrationOperation, ...]
    required_secrets: tuple[SecretReference, ...] = ()
    notes: tuple[str, ...] = ()
    _operation_index: FrozenJsonDict = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Freeze and validate manifest metadata plus nested operations."""

        object.__setattr__(self, "integration_id", _ensure_identifier(self.integration_id, "integration_id"))
        object.__setattr__(self, "domain", _coerce_str_enum(self.domain, IntegrationDomain, "domain"))
        object.__setattr__(self, "title", _ensure_safe_text(self.title, "title"))
        object.__setattr__(self, "summary", _ensure_safe_text(self.summary, "summary"))

        operations = _coerce_tuple(self.operations, "operations")
        if not operations:
            raise ValueError("Integration manifests must define at least one operation.")
        normalized_operations: list[IntegrationOperation] = []
        operation_lookup: dict[str, IntegrationOperation] = {}
        for index, operation in enumerate(operations):
            if not isinstance(operation, IntegrationOperation):
                raise TypeError(f"operations[{index}] must be an IntegrationOperation.")
            normalized_operations.append(operation)
            key = _normalize_identifier_key(operation.operation_id)
            if key in operation_lookup:
                raise ValueError(
                    f"Duplicate operations are not allowed for {self.integration_id}, including case-only duplicates."
                )
            operation_lookup[key] = operation
        object.__setattr__(self, "operations", tuple(normalized_operations))
        object.__setattr__(self, "_operation_index", FrozenJsonDict(operation_lookup))

        required_secrets = _coerce_tuple(self.required_secrets, "required_secrets")
        normalized_secrets: list[SecretReference] = []
        secret_names: list[str] = []
        for index, secret in enumerate(required_secrets):
            if not isinstance(secret, SecretReference):
                raise TypeError(f"required_secrets[{index}] must be a SecretReference.")
            normalized_secrets.append(secret)
            secret_names.append(secret.name.casefold())
        if len(secret_names) != len(set(secret_names)):
            raise ValueError(f"Duplicate secret names are not allowed for {self.integration_id}.")
        object.__setattr__(self, "required_secrets", tuple(normalized_secrets))

        object.__setattr__(self, "notes", _normalize_text_tuple(self.notes, "notes"))

    def operation(self, operation_id: str) -> IntegrationOperation | None:
        """Return one operation by ID when the manifest defines it."""

        if not isinstance(operation_id, str):
            return None
        normalized_operation_id = operation_id.strip()
        if not normalized_operation_id or not _IDENTIFIER_RE.fullmatch(normalized_operation_id):
            return None
        operation = self._operation_index.get(_normalize_identifier_key(normalized_operation_id))
        return operation if isinstance(operation, IntegrationOperation) else None

    def evaluate_request(self, request: "IntegrationRequest") -> "IntegrationDecision":
        """Evaluate a request against this manifest's normalized safety metadata."""

        if not isinstance(request, IntegrationRequest):
            raise TypeError("request must be an IntegrationRequest instance.")

        if _normalize_identifier_key(request.integration_id) != _normalize_identifier_key(self.integration_id):
            return IntegrationDecision.deny(
                f"Request targets integration {request.integration_id!r}, expected {self.integration_id!r}.",
            )

        operation = self.operation(request.operation_id)
        if operation is None:
            return IntegrationDecision.deny(
                f"Operation {request.operation_id!r} is not defined for integration {self.integration_id!r}.",
            )

        payload_size = request.payload_size_bytes()
        if payload_size > operation.safety.max_payload_bytes:
            return IntegrationDecision.deny(
                f"Payload size {payload_size} exceeds the configured limit of "
                f"{operation.safety.max_payload_bytes} bytes for {operation.operation_id!r}.",
            )

        if request.background_trigger and not operation.safety.allow_background_polling:
            return IntegrationDecision.deny(
                f"Operation {operation.operation_id!r} does not allow background triggers.",
            )

        if request.origin is RequestOrigin.REMOTE_SERVICE and not operation.safety.allow_remote_trigger:
            return IntegrationDecision.deny(
                f"Operation {operation.operation_id!r} does not allow remote triggers.",
            )

        if operation.safety.confirmation is ConfirmationMode.USER and not request.explicit_user_confirmation:
            return IntegrationDecision.deny(
                f"Operation {operation.operation_id!r} requires explicit user confirmation.",
                required_confirmation=ConfirmationMode.USER,
            )

        if (
            operation.safety.confirmation is ConfirmationMode.CAREGIVER
            and not request.explicit_caregiver_confirmation
        ):
            return IntegrationDecision.deny(
                f"Operation {operation.operation_id!r} requires explicit caregiver confirmation.",
                required_confirmation=ConfirmationMode.CAREGIVER,
            )

        warnings: list[str] = []
        if request.dry_run:
            warnings.append("dry_run requested; adapters should not perform side effects.")
        if operation.safety.allow_free_text:
            warnings.append("allow_free_text is enabled; adapter-specific validation should still apply.")

        return IntegrationDecision.allow(
            f"Request is compatible with manifest {self.integration_id!r} and operation {operation.operation_id!r}.",
            warnings=tuple(warnings),
        )


@dataclass(slots=True)
class IntegrationRequest(JsonContractMixin):
    """Represent one normalized integration request."""

    integration_id: str
    operation_id: str
    parameters: dict[str, object] = field(default_factory=dict)
    origin: RequestOrigin = RequestOrigin.LOCAL_DEVICE
    explicit_user_confirmation: bool = False
    explicit_caregiver_confirmation: bool = False
    dry_run: bool = False
    background_trigger: bool = False

    def __post_init__(self) -> None:
        """Normalize request identifiers, payload, and confirmation flags."""

        self.integration_id = _ensure_identifier(self.integration_id, "integration_id")
        self.operation_id = _ensure_identifier(self.operation_id, "operation_id")
        self.parameters = _normalize_json_mapping(self.parameters, field_name="parameters")
        self.origin = _coerce_str_enum(self.origin, RequestOrigin, "origin")
        self.explicit_user_confirmation = _ensure_bool(
            self.explicit_user_confirmation,
            "explicit_user_confirmation",
        )
        self.explicit_caregiver_confirmation = _ensure_bool(
            self.explicit_caregiver_confirmation,
            "explicit_caregiver_confirmation",
        )
        self.dry_run = _ensure_bool(self.dry_run, "dry_run")
        self.background_trigger = _ensure_bool(self.background_trigger, "background_trigger")

    def __setattr__(self, name: str, value: object) -> None:
        """Validate assignment so requests stay normalized after construction."""

        if name == "integration_id":
            object.__setattr__(self, name, _ensure_identifier(value, name))
            return
        if name == "operation_id":
            object.__setattr__(self, name, _ensure_identifier(value, name))
            return
        if name == "parameters":
            object.__setattr__(self, name, _normalize_json_mapping(value, field_name=name))
            return
        if name == "origin":
            object.__setattr__(self, name, _coerce_str_enum(value, RequestOrigin, name))
            return
        if name in {
            "explicit_user_confirmation",
            "explicit_caregiver_confirmation",
            "dry_run",
            "background_trigger",
        }:
            object.__setattr__(self, name, _ensure_bool(value, name))
            return
        object.__setattr__(self, name, value)

    @classmethod
    def from_json(cls, json_data: str | bytes | bytearray) -> "IntegrationRequest":
        """Build a request from a JSON object."""

        raw = _orjson.loads(json_data) if _orjson is not None else json.loads(json_data)
        if not isinstance(raw, Mapping):
            raise TypeError("IntegrationRequest JSON must decode to an object.")
        return cls(**dict(raw))

    def clone(self, **changes: object) -> "IntegrationRequest":
        """Return a validated copy with selected fields replaced."""

        return replace(self, **changes)

    def payload_json_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON bytes for the normalized parameter payload."""

        return _canonical_json_bytes(_to_json_primitive(self.parameters))

    def payload_json(self) -> str:
        """Return canonical JSON text for the normalized parameter payload."""

        return self.payload_json_bytes().decode("utf-8")

    def payload_size_bytes(self) -> int:
        """Return the UTF-8 size of the normalized parameter payload."""

        return len(self.payload_json_bytes())

    def redacted_parameters(self, *, extra_sensitive_keys: set[str] | None = None) -> dict[str, object]:
        """Return parameters with sensitive keys recursively redacted and audit-sanitized."""

        sensitive_keys = set(_NORMALIZED_DEFAULT_SENSITIVE_PARAMETER_KEYS)
        if extra_sensitive_keys:
            sensitive_keys.update(
                _normalize_sensitive_key(_ensure_safe_text(key, "extra_sensitive_keys"))
                for key in extra_sensitive_keys
            )

        redacted = _redact_value(self.parameters, sensitive_keys=frozenset(sensitive_keys))
        sanitized = _sanitize_audit_value(redacted)
        if not isinstance(sanitized, dict):  # pragma: no cover - defensive
            raise TypeError("parameters must redact to a dictionary.")
        return sanitized

    def audit_label(self) -> str:
        """Return a log-safe label for the request."""

        return f"{self.integration_id}:{self.operation_id}:{self.origin.value}"


@dataclass(frozen=True, slots=True)
class IntegrationDecision(JsonContractMixin):
    """Represent the policy outcome for one integration request."""

    allowed: bool
    reason: str
    required_confirmation: ConfirmationMode | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the decision payload and reject contradictory states."""

        object.__setattr__(self, "allowed", _ensure_bool(self.allowed, "allowed"))
        object.__setattr__(self, "reason", _ensure_safe_text(self.reason, "reason"))
        if self.required_confirmation is None:
            normalized_confirmation = None
        else:
            normalized_confirmation = _coerce_str_enum(
                self.required_confirmation,
                ConfirmationMode,
                "required_confirmation",
            )
            if normalized_confirmation is ConfirmationMode.NONE:
                normalized_confirmation = None
        object.__setattr__(self, "required_confirmation", normalized_confirmation)
        object.__setattr__(self, "warnings", _normalize_text_tuple(self.warnings, "warnings"))

        if self.allowed and self.required_confirmation is not None:
            raise ValueError("An allowed decision cannot also require confirmation.")

    @classmethod
    def allow(cls, reason: str, *, warnings: tuple[str, ...] = ()) -> "IntegrationDecision":
        """Build an allow decision."""

        return cls(allowed=True, reason=reason, warnings=warnings)

    @classmethod
    def deny(
        cls,
        reason: str,
        *,
        required_confirmation: ConfirmationMode | None = None,
        warnings: tuple[str, ...] = (),
    ) -> "IntegrationDecision":
        """Build a deny decision."""

        return cls(
            allowed=False,
            reason=reason,
            required_confirmation=required_confirmation,
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class IntegrationResult(JsonContractMixin):
    """Represent the normalized result returned by an integration adapter."""

    ok: bool
    summary: str
    details: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    redacted_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the adapter result into immutable safe structures."""

        object.__setattr__(self, "ok", _ensure_bool(self.ok, "ok"))
        object.__setattr__(self, "summary", _ensure_safe_text(self.summary, "summary"))
        object.__setattr__(self, "details", _normalize_json_mapping(self.details, field_name="details"))
        object.__setattr__(self, "warnings", _normalize_text_tuple(self.warnings, "warnings"))
        normalized_redacted_fields = _normalize_text_tuple(self.redacted_fields, "redacted_fields")
        object.__setattr__(self, "redacted_fields", tuple(dict.fromkeys(normalized_redacted_fields)))

    @classmethod
    def from_json(cls, json_data: str | bytes | bytearray) -> "IntegrationResult":
        """Build a result from a JSON object."""

        raw = _orjson.loads(json_data) if _orjson is not None else json.loads(json_data)
        if not isinstance(raw, Mapping):
            raise TypeError("IntegrationResult JSON must decode to an object.")
        return cls(**dict(raw))

    def details_json_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON bytes for the normalized details payload."""

        return _canonical_json_bytes(_to_json_primitive(self.details))

    def details_json(self) -> str:
        """Return canonical JSON text for the normalized details payload."""

        return self.details_json_bytes().decode("utf-8")
