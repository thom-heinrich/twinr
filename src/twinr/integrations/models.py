"""Define the canonical data contracts for Twinr integrations.

These models normalize identifiers, payloads, safety metadata, and redaction
rules so policy, runtime, registry, and provider packages share one contract.
"""

from __future__ import annotations

from collections.abc import Mapping  # AUDIT-FIX(#3,#7): Mapping-basierte Normalisierung für Request-/Result-Payloads.
from dataclasses import dataclass, field
from datetime import date, datetime, time  # AUDIT-FIX(#3): Explizite Behandlung von Datum/Zeit statt stiller str()-Serialisierung.
from decimal import Decimal  # AUDIT-FIX(#3,#7): Deterministische Serialisierung häufiger Scalar-Typen.
from enum import StrEnum
import json
import math  # AUDIT-FIX(#3): NaN/Infinity hart ablehnen für deterministische JSON-Payloads.
from pathlib import PurePath  # AUDIT-FIX(#3,#7): Pfadangaben sicher als Strings serialisieren.
import re  # AUDIT-FIX(#4,#5): Identifier- und Control-Character-Validierung.
from typing import TypeVar  # AUDIT-FIX(#5): StrEnum-Coercion typsicher halten.
from uuid import UUID  # AUDIT-FIX(#3,#7): UUIDs deterministisch serialisieren.


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")  # AUDIT-FIX(#4,#5): Audit- und log-sichere Identifier erzwingen.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")  # AUDIT-FIX(#4,#5): Newlines/NULs in Audit-Texten und Keys blocken.
_SENSITIVE_KEY_NORMALIZATION_RE = re.compile(r"[^a-z0-9]+")  # AUDIT-FIX(#2): accessToken/access_token/access-token gleich behandeln.
_MAX_NESTING_DEPTH = 32  # AUDIT-FIX(#3,#7): Rekursions- und Speicher-Explosionen auf dem RPi begrenzen.
_REDACTED_VALUE = "<redacted>"  # AUDIT-FIX(#2): Einheitlicher Platzhalter für sensible Inhalte.

_StrEnumT = TypeVar("_StrEnumT", bound=StrEnum)


# AUDIT-FIX(#2): Key-Normalisierung verhindert, dass nur eine Schreibweise redacted wird.
def _normalize_sensitive_key(key: str) -> str:
    """Normalize a key for case- and punctuation-insensitive matching."""

    return _SENSITIVE_KEY_NORMALIZATION_RE.sub("", key.casefold())


# AUDIT-FIX(#4,#5,#7): Frühe Validierung verhindert leere Texte, Control-Chars und späte Laufzeitfehler.
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


# AUDIT-FIX(#4,#5): IDs bleiben delimiter-safe und log-injection-resistent.
def _ensure_identifier(value: object, field_name: str) -> str:
    """Validate an audit-safe identifier string."""

    normalized = _ensure_safe_text(value, field_name)
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must match {_IDENTIFIER_RE.pattern!r} to stay audit-safe."
        )
    return normalized


# AUDIT-FIX(#5,#7): Bool-Felder nicht still aus 0/1/"yes" konvertieren.
def _ensure_bool(value: object, field_name: str) -> bool:
    """Require a real boolean instead of a truthy proxy value."""

    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool.")
    return value


# AUDIT-FIX(#1,#3): Integer-Grenzen fail-fast prüfen statt erst im Downstream zu crashen.
def _ensure_positive_int(value: object, field_name: str) -> int:
    """Require an integer greater than zero."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


# AUDIT-FIX(#1,#5,#7): Rohstrings aus Config/JSON sicher in StrEnums überführen.
def _coerce_str_enum(value: object, enum_type: type[_StrEnumT], field_name: str) -> _StrEnumT:
    """Coerce a raw value into a ``StrEnum`` member."""

    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:  # pragma: no cover - trivial branch
            allowed = ", ".join(member.value for member in enum_type)
            raise ValueError(f"{field_name} must be one of: {allowed}.") from exc
    raise TypeError(f"{field_name} must be a {enum_type.__name__} or matching string value.")


# AUDIT-FIX(#6,#7): Listen von außen in immutable Tupel überführen, ohne Set-Reihenfolgen einzuschleusen.
def _coerce_tuple(value: object, field_name: str) -> tuple[object, ...]:
    """Freeze tuple-like input into an immutable tuple."""

    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError(f"{field_name} must be a tuple, list, or None.")


# AUDIT-FIX(#6,#7): Textlisten konsistent validieren und säubern.
def _normalize_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    """Normalize an iterable of user-visible text into a tuple."""

    raw_items = _coerce_tuple(value, field_name)
    return tuple(
        _ensure_safe_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(raw_items)
    )


# AUDIT-FIX(#3,#7): Top-Level-Mappings früh in JSON-sichere Dicts kopieren.
def _normalize_json_mapping(value: object, *, field_name: str) -> dict[str, object]:
    """Normalize a top-level mapping into a JSON-safe ``dict``."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping with string keys.")
    normalized = _normalize_json_value(value, path=field_name, depth=0, seen=set())
    if not isinstance(normalized, dict):  # pragma: no cover - defensive
        raise TypeError(f"{field_name} must normalize to a dict.")
    return normalized


# AUDIT-FIX(#3,#7): Deterministische, serialisierbare und rekursionssichere Payload-Normalisierung.
def _normalize_json_value(
    value: object,
    *,
    path: str,
    depth: int,
    seen: set[int],
) -> object:
    """Normalize nested payload data into deterministic JSON-safe values."""

    if depth > _MAX_NESTING_DEPTH:
        raise ValueError(f"{path} exceeds the maximum supported nesting depth of {_MAX_NESTING_DEPTH}.")

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float value.")
        return value

    if isinstance(value, StrEnum):
        return value.value

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{path} contains a non-finite Decimal value.")
        return str(value)

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, PurePath):
        return str(value)

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{path} contains a timezone-naive datetime.")
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, time):
        if value.tzinfo is not None and value.utcoffset() is None:
            raise ValueError(f"{path} contains a time with an invalid timezone offset.")
        return value.isoformat()

    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{path} contains binary data; pass a file reference instead.")

    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in seen:
            raise ValueError(f"{path} contains a circular reference.")
        seen.add(container_id)
        try:
            normalized: dict[str, object] = {}
            for raw_key, raw_value in value.items():
                if not isinstance(raw_key, str):
                    raise TypeError(f"{path} contains a non-string key: {raw_key!r}.")
                if _CONTROL_CHAR_RE.search(raw_key):
                    raise ValueError(f"{path} contains control characters in key {raw_key!r}.")
                normalized[raw_key] = _normalize_json_value(
                    raw_value,
                    path=f"{path}.{raw_key}",
                    depth=depth + 1,
                    seen=seen,
                )
            return normalized
        finally:
            seen.remove(container_id)

    if isinstance(value, list):
        container_id = id(value)
        if container_id in seen:
            raise ValueError(f"{path} contains a circular reference.")
        seen.add(container_id)
        try:
            return [
                _normalize_json_value(item, path=f"{path}[{index}]", depth=depth + 1, seen=seen)
                for index, item in enumerate(value)
            ]
        finally:
            seen.remove(container_id)

    if isinstance(value, tuple):
        container_id = id(value)
        if container_id in seen:
            raise ValueError(f"{path} contains a circular reference.")
        seen.add(container_id)
        try:
            return tuple(
                _normalize_json_value(item, path=f"{path}[{index}]", depth=depth + 1, seen=seen)
                for index, item in enumerate(value)
            )
        finally:
            seen.remove(container_id)

    if isinstance(value, (set, frozenset)):
        raise TypeError(f"{path} contains an unordered set; use a list or tuple for deterministic payloads.")

    raise TypeError(f"{path} contains an unsupported value of type {type(value).__name__}.")


# AUDIT-FIX(#2): Rekursive Redaction schützt auch verschachtelte Tokens/PII und liefert neue Objekte zurück.
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
        "message",
        "medical_notes",
        "notes",
        "password",
        "refresh_token",
        "secret",
        "session_id",
        "token",
    }
)
_NORMALIZED_DEFAULT_SENSITIVE_PARAMETER_KEYS = frozenset(
    _normalize_sensitive_key(key) for key in DEFAULT_SENSITIVE_PARAMETER_KEYS
)


@dataclass(frozen=True, slots=True)
class SecretReference:
    """Describe one secret dependency required by an integration."""

    name: str
    reference: str
    storage: SecretStorage = SecretStorage.ENV_VAR
    required: bool = True

    def __post_init__(self) -> None:
        """Normalize and validate secret-reference fields at construction time."""

        # AUDIT-FIX(#5): Secret-Metadaten fail-fast validieren statt Downstream-Fehler zu riskieren.
        object.__setattr__(self, "name", _ensure_identifier(self.name, "name"))
        object.__setattr__(self, "reference", _ensure_safe_text(self.reference, "reference"))
        object.__setattr__(self, "storage", _coerce_str_enum(self.storage, SecretStorage, "storage"))
        object.__setattr__(self, "required", _ensure_bool(self.required, "required"))


@dataclass(frozen=True, slots=True)
class SafetyProfile:
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

        # AUDIT-FIX(#1,#5): Policy-Felder normalisieren und Hochrisiko ohne menschliche Freigabe hart verbieten.
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
        object.__setattr__(
            self,
            "max_payload_bytes",
            _ensure_positive_int(self.max_payload_bytes, "max_payload_bytes"),
        )

        if self.risk in {RiskLevel.HIGH, RiskLevel.CRITICAL} and self.confirmation is ConfirmationMode.NONE:
            raise ValueError(
                "High-risk and critical integrations must require user or caregiver confirmation."
            )


@dataclass(frozen=True, slots=True)
class IntegrationOperation:
    """Describe one operation exposed by an integration manifest."""

    operation_id: str
    label: str
    action: IntegrationAction
    summary: str
    safety: SafetyProfile

    def __post_init__(self) -> None:
        """Validate operation identifiers, text, and safety metadata."""

        # AUDIT-FIX(#5): Operationen mit kaputten IDs/Enums/Texten direkt am Rand stoppen.
        object.__setattr__(self, "operation_id", _ensure_identifier(self.operation_id, "operation_id"))
        object.__setattr__(self, "label", _ensure_safe_text(self.label, "label"))
        object.__setattr__(self, "action", _coerce_str_enum(self.action, IntegrationAction, "action"))
        object.__setattr__(self, "summary", _ensure_safe_text(self.summary, "summary"))
        if not isinstance(self.safety, SafetyProfile):
            raise TypeError("safety must be a SafetyProfile instance.")


@dataclass(frozen=True, slots=True)
class IntegrationManifest:
    """Describe one integration and the operations it exposes."""

    integration_id: str
    domain: IntegrationDomain
    title: str
    summary: str
    operations: tuple[IntegrationOperation, ...]
    required_secrets: tuple[SecretReference, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze and validate manifest metadata plus nested operations."""

        # AUDIT-FIX(#4,#5,#6): Manifest-Metadaten validieren, iterables einfrieren, Duplicate-Secrets verhindern.
        object.__setattr__(self, "integration_id", _ensure_identifier(self.integration_id, "integration_id"))
        object.__setattr__(self, "domain", _coerce_str_enum(self.domain, IntegrationDomain, "domain"))
        object.__setattr__(self, "title", _ensure_safe_text(self.title, "title"))
        object.__setattr__(self, "summary", _ensure_safe_text(self.summary, "summary"))

        operations = _coerce_tuple(self.operations, "operations")
        if not operations:
            raise ValueError("Integration manifests must define at least one operation.")
        normalized_operations: list[IntegrationOperation] = []
        operation_ids: list[str] = []
        for index, operation in enumerate(operations):
            if not isinstance(operation, IntegrationOperation):
                raise TypeError(f"operations[{index}] must be an IntegrationOperation.")
            normalized_operations.append(operation)
            operation_ids.append(operation.operation_id)
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError(f"Duplicate operations are not allowed for {self.integration_id}.")
        object.__setattr__(self, "operations", tuple(normalized_operations))

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

        # AUDIT-FIX(#5): Lookup bleibt tolerant bei schlechtem Input, matched aber nur gegen saubere IDs.
        if not isinstance(operation_id, str):
            return None
        normalized_operation_id = operation_id.strip()
        if not normalized_operation_id or not _IDENTIFIER_RE.fullmatch(normalized_operation_id):
            return None
        for operation in self.operations:
            if operation.operation_id == normalized_operation_id:
                return operation
        return None


@dataclass(slots=True)
class IntegrationRequest:
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

        # AUDIT-FIX(#3,#4,#5): Request-Felder sofort normalisieren, audit-sichere IDs erzwingen und JSON-Payload deterministisch machen.
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

    def payload_size_bytes(self) -> int:
        """Return the UTF-8 size of the normalized parameter payload."""

        # AUDIT-FIX(#3): Keine stille default=str()-Magie mehr; nur validierte JSON-Strukturen werden gezählt.
        payload = json.dumps(
            self.parameters,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return len(payload.encode("utf-8"))

    def redacted_parameters(self, *, extra_sensitive_keys: set[str] | None = None) -> dict[str, object]:
        """Return parameters with sensitive keys recursively redacted."""

        # AUDIT-FIX(#2): Rekursive, schreibweisen-robuste Redaction für Logs/Audits.
        sensitive_keys = set(_NORMALIZED_DEFAULT_SENSITIVE_PARAMETER_KEYS)
        if extra_sensitive_keys:
            sensitive_keys.update(
                _normalize_sensitive_key(_ensure_safe_text(key, "extra_sensitive_keys"))
                for key in extra_sensitive_keys
            )

        redacted = _redact_value(self.parameters, sensitive_keys=frozenset(sensitive_keys))
        if not isinstance(redacted, dict):  # pragma: no cover - defensive
            raise TypeError("parameters must redact to a dictionary.")
        return redacted

    def audit_label(self) -> str:
        """Return a log-safe label for the request."""

        # AUDIT-FIX(#4): IDs wurden schon validiert und bleiben damit delimiter-safe für Log-Trails.
        return f"{self.integration_id}:{self.operation_id}:{self.origin.value}"


@dataclass(frozen=True, slots=True)
class IntegrationDecision:
    """Represent the policy outcome for one integration request."""

    allowed: bool
    reason: str
    required_confirmation: ConfirmationMode | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the decision payload and reject contradictory states."""

        # AUDIT-FIX(#7): Widersprüchliche Decision-Zustände und unsaubere Texte blocken.
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
class IntegrationResult:
    """Represent the normalized result returned by an integration adapter."""

    ok: bool
    summary: str
    details: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    redacted_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the adapter result into immutable safe structures."""

        # AUDIT-FIX(#7): Frozen-Resultate trotzdem defensiv kopieren/normalisieren, damit API/Logs später nicht kippen.
        object.__setattr__(self, "ok", _ensure_bool(self.ok, "ok"))
        object.__setattr__(self, "summary", _ensure_safe_text(self.summary, "summary"))
        object.__setattr__(self, "details", _normalize_json_mapping(self.details, field_name="details"))
        object.__setattr__(self, "warnings", _normalize_text_tuple(self.warnings, "warnings"))
        normalized_redacted_fields = _normalize_text_tuple(self.redacted_fields, "redacted_fields")
        object.__setattr__(self, "redacted_fields", tuple(dict.fromkeys(normalized_redacted_fields)))
