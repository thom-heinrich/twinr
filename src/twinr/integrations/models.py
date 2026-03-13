from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import json


class IntegrationDomain(StrEnum):
    CALENDAR = "calendar"
    EMAIL = "email"
    MESSENGER = "messenger"
    SMART_HOME = "smart_home"
    SECURITY = "security"
    HEALTH = "health"


class IntegrationAction(StrEnum):
    READ = "read"
    WRITE = "write"
    SEND = "send"
    QUERY = "query"
    CONTROL = "control"
    ALERT = "alert"


class RequestOrigin(StrEnum):
    LOCAL_DEVICE = "local_device"
    LOCAL_DASHBOARD = "local_dashboard"
    REMOTE_SERVICE = "remote_service"


class RiskLevel(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ConfirmationMode(StrEnum):
    NONE = "none"
    USER = "user"
    CAREGIVER = "caregiver"


class DataSensitivity(StrEnum):
    NORMAL = "normal"
    PERSONAL = "personal"
    SECURITY = "security"
    HEALTH = "health"


class SecretStorage(StrEnum):
    ENV_VAR = "env_var"
    FILE = "file"
    KEYRING = "keyring"
    VAULT = "vault"


DEFAULT_SENSITIVE_PARAMETER_KEYS = frozenset(
    {
        "access_token",
        "attachment",
        "authorization",
        "body",
        "cookie",
        "diagnosis",
        "message",
        "medical_notes",
        "notes",
        "password",
        "secret",
        "token",
    }
)


@dataclass(frozen=True, slots=True)
class SecretReference:
    name: str
    reference: str
    storage: SecretStorage = SecretStorage.ENV_VAR
    required: bool = True


@dataclass(frozen=True, slots=True)
class SafetyProfile:
    risk: RiskLevel
    confirmation: ConfirmationMode = ConfirmationMode.NONE
    sensitivity: DataSensitivity = DataSensitivity.NORMAL
    allow_background_polling: bool = False
    allow_remote_trigger: bool = False
    allow_free_text: bool = False
    max_payload_bytes: int = 4096

    def __post_init__(self) -> None:
        if self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be greater than zero.")


@dataclass(frozen=True, slots=True)
class IntegrationOperation:
    operation_id: str
    label: str
    action: IntegrationAction
    summary: str
    safety: SafetyProfile


@dataclass(frozen=True, slots=True)
class IntegrationManifest:
    integration_id: str
    domain: IntegrationDomain
    title: str
    summary: str
    operations: tuple[IntegrationOperation, ...]
    required_secrets: tuple[SecretReference, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.operations:
            raise ValueError("Integration manifests must define at least one operation.")
        operation_ids = [operation.operation_id for operation in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError(f"Duplicate operations are not allowed for {self.integration_id}.")

    def operation(self, operation_id: str) -> IntegrationOperation | None:
        for operation in self.operations:
            if operation.operation_id == operation_id:
                return operation
        return None


@dataclass(slots=True)
class IntegrationRequest:
    integration_id: str
    operation_id: str
    parameters: dict[str, object] = field(default_factory=dict)
    origin: RequestOrigin = RequestOrigin.LOCAL_DEVICE
    explicit_user_confirmation: bool = False
    explicit_caregiver_confirmation: bool = False
    dry_run: bool = False
    background_trigger: bool = False

    def payload_size_bytes(self) -> int:
        payload = json.dumps(self.parameters, sort_keys=True, default=str)
        return len(payload.encode("utf-8"))

    def redacted_parameters(self, *, extra_sensitive_keys: set[str] | None = None) -> dict[str, object]:
        sensitive_keys = set(DEFAULT_SENSITIVE_PARAMETER_KEYS)
        if extra_sensitive_keys:
            sensitive_keys.update(key.lower() for key in extra_sensitive_keys)

        redacted: dict[str, object] = {}
        for key, value in self.parameters.items():
            if key.lower() in sensitive_keys:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = value
        return redacted

    def audit_label(self) -> str:
        return f"{self.integration_id}:{self.operation_id}:{self.origin.value}"


@dataclass(frozen=True, slots=True)
class IntegrationDecision:
    allowed: bool
    reason: str
    required_confirmation: ConfirmationMode | None = None
    warnings: tuple[str, ...] = ()

    @classmethod
    def allow(cls, reason: str, *, warnings: tuple[str, ...] = ()) -> "IntegrationDecision":
        return cls(allowed=True, reason=reason, warnings=warnings)

    @classmethod
    def deny(
        cls,
        reason: str,
        *,
        required_confirmation: ConfirmationMode | None = None,
        warnings: tuple[str, ...] = (),
    ) -> "IntegrationDecision":
        return cls(
            allowed=False,
            reason=reason,
            required_confirmation=required_confirmation,
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    ok: bool
    summary: str
    details: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    redacted_fields: tuple[str, ...] = ()
