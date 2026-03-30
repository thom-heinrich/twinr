# CHANGELOG: 2026-03-30
# BUG-1: Removed background-eligible smart-home control operations so physical actuation can no longer be scheduled from passive/background flows.
# BUG-2: Normalized manifest/domain/operation lookups to avoid silent misses caused by whitespace, hyphen, or case drift.
# SEC-1: Hard-fail at startup if any WRITE/SEND/ALERT/CONTROL operation is marked background-eligible or if unsafe confirmation policies are introduced.
# IMP-1: Switched defensive cloning to Pydantic round-trip validation when available to avoid model_copy(__dict__) side effects and preserve isolated snapshots.
# IMP-2: Added deterministic catalog/manifest/operation fingerprints, MCP-aligned policy hints, and a serializable audit snapshot for runtime governance.

"""Define Twinr's builtin integration manifest catalog.

This module owns the canonical manifest metadata for built-in integrations,
serves defensive clones so callers cannot mutate shared registry state, and
exports machine-readable policy hints/fingerprints for 2026 agent runtimes.
"""

from __future__ import annotations

# pylint: disable=undefined-all-variable

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Final, TypeVar

from twinr.integrations.models import (
    ConfirmationMode,
    DataSensitivity,
    IntegrationAction,
    IntegrationDomain,
    IntegrationManifest,
    IntegrationOperation,
    RiskLevel,
    SafetyProfile,
    SecretReference,
)

__all__ = (
    "BUILTIN_MANIFESTS",  # noqa: F822 - exported lazily via __getattr__ to preserve defensive copies.
    "BUILTIN_MANIFEST_CATALOG_FINGERPRINT",
    "BUILTIN_MANIFEST_CATALOG_VERSION",
    "OperationPolicyHints",
    "builtin_manifests",
    "catalog_fingerprint",
    "catalog_snapshot",
    "manifest_fingerprint_for_id",
    "manifest_for_id",
    "manifests_for_domain",
    "mcp_tool_annotations_for",
    "operation_for_id",
    "operation_policy_for",
)

TModel = TypeVar("TModel")

BUILTIN_MANIFEST_CATALOG_VERSION: Final[str] = "2026-03-30"

_IDENTIFIER_LOOKUP_SEPARATOR_RE: Final[re.Pattern[str]] = re.compile(r"[\s\-]+")
_ACTIVE_ACTIONS: Final[frozenset[IntegrationAction]] = frozenset(
    {
        IntegrationAction.WRITE,
        IntegrationAction.SEND,
        IntegrationAction.ALERT,
        IntegrationAction.CONTROL,
    }
)
_READ_ONLY_ACTIONS: Final[frozenset[IntegrationAction]] = frozenset(
    {
        IntegrationAction.READ,
        IntegrationAction.QUERY,
    }
)
_OPEN_WORLD_DOMAINS: Final[frozenset[IntegrationDomain]] = frozenset(
    {
        IntegrationDomain.EMAIL,
        IntegrationDomain.MESSENGER,
    }
)
_TRUSTED_RECIPIENT_DOMAINS: Final[frozenset[IntegrationDomain]] = frozenset(
    {
        IntegrationDomain.EMAIL,
        IntegrationDomain.MESSENGER,
        IntegrationDomain.SECURITY,
        IntegrationDomain.HEALTH,
    }
)
_PHYSICAL_EFFECT_DOMAINS: Final[frozenset[IntegrationDomain]] = frozenset(
    {
        IntegrationDomain.SMART_HOME,
    }
)


@dataclass(frozen=True, slots=True)
class OperationPolicyHints:
    """Derived machine-readable policy hints for one builtin operation."""

    integration_id: str
    operation_id: str
    read_only_hint: bool
    destructive_hint: bool
    idempotent_hint: bool
    open_world_hint: bool
    requires_confirmation: bool
    background_eligible: bool
    trusted_recipient_only: bool
    may_trigger_physical_effects: bool
    manifest_fingerprint: str
    operation_fingerprint: str


def _normalize_lookup_token(value: object) -> str | None:
    """Return a permissive lookup token for IDs, enum names, and enum values."""

    text: str | None = None
    if isinstance(value, str):
        text = value
    else:
        raw_value = getattr(value, "value", None)
        if isinstance(raw_value, str):
            text = raw_value
        else:
            raw_name = getattr(value, "name", None)
            if isinstance(raw_name, str):
                text = raw_name

    if text is None:
        return None

    normalized = _IDENTIFIER_LOOKUP_SEPARATOR_RE.sub("_", text.strip().casefold()).strip("_")
    return normalized or None


def _enum_json_value(value: Enum) -> str | int | float | bool:
    """Serialize an enum to a stable JSON-friendly primitive."""

    raw_value = value.value
    if isinstance(raw_value, (str, int, float, bool)):
        return raw_value
    return value.name


def _model_to_mapping(value: object) -> Mapping[str, object] | None:
    """Best-effort conversion of a model-like object to a mapping."""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        for kwargs in (
            {"mode": "python", "round_trip": True},
            {"mode": "python"},
            {},
        ):
            try:
                dumped = model_dump(**kwargs)
            except TypeError:
                continue
            if isinstance(dumped, Mapping):
                return dumped

    if isinstance(value, Mapping):
        return value

    if hasattr(value, "__dict__"):
        dumped = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if dumped:
            return dumped

    return None


def _to_jsonable(value: object) -> object:
    """Convert mixed model/enum structures into a stable JSON-friendly form."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Enum):
        return _enum_json_value(value)

    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}

    mapping = _model_to_mapping(value)
    if mapping is not None:
        return {str(key): _to_jsonable(item) for key, item in mapping.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]

    return str(value)


def _fingerprint(value: object) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-serializable value."""

    payload = json.dumps(
        _to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _clone_model(value: TModel) -> TModel:
    """Return an isolated clone of one model-like value."""

    model_dump = getattr(value, "model_dump", None)
    model_validate = getattr(type(value), "model_validate", None)
    if callable(model_dump) and callable(model_validate):
        for kwargs in (
            {"mode": "python", "round_trip": True},
            {"mode": "python"},
            {},
        ):
            try:
                payload = model_dump(**kwargs)
            except TypeError:
                continue
            try:
                return model_validate(payload)
            except Exception:  # pragma: no cover - best-effort round-trip, then fall back safely.
                break

    model_copy = getattr(value, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(deep=True)
        except TypeError:
            return model_copy()

    return deepcopy(value)


def _clone_manifest(manifest: IntegrationManifest) -> IntegrationManifest:
    """Return a defensive copy of one manifest instance."""

    return _clone_model(manifest)


def _clone_operation(operation: IntegrationOperation) -> IntegrationOperation:
    """Return a defensive copy of one operation instance."""

    return _clone_model(operation)


def _clone_manifests(
    manifests: tuple[IntegrationManifest, ...],
) -> tuple[IntegrationManifest, ...]:
    """Return defensive copies for a manifest tuple."""

    return tuple(_clone_manifest(manifest) for manifest in manifests)


def _require_non_empty_text(field_name: str, value: object) -> None:
    """Reject blank human-facing text."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_clean_human_text(field_name: str, value: object) -> None:
    """Reject blank human-facing text and edge whitespace."""

    _require_non_empty_text(field_name, value)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace")


def _require_normalized_identifier(field_name: str, value: object) -> None:
    """Reject identifiers with blank values or edge whitespace."""

    _require_clean_human_text(field_name, value)


def _operation_minimum_confirmation_requirement(operation: IntegrationOperation) -> bool:
    """Return whether one operation must be confirmation-gated by policy."""

    safety = operation.safety
    if safety is None:
        return True

    return (
        safety.risk == RiskLevel.HIGH
        or operation.action
        in {
            IntegrationAction.SEND,
            IntegrationAction.ALERT,
            IntegrationAction.CONTROL,
        }
        or (
            getattr(safety, "allow_free_text", False)
            and operation.action
            in {
                IntegrationAction.SEND,
                IntegrationAction.ALERT,
                IntegrationAction.WRITE,
            }
        )
    )


def _operation_requires_confirmation(operation: IntegrationOperation) -> bool:
    """Return whether runtime policy should require an explicit user confirmation."""

    safety = operation.safety
    if safety is None:
        return True
    return (
        safety.confirmation == ConfirmationMode.USER
        or _operation_minimum_confirmation_requirement(operation)
    )


def _validate_builtin_manifests(manifests: tuple[IntegrationManifest, ...]) -> None:
    """Validate builtin manifests before exposing them to the runtime."""

    seen_manifest_ids: set[str] = set()
    seen_normalized_manifest_ids: dict[str, str] = {}

    for manifest in manifests:
        _require_normalized_identifier("integration_id", manifest.integration_id)
        _require_clean_human_text(f"{manifest.integration_id}.title", manifest.title)
        _require_clean_human_text(f"{manifest.integration_id}.summary", manifest.summary)

        if manifest.integration_id in seen_manifest_ids:
            raise ValueError(f"Duplicate integration_id: {manifest.integration_id}")
        seen_manifest_ids.add(manifest.integration_id)

        normalized_manifest_id = _normalize_lookup_token(manifest.integration_id)
        if normalized_manifest_id is None:
            raise ValueError(f"{manifest.integration_id} has no normalized lookup token")
        previous_manifest_id = seen_normalized_manifest_ids.get(normalized_manifest_id)
        if previous_manifest_id is not None and previous_manifest_id != manifest.integration_id:
            raise ValueError(
                "Normalized integration_id collision: "
                f"{previous_manifest_id!r} and {manifest.integration_id!r}"
            )
        seen_normalized_manifest_ids[normalized_manifest_id] = manifest.integration_id

        notes = getattr(manifest, "notes", ()) or ()
        for note_index, note in enumerate(notes):
            _require_clean_human_text(f"{manifest.integration_id}.notes[{note_index}]", note)

        if not manifest.operations:
            raise ValueError(f"{manifest.integration_id} must define at least one operation")

        seen_operation_ids: set[str] = set()
        seen_normalized_operation_ids: dict[str, str] = {}

        for operation in manifest.operations:
            _require_normalized_identifier(
                f"{manifest.integration_id}.operations[].operation_id",
                operation.operation_id,
            )
            _require_clean_human_text(
                f"{manifest.integration_id}.{operation.operation_id}.label",
                operation.label,
            )
            _require_clean_human_text(
                f"{manifest.integration_id}.{operation.operation_id}.summary",
                operation.summary,
            )

            if operation.operation_id in seen_operation_ids:
                raise ValueError(
                    f"Duplicate operation_id within {manifest.integration_id}: {operation.operation_id}"
                )
            seen_operation_ids.add(operation.operation_id)

            normalized_operation_id = _normalize_lookup_token(operation.operation_id)
            if normalized_operation_id is None:
                raise ValueError(
                    f"{manifest.integration_id}.{operation.operation_id} has no normalized lookup token"
                )
            previous_operation_id = seen_normalized_operation_ids.get(normalized_operation_id)
            if previous_operation_id is not None and previous_operation_id != operation.operation_id:
                raise ValueError(
                    "Normalized operation_id collision within "
                    f"{manifest.integration_id}: {previous_operation_id!r} and {operation.operation_id!r}"
                )
            seen_normalized_operation_ids[normalized_operation_id] = operation.operation_id

            if operation.safety is None:
                raise ValueError(
                    f"{manifest.integration_id}.{operation.operation_id} must define a safety profile"
                )

            safety = operation.safety
            requires_confirmation = _operation_minimum_confirmation_requirement(operation)
            if (
                requires_confirmation
                and safety.confirmation != ConfirmationMode.USER
            ):
                raise ValueError(
                    f"{manifest.integration_id}.{operation.operation_id} requires explicit user confirmation"
                )

            allow_background_polling = bool(getattr(safety, "allow_background_polling", False))
            allow_free_text = bool(getattr(safety, "allow_free_text", False))

            # BREAKING: From catalog version 2026-03-30 onward, any operation with external or
            # state-changing side effects must remain foreground-only.
            if allow_background_polling and operation.action in _ACTIVE_ACTIONS:
                raise ValueError(
                    f"{manifest.integration_id}.{operation.operation_id} must not be background-eligible"
                )

            if safety.risk == RiskLevel.HIGH and allow_background_polling:
                raise ValueError(
                    f"{manifest.integration_id}.{operation.operation_id} is high-risk and must stay foreground-only"
                )

            if allow_free_text and operation.action == IntegrationAction.CONTROL:
                raise ValueError(
                    f"{manifest.integration_id}.{operation.operation_id} must use structured control parameters, not free text"
                )

            if allow_free_text and operation.action not in {
                IntegrationAction.WRITE,
                IntegrationAction.SEND,
                IntegrationAction.ALERT,
            }:
                raise ValueError(
                    f"{manifest.integration_id}.{operation.operation_id} enables free text for an unsupported action type"
                )


def _derive_operation_policy(
    manifest: IntegrationManifest,
    operation: IntegrationOperation,
    manifest_fingerprint: str,
) -> OperationPolicyHints:
    """Derive machine-readable runtime policy hints for one operation."""

    read_only_hint = operation.action in _READ_ONLY_ACTIONS
    destructive_hint = operation.action in {
        IntegrationAction.SEND,
        IntegrationAction.ALERT,
        IntegrationAction.CONTROL,
    }
    background_eligible = bool(getattr(operation.safety, "allow_background_polling", False))
    open_world_hint = (
        manifest.domain in _OPEN_WORLD_DOMAINS
        or operation.action in {IntegrationAction.SEND, IntegrationAction.ALERT}
    )
    trusted_recipient_only = (
        manifest.domain in _TRUSTED_RECIPIENT_DOMAINS
        and operation.action in {IntegrationAction.SEND, IntegrationAction.ALERT}
    )
    may_trigger_physical_effects = (
        manifest.domain in _PHYSICAL_EFFECT_DOMAINS
        and operation.action == IntegrationAction.CONTROL
    )
    policy_material = {
        "integration_id": manifest.integration_id,
        "operation_id": operation.operation_id,
        "read_only_hint": read_only_hint,
        "destructive_hint": destructive_hint,
        "idempotent_hint": False,
        "open_world_hint": open_world_hint,
        "requires_confirmation": _operation_requires_confirmation(operation),
        "background_eligible": background_eligible,
        "trusted_recipient_only": trusted_recipient_only,
        "may_trigger_physical_effects": may_trigger_physical_effects,
        "manifest_fingerprint": manifest_fingerprint,
        "operation": _to_jsonable(operation),
    }
    operation_fingerprint = _fingerprint(policy_material)
    return OperationPolicyHints(
        integration_id=manifest.integration_id,
        operation_id=operation.operation_id,
        read_only_hint=read_only_hint,
        destructive_hint=destructive_hint,
        idempotent_hint=False,
        open_world_hint=open_world_hint,
        requires_confirmation=_operation_requires_confirmation(operation),
        background_eligible=background_eligible,
        trusted_recipient_only=trusted_recipient_only,
        may_trigger_physical_effects=may_trigger_physical_effects,
        manifest_fingerprint=manifest_fingerprint,
        operation_fingerprint=operation_fingerprint,
    )


def _build_domain_aliases(
    manifests: tuple[IntegrationManifest, ...],
) -> Mapping[str, IntegrationDomain]:
    """Build normalized aliases for manifest domains."""

    aliases: dict[str, IntegrationDomain] = {}

    def register(candidate: object, domain: IntegrationDomain) -> None:
        normalized = _normalize_lookup_token(candidate)
        if normalized is None:
            return
        previous = aliases.get(normalized)
        if previous is not None and previous != domain:
            raise ValueError(
                f"Normalized domain alias collision: {candidate!r} resolves to both {previous!r} and {domain!r}"
            )
        aliases[normalized] = domain

    for manifest in manifests:
        register(manifest.domain, manifest.domain)
        register(getattr(manifest.domain, "name", None), manifest.domain)
        register(getattr(manifest.domain, "value", None), manifest.domain)

    return MappingProxyType(aliases)


def _build_catalog_indexes(
    manifests: tuple[IntegrationManifest, ...],
) -> tuple[
    Mapping[str, IntegrationManifest],
    Mapping[str, str],
    Mapping[IntegrationDomain, tuple[IntegrationManifest, ...]],
    Mapping[tuple[str, str], IntegrationOperation],
    Mapping[tuple[str, str], tuple[str, str]],
    Mapping[tuple[str, str], OperationPolicyHints],
    Mapping[tuple[str, str], Mapping[str, object]],
    Mapping[str, str],
    Mapping[str, IntegrationDomain],
    dict[str, object],
]:
    """Build immutable lookup indexes, fingerprints, and audit snapshots."""

    manifests_by_id: dict[str, IntegrationManifest] = {}
    manifest_ids_by_normalized: dict[str, str] = {}
    manifests_by_domain: dict[IntegrationDomain, list[IntegrationManifest]] = {}
    operations_by_key: dict[tuple[str, str], IntegrationOperation] = {}
    operation_keys_by_normalized: dict[tuple[str, str], tuple[str, str]] = {}
    policies_by_operation_key: dict[tuple[str, str], OperationPolicyHints] = {}
    mcp_annotations_by_operation_key: dict[tuple[str, str], Mapping[str, object]] = {}
    manifest_fingerprints_by_id: dict[str, str] = {}
    manifest_snapshots: list[dict[str, object]] = []

    for manifest in manifests:
        manifests_by_id[manifest.integration_id] = manifest
        manifest_ids_by_normalized[_normalize_lookup_token(manifest.integration_id) or manifest.integration_id] = (
            manifest.integration_id
        )
        manifests_by_domain.setdefault(manifest.domain, []).append(manifest)

        manifest_fingerprint = _fingerprint(
            {
                "catalog_version": BUILTIN_MANIFEST_CATALOG_VERSION,
                "manifest": _to_jsonable(manifest),
            }
        )
        manifest_fingerprints_by_id[manifest.integration_id] = manifest_fingerprint

        operation_snapshots: list[dict[str, object]] = []
        for operation in manifest.operations:
            operation_key = (manifest.integration_id, operation.operation_id)
            operations_by_key[operation_key] = operation
            operation_keys_by_normalized[
                (
                    _normalize_lookup_token(manifest.integration_id) or manifest.integration_id,
                    _normalize_lookup_token(operation.operation_id) or operation.operation_id,
                )
            ] = operation_key

            policy = _derive_operation_policy(manifest, operation, manifest_fingerprint)
            policies_by_operation_key[operation_key] = policy

            mcp_annotations = MappingProxyType(
                {
                    "title": operation.label,
                    "readOnlyHint": policy.read_only_hint,
                    "destructiveHint": policy.destructive_hint,
                    "idempotentHint": policy.idempotent_hint,
                    "openWorldHint": policy.open_world_hint,
                }
            )
            mcp_annotations_by_operation_key[operation_key] = mcp_annotations

            operation_snapshot = deepcopy(_to_jsonable(operation))
            if not isinstance(operation_snapshot, dict):
                operation_snapshot = {"operation": operation_snapshot}
            operation_snapshot["operation_fingerprint"] = policy.operation_fingerprint
            operation_snapshot["policy_hints"] = _to_jsonable(policy)
            operation_snapshot["mcp_annotations"] = dict(mcp_annotations)
            operation_snapshots.append(operation_snapshot)

        manifest_snapshot = deepcopy(_to_jsonable(manifest))
        if not isinstance(manifest_snapshot, dict):
            manifest_snapshot = {"manifest": manifest_snapshot}
        manifest_snapshot["manifest_fingerprint"] = manifest_fingerprint
        manifest_snapshot["operations"] = operation_snapshots
        manifest_snapshots.append(manifest_snapshot)

    catalog_fingerprint_value = _fingerprint(
        {
            "catalog_version": BUILTIN_MANIFEST_CATALOG_VERSION,
            "manifests": manifest_snapshots,
        }
    )
    catalog_snapshot_value = {
        "catalog_version": BUILTIN_MANIFEST_CATALOG_VERSION,
        "catalog_fingerprint": catalog_fingerprint_value,
        "manifests": manifest_snapshots,
    }

    return (
        MappingProxyType(manifests_by_id),
        MappingProxyType(manifest_ids_by_normalized),
        MappingProxyType(
            {
                domain: tuple(domain_manifests)
                for domain, domain_manifests in manifests_by_domain.items()
            }
        ),
        MappingProxyType(operations_by_key),
        MappingProxyType(operation_keys_by_normalized),
        MappingProxyType(policies_by_operation_key),
        MappingProxyType(mcp_annotations_by_operation_key),
        MappingProxyType(manifest_fingerprints_by_id),
        _build_domain_aliases(manifests),
        catalog_snapshot_value,
    )


def _resolve_integration_id(integration_id: object) -> str | None:
    """Resolve an integration identifier using normalized aliases."""

    normalized = _normalize_lookup_token(integration_id)
    if normalized is None:
        return None
    return _MANIFEST_IDS_BY_NORMALIZED.get(normalized)


def _resolve_domain(domain: object) -> IntegrationDomain | None:
    """Resolve an integration domain from enum, enum name, or normalized string."""

    if domain in _MANIFESTS_BY_DOMAIN:
        return domain  # type: ignore[return-value]
    normalized = _normalize_lookup_token(domain)
    if normalized is None:
        return None
    return _DOMAIN_ALIASES.get(normalized)


def _resolve_operation_key(
    integration_id: object,
    operation_id: object,
) -> tuple[str, str] | None:
    """Resolve one operation key using normalized integration and operation IDs."""

    resolved_integration_id = _resolve_integration_id(integration_id)
    normalized_operation_id = _normalize_lookup_token(operation_id)
    if resolved_integration_id is None or normalized_operation_id is None:
        return None
    return _OPERATION_KEYS_BY_NORMALIZED.get(
        (_normalize_lookup_token(resolved_integration_id) or resolved_integration_id, normalized_operation_id)
    )


_CANONICAL_BUILTIN_MANIFESTS: Final[tuple[IntegrationManifest, ...]] = (
    IntegrationManifest(
        integration_id="calendar_agenda",
        domain=IntegrationDomain.CALENDAR,
        title="Calendar Agenda",
        summary="Read a trusted agenda feed and prepare short read-only summaries for the user.",
        operations=(
            IntegrationOperation(
                operation_id="read_today",
                label="Read today's agenda",
                action=IntegrationAction.READ,
                summary="Read today's appointments and visits from a trusted calendar source.",
                safety=SafetyProfile(
                    risk=RiskLevel.LOW,
                    sensitivity=DataSensitivity.PERSONAL,
                    allow_background_polling=True,
                ),
            ),
            IntegrationOperation(
                operation_id="read_upcoming",
                label="Read upcoming events",
                action=IntegrationAction.QUERY,
                summary="Read the next days of calendar events without changing the source calendar.",
                safety=SafetyProfile(
                    risk=RiskLevel.LOW,
                    sensitivity=DataSensitivity.PERSONAL,
                    allow_background_polling=True,
                ),
            ),
            IntegrationOperation(
                operation_id="read_next_event",
                label="Read next event",
                action=IntegrationAction.QUERY,
                summary="Read only the next known event from the trusted calendar source.",
                safety=SafetyProfile(
                    risk=RiskLevel.LOW,
                    sensitivity=DataSensitivity.PERSONAL,
                    allow_background_polling=True,
                ),
            ),
        ),
        notes=(
            "Phase 1 is read-only.",
            "Calendar writes, invitations, and attendee changes are intentionally excluded.",
        ),
    ),
    IntegrationManifest(
        integration_id="email_mailbox",
        domain=IntegrationDomain.EMAIL,
        title="Email Mailbox",
        summary="Read recent email and prepare carefully confirmed replies for known contacts.",
        required_secrets=(
            SecretReference("email_app_password", "TWINR_INTEGRATION_EMAIL_APP_PASSWORD"),
        ),
        operations=(
            IntegrationOperation(
                operation_id="read_recent",
                label="Read recent email",
                action=IntegrationAction.READ,
                summary="Fetch a short summary of recent messages.",
                safety=SafetyProfile(
                    risk=RiskLevel.LOW,
                    sensitivity=DataSensitivity.PERSONAL,
                    allow_background_polling=True,
                ),
            ),
            IntegrationOperation(
                operation_id="draft_reply",
                label="Draft reply",
                action=IntegrationAction.WRITE,
                summary="Prepare a reply draft without sending it.",
                safety=SafetyProfile(
                    risk=RiskLevel.MODERATE,
                    confirmation=ConfirmationMode.USER,
                    sensitivity=DataSensitivity.PERSONAL,
                    allow_free_text=True,
                ),
            ),
            IntegrationOperation(
                operation_id="send_message",
                label="Send email",
                action=IntegrationAction.SEND,
                summary="Send an email reply after explicit user confirmation.",
                safety=SafetyProfile(
                    risk=RiskLevel.HIGH,
                    confirmation=ConfirmationMode.USER,
                    sensitivity=DataSensitivity.PERSONAL,
                    allow_free_text=True,
                ),
            ),
        ),
        notes=(
            "Restrict sending to approved contacts.",
            "Never store raw mailbox credentials in Twinr logs or artifact stores.",
        ),
    ),
    IntegrationManifest(
        integration_id="messenger_bridge",
        domain=IntegrationDomain.MESSENGER,
        title="Messenger Bridge",
        summary="Read short message summaries and send carefully confirmed check-ins.",
        required_secrets=(
            SecretReference("messenger_access_token", "TWINR_MESSENGER_ACCESS_TOKEN"),
        ),
        operations=(
            IntegrationOperation(
                operation_id="read_recent_thread",
                label="Read recent thread",
                action=IntegrationAction.READ,
                summary="Read a brief summary from a trusted thread.",
                safety=SafetyProfile(
                    risk=RiskLevel.LOW,
                    sensitivity=DataSensitivity.PERSONAL,
                    allow_background_polling=True,
                ),
            ),
            IntegrationOperation(
                operation_id="send_message",
                label="Send message",
                action=IntegrationAction.SEND,
                summary="Send a short text message after explicit user confirmation.",
                safety=SafetyProfile(
                    risk=RiskLevel.HIGH,
                    confirmation=ConfirmationMode.USER,
                    sensitivity=DataSensitivity.PERSONAL,
                    allow_free_text=True,
                ),
            ),
            IntegrationOperation(
                operation_id="send_check_in",
                label="Send caregiver check-in",
                action=IntegrationAction.ALERT,
                summary="Send a short well-being check-in to a trusted contact.",
                safety=SafetyProfile(
                    risk=RiskLevel.MODERATE,
                    confirmation=ConfirmationMode.USER,
                    sensitivity=DataSensitivity.PERSONAL,
                ),
            ),
        ),
        notes=(
            "Group chats should be opt-in only.",
            "Do not send messages without explicit confirmation.",
        ),
    ),
    IntegrationManifest(
        integration_id="smart_home_hub",
        domain=IntegrationDomain.SMART_HOME,
        title="Smart Home Hub",
        summary="Read device state, control approved low-risk devices, and consume bounded sensor streams.",
        required_secrets=(
            SecretReference("smart_home_endpoint", "TWINR_SMART_HOME_ENDPOINT"),
            SecretReference("smart_home_token", "TWINR_SMART_HOME_TOKEN"),
        ),
        operations=(
            IntegrationOperation(
                operation_id="read_device_state",
                label="Read device state",
                action=IntegrationAction.QUERY,
                summary="Read current state for known smart-home entities such as lights or sensors.",
                safety=SafetyProfile(
                    risk=RiskLevel.LOW,
                    sensitivity=DataSensitivity.NORMAL,
                    allow_background_polling=True,
                ),
            ),
            IntegrationOperation(
                operation_id="list_entities",
                label="List smart-home entities",
                action=IntegrationAction.QUERY,
                summary="List bounded smart-home entities with their current state and control capabilities.",
                safety=SafetyProfile(
                    risk=RiskLevel.LOW,
                    sensitivity=DataSensitivity.NORMAL,
                    allow_background_polling=True,
                ),
            ),
            IntegrationOperation(
                operation_id="read_sensor_stream",
                label="Read sensor stream",
                action=IntegrationAction.READ,
                summary="Read a bounded batch of normalized smart-home sensor or device events.",
                safety=SafetyProfile(
                    risk=RiskLevel.LOW,
                    sensitivity=DataSensitivity.NORMAL,
                    allow_background_polling=True,
                ),
            ),
            IntegrationOperation(
                operation_id="run_safe_scene",
                label="Run safe scene",
                action=IntegrationAction.CONTROL,
                summary="Trigger a pre-approved low-risk routine such as lights on.",
                safety=SafetyProfile(
                    risk=RiskLevel.MODERATE,
                    confirmation=ConfirmationMode.USER,
                    # BREAKING: Physical actuation is now foreground-only; background-triggered scenes are no longer catalog-valid.
                    allow_background_polling=False,
                ),
            ),
            IntegrationOperation(
                operation_id="control_entities",
                label="Control smart-home entities",
                action=IntegrationAction.CONTROL,
                summary="Control approved low-risk smart-home entities such as lights, light groups, or scenes.",
                safety=SafetyProfile(
                    risk=RiskLevel.MODERATE,
                    confirmation=ConfirmationMode.USER,
                    # BREAKING: Physical actuation is now foreground-only; background-triggered control is no longer catalog-valid.
                    allow_background_polling=False,
                ),
            ),
        ),
        notes=(
            "Critical actuations such as door unlock or alarm disarm are intentionally excluded.",
            "Background observation is allowed; actuation remains foreground-only and confirmation-gated.",
        ),
    ),
    IntegrationManifest(
        integration_id="security_monitor",
        domain=IntegrationDomain.SECURITY,
        title="Security Monitor",
        summary="Read security status and raise human-visible alerts without exposing dangerous controls.",
        required_secrets=(
            SecretReference("security_endpoint", "TWINR_SECURITY_ENDPOINT"),
            SecretReference("security_token", "TWINR_SECURITY_TOKEN"),
        ),
        operations=(
            IntegrationOperation(
                operation_id="read_status",
                label="Read security status",
                action=IntegrationAction.QUERY,
                summary="Read a short status summary for sensors and recent alerts.",
                safety=SafetyProfile(
                    risk=RiskLevel.MODERATE,
                    confirmation=ConfirmationMode.USER,
                    sensitivity=DataSensitivity.SECURITY,
                    allow_background_polling=True,
                ),
            ),
            IntegrationOperation(
                operation_id="read_camera_snapshot",
                label="Read camera snapshot",
                action=IntegrationAction.READ,
                summary="Fetch a recent camera still for human review on a trusted screen.",
                safety=SafetyProfile(
                    risk=RiskLevel.HIGH,
                    confirmation=ConfirmationMode.USER,
                    sensitivity=DataSensitivity.SECURITY,
                ),
            ),
            IntegrationOperation(
                operation_id="send_help_alert",
                label="Send help alert",
                action=IntegrationAction.ALERT,
                summary="Notify a trusted contact that the user asked for help.",
                safety=SafetyProfile(
                    risk=RiskLevel.HIGH,
                    confirmation=ConfirmationMode.USER,
                    sensitivity=DataSensitivity.SECURITY,
                ),
            ),
        ),
        notes=(
            "No unlock, disarm, or suppression operations belong in the generic catalog.",
        ),
    ),
    IntegrationManifest(
        integration_id="health_records",
        domain=IntegrationDomain.HEALTH,
        title="Health Records",
        summary="Read health summaries and share explicit updates without medication-control actions.",
        required_secrets=(
            SecretReference("health_endpoint", "TWINR_HEALTH_ENDPOINT"),
            SecretReference("health_token", "TWINR_HEALTH_TOKEN"),
        ),
        operations=(
            IntegrationOperation(
                operation_id="read_daily_summary",
                label="Read daily summary",
                action=IntegrationAction.READ,
                summary="Read a short daily summary such as appointments or measurements.",
                safety=SafetyProfile(
                    risk=RiskLevel.MODERATE,
                    confirmation=ConfirmationMode.USER,
                    sensitivity=DataSensitivity.HEALTH,
                ),
            ),
            IntegrationOperation(
                operation_id="read_medication_schedule",
                label="Read medication schedule",
                action=IntegrationAction.QUERY,
                summary="Read the stored medication schedule without changing it.",
                safety=SafetyProfile(
                    risk=RiskLevel.MODERATE,
                    confirmation=ConfirmationMode.USER,
                    sensitivity=DataSensitivity.HEALTH,
                ),
            ),
            IntegrationOperation(
                operation_id="send_caregiver_update",
                label="Send caregiver update",
                action=IntegrationAction.SEND,
                summary="Send a short health update after explicit user confirmation.",
                safety=SafetyProfile(
                    risk=RiskLevel.HIGH,
                    confirmation=ConfirmationMode.USER,
                    sensitivity=DataSensitivity.HEALTH,
                    allow_free_text=True,
                ),
            ),
        ),
        notes=(
            "Medication changes and diagnosis edits require a dedicated reviewed adapter, not the generic layer.",
        ),
    ),
)

try:
    _validate_builtin_manifests(_CANONICAL_BUILTIN_MANIFESTS)
except ValueError as exc:
    raise RuntimeError(f"Invalid builtin integration manifest registry: {exc}") from exc

(
    _MANIFESTS_BY_ID,
    _MANIFEST_IDS_BY_NORMALIZED,
    _MANIFESTS_BY_DOMAIN,
    _OPERATIONS_BY_KEY,
    _OPERATION_KEYS_BY_NORMALIZED,
    _POLICIES_BY_OPERATION_KEY,
    _MCP_ANNOTATIONS_BY_OPERATION_KEY,
    _MANIFEST_FINGERPRINTS_BY_ID,
    _DOMAIN_ALIASES,
    _CATALOG_SNAPSHOT,
) = _build_catalog_indexes(_CANONICAL_BUILTIN_MANIFESTS)

BUILTIN_MANIFEST_CATALOG_FINGERPRINT: Final[str] = _CATALOG_SNAPSHOT["catalog_fingerprint"]  # type: ignore[assignment]


def __getattr__(name: str) -> object:
    """Serve compatibility exports that are materialized on demand."""

    if name == "BUILTIN_MANIFESTS":
        return _clone_manifests(_CANONICAL_BUILTIN_MANIFESTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Keep compatibility exports visible to discovery tools."""

    return sorted(set(globals()) | {"BUILTIN_MANIFESTS"})


def builtin_manifests() -> tuple[IntegrationManifest, ...]:
    """Return cloned builtin manifests for callers that need a snapshot."""

    return _clone_manifests(_CANONICAL_BUILTIN_MANIFESTS)


def catalog_fingerprint() -> str:
    """Return the deterministic fingerprint for the current builtin catalog."""

    return BUILTIN_MANIFEST_CATALOG_FINGERPRINT


def catalog_snapshot() -> dict[str, object]:
    """Return a serializable audit snapshot enriched with policy hints."""

    return deepcopy(_CATALOG_SNAPSHOT)


def manifest_for_id(integration_id: str) -> IntegrationManifest | None:
    """Return one cloned builtin manifest by integration ID."""

    resolved_integration_id = _resolve_integration_id(integration_id)
    if resolved_integration_id is None:
        return None
    manifest = _MANIFESTS_BY_ID.get(resolved_integration_id)
    if manifest is None:
        return None
    return _clone_manifest(manifest)


def manifest_fingerprint_for_id(integration_id: str) -> str | None:
    """Return the deterministic fingerprint for one manifest."""

    resolved_integration_id = _resolve_integration_id(integration_id)
    if resolved_integration_id is None:
        return None
    return _MANIFEST_FINGERPRINTS_BY_ID.get(resolved_integration_id)


def manifests_for_domain(domain: IntegrationDomain | str) -> tuple[IntegrationManifest, ...]:
    """Return cloned builtin manifests for one integration domain."""

    resolved_domain = _resolve_domain(domain)
    if resolved_domain is None:
        return ()
    manifests = _MANIFESTS_BY_DOMAIN.get(resolved_domain, ())
    return _clone_manifests(manifests)


def operation_for_id(
    integration_id: str,
    operation_id: str,
) -> IntegrationOperation | None:
    """Return one cloned builtin operation by integration and operation ID."""

    operation_key = _resolve_operation_key(integration_id, operation_id)
    if operation_key is None:
        return None
    operation = _OPERATIONS_BY_KEY.get(operation_key)
    if operation is None:
        return None
    return _clone_operation(operation)


def operation_policy_for(
    integration_id: str,
    operation_id: str,
) -> OperationPolicyHints | None:
    """Return derived policy hints for one builtin operation."""

    operation_key = _resolve_operation_key(integration_id, operation_id)
    if operation_key is None:
        return None
    return _POLICIES_BY_OPERATION_KEY.get(operation_key)


def mcp_tool_annotations_for(
    integration_id: str,
    operation_id: str,
) -> dict[str, object] | None:
    """Return MCP-aligned tool annotations for one builtin operation."""

    operation_key = _resolve_operation_key(integration_id, operation_id)
    if operation_key is None:
        return None
    annotations = _MCP_ANNOTATIONS_BY_OPERATION_KEY.get(operation_key)
    if annotations is None:
        return None
    return dict(annotations)