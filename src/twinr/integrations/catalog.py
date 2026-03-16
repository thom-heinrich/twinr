from __future__ import annotations

from collections.abc import Mapping  # AUDIT-FIX(#3): Type immutable lookup indexes precisely in Python 3.11.2.
from copy import deepcopy  # AUDIT-FIX(#2): Defensive-copy manifests before exposing them outside this module.
from types import MappingProxyType  # AUDIT-FIX(#3): Freeze registry indexes against accidental mutation.
from typing import Final  # AUDIT-FIX(#2): Mark canonical registry data as immutable by contract.

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
    SecretStorage,  # AUDIT-FIX(#5): Kept intentionally per compatibility constraint; storage policy stays schema-driven.
)

__all__ = (
    "BUILTIN_MANIFESTS",
    "builtin_manifests",
    "manifest_for_id",
    "manifests_for_domain",
)


def _clone_manifest(manifest: IntegrationManifest) -> IntegrationManifest:
    # AUDIT-FIX(#2): Return a defensive clone so callers cannot mutate shared registry state across requests.
    model_copy = getattr(manifest, "model_copy", None)
    if callable(model_copy):
        return model_copy(deep=True)
    return deepcopy(manifest)


def _clone_manifests(
    manifests: tuple[IntegrationManifest, ...],
) -> tuple[IntegrationManifest, ...]:
    # AUDIT-FIX(#2): Clone each manifest on export so every caller gets an isolated snapshot.
    return tuple(_clone_manifest(manifest) for manifest in manifests)


def _require_non_empty_text(field_name: str, value: object) -> None:
    # AUDIT-FIX(#4): Fail fast on blank human-visible text instead of shipping malformed UI/voice labels.
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_normalized_identifier(field_name: str, value: object) -> None:
    # AUDIT-FIX(#4): Reject leading/trailing whitespace in identifiers to avoid silent lookup misses.
    _require_non_empty_text(field_name, value)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace")


def _validate_builtin_manifests(manifests: tuple[IntegrationManifest, ...]) -> None:
    # AUDIT-FIX(#3): Validate uniqueness and completeness at startup so registry mistakes fail closed.
    seen_manifest_ids: set[str] = set()

    for manifest in manifests:
        _require_normalized_identifier("integration_id", manifest.integration_id)
        _require_non_empty_text(f"{manifest.integration_id}.title", manifest.title)
        _require_non_empty_text(f"{manifest.integration_id}.summary", manifest.summary)

        if manifest.integration_id in seen_manifest_ids:
            raise ValueError(f"Duplicate integration_id: {manifest.integration_id}")
        seen_manifest_ids.add(manifest.integration_id)

        if not manifest.operations:
            raise ValueError(f"{manifest.integration_id} must define at least one operation")

        seen_operation_ids: set[str] = set()
        for operation in manifest.operations:
            _require_normalized_identifier(
                f"{manifest.integration_id}.operations[].operation_id",
                operation.operation_id,
            )
            _require_non_empty_text(
                f"{manifest.integration_id}.{operation.operation_id}.label",
                operation.label,
            )
            _require_non_empty_text(
                f"{manifest.integration_id}.{operation.operation_id}.summary",
                operation.summary,
            )

            if operation.operation_id in seen_operation_ids:
                raise ValueError(
                    f"Duplicate operation_id within {manifest.integration_id}: {operation.operation_id}"
                )
            seen_operation_ids.add(operation.operation_id)

            if operation.safety is None:
                raise ValueError(
                    f"{manifest.integration_id}.{operation.operation_id} must define a safety profile"
                )

            requires_confirmation = (
                operation.safety.risk == RiskLevel.HIGH
                or operation.action
                in {
                    IntegrationAction.SEND,
                    IntegrationAction.ALERT,
                    IntegrationAction.CONTROL,
                }
                or (
                    getattr(operation.safety, "allow_free_text", False)
                    and operation.action
                    in {
                        IntegrationAction.SEND,
                        IntegrationAction.ALERT,
                        IntegrationAction.CONTROL,
                        IntegrationAction.WRITE,
                    }
                )
            )
            if (
                requires_confirmation
                and operation.safety.confirmation != ConfirmationMode.USER
            ):
                # AUDIT-FIX(#1): Dangerous or free-text actions must never bypass explicit user confirmation.
                raise ValueError(
                    f"{manifest.integration_id}.{operation.operation_id} requires explicit user confirmation"
                )


def _build_manifest_indexes(
    manifests: tuple[IntegrationManifest, ...],
) -> tuple[
    Mapping[str, IntegrationManifest],
    Mapping[IntegrationDomain, tuple[IntegrationManifest, ...]],
]:
    # AUDIT-FIX(#3): Precompute immutable lookup indexes once after validation to avoid silent overwrite behavior.
    manifests_by_id: dict[str, IntegrationManifest] = {}
    manifests_by_domain: dict[IntegrationDomain, list[IntegrationManifest]] = {}

    for manifest in manifests:
        manifests_by_id[manifest.integration_id] = manifest
        manifests_by_domain.setdefault(manifest.domain, []).append(manifest)

    return (
        MappingProxyType(manifests_by_id),
        MappingProxyType(
            {
                domain: tuple(domain_manifests)
                for domain, domain_manifests in manifests_by_domain.items()
            }
        ),
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
        summary="Read device state and trigger non-critical routines with explicit confirmation.",
        required_secrets=(
            SecretReference("smart_home_endpoint", "TWINR_SMART_HOME_ENDPOINT"),
            SecretReference("smart_home_token", "TWINR_SMART_HOME_TOKEN"),
        ),
        operations=(
            IntegrationOperation(
                operation_id="read_device_state",
                label="Read device state",
                action=IntegrationAction.QUERY,
                summary="Read current state for lights, temperature, or safe appliances.",
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
                ),
            ),
        ),
        notes=(
            "Critical actuations such as door unlock or alarm disarm are intentionally excluded.",
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
    # AUDIT-FIX(#1): Crash safely at startup on unsafe registry data instead of serving dangerous defaults.
    raise RuntimeError(f"Invalid builtin integration manifest registry: {exc}") from exc

_MANIFESTS_BY_ID, _MANIFESTS_BY_DOMAIN = _build_manifest_indexes(
    _CANONICAL_BUILTIN_MANIFESTS
)


def __getattr__(name: str) -> object:
    if name == "BUILTIN_MANIFESTS":
        # AUDIT-FIX(#2): Preserve the old export name while returning a fresh immutable-by-convention snapshot.
        return _clone_manifests(_CANONICAL_BUILTIN_MANIFESTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # AUDIT-FIX(#2): Keep BUILTIN_MANIFESTS discoverable even though it is now served via __getattr__.
    return sorted(set(globals()) | {"BUILTIN_MANIFESTS"})


def builtin_manifests() -> tuple[IntegrationManifest, ...]:
    # AUDIT-FIX(#2): Always return clones so one caller cannot alter another caller's view of the catalog.
    return _clone_manifests(_CANONICAL_BUILTIN_MANIFESTS)


def manifest_for_id(integration_id: str) -> IntegrationManifest | None:
    manifest = _MANIFESTS_BY_ID.get(integration_id)
    if manifest is None:
        return None
    # AUDIT-FIX(#2): Return a clone of the canonical manifest to prevent shared-state mutation.
    return _clone_manifest(manifest)


def manifests_for_domain(domain: IntegrationDomain) -> tuple[IntegrationManifest, ...]:
    manifests = _MANIFESTS_BY_DOMAIN.get(domain, ())
    # AUDIT-FIX(#2): Read from the immutable domain index and clone results before returning them.
    return _clone_manifests(manifests)