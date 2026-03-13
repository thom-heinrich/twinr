from __future__ import annotations

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
    SecretStorage,
)


BUILTIN_MANIFESTS = (
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

_MANIFESTS_BY_ID = {manifest.integration_id: manifest for manifest in BUILTIN_MANIFESTS}


def builtin_manifests() -> tuple[IntegrationManifest, ...]:
    return BUILTIN_MANIFESTS


def manifest_for_id(integration_id: str) -> IntegrationManifest | None:
    return _MANIFESTS_BY_ID.get(integration_id)


def manifests_for_domain(domain: IntegrationDomain) -> tuple[IntegrationManifest, ...]:
    return tuple(manifest for manifest in BUILTIN_MANIFESTS if manifest.domain == domain)
