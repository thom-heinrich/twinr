from __future__ import annotations

from twinr.integrations import ManagedIntegrationConfig, validate_calendar_source
from twinr.web.contracts import IntegrationOverviewRow, SettingsSection
from twinr.web.forms import _select_field, _text_field, _textarea_field
from twinr.web.store import FileBackedSetting
from twinr.web.viewmodels_common import (
    _BOOL_OPTIONS,
    _CALENDAR_SOURCE_OPTIONS,
    _EMAIL_PROFILE_OPTIONS,
    _EMAIL_SECRET_KEY,
    _credential_state_label,
)


def _integration_overview_rows(
    readiness_items,
) -> tuple[IntegrationOverviewRow, ...]:
    return tuple(
        IntegrationOverviewRow(
            label=item.label,
            status=item.status,
            summary=item.summary,
            detail=item.detail,
        )
        for item in readiness_items
    )


def _email_integration_sections(
    record: ManagedIntegrationConfig,
    env_values: dict[str, str],
) -> tuple[SettingsSection, ...]:
    values = dict(record.settings)
    profile = values.get("profile", "gmail")
    account_email = values.get("account_email", "")
    gmail_default = profile == "gmail"
    return (
        SettingsSection(
            title="Email",
            description="Set up a mailbox connection for reading mail summaries and preparing approved replies.",
            fields=(
                _select_field(
                    "enabled",
                    "Email enabled",
                    values,
                    _BOOL_OPTIONS,
                    "true" if record.enabled else "false",
                    tooltip_text="Enable only after the account and password are configured.",
                ),
                _select_field(
                    "profile",
                    "Profile",
                    values,
                    _EMAIL_PROFILE_OPTIONS,
                    profile,
                    tooltip_text="Gmail pre-fills the standard IMAP/SMTP defaults. Generic keeps everything manual.",
                ),
                _text_field(
                    "account_email",
                    "Account email",
                    values,
                    account_email,
                    placeholder="name@gmail.com",
                    tooltip_text="The mailbox address Twinr will read from and usually also send from.",
                ),
                _text_field(
                    "from_address",
                    "From address",
                    values,
                    values.get("from_address", account_email),
                    placeholder="name@gmail.com",
                    tooltip_text="Outgoing sender address. Leave it equal to the account unless you know you need a different sender.",
                ),
                FileBackedSetting(
                    key=_EMAIL_SECRET_KEY,
                    label="App password",
                    value="",
                    help_text=(
                        f"Credential state: {_credential_state_label(env_values.get(_EMAIL_SECRET_KEY))}. "
                        "Leave blank to keep it unchanged."
                    ),
                    tooltip_text="For Gmail use a Google app password, not the normal Google account password.",
                    input_type="password",
                    placeholder="16-character app password",
                    secret=True,
                ),
            ),
        ),
        SettingsSection(
            title="Mailbox transport",
            description="These values are stored locally for the future live adapter wiring. Gmail works with the defaults shown here.",
            fields=(
                _text_field(
                    "imap_host",
                    "IMAP host",
                    values,
                    values.get("imap_host", "imap.gmail.com" if gmail_default else ""),
                    placeholder="imap.gmail.com",
                    tooltip_text="Incoming mail server hostname.",
                ),
                _text_field(
                    "imap_port",
                    "IMAP port",
                    values,
                    values.get("imap_port", "993" if gmail_default else ""),
                    tooltip_text="Incoming mail server port. Gmail uses 993.",
                ),
                _text_field(
                    "imap_mailbox",
                    "Mailbox",
                    values,
                    values.get("imap_mailbox", "INBOX"),
                    tooltip_text="Mailbox folder Twinr should read from.",
                ),
                _text_field(
                    "smtp_host",
                    "SMTP host",
                    values,
                    values.get("smtp_host", "smtp.gmail.com" if gmail_default else ""),
                    placeholder="smtp.gmail.com",
                    tooltip_text="Outgoing mail server hostname.",
                ),
                _text_field(
                    "smtp_port",
                    "SMTP port",
                    values,
                    values.get("smtp_port", "587" if gmail_default else ""),
                    tooltip_text="Outgoing mail server port. Gmail uses 587 with STARTTLS.",
                ),
            ),
        ),
        SettingsSection(
            title="Guardrails",
            description="Reads are open by default. Drafts and sends still need explicit approval in Twinr. The two strict toggles below are optional extra fences.",
            fields=(
                _select_field(
                    "unread_only_default",
                    "Read unread only",
                    values,
                    _BOOL_OPTIONS,
                    values.get("unread_only_default", "true"),
                    tooltip_text="When enabled, Twinr prefers unread mail for summaries by default.",
                ),
                _select_field(
                    "restrict_reads_to_known_senders",
                    "Restrict reads to known senders",
                    values,
                    _BOOL_OPTIONS,
                    values.get("restrict_reads_to_known_senders", "false"),
                    tooltip_text="Optional extra fence. Leave off if Twinr should summarize any mailbox sender.",
                ),
                _select_field(
                    "restrict_recipients_to_known_contacts",
                    "Restrict send to known contacts",
                    values,
                    _BOOL_OPTIONS,
                    values.get("restrict_recipients_to_known_contacts", "false"),
                    tooltip_text="Optional extra fence. Leave off if explicit approval should be enough for sending.",
                ),
                _textarea_field(
                    "known_contacts_text",
                    "Known contacts",
                    values,
                    values.get("known_contacts_text", ""),
                    placeholder="Anna <anna@example.com>\nDoctor <doctor@example.com>",
                    tooltip_text="Optional future contact hints. One contact per line.",
                    rows=5,
                ),
            ),
        ),
    )


def _calendar_integration_sections(record: ManagedIntegrationConfig) -> tuple[SettingsSection, ...]:
    values = dict(record.settings)
    return (
        SettingsSection(
            title="Calendar",
            description="Read-only agenda setup for day plans, appointment summaries, and later reminder synchronization.",
            fields=(
                _select_field(
                    "enabled",
                    "Calendar enabled",
                    values,
                    _BOOL_OPTIONS,
                    "true" if record.enabled else "false",
                    tooltip_text="Enable only when a local ICS file or feed URL is configured.",
                ),
                _select_field(
                    "source_kind",
                    "Source type",
                    values,
                    _CALENDAR_SOURCE_OPTIONS,
                    values.get("source_kind", "ics_file"),
                    tooltip_text="Phase 1 uses a simple ICS file or ICS feed only.",
                ),
                _text_field(
                    "source_value",
                    "ICS path or URL",
                    values,
                    values.get("source_value", ""),
                    placeholder="state/calendar.ics or https://...",
                    tooltip_text="Relative file paths are resolved from the Twinr project root.",
                ),
                _text_field(
                    "timezone",
                    "Timezone",
                    values,
                    values.get("timezone", "Europe/Berlin"),
                    placeholder="Europe/Berlin",
                    tooltip_text="Used for all-day events and local agenda rendering.",
                ),
                _text_field(
                    "default_upcoming_days",
                    "Upcoming days",
                    values,
                    values.get("default_upcoming_days", "7"),
                    tooltip_text="Default look-ahead window for upcoming agenda summaries.",
                ),
                _text_field(
                    "max_events",
                    "Max events",
                    values,
                    values.get("max_events", "12"),
                    tooltip_text="Upper bound for one agenda readout so the device stays short and readable.",
                ),
            ),
        ),
    )


def _build_email_integration_record(
    form: dict[str, str],
    env_values: dict[str, str],
) -> tuple[ManagedIntegrationConfig, dict[str, str]]:
    enabled = form.get("enabled", "false") == "true"
    profile = (form.get("profile", "gmail") or "gmail").strip()
    account_email = form.get("account_email", "").strip()
    from_address = form.get("from_address", "").strip() or account_email
    imap_host = form.get("imap_host", "").strip() or ("imap.gmail.com" if profile == "gmail" else "")
    imap_port = form.get("imap_port", "").strip() or ("993" if profile == "gmail" else "")
    imap_mailbox = form.get("imap_mailbox", "").strip() or "INBOX"
    smtp_host = form.get("smtp_host", "").strip() or ("smtp.gmail.com" if profile == "gmail" else "")
    smtp_port = form.get("smtp_port", "").strip() or ("587" if profile == "gmail" else "")
    if enabled and not account_email:
        raise ValueError("Email account address is required when email is enabled.")
    if enabled and not (form.get(_EMAIL_SECRET_KEY, "").strip() or env_values.get(_EMAIL_SECRET_KEY)):
        raise ValueError("Email app password is required when email is enabled.")

    env_updates: dict[str, str] = {}
    secret_value = form.get(_EMAIL_SECRET_KEY, "").strip()
    if secret_value:
        env_updates[_EMAIL_SECRET_KEY] = secret_value

    record = ManagedIntegrationConfig(
        integration_id="email_mailbox",
        enabled=enabled,
        settings={
            "profile": profile,
            "account_email": account_email,
            "from_address": from_address,
            "imap_host": imap_host,
            "imap_port": imap_port,
            "imap_mailbox": imap_mailbox,
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
            "unread_only_default": "true" if form.get("unread_only_default", "true") == "true" else "false",
            "restrict_reads_to_known_senders": (
                "true" if form.get("restrict_reads_to_known_senders", "false") == "true" else "false"
            ),
            "restrict_recipients_to_known_contacts": (
                "true" if form.get("restrict_recipients_to_known_contacts", "false") == "true" else "false"
            ),
            "known_contacts_text": form.get("known_contacts_text", "").strip(),
        },
    )
    return record, env_updates


def _build_calendar_integration_record(form: dict[str, str]) -> tuple[ManagedIntegrationConfig, dict[str, str]]:
    enabled = form.get("enabled", "false") == "true"
    source_kind = (form.get("source_kind", "ics_file") or "ics_file").strip()
    source_value = form.get("source_value", "").strip()
    if enabled and not source_value:
        raise ValueError("Calendar source path or URL is required when calendar is enabled.")
    if enabled:
        validate_calendar_source(source_kind=source_kind, source_value=source_value)

    record = ManagedIntegrationConfig(
        integration_id="calendar_agenda",
        enabled=enabled,
        settings={
            "source_kind": source_kind,
            "source_value": source_value,
            "timezone": form.get("timezone", "").strip() or "Europe/Berlin",
            "default_upcoming_days": form.get("default_upcoming_days", "").strip() or "7",
            "max_events": form.get("max_events", "").strip() or "12",
        },
    )
    return record, {}
