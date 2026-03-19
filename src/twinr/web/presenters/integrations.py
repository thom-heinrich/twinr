"""Build Twinr integration presenter models and validated form records.

This module keeps the web layer thin by turning managed integration state into
``SettingsSection`` view models and by validating submitted email/calendar
forms before they become persisted integration records.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable, Mapping
from email.utils import parseaddr
from pathlib import Path, PurePath
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.base_agent import TwinrConfig
from twinr.integrations import ManagedIntegrationConfig, validate_calendar_source
from twinr.web.support.channel_onboarding import ChannelPairingSnapshot
from twinr.web.support.contracts import IntegrationOverviewRow, SettingsSection, WizardCheckRow
from twinr.web.support.forms import _select_field, _text_field, _textarea_field
from twinr.web.support.store import FileBackedSetting
from twinr.web.support.whatsapp import canonicalize_whatsapp_allow_from, probe_whatsapp_runtime
from twinr.web.presenters.common import (
    _BOOL_OPTIONS,
    _CALENDAR_SOURCE_OPTIONS,
    _EMAIL_PROFILE_OPTIONS,
    _EMAIL_SECRET_KEY,
    _credential_state_label,
)

_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off", ""})
_EMAIL_LOCAL_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+$")
_HOST_LABEL_RE = re.compile(r"^[A-Za-z0-9-]+$")
_MAX_KNOWN_CONTACTS_TEXT_LENGTH = 8192
_MAX_MAILBOX_LENGTH = 255


def _coerce_text(value: object, default: str = "") -> str:
    """Convert optional values into strings while preserving a fallback."""

    return default if value is None else (value if isinstance(value, str) else str(value))


def _string_settings(settings: object) -> dict[str, str]:
    """Snapshot mapping-like integration settings into a string-only dictionary."""

    # AUDIT-FIX(#6): Degrade safely when persisted settings contain non-string or malformed values.
    if not settings:
        return {}
    try:
        items = dict(settings).items()
    except (TypeError, ValueError):
        return {}
    return {str(key): _coerce_text(value) for key, value in items}


def _reject_control_characters(value: str, field_label: str, *, allow_newlines: bool = False) -> str:
    """Reject unsupported control characters in form-backed text fields."""

    # AUDIT-FIX(#1): Block control characters so env/file persistence cannot be corrupted via CR/LF injection.
    for char in value:
        codepoint = ord(char)
        if char in "\r\n" and allow_newlines:
            continue
        if codepoint < 32 or codepoint == 127:
            raise ValueError(f"{field_label} contains unsupported control characters.")
    return value


def _normalize_multiline_text(value: str, field_label: str, *, max_length: int) -> str:
    """Normalize multiline text input before it is persisted."""

    # AUDIT-FIX(#4): Normalize multiline free-text so later parsers do not receive hostile or oversized payloads.
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    _reject_control_characters(normalized, field_label, allow_newlines=True)
    if len(normalized) > max_length:
        raise ValueError(f"{field_label} is too long.")
    return normalized


def _parse_bool_choice(raw: object, field_label: str, *, default: bool = False) -> bool:
    """Parse a boolean form field from common text encodings."""

    # AUDIT-FIX(#3): Accept common boolean encodings and reject garbage instead of silently flipping to false.
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    value = _coerce_text(raw).strip().lower()
    if not value:
        return default
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"{field_label} must be true or false.")


def _safe_bool_string(raw: object, *, default: bool) -> str:
    """Render a stored boolean-like value as canonical ``true`` or ``false``."""

    # AUDIT-FIX(#6): Keep the settings page renderable even when legacy boolean values were stored badly.
    try:
        return "true" if _parse_bool_choice(raw, "Boolean value", default=default) else "false"
    except ValueError:
        return "true" if default else "false"


def _option_value_map(options: object, *, fallback: Iterable[str] = ()) -> dict[str, str]:
    """Build a case-insensitive map of canonical option values."""

    value_map: dict[str, str] = {}

    def register(candidate: object) -> None:
        text = _coerce_text(candidate).strip()
        if text:
            value_map.setdefault(text.casefold(), text)

    for candidate in fallback:
        register(candidate)

    if isinstance(options, Mapping):
        for key, value in options.items():
            register(key)
            if isinstance(value, (str, int, bool)):
                register(value)
            elif hasattr(value, "value"):
                register(getattr(value, "value"))
        return value_map

    if isinstance(options, Iterable) and not isinstance(options, (str, bytes)):
        for option in options:
            if isinstance(option, (str, int, bool)):
                register(option)
                continue
            if hasattr(option, "value"):
                register(getattr(option, "value"))
            if isinstance(option, Mapping):
                for key in ("value", "key", "id", "name"):
                    if key in option:
                        register(option[key])
            elif isinstance(option, (tuple, list)):
                for part in option[:2]:
                    if isinstance(part, (str, int, bool)):
                        register(part)
                    elif hasattr(part, "value"):
                        register(getattr(part, "value"))
    return value_map


def _normalize_choice(
    raw: object,
    field_label: str,
    *,
    default: str,
    options: object,
    fallback: Iterable[str] = (),
) -> str:
    """Validate and canonicalize one select-backed form value."""

    # AUDIT-FIX(#4): Reject unsupported choice values before they become persisted broken state.
    value = _coerce_text(raw, default).strip() or default
    option_map = _option_value_map(options, fallback=(default, *tuple(fallback)))
    canonical = option_map.get(value.casefold())
    if canonical is None:
        raise ValueError(f"{field_label} has an unsupported value.")
    return canonical


def _safe_normalize_choice(
    raw: object,
    *,
    default: str,
    options: object,
    fallback: Iterable[str] = (),
) -> str:
    """Return a canonical choice or the safe default for stale stored data."""

    # AUDIT-FIX(#6): Fall back to safe defaults while rendering stale or corrupted persisted choices.
    try:
        return _normalize_choice(
            raw,
            "Choice",
            default=default,
            options=options,
            fallback=fallback,
        )
    except ValueError:
        return default


def _normalize_secret_value(raw: object, *, profile: str) -> str:
    """Normalize the email secret field before it is written to ``.env``."""

    # AUDIT-FIX(#1): Validate and normalize secrets before passing them to env persistence.
    secret = _coerce_text(raw).strip()
    _reject_control_characters(secret, "Email app password")
    if profile == "gmail":
        secret = "".join(secret.split())
    return secret


def _normalize_persisted_secret(raw: object) -> str:
    """Normalize a stored secret only enough to derive presence state safely."""

    # AUDIT-FIX(#6): Ignore malformed stored secrets when computing UI state so the page stays truthful.
    secret = _coerce_text(raw).strip()
    try:
        return _reject_control_characters(secret, "Email app password")
    except ValueError:
        return ""


def _validate_host(value: str, field_label: str) -> str:
    """Validate a hostname or IP address without scheme, path, or port."""

    # AUDIT-FIX(#4): Validate mail server hosts early so later network adapters do not fail unpredictably.
    host = value.strip()
    _reject_control_characters(host, field_label)
    if not host:
        raise ValueError(f"{field_label} is required.")
    if "://" in host or "/" in host or "?" in host or "#" in host or "@" in host:
        raise ValueError(f"{field_label} must be a hostname or IP address without a scheme or path.")
    if host.startswith("["):
        if not host.endswith("]"):
            raise ValueError(f"{field_label} must be a valid hostname or IP address.")
        literal = host[1:-1]
        try:
            ipaddress.ip_address(literal)
        except ValueError as exc:
            raise ValueError(f"{field_label} must be a valid hostname or IP address.") from exc
        return host
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    if ":" in host:
        raise ValueError(f"{field_label} must not include a port.")
    if len(host) > 253:
        raise ValueError(f"{field_label} is too long.")
    labels = host.split(".")
    if any(not label or len(label) > 63 for label in labels):
        raise ValueError(f"{field_label} must be a valid hostname or IP address.")
    for label in labels:
        if not _HOST_LABEL_RE.fullmatch(label) or label.startswith("-") or label.endswith("-"):
            raise ValueError(f"{field_label} must be a valid hostname or IP address.")
    return host.lower()


def _validate_port(value: str, field_label: str) -> str:
    """Validate a TCP port and return it as a canonical string."""

    # AUDIT-FIX(#4): Force transport ports into a canonical safe range before persisting them.
    port_text = value.strip()
    _reject_control_characters(port_text, field_label)
    if not port_text.isdigit():
        raise ValueError(f"{field_label} must be a whole number between 1 and 65535.")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError(f"{field_label} must be a whole number between 1 and 65535.")
    return str(port)


def _validate_bounded_int(value: str, field_label: str, *, minimum: int, maximum: int) -> str:
    """Validate a bounded integer setting and return the canonical string form."""

    # AUDIT-FIX(#5): Bound calendar fan-out values so agenda rendering stays predictable on the RPi.
    int_text = value.strip()
    _reject_control_characters(int_text, field_label)
    if not int_text.isdigit():
        raise ValueError(f"{field_label} must be a whole number between {minimum} and {maximum}.")
    number = int(int_text)
    if number < minimum or number > maximum:
        raise ValueError(f"{field_label} must be a whole number between {minimum} and {maximum}.")
    return str(number)


def _validate_email_address(value: str, field_label: str) -> str:
    """Validate a plain mailbox address without display-name syntax."""

    # AUDIT-FIX(#4): Reject malformed addresses and header-injection characters before later SMTP use.
    address = value.strip()
    _reject_control_characters(address, field_label)
    display_name, parsed_email = parseaddr(address)
    if display_name or parsed_email != address:
        raise ValueError(f"{field_label} must be a plain email address like name@example.com.")
    if address.count("@") != 1:
        raise ValueError(f"{field_label} must be a plain email address like name@example.com.")
    local_part, domain = address.rsplit("@", 1)
    if not local_part or not domain or local_part.startswith(".") or local_part.endswith(".") or ".." in local_part:
        raise ValueError(f"{field_label} must be a plain email address like name@example.com.")
    if not _EMAIL_LOCAL_RE.fullmatch(local_part):
        raise ValueError(f"{field_label} must be a plain email address like name@example.com.")
    _validate_host(domain, f"{field_label} domain")
    return address


def _validate_mailbox(value: str) -> str:
    """Validate an IMAP mailbox name for safe persistence."""

    # AUDIT-FIX(#4): Reject mailbox names with control characters that can break downstream IMAP calls.
    mailbox = value.strip()
    _reject_control_characters(mailbox, "Mailbox")
    if not mailbox:
        raise ValueError("Mailbox is required when email is enabled.")
    if len(mailbox) > _MAX_MAILBOX_LENGTH:
        raise ValueError("Mailbox is too long.")
    return mailbox


def _validate_timezone(value: str) -> str:
    """Validate and return one IANA timezone identifier."""

    # AUDIT-FIX(#5): Store only valid IANA timezones so DST/all-day agenda rendering remains stable.
    timezone = value.strip()
    _reject_control_characters(timezone, "Timezone")
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("Timezone must be a valid IANA timezone like Europe/Berlin.") from exc
    return timezone


def _looks_like_url(value: str) -> bool:
    """Return whether a calendar source string looks like an HTTP(S) URL."""

    return value.casefold().startswith(("http://", "https://"))


def _is_local_calendar_source(source_kind: str, source_value: str) -> bool:
    """Return whether the calendar source should be treated as a local path."""

    if _looks_like_url(source_value):
        return False
    lowered_kind = source_kind.casefold()
    if "url" in lowered_kind or "feed" in lowered_kind:
        return False
    return True


def _validate_calendar_local_path(value: str) -> str:
    """Validate a project-relative local ICS path without traversal or symlinks."""

    # AUDIT-FIX(#2): Constrain local ICS paths to project-relative, non-symlinked locations.
    local_path = value.strip()
    _reject_control_characters(local_path, "Calendar source path")
    if Path(local_path).is_absolute():
        raise ValueError("Calendar ICS file must be a relative path inside the Twinr project folder.")
    pure_path = PurePath(local_path)
    if any(part == ".." for part in pure_path.parts):
        raise ValueError("Calendar ICS file must stay inside the Twinr project folder.")
    current_path = Path.cwd()
    for part in pure_path.parts:
        current_path = current_path / part
        try:
            if current_path.is_symlink():
                raise ValueError("Calendar ICS file must not use symlinks.")
        except OSError as exc:
            raise ValueError("Calendar ICS file path could not be checked safely.") from exc
    return Path(*pure_path.parts).as_posix()


def _normalize_calendar_source_value(source_kind: str, raw_value: object) -> str:
    """Normalize a calendar source value before source-specific validation."""

    # AUDIT-FIX(#2): Normalize calendar sources before validation so disabled configs cannot stash dangerous paths.
    source_value = _coerce_text(raw_value).strip()
    _reject_control_characters(source_value, "Calendar source")
    if not source_value:
        return ""
    if _is_local_calendar_source(source_kind, source_value):
        return _validate_calendar_local_path(source_value)
    return source_value


def _integration_overview_rows(
    readiness_items: Iterable[Any],
) -> tuple[IntegrationOverviewRow, ...]:
    """Convert readiness objects into dashboard overview rows."""

    return tuple(
        IntegrationOverviewRow(
            label=item.label,
            status=item.status,
            summary=item.summary,
            detail=item.detail,
        )
        for item in readiness_items
    )


def _whatsapp_integration_context(
    config: TwinrConfig,
    env_values: Mapping[str, object],
    *,
    env_path: str | Path,
    pairing_snapshot: ChannelPairingSnapshot | None = None,
) -> dict[str, object]:
    """Build the compact WhatsApp setup summary shown on ``/integrations``."""

    probe = probe_whatsapp_runtime(config, env_path=Path(env_path))
    snapshot = pairing_snapshot if isinstance(pairing_snapshot, ChannelPairingSnapshot) else ChannelPairingSnapshot.initial("whatsapp")
    allow_from_label = _display_whatsapp_allow_from(env_values.get("TWINR_WHATSAPP_ALLOW_FROM"))
    allow_from_ready = allow_from_label != "Not set yet"
    guardrails_ready = bool(config.whatsapp_self_chat_mode) and not bool(config.whatsapp_groups_enabled)
    runtime_ready = probe.node_ready and probe.worker_ready
    linked_device_summary, linked_device_detail, linked_device_status = _whatsapp_linked_device_display(
        probe,
        snapshot,
        pairing_ready=(probe.paired or snapshot.paired) and not snapshot.auth_repair_needed,
    )

    if snapshot.auth_repair_needed:
        status = "fail"
        status_label = "Repair needed"
        detail = "The saved WhatsApp session needs repair. Open the wizard and start a fresh QR pairing."
    elif snapshot.running and snapshot.qr_needed:
        status = "warn"
        status_label = "QR needed"
        detail = "A pairing window is open right now. Open the wizard to scan the QR directly in the portal from WhatsApp Linked devices."
    elif snapshot.running:
        status = "warn"
        status_label = "Pairing live"
        detail = "Twinr is opening the temporary WhatsApp worker and waiting for the next pairing status update."
    elif allow_from_ready and guardrails_ready and runtime_ready and linked_device_status == "ok":
        status = "ok"
        status_label = "Ready"
        detail = "Twinr is ready for one internal self-chat with your own WhatsApp number."
    elif allow_from_ready and guardrails_ready and runtime_ready:
        status = "warn"
        status_label = "Needs pairing"
        detail = "The saved number and worker runtime are ready. Open the wizard once to finish QR pairing."
    else:
        status = "blocked"
        status_label = "Needs setup"
        detail = "Use the wizard to save one allowed chat, keep self-chat mode on, block groups, and prepare the worker runtime."

    if guardrails_ready:
        guardrail_summary = "Self-chat on, groups blocked"
        guardrail_detail = "Twinr stays limited to your own direct chat for this internal test."
        guardrail_status = "ok"
    else:
        guardrail_summary = "Needs fix"
        guardrail_detail = "The wizard should keep self-chat mode on and group chats blocked before you pair."
        guardrail_status = "fail" if allow_from_ready else "warn"

    runtime_summary = "Ready" if runtime_ready else "Needs check"
    runtime_detail = f"{probe.node_detail} {probe.worker_detail}".strip()

    return {
        "title": "WhatsApp self-chat",
        "status": status,
        "status_label": status_label,
        "detail": detail,
        "action_href": "/connect/whatsapp",
        "action_label": "Open WhatsApp wizard",
        "checks": (
            WizardCheckRow(
                label="Allowed chat",
                summary=allow_from_label,
                detail="Twinr accepts only this one direct-message sender.",
                status="ok" if allow_from_ready else "warn",
            ),
            WizardCheckRow(
                label="Guardrails",
                summary=guardrail_summary,
                detail=guardrail_detail,
                status=guardrail_status,
            ),
            WizardCheckRow(
                label="Worker runtime",
                summary=runtime_summary,
                detail=runtime_detail,
                status="ok" if runtime_ready else "warn",
            ),
            WizardCheckRow(
                label="Linked device",
                summary=linked_device_summary,
                detail=linked_device_detail,
                status=linked_device_status,
            ),
        ),
    }


def _display_whatsapp_allow_from(raw_value: object) -> str:
    """Return a safe operator-facing label for the saved WhatsApp number."""

    normalized = _coerce_text(raw_value).strip()
    if not normalized:
        return "Not set yet"
    try:
        return canonicalize_whatsapp_allow_from(normalized)
    except (TypeError, ValueError):
        return normalized


def _whatsapp_linked_device_display(
    probe: Any,
    snapshot: ChannelPairingSnapshot,
    *,
    pairing_ready: bool,
) -> tuple[str, str, str]:
    """Summarize the current linked-device state for the integrations page."""

    if snapshot.auth_repair_needed:
        return "Repair needed", snapshot.detail, "fail"
    if snapshot.running and snapshot.qr_needed:
        return "QR needed", snapshot.detail, "warn"
    if snapshot.running:
        return snapshot.summary, snapshot.detail, "warn"
    if pairing_ready and probe.paired:
        return "Stored", probe.pair_detail, "ok"
    if pairing_ready:
        return "Paired", snapshot.detail, "ok"
    if probe.auth_dir_exists:
        return "Missing", probe.pair_detail, "warn"
    return "Missing", probe.pair_detail, "muted"


def _email_integration_sections(
    record: ManagedIntegrationConfig,
    env_values: Mapping[str, object],
) -> tuple[SettingsSection, ...]:
    """Build the email integration form sections for the dashboard."""

    values = _string_settings(getattr(record, "settings", {}))
    profile = _safe_normalize_choice(
        values.get("profile", "gmail"),
        default="gmail",
        options=_EMAIL_PROFILE_OPTIONS,
        fallback=("generic",),
    )
    account_email = values.get("account_email", "").strip()
    values["profile"] = profile  # AUDIT-FIX(#6): Canonicalize select-backed state before rendering the form.
    values["account_email"] = account_email  # AUDIT-FIX(#6): Render trimmed persisted values consistently.
    values["from_address"] = values.get("from_address", "").strip() or account_email  # AUDIT-FIX(#6): Avoid blank sender defaults.
    values["unread_only_default"] = _safe_bool_string(values.get("unread_only_default", "true"), default=True)
    values["restrict_reads_to_known_senders"] = _safe_bool_string(
        values.get("restrict_reads_to_known_senders", "false"),
        default=False,
    )
    values["restrict_recipients_to_known_contacts"] = _safe_bool_string(
        values.get("restrict_recipients_to_known_contacts", "false"),
        default=False,
    )
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
                    values["from_address"],
                    placeholder="name@gmail.com",
                    tooltip_text="Outgoing sender address. Leave it equal to the account unless you know you need a different sender.",
                ),
                FileBackedSetting(
                    key=_EMAIL_SECRET_KEY,
                    label="App password",
                    value="",
                    help_text=(
                        # AUDIT-FIX(#6): Derive credential state from a sanitized persisted secret so the UI stays accurate.
                        f"Credential state: {_credential_state_label(_normalize_persisted_secret(env_values.get(_EMAIL_SECRET_KEY)))}. "
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
                    values.get("imap_host", "imap.gmail.com" if gmail_default else "").strip(),
                    placeholder="imap.gmail.com",
                    tooltip_text="Incoming mail server hostname.",
                ),
                _text_field(
                    "imap_port",
                    "IMAP port",
                    values,
                    values.get("imap_port", "993" if gmail_default else "").strip(),
                    tooltip_text="Incoming mail server port. Gmail uses 993.",
                ),
                _text_field(
                    "imap_mailbox",
                    "Mailbox",
                    values,
                    values.get("imap_mailbox", "INBOX").strip(),
                    tooltip_text="Mailbox folder Twinr should read from.",
                ),
                _text_field(
                    "smtp_host",
                    "SMTP host",
                    values,
                    values.get("smtp_host", "smtp.gmail.com" if gmail_default else "").strip(),
                    placeholder="smtp.gmail.com",
                    tooltip_text="Outgoing mail server hostname.",
                ),
                _text_field(
                    "smtp_port",
                    "SMTP port",
                    values,
                    values.get("smtp_port", "587" if gmail_default else "").strip(),
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
                    values["unread_only_default"],
                    tooltip_text="When enabled, Twinr prefers unread mail for summaries by default.",
                ),
                _select_field(
                    "restrict_reads_to_known_senders",
                    "Restrict reads to known senders",
                    values,
                    _BOOL_OPTIONS,
                    values["restrict_reads_to_known_senders"],
                    tooltip_text="Optional extra fence. Leave off if Twinr should summarize any mailbox sender.",
                ),
                _select_field(
                    "restrict_recipients_to_known_contacts",
                    "Restrict send to known contacts",
                    values,
                    _BOOL_OPTIONS,
                    values["restrict_recipients_to_known_contacts"],
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
    """Build the calendar integration form section for the dashboard."""

    values = _string_settings(getattr(record, "settings", {}))
    source_kind = _safe_normalize_choice(
        values.get("source_kind", "ics_file"),
        default="ics_file",
        options=_CALENDAR_SOURCE_OPTIONS,
        fallback=("ics_url", "ics_feed"),
    )
    values["source_kind"] = source_kind  # AUDIT-FIX(#6): Canonicalize source type before rendering the select field.
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
                    source_kind,
                    tooltip_text="Phase 1 uses a simple ICS file or ICS feed only.",
                ),
                _text_field(
                    "source_value",
                    "ICS path or URL",
                    values,
                    values.get("source_value", "").strip(),
                    placeholder="state/calendar.ics or https://...",
                    tooltip_text="Relative file paths are resolved from the Twinr project root.",
                ),
                _text_field(
                    "timezone",
                    "Timezone",
                    values,
                    values.get("timezone", "Europe/Berlin").strip(),
                    placeholder="Europe/Berlin",
                    tooltip_text="Used for all-day events and local agenda rendering.",
                ),
                _text_field(
                    "default_upcoming_days",
                    "Upcoming days",
                    values,
                    values.get("default_upcoming_days", "7").strip(),
                    tooltip_text="Default look-ahead window for upcoming agenda summaries.",
                ),
                _text_field(
                    "max_events",
                    "Max events",
                    values,
                    values.get("max_events", "12").strip(),
                    tooltip_text="Upper bound for one agenda readout so the device stays short and readable.",
                ),
            ),
        ),
    )


def _build_email_integration_record(
    form: Mapping[str, object],
    env_values: Mapping[str, object],
) -> tuple[ManagedIntegrationConfig, dict[str, str]]:
    """Validate posted email form data into a managed integration record.

    Args:
        form: Submitted web form values.
        env_values: Current env-backed secret values used to keep stored
            credentials when the form leaves the secret blank.

    Returns:
        Tuple of validated integration record and sanitized env updates.

    Raises:
        ValueError: If the submitted data is incomplete or unsafe to persist.
    """

    enabled = _parse_bool_choice(form.get("enabled", "false"), "Email enabled", default=False)  # AUDIT-FIX(#3): Parse booleans strictly.
    profile = _normalize_choice(  # AUDIT-FIX(#4): Accept only supported profile values.
        form.get("profile", "gmail"),
        "Profile",
        default="gmail",
        options=_EMAIL_PROFILE_OPTIONS,
        fallback=("generic",),
    )
    account_email = _coerce_text(form.get("account_email", "")).strip()
    if account_email:
        account_email = _validate_email_address(account_email, "Account email")  # AUDIT-FIX(#4): Validate addresses before persisting.
    from_address = _coerce_text(form.get("from_address", "")).strip() or account_email
    if from_address:
        from_address = _validate_email_address(from_address, "From address")  # AUDIT-FIX(#4): Prevent malformed sender addresses.

    imap_host = _coerce_text(form.get("imap_host", "")).strip() or ("imap.gmail.com" if profile == "gmail" else "")
    smtp_host = _coerce_text(form.get("smtp_host", "")).strip() or ("smtp.gmail.com" if profile == "gmail" else "")
    imap_port = _coerce_text(form.get("imap_port", "")).strip() or ("993" if profile == "gmail" else "")
    smtp_port = _coerce_text(form.get("smtp_port", "")).strip() or ("587" if profile == "gmail" else "")
    imap_mailbox = _coerce_text(form.get("imap_mailbox", "")).strip() or "INBOX"

    if enabled and not account_email:
        raise ValueError("Email account address is required when email is enabled.")
    if enabled and not imap_host:
        raise ValueError("IMAP host is required when email is enabled.")
    if enabled and not smtp_host:
        raise ValueError("SMTP host is required when email is enabled.")
    if enabled and not imap_port:
        raise ValueError("IMAP port is required when email is enabled.")
    if enabled and not smtp_port:
        raise ValueError("SMTP port is required when email is enabled.")

    if imap_host:
        imap_host = _validate_host(imap_host, "IMAP host")  # AUDIT-FIX(#4): Validate transport hosts early.
    if smtp_host:
        smtp_host = _validate_host(smtp_host, "SMTP host")  # AUDIT-FIX(#4): Validate transport hosts early.
    if imap_port:
        imap_port = _validate_port(imap_port, "IMAP port")  # AUDIT-FIX(#4): Enforce valid numeric port ranges.
    if smtp_port:
        smtp_port = _validate_port(smtp_port, "SMTP port")  # AUDIT-FIX(#4): Enforce valid numeric port ranges.
    imap_mailbox = _validate_mailbox(imap_mailbox)  # AUDIT-FIX(#4): Prevent malformed mailbox names.

    existing_secret = _normalize_persisted_secret(env_values.get(_EMAIL_SECRET_KEY))
    secret_value = _normalize_secret_value(form.get(_EMAIL_SECRET_KEY, ""), profile=profile)  # AUDIT-FIX(#1): Sanitize env-backed secret input.
    if enabled and not (secret_value or existing_secret):
        raise ValueError("Email app password is required when email is enabled.")

    env_updates: dict[str, str] = {}
    if secret_value:
        env_updates[_EMAIL_SECRET_KEY] = secret_value  # AUDIT-FIX(#1): Persist only sanitized secret material.

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
            "unread_only_default": (
                "true"
                if _parse_bool_choice(form.get("unread_only_default", "true"), "Read unread only", default=True)
                else "false"
            ),
            "restrict_reads_to_known_senders": (
                "true"
                if _parse_bool_choice(
                    form.get("restrict_reads_to_known_senders", "false"),
                    "Restrict reads to known senders",
                    default=False,
                )
                else "false"
            ),
            "restrict_recipients_to_known_contacts": (
                "true"
                if _parse_bool_choice(
                    form.get("restrict_recipients_to_known_contacts", "false"),
                    "Restrict send to known contacts",
                    default=False,
                )
                else "false"
            ),
            "known_contacts_text": _normalize_multiline_text(
                _coerce_text(form.get("known_contacts_text", "")),
                "Known contacts",
                max_length=_MAX_KNOWN_CONTACTS_TEXT_LENGTH,
            ),
        },
    )
    return record, env_updates


def _build_calendar_integration_record(form: Mapping[str, object]) -> tuple[ManagedIntegrationConfig, dict[str, str]]:
    """Validate posted calendar form data into a managed integration record.

    Args:
        form: Submitted web form values.

    Returns:
        Tuple of validated calendar integration record and env updates. Calendar
        records do not currently emit env updates, so the second item is empty.

    Raises:
        ValueError: If the submitted data is incomplete or unsafe to persist.
    """

    enabled = _parse_bool_choice(form.get("enabled", "false"), "Calendar enabled", default=False)  # AUDIT-FIX(#3): Parse booleans strictly.
    source_kind = _normalize_choice(  # AUDIT-FIX(#4): Accept only supported calendar source types.
        form.get("source_kind", "ics_file"),
        "Source type",
        default="ics_file",
        options=_CALENDAR_SOURCE_OPTIONS,
        fallback=("ics_url", "ics_feed"),
    )
    source_value = _normalize_calendar_source_value(source_kind, form.get("source_value", ""))  # AUDIT-FIX(#2): Block traversal and unsafe local paths.
    if enabled and not source_value:
        raise ValueError("Calendar source path or URL is required when calendar is enabled.")
    if source_value:
        validate_calendar_source(source_kind=source_kind, source_value=source_value)  # AUDIT-FIX(#2): Validate all non-empty sources, even while disabled.

    record = ManagedIntegrationConfig(
        integration_id="calendar_agenda",
        enabled=enabled,
        settings={
            "source_kind": source_kind,
            "source_value": source_value,
            "timezone": _validate_timezone(_coerce_text(form.get("timezone", "")).strip() or "Europe/Berlin"),  # AUDIT-FIX(#5): Enforce valid IANA timezones.
            "default_upcoming_days": _validate_bounded_int(
                _coerce_text(form.get("default_upcoming_days", "")).strip() or "7",
                "Upcoming days",
                minimum=1,
                maximum=365,
            ),
            "max_events": _validate_bounded_int(
                _coerce_text(form.get("max_events", "")).strip() or "12",
                "Max events",
                minimum=1,
                maximum=100,
            ),
        },
    )
    return record, {}
