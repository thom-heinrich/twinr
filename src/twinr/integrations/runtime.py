"""Assemble managed Twinr integrations from config, env, and provider code.

This module wires the shared integration layer to the built-in email and
calendar packages while keeping setup validation and readiness reporting
centralized.
"""

from __future__ import annotations

import errno
import ipaddress
import os
import socket
import stat
from dataclasses import dataclass
from email.utils import parseaddr
from pathlib import Path
from collections.abc import Mapping
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from zoneinfo import ZoneInfo

from twinr.integrations.calendar import CalendarAdapterSettings, ICSCalendarSource, ReadOnlyCalendarAdapter
from twinr.integrations.catalog import manifest_for_id
from twinr.integrations.email import (
    ApprovedEmailContacts,
    EmailAdapterSettings,
    EmailContact,
    EmailMailboxAdapter,
    IMAPMailboxConfig,
    IMAPMailboxReader,
    SMTPMailSender,
    SMTPMailSenderConfig,
    normalize_email,
)
from twinr.integrations.smarthome import (
    AggregatedSmartHomeProvider,
    RoutedSmartHomeProvider,
    SmartHomeAdapterSettings,
    SmartHomeIntegrationAdapter,
)
from twinr.integrations.smarthome.hue import HueBridgeClient, HueBridgeConfig, HueSmartHomeProvider, build_hue_smart_home_provider
from twinr.integrations.store import ManagedIntegrationConfig, TwinrIntegrationStore

EMAIL_MAILBOX_INTEGRATION_ID = "email_mailbox"
CALENDAR_AGENDA_INTEGRATION_ID = "calendar_agenda"
SMART_HOME_HUB_INTEGRATION_ID = "smart_home_hub"
SMART_HOME_HUE_PROVIDER = "hue"
EMAIL_APP_PASSWORD_ENV_KEY = "TWINR_INTEGRATION_EMAIL_APP_PASSWORD"
HUE_APPLICATION_KEY_ENV_KEY = "TWINR_INTEGRATION_HUE_APPLICATION_KEY"
HUE_ADDITIONAL_BRIDGE_HOSTS_SETTING_KEY = "additional_bridge_hosts"
CALENDAR_ALLOWED_ROOT_ENV_KEY = "TWINR_INTEGRATION_CALENDAR_ALLOWED_ROOT"  # AUDIT-FIX(#1): Optional allowlist root for local ICS files; defaults to the project root.
_MAX_ICS_DOWNLOAD_BYTES = 2 * 1024 * 1024
_MAX_ENV_FILE_BYTES = 64 * 1024  # AUDIT-FIX(#7): Bound .env reads so a damaged or wrong file cannot consume unbounded memory on RPi.
_ICS_URL_FETCH_TIMEOUT_S = 10.0
_MAX_ICS_URL_REDIRECTS = 3  # AUDIT-FIX(#2): Follow only a small number of validated redirects for remote ICS feeds.
_ICS_HTTP_USER_AGENT = "Twinr/0.1"


@dataclass(frozen=True, slots=True)
class IntegrationReadiness:
    """Describe whether one managed integration is ready for runtime use."""

    integration_id: str
    label: str
    status: str
    summary: str
    detail: str
    warnings: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        """Return whether the readiness status represents a usable adapter."""

        return self.status == "ok"


@dataclass(frozen=True, slots=True)
class ManagedIntegrationsRuntime:
    """Bundle the built-in managed adapters plus their readiness state."""

    email_mailbox: EmailMailboxAdapter | None = None
    calendar_agenda: ReadOnlyCalendarAdapter | None = None
    smart_home_hub: SmartHomeIntegrationAdapter | None = None
    readiness: tuple[IntegrationReadiness, ...] = ()

    def readiness_for(self, integration_id: str) -> IntegrationReadiness | None:
        """Return readiness metadata for one integration ID."""

        for item in self.readiness:
            if item.integration_id == integration_id:
                return item
        return None


class _NoRedirectHandler(HTTPRedirectHandler):
    """Disable implicit redirects so each hop can be revalidated."""

    # AUDIT-FIX(#2): Reject urllib's implicit redirect behavior so every hop is revalidated before any request is sent.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Tell urllib to stop automatic redirect handling for this request."""

        return None


def build_managed_integrations(
    project_root: str | Path,
    *,
    env_path: str | Path | None = None,
    url_text_loader: Callable[[str], str] | None = None,
) -> ManagedIntegrationsRuntime:
    """Build the managed email, calendar, and smart-home adapters for one project root."""

    project_path = Path(project_root).resolve()
    env_values = _read_env_values(_resolve_env_path(project_path, env_path))

    try:
        # AUDIT-FIX(#4): Do not let a damaged file-backed integration store crash the whole runtime during startup.
        store = TwinrIntegrationStore.from_project_root(project_path)
        email_record = store.get(EMAIL_MAILBOX_INTEGRATION_ID)
        calendar_record = store.get(CALENDAR_AGENDA_INTEGRATION_ID)
        smart_home_record = store.get(SMART_HOME_HUB_INTEGRATION_ID)
    except Exception:
        detail = (
            "Integration settings could not be loaded from disk. "
            "Check file permissions and restore the project state if it was damaged."
        )
        return ManagedIntegrationsRuntime(
            readiness=(
                IntegrationReadiness(
                    integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
                    label="Email",
                    status="warn",
                    summary="Unavailable",
                    detail=detail,
                ),
                IntegrationReadiness(
                    integration_id=CALENDAR_AGENDA_INTEGRATION_ID,
                    label="Calendar",
                    status="warn",
                    summary="Unavailable",
                    detail=detail,
                ),
                IntegrationReadiness(
                    integration_id=SMART_HOME_HUB_INTEGRATION_ID,
                    label="Smart Home",
                    status="warn",
                    summary="Unavailable",
                    detail=detail,
                ),
            ),
        )

    email_adapter, email_readiness = _safe_build_email_mailbox_runtime(
        email_record,
        env_values=env_values,
    )
    calendar_adapter, calendar_readiness = _safe_build_calendar_agenda_runtime(
        project_path,
        calendar_record,
        env_values=env_values,
        url_text_loader=url_text_loader or _fetch_ics_url,
    )
    smart_home_adapter, smart_home_readiness = _safe_build_smart_home_hub_runtime(
        smart_home_record,
        env_values=env_values,
    )
    return ManagedIntegrationsRuntime(
        email_mailbox=email_adapter,
        calendar_agenda=calendar_adapter,
        smart_home_hub=smart_home_adapter,
        readiness=(email_readiness, calendar_readiness, smart_home_readiness),
    )


def build_email_mailbox_adapter(
    project_root: str | Path,
    *,
    env_path: str | Path | None = None,
) -> EmailMailboxAdapter | None:
    """Build only the managed email adapter for one project root."""

    return build_managed_integrations(project_root, env_path=env_path).email_mailbox


def build_calendar_agenda_adapter(
    project_root: str | Path,
    *,
    env_path: str | Path | None = None,
    url_text_loader: Callable[[str], str] | None = None,
) -> ReadOnlyCalendarAdapter | None:
    """Build only the managed calendar adapter for one project root."""

    return build_managed_integrations(
        project_root,
        env_path=env_path,
        url_text_loader=url_text_loader,
    ).calendar_agenda


def build_smart_home_hub_adapter(
    project_root: str | Path,
    *,
    env_path: str | Path | None = None,
) -> SmartHomeIntegrationAdapter | None:
    """Build only the managed smart-home adapter for one project root."""

    return build_managed_integrations(project_root, env_path=env_path).smart_home_hub


def hue_application_key_env_key_for_host(host: str) -> str:
    """Return the per-bridge Hue application-key env key for one host."""

    normalized_host = _validate_network_host(host, label="Hue bridge host")
    host_suffix = "".join(
        character.upper() if character.isalnum() else "_"
        for character in normalized_host
    ).strip("_")
    if not host_suffix:
        host_suffix = "HOST"
    return f"TWINR_INTEGRATION_HUE_BRIDGE_{host_suffix}_APPLICATION_KEY"


def validate_calendar_source(*, source_kind: str, source_value: str) -> None:
    """Validate one calendar source selection from operator settings."""

    normalized_source_kind = _coerce_text(source_kind).strip().lower()  # AUDIT-FIX(#8): Normalize source kind so harmless casing differences do not bypass validation or produce inconsistent behavior.
    if normalized_source_kind == "ics_file":
        return
    if normalized_source_kind != "ics_url":
        raise ValueError(f"Unsupported calendar source type: {source_kind}")

    text = _coerce_text(source_value).strip()
    _validate_calendar_url_text(text)  # AUDIT-FIX(#2): Centralize strict URL validation before any remote fetch path is constructed.


def _safe_build_email_mailbox_runtime(
    record: ManagedIntegrationConfig,
    *,
    env_values: dict[str, str],
) -> tuple[EmailMailboxAdapter | None, IntegrationReadiness]:
    """Build the email runtime and degrade cleanly on unexpected failure."""

    try:
        # AUDIT-FIX(#4): Degrade this single integration cleanly if an unexpected constructor/runtime error occurs.
        return _build_email_mailbox_runtime(record, env_values=env_values)
    except Exception:
        return None, IntegrationReadiness(
            integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
            label="Email",
            status="warn",
            summary="Unavailable",
            detail="Email integration could not be started. Check the mailbox settings and restore packaged files if needed.",
        )


def _safe_build_calendar_agenda_runtime(
    project_root: Path,
    record: ManagedIntegrationConfig,
    *,
    env_values: dict[str, str],
    url_text_loader: Callable[[str], str],
) -> tuple[ReadOnlyCalendarAdapter | None, IntegrationReadiness]:
    """Build the calendar runtime and degrade cleanly on unexpected failure."""

    try:
        # AUDIT-FIX(#4): Keep calendar startup failures isolated so one broken source does not take down the process.
        return _build_calendar_agenda_runtime(
            project_root,
            record,
            env_values=env_values,
            url_text_loader=url_text_loader,
        )
    except Exception:
        return None, IntegrationReadiness(
            integration_id=CALENDAR_AGENDA_INTEGRATION_ID,
            label="Calendar",
            status="warn",
            summary="Unavailable",
            detail="Calendar integration could not be started. Check the source path or URL and restore packaged files if needed.",
        )


def _safe_build_smart_home_hub_runtime(
    record: ManagedIntegrationConfig,
    *,
    env_values: dict[str, str],
) -> tuple[SmartHomeIntegrationAdapter | None, IntegrationReadiness]:
    """Build the smart-home runtime and degrade cleanly on unexpected failure."""

    try:
        return _build_smart_home_hub_runtime(record, env_values=env_values)
    except Exception:
        return None, IntegrationReadiness(
            integration_id=SMART_HOME_HUB_INTEGRATION_ID,
            label="Smart Home",
            status="warn",
            summary="Unavailable",
            detail="Smart-home integration could not be started. Check the bridge settings and restore packaged files if needed.",
        )


def _build_email_mailbox_runtime(
    record: ManagedIntegrationConfig,
    *,
    env_values: dict[str, str],
) -> tuple[EmailMailboxAdapter | None, IntegrationReadiness]:
    """Build the email adapter and readiness summary from one config record."""

    if not record.enabled:
        return None, IntegrationReadiness(
            integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
            label="Email",
            status="muted",
            summary="Disabled",
            detail="No mailbox connection is active. Configure account data and enable it when ready.",
        )

    profile = (_coerce_text(record.value("profile", "gmail"), default="gmail").strip().lower() or "gmail")  # AUDIT-FIX(#5): Coerce config values to text and normalize casing before calling string methods.
    account_email = _coerce_text(record.value("account_email", "")).strip()  # AUDIT-FIX(#5): Avoid AttributeError when the config store returns None or non-string values.
    from_address = _coerce_text(record.value("from_address", "")).strip() or account_email  # AUDIT-FIX(#5): Same guard for sender addresses.
    imap_host = _coerce_text(
        record.value("imap_host", "imap.gmail.com" if profile == "gmail" else "")
    ).strip()  # AUDIT-FIX(#5): Same guard for IMAP host values.
    smtp_host = _coerce_text(
        record.value("smtp_host", "smtp.gmail.com" if profile == "gmail" else "")
    ).strip()  # AUDIT-FIX(#5): Same guard for SMTP host values.
    secret_value = _coerce_text(env_values.get(EMAIL_APP_PASSWORD_ENV_KEY, "")).strip()

    missing: list[str] = []
    if not account_email:
        missing.append("account")
    if not imap_host:
        missing.append("IMAP host")
    if not smtp_host:
        missing.append("SMTP host")
    if not secret_value:
        missing.append("credential")
    if missing:
        return None, IntegrationReadiness(
            integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
            label="Email",
            status="warn",
            summary="Needs setup",
            detail=f"Missing: {', '.join(missing)}.",
        )

    try:
        normalized_account = _validate_email_address(account_email, label="Email account address")
        normalized_from = _validate_email_address(from_address, label="Email sender address")
        imap_port = _parse_positive_int(record.value("imap_port", "993"), label="IMAP port", max_value=65535)  # AUDIT-FIX(#6): Reject invalid TCP ports during config parsing instead of failing later at connection time.
        smtp_port = _parse_positive_int(record.value("smtp_port", "587"), label="SMTP port", max_value=65535)  # AUDIT-FIX(#6): Same range check for SMTP.
        contacts, contact_warnings = _parse_known_contacts(_coerce_text(record.value("known_contacts_text", "")))  # AUDIT-FIX(#5): Parse contacts from a safe text coercion path.
        unread_only_default = _parse_bool(
            record.value("unread_only_default", "true"),
            default=True,
            label="Unread-only default",
        )  # AUDIT-FIX(#5): Parse booleans inside the guarded validation block so bad values return readiness warnings instead of crashing.
        restrict_reads_to_known_senders = _parse_bool(
            record.value("restrict_reads_to_known_senders", "false"),
            default=False,
            label="Restrict reads to known senders",
        )  # AUDIT-FIX(#5): Same guarded boolean parsing for sender restrictions.
        restrict_recipients_to_known_contacts = _parse_bool(
            record.value("restrict_recipients_to_known_contacts", "true"),
            default=True,
            label="Restrict recipients to known contacts",
        )  # AUDIT-FIX(#3): Default outbound mail to approved contacts only; this is the safer baseline for a senior voice agent.
    except ValueError as exc:
        return None, IntegrationReadiness(
            integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
            label="Email",
            status="warn",
            summary="Invalid config",
            detail=str(exc),
        )

    manifest = manifest_for_id(EMAIL_MAILBOX_INTEGRATION_ID)
    if manifest is None:
        return None, IntegrationReadiness(
            integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
            label="Email",
            status="warn",
            summary="Unavailable",
            detail="Email integration files are incomplete. Restore the Twinr package before enabling mail access.",
        )  # AUDIT-FIX(#4): Treat missing built-in manifests as a degraded runtime state instead of crashing startup.

    warnings = list(contact_warnings)
    if restrict_recipients_to_known_contacts and not contacts:
        warnings.append(
            "No approved contacts are configured. Outbound email will stay blocked until you add contacts."
        )  # AUDIT-FIX(#3): Make the safer recipient restriction operationally visible so setup does not silently fail later.
    if restrict_reads_to_known_senders and not contacts:
        warnings.append(
            "Known-sender restriction is enabled but no approved contacts are configured."
        )  # AUDIT-FIX(#5): Surface contact-dependent read restrictions as a configuration warning instead of a confusing empty inbox.
    if not restrict_recipients_to_known_contacts:
        warnings.append(
            "Outbound email is not restricted to approved contacts. This increases the risk of misaddressed or unsafe sends."
        )  # AUDIT-FIX(#3): Surface an explicit safety warning when an operator opts out of the safer default.

    adapter = EmailMailboxAdapter(
        manifest=manifest,
        contacts=contacts,
        mailbox_reader=IMAPMailboxReader(
            IMAPMailboxConfig(
                host=imap_host,
                port=imap_port,
                username=normalized_account,
                password=secret_value,
                mailbox=_coerce_text(record.value("imap_mailbox", "INBOX"), default="INBOX").strip() or "INBOX",  # AUDIT-FIX(#5): Guard mailbox name parsing the same way as the rest of the config surface.
                use_ssl=True,
            )
        ),
        mail_sender=SMTPMailSender(
            SMTPMailSenderConfig(
                host=smtp_host,
                port=smtp_port,
                username=normalized_account,
                password=secret_value,
                from_address=normalized_from,
                use_starttls=True,
            )
        ),
        settings=EmailAdapterSettings(
            unread_only_default=unread_only_default,
            restrict_reads_to_known_senders=restrict_reads_to_known_senders,
            restrict_recipients_to_known_contacts=restrict_recipients_to_known_contacts,
        ),
    )
    warning_items = tuple(warnings)
    status = "warn" if warning_items else "ok"
    summary = "Ready with warnings" if warning_items else "Ready"
    detail = (
        f"{normalized_account} via {profile.replace('_', ' ')} · "
        f"IMAP {imap_host}:{imap_port} · SMTP {smtp_host}:{smtp_port} · "
        "credential stored separately in .env"
    )
    if warning_items:
        detail = f"{detail} · {warning_items[0]}"
    return adapter, IntegrationReadiness(
        integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
        label="Email",
        status=status,
        summary=summary,
        detail=detail,
        warnings=warning_items,
    )


def _build_calendar_agenda_runtime(
    project_root: Path,
    record: ManagedIntegrationConfig,
    *,
    env_values: dict[str, str],
    url_text_loader: Callable[[str], str],
) -> tuple[ReadOnlyCalendarAdapter | None, IntegrationReadiness]:
    """Build the calendar adapter and readiness summary from one config record."""

    if not record.enabled:
        return None, IntegrationReadiness(
            integration_id=CALENDAR_AGENDA_INTEGRATION_ID,
            label="Calendar",
            status="muted",
            summary="Disabled",
            detail="No agenda source is active. Configure an ICS file or feed when ready.",
        )

    source_kind = (_coerce_text(record.value("source_kind", "ics_file"), default="ics_file").strip().lower() or "ics_file")  # AUDIT-FIX(#8): Normalize source kind consistently across validation and runtime construction.
    source_value = _coerce_text(record.value("source_value", "")).strip()  # AUDIT-FIX(#5): Avoid AttributeError on mis-typed or missing source values.
    if not source_value:
        return None, IntegrationReadiness(
            integration_id=CALENDAR_AGENDA_INTEGRATION_ID,
            label="Calendar",
            status="warn",
            summary="Needs setup",
            detail="Missing calendar source path or URL.",
        )

    try:
        default_timezone = _resolve_timezone(record.value("timezone", "Europe/Berlin") or "Europe/Berlin")
        settings = CalendarAdapterSettings(
            max_events=min(_parse_positive_int(record.value("max_events", "12"), label="Max events"), 24),
            default_upcoming_days=min(
                _parse_positive_int(record.value("default_upcoming_days", "7"), label="Upcoming days"),
                30,
            ),
        )
        allowed_calendar_root = _resolve_allowed_calendar_root(project_root, env_values)  # AUDIT-FIX(#1): Constrain local ICS reads to an explicit allowlist root, defaulting to the project root.
        reader, detail, warnings = _build_calendar_reader(
            project_root,
            source_kind=source_kind,
            source_value=source_value,
            allowed_calendar_root=allowed_calendar_root,
            default_timezone=default_timezone,
            url_text_loader=url_text_loader,
        )
    except ValueError as exc:
        return None, IntegrationReadiness(
            integration_id=CALENDAR_AGENDA_INTEGRATION_ID,
            label="Calendar",
            status="warn",
            summary="Invalid config",
            detail=str(exc),
        )

    manifest = manifest_for_id(CALENDAR_AGENDA_INTEGRATION_ID)
    if manifest is None:
        return None, IntegrationReadiness(
            integration_id=CALENDAR_AGENDA_INTEGRATION_ID,
            label="Calendar",
            status="warn",
            summary="Unavailable",
            detail="Calendar integration files are incomplete. Restore the Twinr package before enabling agenda access.",
        )  # AUDIT-FIX(#4): Treat missing built-in manifests as a degraded runtime state instead of crashing startup.

    adapter = ReadOnlyCalendarAdapter(
        manifest=manifest,
        calendar_reader=reader,
        settings=settings,
    )
    status = "warn" if warnings else "ok"
    summary = "Ready with warnings" if warnings else "Ready"
    if warnings:
        detail = f"{detail} · {warnings[0]}"
    return adapter, IntegrationReadiness(
        integration_id=CALENDAR_AGENDA_INTEGRATION_ID,
        label="Calendar",
        status=status,
        summary=summary,
        detail=detail,
        warnings=warnings,
    )


def _build_calendar_reader(
    project_root: Path,
    *,
    source_kind: str,
    source_value: str,
    allowed_calendar_root: Path,
    default_timezone: ZoneInfo,
    url_text_loader: Callable[[str], str],
) -> tuple[ICSCalendarSource, str, tuple[str, ...]]:
    """Build an ``ICSCalendarSource`` plus readiness detail text."""

    if source_kind == "ics_file":
        resolved_path, relative_parts = _resolve_local_calendar_file(
            project_root,
            source_value=source_value,
            allowed_calendar_root=allowed_calendar_root,
        )  # AUDIT-FIX(#1): Resolve the local ICS file once under the allowed root and keep a relative component path for safe reopen.
        return (
            ICSCalendarSource(
                loader=lambda root=allowed_calendar_root, parts=relative_parts: _read_local_ics_text(root, parts),  # AUDIT-FIX(#1): Reopen local ICS files through a no-symlink path traversal routine on every load.
                default_timezone=default_timezone,
            ),
            f"ICS file {resolved_path} · timezone {_timezone_label(default_timezone)}",
            (),
        )

    if source_kind != "ics_url":
        raise ValueError(f"Unsupported calendar source type: {source_kind}")

    validate_calendar_source(source_kind=source_kind, source_value=source_value)
    display_url = _display_url(source_value)
    warnings: tuple[str, ...] = ()
    if urlsplit(source_value).scheme.lower() != "https":
        warnings = ("Plain HTTP calendar feeds are allowed but less private than HTTPS or a local ICS file.",)
    return (
        ICSCalendarSource(
            loader=lambda: url_text_loader(source_value),
            default_timezone=default_timezone,
        ),
        f"ICS URL {display_url} · timezone {_timezone_label(default_timezone)}",
        warnings,
    )


def _build_smart_home_hub_runtime(
    record: ManagedIntegrationConfig,
    *,
    env_values: dict[str, str],
) -> tuple[SmartHomeIntegrationAdapter | None, IntegrationReadiness]:
    """Build the smart-home adapter and readiness summary from one config record."""

    if not record.enabled:
        return None, IntegrationReadiness(
            integration_id=SMART_HOME_HUB_INTEGRATION_ID,
            label="Smart Home",
            status="muted",
            summary="Disabled",
            detail="No smart-home bridge is active. Configure a local bridge connection when ready.",
        )

    provider_name = _coerce_text(record.value("provider", SMART_HOME_HUE_PROVIDER)).strip().lower() or SMART_HOME_HUE_PROVIDER
    if provider_name != SMART_HOME_HUE_PROVIDER:
        return None, IntegrationReadiness(
            integration_id=SMART_HOME_HUB_INTEGRATION_ID,
            label="Smart Home",
            status="warn",
            summary="Invalid config",
            detail=f"Unsupported smart-home provider: {provider_name}",
        )

    try:
        configured_hosts = _configured_hue_bridge_hosts(record)
        if not configured_hosts:
            raise ValueError("Missing Hue bridge host.")
        verify_tls = _parse_bool(
            record.value("verify_tls", "false"),
            default=False,
            label="Hue bridge TLS verification",
        )
        request_timeout_s = min(
            _parse_positive_float(
                record.value("request_timeout_s", "10.0"),
                label="Hue bridge timeout",
            ),
            30.0,
        )
        event_timeout_s = min(
            _parse_positive_float(
                record.value("event_timeout_s", "2.0"),
                label="Hue event timeout",
            ),
            10.0,
        )
    except ValueError as exc:
        return None, IntegrationReadiness(
            integration_id=SMART_HOME_HUB_INTEGRATION_ID,
            label="Smart Home",
            status="warn",
            summary="Invalid config",
            detail=str(exc),
        )

    manifest = manifest_for_id(SMART_HOME_HUB_INTEGRATION_ID)
    if manifest is None:
        return None, IntegrationReadiness(
            integration_id=SMART_HOME_HUB_INTEGRATION_ID,
            label="Smart Home",
            status="warn",
            summary="Unavailable",
            detail="Smart-home integration files are incomplete. Restore the Twinr package before enabling bridge access.",
        )

    try:
        providers: list[RoutedSmartHomeProvider] = []
        ready_hosts: list[str] = []
        warnings: list[str] = []
        for index, bridge_host in enumerate(configured_hosts):
            resolved_application_key, secret_env_key = _resolve_hue_application_key(
                env_values,
                bridge_host,
                allow_legacy_fallback=index == 0,
            )
            if not resolved_application_key:
                warnings.append(f"Missing Hue application key in .env for bridge {bridge_host} ({secret_env_key}).")
                continue
            client = HueBridgeClient(
                HueBridgeConfig(
                    bridge_host=bridge_host,
                    application_key=resolved_application_key,
                    verify_tls=verify_tls,
                    timeout_s=request_timeout_s,
                )
            )
            provider = build_hue_smart_home_provider(client)
            provider.event_timeout_s = event_timeout_s
            providers.append(
                RoutedSmartHomeProvider(
                    route_id=bridge_host,
                    entity_provider=provider,
                )
            )
            ready_hosts.append(bridge_host)
    except (TypeError, ValueError) as exc:
        return None, IntegrationReadiness(
            integration_id=SMART_HOME_HUB_INTEGRATION_ID,
            label="Smart Home",
            status="warn",
            summary="Invalid config",
            detail=str(exc),
        )

    if not providers:
        summary = "Needs secret" if configured_hosts else "Needs setup"
        detail = warnings[0] if warnings else "No Hue bridge is ready."
        return None, IntegrationReadiness(
            integration_id=SMART_HOME_HUB_INTEGRATION_ID,
            label="Smart Home",
            status="warn",
            summary=summary,
            detail=detail,
            warnings=tuple(warnings),
        )

    routed_provider: AggregatedSmartHomeProvider | HueSmartHomeProvider
    if len(configured_hosts) > 1:
        routed_provider = AggregatedSmartHomeProvider(tuple(providers))
    else:
        routed_provider = providers[0].entity_provider
        if not isinstance(routed_provider, HueSmartHomeProvider):  # pragma: no cover - defensive for future refactors.
            raise TypeError("Single Hue bridge provider must be a HueSmartHomeProvider.")

    adapter = SmartHomeIntegrationAdapter(
        manifest=manifest,
        entity_provider=routed_provider,
        controller=routed_provider,
        sensor_stream=routed_provider,
        settings=SmartHomeAdapterSettings(),
    )
    tls_label = "TLS verify on" if verify_tls else "TLS verify off"
    ready_hosts_label = ", ".join(ready_hosts)
    bridges_label = "bridges" if len(ready_hosts) != 1 else "bridge"
    detail = f"Hue {bridges_label} {ready_hosts_label} · local app key stored separately in .env · {tls_label}"
    if len(configured_hosts) > 1:
        detail = f"{detail} · {len(ready_hosts)}/{len(configured_hosts)} bridges ready"
    if warnings:
        detail = f"{detail} · {warnings[0]}"
    return adapter, IntegrationReadiness(
        integration_id=SMART_HOME_HUB_INTEGRATION_ID,
        label="Smart Home",
        status="warn" if warnings else "ok",
        summary="Ready with warnings" if warnings else "Ready",
        detail=detail,
        warnings=tuple(warnings),
    )


def _configured_hue_bridge_hosts(record: ManagedIntegrationConfig) -> tuple[str, ...]:
    """Return the normalized ordered list of configured Hue bridge hosts."""

    primary_host = _coerce_text(record.value("bridge_host", "")).strip()
    additional_hosts_text = _coerce_text(record.value(HUE_ADDITIONAL_BRIDGE_HOSTS_SETTING_KEY, "")).strip()
    collected: list[str] = []
    seen: set[str] = set()
    for raw_host in (primary_host, *_split_hue_bridge_hosts_text(additional_hosts_text)):
        if not raw_host:
            continue
        normalized_host = _validate_network_host(raw_host, label="Hue bridge host")
        if normalized_host in seen:
            continue
        seen.add(normalized_host)
        collected.append(normalized_host)
    return tuple(collected)


def _split_hue_bridge_hosts_text(value: str) -> tuple[str, ...]:
    """Split newline- or comma-delimited bridge host text into raw items."""

    if not value:
        return ()
    return tuple(
        line.strip()
        for line in value.replace(",", "\n").splitlines()
        if line.strip()
    )


def _resolve_hue_application_key(
    env_values: Mapping[str, str],
    bridge_host: str,
    *,
    allow_legacy_fallback: bool,
) -> tuple[str, str]:
    """Resolve the best available Hue application key for one bridge."""

    host_env_key = hue_application_key_env_key_for_host(bridge_host)
    host_key = _coerce_text(env_values.get(host_env_key, "")).strip()
    if host_key:
        return host_key, host_env_key
    if allow_legacy_fallback:
        legacy_key = _coerce_text(env_values.get(HUE_APPLICATION_KEY_ENV_KEY, "")).strip()
        if legacy_key:
            return legacy_key, HUE_APPLICATION_KEY_ENV_KEY
    return "", host_env_key


def _parse_known_contacts(text: str) -> tuple[ApprovedEmailContacts, tuple[str, ...]]:
    """Parse approved contacts from operator-supplied multiline text."""

    contacts: list[EmailContact] = []
    warnings: list[str] = []
    seen_emails: set[str] = set()  # AUDIT-FIX(#5): Deduplicate approved contacts to keep recipient matching deterministic.
    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        display_name, email_address = parseaddr(line)
        normalized = normalize_email(email_address)
        if not normalized or "@" not in normalized:
            warnings.append(f"Ignored malformed contact line {index}.")
            continue
        if normalized in seen_emails:
            warnings.append(f"Ignored duplicate contact line {index}.")
            continue
        seen_emails.add(normalized)
        contacts.append(
            EmailContact(
                email=normalized,
                display_name=_single_line_text(display_name.strip()) or normalized,  # AUDIT-FIX(#5): Collapse control whitespace in display names before surfacing them in UI/readiness text.
            )
        )
    return ApprovedEmailContacts(tuple(contacts)), tuple(warnings)


def _validate_email_address(value: object, *, label: str) -> str:
    """Validate one email address from integration settings."""

    normalized = normalize_email(_coerce_text(value).strip())  # AUDIT-FIX(#5): Accept None/non-string config values and fail with a clean validation error instead of AttributeError.
    if not normalized or "@" not in normalized:
        raise ValueError(f"{label} must be a valid email address.")
    return normalized


def _parse_positive_int(value: object, *, label: str, max_value: int | None = None) -> int:
    """Parse one positive integer setting with optional upper bound."""

    text = _coerce_text(value).strip()  # AUDIT-FIX(#5): Parse integers from normalized text so invalid types do not crash the validator.
    if not text:
        raise ValueError(f"{label} must be a whole number.")
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number.") from None
    if parsed <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{label} must be at most {max_value}.")  # AUDIT-FIX(#6): Enforce valid port ranges and other bounded integer settings during config validation.
    return parsed


def _parse_positive_float(value: object, *, label: str) -> float:
    """Parse one positive float setting."""

    text = _coerce_text(value).strip()
    if not text:
        raise ValueError(f"{label} must be a number.")
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if parsed <= 0.0:
        raise ValueError(f"{label} must be greater than zero.")
    return parsed


def _parse_bool(value: object, *, default: bool, label: str = "Boolean value") -> bool:
    """Parse one boolean setting from tolerant text input."""

    normalized = _coerce_text(value).strip().lower()  # AUDIT-FIX(#5): Parse booleans from a tolerant text conversion path.
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{label} is invalid: {value}")


def _validate_network_host(value: object, *, label: str) -> str:
    """Validate one hostname or IP address without scheme, path, or port."""

    host = _coerce_text(value).strip()
    if not host:
        raise ValueError(f"{label} is required.")
    if "://" in host or "/" in host or "?" in host or "#" in host or "@" in host:
        raise ValueError(f"{label} must be a hostname or IP address without a scheme or path.")
    if host.startswith("["):
        if not host.endswith("]"):
            raise ValueError(f"{label} must be a valid hostname or IP address.")
        literal = host[1:-1]
        try:
            ipaddress.ip_address(literal)
        except ValueError as exc:
            raise ValueError(f"{label} must be a valid hostname or IP address.") from exc
        return host
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    if ":" in host:
        raise ValueError(f"{label} must not include a port.")
    if len(host) > 253:
        raise ValueError(f"{label} is too long.")
    labels = host.split(".")
    for label_text in labels:
        if not label_text or len(label_text) > 63:
            raise ValueError(f"{label} must be a valid hostname or IP address.")
        if label_text.startswith("-") or label_text.endswith("-"):
            raise ValueError(f"{label} must be a valid hostname or IP address.")
        for character in label_text:
            if not (character.isalnum() or character == "-"):
                raise ValueError(f"{label} must be a valid hostname or IP address.")
    return host.lower()


def _resolve_timezone(value: object) -> ZoneInfo:
    """Resolve one timezone string into a ``ZoneInfo`` instance."""

    try:
        return ZoneInfo(_coerce_text(value).strip())
    except Exception:
        raise ValueError(f"Timezone is invalid: {value}") from None


def _timezone_label(value: ZoneInfo) -> str:
    """Return a stable display label for a timezone object."""

    return getattr(value, "key", str(value))


def _resolve_env_path(project_root: Path, env_path: str | Path | None) -> Path:
    """Resolve the `.env` path used for managed integration secrets."""

    if env_path is None:
        return project_root / ".env"
    path = Path(env_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _read_env_values(path: Path) -> dict[str, str]:
    """Read a bounded dotenv-style file into a key/value mapping."""

    try:
        if not path.exists():
            return {}
        if not path.is_file():
            return {}
        if path.stat().st_size > _MAX_ENV_FILE_BYTES:
            return {}
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {}  # AUDIT-FIX(#7): Fail closed to an empty env mapping if .env is missing, unreadable, or not a regular file.

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()  # AUDIT-FIX(#7): Accept common dotenv syntax without introducing an extra dependency.
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _parse_env_value(raw_value)  # AUDIT-FIX(#7): Parse quotes and inline comments more safely than naive strip calls.
    return values


def _parse_env_value(raw_value: str) -> str:
    """Parse one dotenv value with simple quote and comment handling."""

    value = raw_value.strip()
    if not value:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def _resolve_allowed_calendar_root(project_root: Path, env_values: dict[str, str]) -> Path:
    """Resolve the allowed root directory for local ICS files."""

    configured_root = _coerce_text(env_values.get(CALENDAR_ALLOWED_ROOT_ENV_KEY, "")).strip()
    root_path = Path(configured_root) if configured_root else project_root
    if not root_path.is_absolute():
        root_path = (project_root / root_path).resolve()
    else:
        root_path = root_path.resolve()
    if not root_path.exists():
        raise ValueError(f"Configured calendar root does not exist: {root_path}")
    if not root_path.is_dir():
        raise ValueError(f"Configured calendar root must be a directory: {root_path}")
    return root_path


def _resolve_local_calendar_file(
    project_root: Path,
    *,
    source_value: str,
    allowed_calendar_root: Path,
) -> tuple[Path, tuple[str, ...]]:
    """Resolve and validate one local ICS file beneath the allowed root."""

    raw_path = Path(source_value)
    candidate_path = raw_path if raw_path.is_absolute() else (project_root / raw_path)
    try:
        resolved_path = candidate_path.resolve(strict=True)
    except FileNotFoundError:
        raise ValueError(f"Configured ICS file does not exist: {candidate_path}") from None
    except OSError:
        raise ValueError(f"Configured ICS file is not readable: {candidate_path}") from None

    if not resolved_path.is_file():
        raise ValueError(f"Configured ICS file must be a regular file: {resolved_path}")  # AUDIT-FIX(#1): Reject directories and device nodes early.
    try:
        relative_path = resolved_path.relative_to(allowed_calendar_root)
    except ValueError:
        raise ValueError(
            f"Configured ICS file must stay inside {allowed_calendar_root}."
        ) from None  # AUDIT-FIX(#1): Block traversal and arbitrary absolute-path reads outside the configured allowlist root.

    size_bytes = resolved_path.stat().st_size
    if size_bytes > _MAX_ICS_DOWNLOAD_BYTES:
        raise ValueError(
            f"Configured ICS file is too large ({size_bytes} bytes)."
        )  # AUDIT-FIX(#1): Apply the same size cap to local calendar files that already exists for remote feeds.
    return resolved_path, relative_path.parts


def _read_local_ics_text(allowed_calendar_root: Path, relative_parts: tuple[str, ...]) -> str:
    """Read one local ICS file through a no-symlink directory walk."""

    root_fd = None
    current_fd = None
    file_fd = None
    try:
        root_fd = os.open(
            allowed_calendar_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        current_fd = root_fd
        for part in relative_parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current_fd,
            )
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd

        file_fd = os.open(
            relative_parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current_fd,
        )  # AUDIT-FIX(#1): Traverse every path component from the allowlisted root with O_NOFOLLOW to block symlink swaps at load time.
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError("Configured ICS file is not a regular file.")
        if file_stat.st_size > _MAX_ICS_DOWNLOAD_BYTES:
            raise RuntimeError("Configured ICS file is too large.")

        with os.fdopen(file_fd, "rb", closefd=True) as handle:
            file_fd = None
            payload = handle.read(_MAX_ICS_DOWNLOAD_BYTES + 1)
        if len(payload) > _MAX_ICS_DOWNLOAD_BYTES:
            raise RuntimeError("Configured ICS file is too large.")
        return payload.decode("utf-8", errors="replace")
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise RuntimeError("Configured ICS file path contains a symlink.") from None
        raise RuntimeError("Configured ICS file could not be read.") from exc
    finally:
        for fd in {fd for fd in (file_fd, current_fd, root_fd) if fd is not None}:
            try:
                os.close(fd)
            except OSError:
                pass


def _fetch_ics_url(url: str) -> str:
    """Download one remote ICS feed with bounded redirects and size."""

    opener = build_opener(_NoRedirectHandler())
    current_url = url
    for _ in range(_MAX_ICS_URL_REDIRECTS + 1):
        current_url = _validate_remote_calendar_url(current_url)  # AUDIT-FIX(#2): Revalidate every URL hop before any outbound request is sent.
        request = Request(
            current_url,
            headers={
                "User-Agent": _ICS_HTTP_USER_AGENT,
                "Accept": "text/calendar, text/plain;q=0.9, */*;q=0.1",
            },
        )
        try:
            with opener.open(request, timeout=_ICS_URL_FETCH_TIMEOUT_S) as response:
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        declared_size = int(content_length.strip())
                    except ValueError:
                        raise RuntimeError("Calendar feed size header is invalid.") from None
                    if declared_size < 0:
                        raise RuntimeError("Calendar feed size header is invalid.")
                    if declared_size > _MAX_ICS_DOWNLOAD_BYTES:
                        raise RuntimeError("Calendar feed is too large.")  # AUDIT-FIX(#2): Reject oversized responses before reading them into memory.
                payload = response.read(_MAX_ICS_DOWNLOAD_BYTES + 1)
                if len(payload) > _MAX_ICS_DOWNLOAD_BYTES:
                    raise RuntimeError("Calendar feed is too large.")
                charset = response.headers.get_content_charset() or "utf-8"
                try:
                    return payload.decode(charset, errors="replace")
                except LookupError:
                    return payload.decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                raise RuntimeError("Calendar feed could not be downloaded.") from exc
            location = exc.headers.get("Location", "").strip()
            if not location:
                raise RuntimeError("Calendar feed redirect is missing a target URL.") from exc
            current_url = urljoin(current_url, location)
        except (URLError, OSError):
            raise RuntimeError("Calendar feed could not be downloaded.") from None
    raise RuntimeError("Calendar feed redirected too many times.")


def _validate_remote_calendar_url(url: str) -> str:
    """Validate one remote calendar URL and return the normalized text."""

    text = _coerce_text(url).strip()
    _validate_calendar_url_text(text)
    hostname = urlsplit(text).hostname or ""
    if not hostname:
        raise RuntimeError("Calendar URL hostname is invalid.")
    _assert_public_remote_host(hostname)  # AUDIT-FIX(#2): Block loopback, link-local, and RFC1918/private targets to close the SSRF hole.
    return text


def _validate_calendar_url_text(url: str) -> None:
    """Validate the textual shape of a remote calendar URL."""

    if not url:
        raise ValueError("Calendar URL is required.")
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in url):
        raise ValueError("Calendar URL must not contain whitespace or control characters.")  # AUDIT-FIX(#8): Reject malformed URLs with embedded whitespace/control characters early.
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError("Calendar URL must start with http:// or https://.")
    try:
        _ = parts.port
    except ValueError:
        raise ValueError("Calendar URL port is invalid.") from None
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(
            "Calendar URL must not include embedded credentials, query tokens, or fragments. "
            "Use a plain feed URL or a local ICS file."
        )


def _assert_public_remote_host(hostname: str) -> None:
    """Reject hostnames that resolve to non-public addresses."""

    literal_ip = _parse_ip_address(hostname)
    if literal_ip is not None:
        if not literal_ip.is_global:
            raise RuntimeError("Calendar URL must point to a public internet host.")
        return

    try:
        address_info = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise RuntimeError("Calendar URL hostname could not be resolved.") from None

    if not address_info:
        raise RuntimeError("Calendar URL hostname could not be resolved.")

    for _, _, _, _, sockaddr in address_info:
        candidate_ip = _parse_ip_address(sockaddr[0])
        if candidate_ip is None or not candidate_ip.is_global:
            raise RuntimeError("Calendar URL must point to a public internet host.")


def _parse_ip_address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse a literal IP address if the hostname already is one."""

    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _display_url(url: str) -> str:
    """Render a remote URL without credentials, query, or fragment data."""

    parts = urlsplit(url)
    hostname = parts.hostname or ""
    netloc = hostname
    if parts.port is not None:
        netloc = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _coerce_text(value: object | None, default: str = "") -> str:
    """Coerce a possibly missing value to text."""

    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _single_line_text(value: str) -> str:
    """Collapse internal whitespace to a single display line."""

    return " ".join(value.split())
