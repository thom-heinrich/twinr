# CHANGELOG: 2026-03-30
# BUG-1: IntegrationReadiness.ready now treats "Ready with warnings" adapters as usable.
# BUG-2: Real process environment now overrides .env, and secrets can be loaded from *_FILE or systemd credentials.
# BUG-3: Email transport/security is now validated explicitly so broken 465/143 combinations fail fast instead of later at connection time.
# SEC-1: Remote ICS fetching now pins the validated public IP at connect-time, closing the DNS-rebinding gap in the old SSRF defense.
# SEC-2: Plain-HTTP calendar feeds now require explicit opt-in, and Hue TLS verification defaults to on.
# IMP-1: Remote ICS feeds now use thread-safe conditional revalidation (ETag/Last-Modified), bounded freshness caching, and stale-on-error fallback.
# IMP-2: Query-string ICS subscription URLs, secret files, and systemd credential directories are supported without leaking tokens in readiness text.

"""Assemble managed Twinr integrations from config, env, and provider code.

This module wires the shared integration layer to the built-in email and
calendar packages while keeping setup validation and readiness reporting
centralized.
"""

from __future__ import annotations

import errno
import http.client
import inspect
import ipaddress
import os
import socket
import ssl
import stat
import threading
import time
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit
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
from twinr.integrations.email.profiles import DEFAULT_EMAIL_PROFILE_ID, email_provider_profile
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
CALENDAR_ALLOWED_ROOT_ENV_KEY = "TWINR_INTEGRATION_CALENDAR_ALLOWED_ROOT"
CALENDAR_ALLOW_HTTP_ENV_KEY = "TWINR_INTEGRATION_CALENDAR_ALLOW_HTTP"
CALENDAR_REFRESH_INTERVAL_ENV_KEY = "TWINR_INTEGRATION_CALENDAR_REFRESH_INTERVAL_S"
CALENDAR_STALE_IF_ERROR_ENV_KEY = "TWINR_INTEGRATION_CALENDAR_STALE_IF_ERROR_S"
_CREDENTIALS_DIRECTORY_ENV_KEY = "CREDENTIALS_DIRECTORY"
_INTERNAL_ENV_BASE_DIR_KEY = "_TWINR_INTERNAL_ENV_BASE_DIR"
_MAX_ICS_DOWNLOAD_BYTES = 2 * 1024 * 1024
_MAX_ENV_FILE_BYTES = 64 * 1024
_MAX_SECRET_FILE_BYTES = 8 * 1024
_ICS_URL_FETCH_TIMEOUT_S = 10.0
_MAX_ICS_URL_REDIRECTS = 3
_ICS_HTTP_USER_AGENT = "Twinr/0.2"
_DEFAULT_REMOTE_ICS_REFRESH_INTERVAL_S = 300.0
_MAX_REMOTE_ICS_REFRESH_INTERVAL_S = 6 * 60 * 60.0
_DEFAULT_REMOTE_ICS_STALE_IF_ERROR_S = 12 * 60 * 60.0
_MAX_REMOTE_ICS_STALE_IF_ERROR_S = 7 * 24 * 60 * 60.0
_REMOTE_ICS_FETCH_ATTEMPTS = 2
_RETRYABLE_REMOTE_ICS_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


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
        """Return whether the readiness entry represents a usable adapter."""

        return self.summary.startswith("Ready")


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


@dataclass(frozen=True, slots=True)
class _ResolvedEndpoint:
    """Represent one concrete resolved remote socket endpoint."""

    family: int
    socktype: int
    proto: int
    sockaddr: tuple[Any, ...]
    address_text: str


@dataclass(frozen=True, slots=True)
class _PinnedRemoteTarget:
    """Describe one validated remote HTTP(S) target plus its pinned endpoints."""

    normalized_url: str
    scheme: str
    hostname: str
    port: int
    request_target: str
    host_header: str
    endpoints: tuple[_ResolvedEndpoint, ...]


@dataclass(frozen=True, slots=True)
class _FetchedRemoteCalendarText:
    """Carry one fetched remote calendar payload plus cache validators."""

    text: str = ""
    etag: str = ""
    last_modified: str = ""
    refresh_after_s: float = _DEFAULT_REMOTE_ICS_REFRESH_INTERVAL_S
    not_modified: bool = False


@dataclass(slots=True)
class _RemoteICSTextLoader:
    """Thread-safe cached loader for one remote ICS feed."""

    url: str
    allow_insecure_http: bool
    default_refresh_interval_s: float
    stale_if_error_s: float
    custom_text_loader: Callable[[str], str] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _has_cached_value: bool = False
    _text: str = ""
    _etag: str = ""
    _last_modified: str = ""
    _fresh_until_monotonic: float = 0.0
    _stale_until_monotonic: float = 0.0

    def __call__(self) -> str:
        """Return cached or freshly fetched calendar text."""

        now = time.monotonic()
        with self._lock:
            if self._has_cached_value and now < self._fresh_until_monotonic:
                return self._text

            try:
                refreshed_text = self._refresh_locked(now)
            except Exception:
                if self._has_cached_value and now < self._stale_until_monotonic:
                    return self._text
                raise
            return refreshed_text

    def _refresh_locked(self, now: float) -> str:
        """Refresh the cached payload while holding the loader lock."""

        if self.custom_text_loader is None:
            fetched = _fetch_ics_url_response(
                self.url,
                etag=self._etag,
                last_modified=self._last_modified,
                allow_insecure_http=self.allow_insecure_http,
                default_refresh_interval_s=self.default_refresh_interval_s,
            )
            if fetched.not_modified:
                if not self._has_cached_value:
                    raise RuntimeError("Calendar feed returned 304 without a cached payload.")
                self._fresh_until_monotonic = now + fetched.refresh_after_s
                self._stale_until_monotonic = self._fresh_until_monotonic + self.stale_if_error_s
                return self._text

            self._text = fetched.text
            self._etag = fetched.etag
            self._last_modified = fetched.last_modified
            self._has_cached_value = True
            self._fresh_until_monotonic = now + fetched.refresh_after_s
            self._stale_until_monotonic = self._fresh_until_monotonic + self.stale_if_error_s
            return self._text

        fetched_text = self.custom_text_loader(self.url)
        self._text = _coerce_text(fetched_text)
        self._has_cached_value = True
        self._fresh_until_monotonic = now + self.default_refresh_interval_s
        self._stale_until_monotonic = self._fresh_until_monotonic + self.stale_if_error_s
        return self._text


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that connects to a pre-resolved endpoint."""

    def __init__(self, *, host: str, port: int, endpoint: _ResolvedEndpoint, timeout: float):
        super().__init__(host=host, port=port, timeout=timeout)
        self._endpoint = endpoint

    def connect(self) -> None:
        """Open the socket to the pinned endpoint."""

        sock = socket.socket(self._endpoint.family, self._endpoint.socktype, self._endpoint.proto)
        try:
            if self.timeout is not None:
                sock.settimeout(self.timeout)
            sock.connect(self._endpoint.sockaddr)
        except Exception:
            sock.close()
            raise
        self.sock = sock


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that validates TLS for the logical hostname but connects to a pinned endpoint."""

    def __init__(self, *, host: str, port: int, endpoint: _ResolvedEndpoint, timeout: float, context: ssl.SSLContext):
        super().__init__(host=host, port=port, timeout=timeout, context=context)
        self._endpoint = endpoint

    def connect(self) -> None:
        """Open the TLS socket to the pinned endpoint."""

        raw_sock = socket.socket(self._endpoint.family, self._endpoint.socktype, self._endpoint.proto)
        try:
            if self.timeout is not None:
                raw_sock.settimeout(self.timeout)
            raw_sock.connect(self._endpoint.sockaddr)
            self.sock = self._context.wrap_socket(raw_sock, server_hostname=self.host)
        except Exception:
            raw_sock.close()
            raise


def build_managed_integrations(
    project_root: str | Path,
    *,
    env_path: str | Path | None = None,
    url_text_loader: Callable[[str], str] | None = None,
) -> ManagedIntegrationsRuntime:
    """Build the managed email, calendar, and smart-home adapters for one project root."""

    project_path = Path(project_root).resolve()
    env_values = _read_effective_env_values(_resolve_env_path(project_path, env_path))

    try:
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


def validate_calendar_source(*, source_kind: str, source_value: str, allow_insecure_http: bool = False) -> None:
    """Validate one calendar source selection from operator settings."""

    normalized_source_kind = _coerce_text(source_kind).strip().lower()
    if normalized_source_kind == "ics_file":
        return
    if normalized_source_kind != "ics_url":
        raise ValueError(f"Unsupported calendar source type: {source_kind}")

    text = _coerce_text(source_value).strip()
    _validate_calendar_url_text(text, allow_insecure_http=allow_insecure_http)


def _safe_build_email_mailbox_runtime(
    record: ManagedIntegrationConfig,
    *,
    env_values: dict[str, str],
) -> tuple[EmailMailboxAdapter | None, IntegrationReadiness]:
    """Build the email runtime and degrade cleanly on unexpected failure."""

    try:
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

    profile_config = email_provider_profile(
        record.value("profile", DEFAULT_EMAIL_PROFILE_ID),
        default=DEFAULT_EMAIL_PROFILE_ID,
    )
    account_email = _coerce_text(record.value("account_email", "")).strip()
    from_address = _coerce_text(record.value("from_address", "")).strip() or account_email
    imap_host = _coerce_text(
        record.value("imap_host", profile_config.default_imap_host)
    ).strip()
    smtp_host = _coerce_text(
        record.value("smtp_host", profile_config.default_smtp_host)
    ).strip()

    if not profile_config.supported:
        return None, IntegrationReadiness(
            integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
            label="Email",
            status="warn",
            summary="Needs OAuth2",
            detail=profile_config.support_detail,
        )
    if profile_config.requires_oauth_sign_in:
        detail = f"{profile_config.label} requires {profile_config.secret_placeholder}."
        if profile_config.secret_help_text:
            detail = f"{detail} {profile_config.secret_help_text}"
        return None, IntegrationReadiness(
            integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
            label="Email",
            status="warn",
            summary="Needs OAuth2",
            detail=detail,
        )

    missing: list[str] = []
    if not account_email:
        missing.append("account")
    if not imap_host:
        missing.append("IMAP host")
    if not smtp_host:
        missing.append("SMTP host")
    if not _has_secret_value(env_values, EMAIL_APP_PASSWORD_ENV_KEY):
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
        imap_port = _parse_positive_int(
            record.value("imap_port", profile_config.default_imap_port or "993"),
            label="IMAP port",
            max_value=65535,
        )
        smtp_port = _parse_positive_int(
            record.value("smtp_port", profile_config.default_smtp_port or "587"),
            label="SMTP port",
            max_value=65535,
        )
        contacts, contact_warnings = _parse_known_contacts(_coerce_text(record.value("known_contacts_text", "")))
        unread_only_default = _parse_bool(
            record.value("unread_only_default", "true"),
            default=True,
            label="Unread-only default",
        )
        restrict_reads_to_known_senders = _parse_bool(
            record.value("restrict_reads_to_known_senders", "false"),
            default=False,
            label="Restrict reads to known senders",
        )
        restrict_recipients_to_known_contacts = _parse_bool(
            record.value("restrict_recipients_to_known_contacts", "true"),
            default=True,
            label="Restrict recipients to known contacts",
        )
        imap_security = _parse_email_security_mode(
            record.value("imap_security", "auto"),
            label="IMAP security",
        )
        smtp_security = _parse_email_security_mode(
            record.value("smtp_security", "auto"),
            label="SMTP security",
        )
        secret_value, _ = _resolve_secret_value(env_values, EMAIL_APP_PASSWORD_ENV_KEY)
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
        )

    warnings = list(contact_warnings)
    if restrict_recipients_to_known_contacts and not contacts:
        warnings.append(
            "No approved contacts are configured. Outbound email will stay blocked until you add contacts."
        )
    if restrict_reads_to_known_senders and not contacts:
        warnings.append(
            "Known-sender restriction is enabled but no approved contacts are configured."
        )
    if not restrict_recipients_to_known_contacts:
        warnings.append(
            "Outbound email is not restricted to approved contacts. This increases the risk of misaddressed or unsafe sends."
        )

    try:
        mailbox_reader, imap_transport_label = _build_imap_mailbox_reader(
            host=imap_host,
            port=imap_port,
            username=normalized_account,
            password=secret_value,
            mailbox=_coerce_text(
                record.value("imap_mailbox", profile_config.default_mailbox),
                default=profile_config.default_mailbox,
            ).strip()
            or profile_config.default_mailbox,
            security_mode=imap_security,
        )
        mail_sender, smtp_transport_label = _build_smtp_mail_sender(
            host=smtp_host,
            port=smtp_port,
            username=normalized_account,
            password=secret_value,
            from_address=normalized_from,
            security_mode=smtp_security,
        )
    except ValueError as exc:
        return None, IntegrationReadiness(
            integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
            label="Email",
            status="warn",
            summary="Invalid config",
            detail=str(exc),
        )

    adapter = EmailMailboxAdapter(
        manifest=manifest,
        contacts=contacts,
        mailbox_reader=mailbox_reader,
        mail_sender=mail_sender,
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
        f"{normalized_account} via {profile_config.label.lower()} · "
        f"{imap_transport_label} {imap_host}:{imap_port} · {smtp_transport_label} {smtp_host}:{smtp_port} · "
        "credential stored outside config"
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

    source_kind = (_coerce_text(record.value("source_kind", "ics_file"), default="ics_file").strip().lower() or "ics_file")
    source_value = _coerce_text(record.value("source_value", "")).strip()
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
        # BREAKING: Plain-HTTP calendar feeds are no longer accepted by default.
        allow_insecure_http = _parse_bool(
            record.value("allow_insecure_http", env_values.get(CALENDAR_ALLOW_HTTP_ENV_KEY, "false")),
            default=False,
            label="Calendar allow insecure HTTP",
        )
        remote_refresh_interval_s = min(
            _parse_positive_float(
                record.value(
                    "refresh_interval_s",
                    env_values.get(CALENDAR_REFRESH_INTERVAL_ENV_KEY, str(_DEFAULT_REMOTE_ICS_REFRESH_INTERVAL_S)),
                ),
                label="Calendar refresh interval",
            ),
            _MAX_REMOTE_ICS_REFRESH_INTERVAL_S,
        )
        remote_stale_if_error_s = min(
            _parse_positive_float(
                record.value(
                    "stale_if_error_s",
                    env_values.get(CALENDAR_STALE_IF_ERROR_ENV_KEY, str(_DEFAULT_REMOTE_ICS_STALE_IF_ERROR_S)),
                ),
                label="Calendar stale-on-error window",
            ),
            _MAX_REMOTE_ICS_STALE_IF_ERROR_S,
        )
        allowed_calendar_root = _resolve_allowed_calendar_root(project_root, env_values)
        reader, detail, warnings = _build_calendar_reader(
            project_root,
            source_kind=source_kind,
            source_value=source_value,
            allowed_calendar_root=allowed_calendar_root,
            default_timezone=default_timezone,
            allow_insecure_http=allow_insecure_http,
            remote_refresh_interval_s=remote_refresh_interval_s,
            remote_stale_if_error_s=remote_stale_if_error_s,
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
        )

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
    allow_insecure_http: bool,
    remote_refresh_interval_s: float,
    remote_stale_if_error_s: float,
    url_text_loader: Callable[[str], str],
) -> tuple[ICSCalendarSource, str, tuple[str, ...]]:
    """Build an ``ICSCalendarSource`` plus readiness detail text."""

    if source_kind == "ics_file":
        resolved_path, relative_parts = _resolve_local_calendar_file(
            project_root,
            source_value=source_value,
            allowed_calendar_root=allowed_calendar_root,
        )
        return (
            ICSCalendarSource(
                loader=lambda root=allowed_calendar_root, parts=relative_parts: _read_local_ics_text(root, parts),
                default_timezone=default_timezone,
            ),
            f"ICS file {resolved_path} · timezone {_timezone_label(default_timezone)}",
            (),
        )

    if source_kind != "ics_url":
        raise ValueError(f"Unsupported calendar source type: {source_kind}")

    validate_calendar_source(
        source_kind=source_kind,
        source_value=source_value,
        allow_insecure_http=allow_insecure_http,
    )
    display_url = _display_url(source_value)
    warnings: list[str] = []
    if urlsplit(source_value).scheme.lower() == "http":
        warnings.append("Plain HTTP calendar feeds are explicitly enabled and are less private than HTTPS or a local ICS file.")

    custom_loader = None if url_text_loader is _fetch_ics_url else url_text_loader
    loader = _RemoteICSTextLoader(
        url=source_value,
        allow_insecure_http=allow_insecure_http,
        default_refresh_interval_s=remote_refresh_interval_s,
        stale_if_error_s=remote_stale_if_error_s,
        custom_text_loader=custom_loader,
    )
    return (
        ICSCalendarSource(
            loader=loader,
            default_timezone=default_timezone,
        ),
        f"ICS URL {display_url} · timezone {_timezone_label(default_timezone)} · refresh ≤ {int(remote_refresh_interval_s)}s",
        tuple(warnings),
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
        # BREAKING: Hue TLS verification now defaults to on.
        verify_tls = _parse_bool(
            record.value("verify_tls", "true"),
            default=True,
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
        if not verify_tls:
            warnings.append(
                "Hue bridge TLS verification is disabled. This weakens protection against local-network interception."
            )
        for index, bridge_host in enumerate(configured_hosts):
            resolved_application_key, secret_env_key = _resolve_hue_application_key(
                env_values,
                bridge_host,
                allow_legacy_fallback=index == 0,
            )
            if not resolved_application_key:
                warnings.append(f"Missing Hue application key outside config for bridge {bridge_host} ({secret_env_key}).")
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
    detail = f"Hue {bridges_label} {ready_hosts_label} · local app key stored outside config · {tls_label}"
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
    host_key, host_source = _resolve_secret_value(env_values, host_env_key)
    if host_key:
        return host_key, host_source
    if allow_legacy_fallback:
        legacy_key, legacy_source = _resolve_secret_value(env_values, HUE_APPLICATION_KEY_ENV_KEY)
        if legacy_key:
            return legacy_key, legacy_source
    return "", host_env_key


def _parse_known_contacts(text: str) -> tuple[ApprovedEmailContacts, tuple[str, ...]]:
    """Parse approved contacts from operator-supplied multiline text."""

    contacts: list[EmailContact] = []
    warnings: list[str] = []
    seen_emails: set[str] = set()
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
                display_name=_single_line_text(display_name.strip()) or normalized,
            )
        )
    return ApprovedEmailContacts(tuple(contacts)), tuple(warnings)


def _validate_email_address(value: object, *, label: str) -> str:
    """Validate one email address from integration settings."""

    normalized = normalize_email(_coerce_text(value).strip())
    if not normalized or "@" not in normalized:
        raise ValueError(f"{label} must be a valid email address.")
    return normalized


def _parse_positive_int(value: object, *, label: str, max_value: int | None = None) -> int:
    """Parse one positive integer setting with optional upper bound."""

    text = _coerce_text(value).strip()
    if not text:
        raise ValueError(f"{label} must be a whole number.")
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number.") from None
    if parsed <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{label} must be at most {max_value}.")
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

    normalized = _coerce_text(value).strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{label} is invalid: {value}")


def _parse_email_security_mode(value: object, *, label: str) -> str:
    """Parse one transport-security mode for IMAP or SMTP."""

    normalized = _coerce_text(value).strip().lower()
    if not normalized:
        return "auto"
    aliases = {
        "auto": "auto",
        "ssl": "ssl",
        "tls": "ssl",
        "implicit_tls": "ssl",
        "implicit-tls": "ssl",
        "imaps": "ssl",
        "smtps": "ssl",
        "starttls": "starttls",
    }
    parsed = aliases.get(normalized)
    if parsed is None:
        raise ValueError(f"{label} is invalid: {value}")
    return parsed


def _build_imap_mailbox_reader(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    mailbox: str,
    security_mode: str,
) -> tuple[IMAPMailboxReader, str]:
    """Build the configured IMAP reader with transport validation."""

    resolved_mode = _resolve_imap_security_mode(security_mode, port=port)
    config_kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "mailbox": mailbox,
    }

    ssl_keyword = _first_supported_keyword(IMAPMailboxConfig, ("use_ssl", "ssl"))
    starttls_keyword = _first_supported_keyword(IMAPMailboxConfig, ("use_starttls", "starttls"))
    if resolved_mode == "ssl":
        if ssl_keyword is not None:
            config_kwargs[ssl_keyword] = True
        if starttls_keyword is not None:
            config_kwargs[starttls_keyword] = False
    else:
        if starttls_keyword is None:
            raise ValueError(
                "IMAP STARTTLS is not supported by the installed Twinr backend. Use port 993 or upgrade the backend."
            )
        config_kwargs[starttls_keyword] = True
        if ssl_keyword is not None:
            config_kwargs[ssl_keyword] = False

    return IMAPMailboxReader(IMAPMailboxConfig(**config_kwargs)), (
        "IMAPS" if resolved_mode == "ssl" else "IMAP STARTTLS"
    )


def _build_smtp_mail_sender(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    from_address: str,
    security_mode: str,
) -> tuple[SMTPMailSender, str]:
    """Build the configured SMTP sender with transport validation."""

    resolved_mode = _resolve_smtp_security_mode(security_mode, port=port)
    config_kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "from_address": from_address,
    }

    ssl_keyword = _first_supported_keyword(SMTPMailSenderConfig, ("use_ssl", "ssl"))
    starttls_keyword = _first_supported_keyword(SMTPMailSenderConfig, ("use_starttls", "starttls"))
    if resolved_mode == "ssl":
        if ssl_keyword is None:
            raise ValueError(
                "SMTP implicit TLS on port 465 is not supported by the installed Twinr backend. Use STARTTLS on port 587 or upgrade the backend."
            )
        config_kwargs[ssl_keyword] = True
        if starttls_keyword is not None:
            config_kwargs[starttls_keyword] = False
    else:
        if starttls_keyword is None:
            raise ValueError("SMTP STARTTLS is not supported by the installed Twinr backend.")
        config_kwargs[starttls_keyword] = True
        if ssl_keyword is not None:
            config_kwargs[ssl_keyword] = False

    return SMTPMailSender(SMTPMailSenderConfig(**config_kwargs)), (
        "SMTP implicit TLS" if resolved_mode == "ssl" else "SMTP STARTTLS"
    )


def _resolve_imap_security_mode(mode: str, *, port: int) -> str:
    """Resolve IMAP security mode from config plus port."""

    if mode != "auto":
        return mode
    if port == 143:
        return "starttls"
    return "ssl"


def _resolve_smtp_security_mode(mode: str, *, port: int) -> str:
    """Resolve SMTP security mode from config plus port."""

    if mode != "auto":
        return mode
    if port == 465:
        return "ssl"
    return "starttls"


def _supports_keyword(factory: object, keyword: str) -> bool:
    """Return whether a callable/class constructor accepts one keyword."""

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        annotations = getattr(factory, "__annotations__", {})
        return keyword in annotations
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters


def _first_supported_keyword(factory: object, candidates: tuple[str, ...]) -> str | None:
    """Return the first accepted keyword from a list of aliases."""

    for candidate in candidates:
        if _supports_keyword(factory, candidate):
            return candidate
    return None


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
            ip_value = ipaddress.ip_address(literal)
        except ValueError as exc:
            raise ValueError(f"{label} must be a valid hostname or IP address.") from exc
        return f"[{ip_value.compressed.lower()}]"
    try:
        return ipaddress.ip_address(host).compressed.lower()
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


def _read_effective_env_values(path: Path) -> dict[str, str]:
    """Read file-backed env values and overlay runtime environment variables."""

    values = _read_env_values(path)
    values[_INTERNAL_ENV_BASE_DIR_KEY] = str(path.parent)
    for key, value in os.environ.items():
        if key.startswith("TWINR_") or key == _CREDENTIALS_DIRECTORY_ENV_KEY:
            values[key] = value
    return values


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
        return {}

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _parse_env_value(raw_value)
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


def _resolve_secret_value(env_values: Mapping[str, str], key: str) -> tuple[str, str]:
    """Resolve one secret from direct env, *_FILE, or systemd credentials."""

    direct_value = _coerce_text(env_values.get(key, "")).strip()
    if direct_value:
        return direct_value, key

    file_key = f"{key}_FILE"
    file_value = _coerce_text(env_values.get(file_key, "")).strip()
    if file_value:
        secret_path = _resolve_secret_path(env_values, file_value)
        return _read_secret_file(secret_path), file_key

    credentials_directory = _coerce_text(env_values.get(_CREDENTIALS_DIRECTORY_ENV_KEY, "")).strip()
    if credentials_directory:
        credential_path = Path(credentials_directory) / key
        if credential_path.exists():
            return _read_secret_file(credential_path), f"{key}@credentials"

    return "", key


def _has_secret_value(env_values: Mapping[str, str], key: str) -> bool:
    """Return whether a secret exists in any supported source without throwing."""

    try:
        secret_value, _ = _resolve_secret_value(env_values, key)
    except ValueError:
        return True
    return bool(secret_value)


def _resolve_secret_path(env_values: Mapping[str, str], raw_path: str) -> Path:
    """Resolve one secret file path relative to the env file when needed."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    base_dir = _coerce_text(env_values.get(_INTERNAL_ENV_BASE_DIR_KEY, "")).strip()
    if not base_dir:
        return path.resolve()
    return (Path(base_dir) / path).resolve()


def _read_secret_file(path: Path) -> str:
    """Read one small text secret file."""

    try:
        if not path.exists():
            raise ValueError(f"Secret file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Secret file must be a regular file: {path}")
        if path.stat().st_size > _MAX_SECRET_FILE_BYTES:
            raise ValueError(f"Secret file is too large: {path}")
        return path.read_text(encoding="utf-8-sig").rstrip("\r\n")
    except OSError:
        raise ValueError(f"Secret file could not be read: {path}") from None


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
    if not raw_path.is_absolute() and not candidate_path.exists():
        candidate_path = allowed_calendar_root / raw_path
    try:
        resolved_path = candidate_path.resolve(strict=True)
    except FileNotFoundError:
        raise ValueError(f"Configured ICS file does not exist: {candidate_path}") from None
    except OSError:
        raise ValueError(f"Configured ICS file is not readable: {candidate_path}") from None

    if not resolved_path.is_file():
        raise ValueError(f"Configured ICS file must be a regular file: {resolved_path}")
    try:
        relative_path = resolved_path.relative_to(allowed_calendar_root)
    except ValueError:
        raise ValueError(
            f"Configured ICS file must stay inside {allowed_calendar_root}."
        ) from None

    size_bytes = resolved_path.stat().st_size
    if size_bytes > _MAX_ICS_DOWNLOAD_BYTES:
        raise ValueError(
            f"Configured ICS file is too large ({size_bytes} bytes)."
        )
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
        )
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
    """Download one remote ICS feed with redirect validation and caching-friendly semantics."""

    return _fetch_ics_url_response(url).text


def _fetch_ics_url_response(
    url: str,
    *,
    etag: str = "",
    last_modified: str = "",
    allow_insecure_http: bool = False,
    default_refresh_interval_s: float = _DEFAULT_REMOTE_ICS_REFRESH_INTERVAL_S,
) -> _FetchedRemoteCalendarText:
    """Download one remote ICS feed with pinned DNS resolution and validators."""

    current_url = url
    for _ in range(_MAX_ICS_URL_REDIRECTS + 1):
        target = _build_pinned_remote_target(current_url, allow_insecure_http=allow_insecure_http)
        status, headers, payload = _request_pinned_remote_target(
            target,
            etag=etag,
            last_modified=last_modified,
        )
        refresh_after_s = _compute_remote_refresh_interval(headers, default_refresh_interval_s)
        if status == http.client.NOT_MODIFIED:
            return _FetchedRemoteCalendarText(
                etag=etag,
                last_modified=last_modified,
                refresh_after_s=refresh_after_s,
                not_modified=True,
            )
        if status in {301, 302, 303, 307, 308}:
            location = headers.get("Location", "").strip()
            if not location:
                raise RuntimeError("Calendar feed redirect is missing a target URL.")
            current_url = urljoin(current_url, location)
            continue
        charset = headers.get_content_charset() or "utf-8"
        try:
            decoded_payload = payload.decode(charset, errors="replace")
        except LookupError:
            decoded_payload = payload.decode("utf-8", errors="replace")
        return _FetchedRemoteCalendarText(
            text=decoded_payload,
            etag=headers.get("ETag", "").strip(),
            last_modified=headers.get("Last-Modified", "").strip(),
            refresh_after_s=refresh_after_s,
        )
    raise RuntimeError("Calendar feed redirected too many times.")


def _request_pinned_remote_target(
    target: _PinnedRemoteTarget,
    *,
    etag: str,
    last_modified: str,
) -> tuple[int, http.client.HTTPMessage, bytes]:
    """Issue one GET request against a validated and pinned remote target."""

    last_error: Exception | None = None
    retry_after_s = 0.0
    should_retry = False
    for attempt in range(_REMOTE_ICS_FETCH_ATTEMPTS):
        should_retry = False
        for endpoint in target.endpoints:
            connection: http.client.HTTPConnection
            if target.scheme == "https":
                connection = _PinnedHTTPSConnection(
                    host=target.hostname,
                    port=target.port,
                    endpoint=endpoint,
                    timeout=_ICS_URL_FETCH_TIMEOUT_S,
                    context=ssl.create_default_context(),
                )
            else:
                connection = _PinnedHTTPConnection(
                    host=target.hostname,
                    port=target.port,
                    endpoint=endpoint,
                    timeout=_ICS_URL_FETCH_TIMEOUT_S,
                )
            try:
                connection.putrequest("GET", target.request_target, skip_host=True, skip_accept_encoding=True)
                connection.putheader("Host", target.host_header)
                connection.putheader("User-Agent", _ICS_HTTP_USER_AGENT)
                connection.putheader("Accept", "text/calendar, text/plain;q=0.9, */*;q=0.1")
                connection.putheader("Accept-Encoding", "identity")
                connection.putheader("Connection", "close")
                if etag:
                    connection.putheader("If-None-Match", etag)
                if last_modified:
                    connection.putheader("If-Modified-Since", last_modified)
                connection.endheaders()
                response = connection.getresponse()
                status = response.status
                headers = response.headers
                if status == http.client.NOT_MODIFIED or status in {301, 302, 303, 307, 308}:
                    response.read()
                    return status, headers, b""
                if 200 <= status < 300:
                    _validate_remote_content_length(headers)
                    payload = response.read(_MAX_ICS_DOWNLOAD_BYTES + 1)
                    if len(payload) > _MAX_ICS_DOWNLOAD_BYTES:
                        raise RuntimeError("Calendar feed is too large.")
                    return status, headers, payload
                if status in _RETRYABLE_REMOTE_ICS_HTTP_STATUSES:
                    should_retry = True
                    retry_after_s = max(retry_after_s, _parse_retry_after_seconds(headers.get("Retry-After", "")))
                    last_error = RuntimeError(f"Calendar feed returned transient HTTP {status}.")
                    continue
                raise RuntimeError(f"Calendar feed returned HTTP {status}.")
            except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
                should_retry = True
                last_error = exc
                continue
            finally:
                try:
                    connection.close()
                except Exception:
                    pass
        if should_retry and attempt + 1 < _REMOTE_ICS_FETCH_ATTEMPTS:
            time.sleep(min(max(retry_after_s, 0.2 * (attempt + 1)), 1.0))
            continue
        break

    raise RuntimeError("Calendar feed could not be downloaded.") from last_error


def _build_pinned_remote_target(url: str, *, allow_insecure_http: bool) -> _PinnedRemoteTarget:
    """Validate one remote URL and bind it to a concrete endpoint set."""

    text = _coerce_text(url).strip()
    try:
        _validate_calendar_url_text(text, allow_insecure_http=allow_insecure_http)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None
    parts = urlsplit(text)
    hostname = parts.hostname or ""
    if not hostname:
        raise RuntimeError("Calendar URL hostname is invalid.")
    normalized_hostname = _normalize_remote_hostname(hostname)
    scheme = parts.scheme.lower()
    port = parts.port or (443 if scheme == "https" else 80)
    endpoints = _resolve_public_remote_endpoints(normalized_hostname, port)
    normalized_netloc = _format_authority(normalized_hostname, parts.port)
    normalized_url = urlunsplit((scheme, normalized_netloc, parts.path, parts.query, ""))
    return _PinnedRemoteTarget(
        normalized_url=normalized_url,
        scheme=scheme,
        hostname=normalized_hostname,
        port=port,
        request_target=_build_request_target(parts),
        host_header=_format_authority(normalized_hostname, parts.port),
        endpoints=endpoints,
    )


def _normalize_remote_hostname(hostname: str) -> str:
    """Normalize one remote hostname for sockets and Host headers."""

    literal_ip = _parse_ip_address(hostname)
    if literal_ip is not None:
        return literal_ip.compressed.lower()
    try:
        return hostname.encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise RuntimeError("Calendar URL hostname is invalid.") from None


def _resolve_public_remote_endpoints(hostname: str, port: int) -> tuple[_ResolvedEndpoint, ...]:
    """Resolve one hostname to a deduplicated list of public endpoints."""

    literal_ip = _parse_ip_address(hostname)
    if literal_ip is not None:
        if not literal_ip.is_global:
            raise RuntimeError("Calendar URL must point to a public internet host.")
        family = socket.AF_INET6 if literal_ip.version == 6 else socket.AF_INET
        sockaddr: tuple[Any, ...]
        if literal_ip.version == 6:
            sockaddr = (literal_ip.compressed.lower(), port, 0, 0)
        else:
            sockaddr = (literal_ip.compressed.lower(), port)
        return (
            _ResolvedEndpoint(
                family=family,
                socktype=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
                sockaddr=sockaddr,
                address_text=literal_ip.compressed.lower(),
            ),
        )

    try:
        address_info = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise RuntimeError("Calendar URL hostname could not be resolved.") from None

    if not address_info:
        raise RuntimeError("Calendar URL hostname could not be resolved.")

    endpoints: list[_ResolvedEndpoint] = []
    seen: set[tuple[int, str, int]] = set()
    for family, socktype, proto, _, sockaddr in address_info:
        candidate_ip = _parse_ip_address(sockaddr[0])
        if candidate_ip is None or not candidate_ip.is_global:
            raise RuntimeError("Calendar URL must point to a public internet host.")
        key = (family, candidate_ip.compressed.lower(), port)
        if key in seen:
            continue
        seen.add(key)
        endpoints.append(
            _ResolvedEndpoint(
                family=family,
                socktype=socktype,
                proto=proto,
                sockaddr=sockaddr,
                address_text=candidate_ip.compressed.lower(),
            )
        )
    return tuple(endpoints)


def _build_request_target(parts: SplitResult) -> str:
    """Return the request-target for an HTTP request."""

    path = parts.path or "/"
    if parts.query:
        return f"{path}?{parts.query}"
    return path


def _format_authority(hostname: str, explicit_port: int | None) -> str:
    """Format host[:port] with IPv6 bracket rules."""

    if ":" in hostname and not hostname.startswith("["):
        host_label = f"[{hostname}]"
    else:
        host_label = hostname
    if explicit_port is None:
        return host_label
    return f"{host_label}:{explicit_port}"


def _validate_remote_content_length(headers: http.client.HTTPMessage) -> None:
    """Validate the declared content length if present."""

    content_length = headers.get("Content-Length")
    if not content_length:
        return
    try:
        declared_size = int(content_length.strip())
    except ValueError:
        raise RuntimeError("Calendar feed size header is invalid.") from None
    if declared_size < 0:
        raise RuntimeError("Calendar feed size header is invalid.")
    if declared_size > _MAX_ICS_DOWNLOAD_BYTES:
        raise RuntimeError("Calendar feed is too large.")


def _compute_remote_refresh_interval(headers: http.client.HTTPMessage, default_refresh_interval_s: float) -> float:
    """Compute the next refresh interval from response headers."""

    cache_control = headers.get("Cache-Control", "")
    directives = [directive.strip().lower() for directive in cache_control.split(",") if directive.strip()]
    for directive in directives:
        if directive in {"no-cache", "no-store"}:
            return 0.0
        if directive.startswith("max-age="):
            try:
                max_age = float(directive.split("=", 1)[1].strip())
            except ValueError:
                break
            return min(max(max_age, 0.0), _MAX_REMOTE_ICS_REFRESH_INTERVAL_S)

    expires_header = headers.get("Expires", "").strip()
    if expires_header:
        try:
            expires_at = parsedate_to_datetime(expires_header)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            freshness_s = max(0.0, expires_at.timestamp() - time.time())
            return min(freshness_s, _MAX_REMOTE_ICS_REFRESH_INTERVAL_S)
        except (TypeError, ValueError, OverflowError):
            pass

    return min(max(default_refresh_interval_s, 0.0), _MAX_REMOTE_ICS_REFRESH_INTERVAL_S)


def _parse_retry_after_seconds(raw_value: str) -> float:
    """Parse a Retry-After header into bounded seconds."""

    text = raw_value.strip()
    if not text:
        return 0.0
    try:
        return min(max(float(text), 0.0), 5.0)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(text)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return min(max(retry_at.timestamp() - time.time(), 0.0), 5.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _validate_calendar_url_text(url: str, *, allow_insecure_http: bool = False) -> None:
    """Validate the textual shape of a remote calendar URL."""

    if not url:
        raise ValueError("Calendar URL is required.")
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in url):
        raise ValueError("Calendar URL must not contain whitespace or control characters.")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("Calendar URL must start with http:// or https://.")
    if scheme == "http" and not allow_insecure_http:
        raise ValueError("Calendar URL must use HTTPS unless insecure HTTP is explicitly enabled.")
    try:
        _ = parts.port
    except ValueError:
        raise ValueError("Calendar URL port is invalid.") from None
    if parts.username or parts.password or parts.fragment:
        raise ValueError(
            "Calendar URL must not include embedded credentials or fragments. "
            "Use a plain feed URL or a local ICS file."
        )


def _display_url(url: str) -> str:
    """Render a remote URL without credentials, query, or fragment data."""

    parts = urlsplit(url)
    hostname = parts.hostname or ""
    netloc = _format_authority(hostname, parts.port)
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _parse_ip_address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse a literal IP address if the hostname already is one."""

    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


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
