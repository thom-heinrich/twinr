from __future__ import annotations

from dataclasses import dataclass
from email.utils import parseaddr
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
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
from twinr.integrations.store import ManagedIntegrationConfig, TwinrIntegrationStore

EMAIL_MAILBOX_INTEGRATION_ID = "email_mailbox"
CALENDAR_AGENDA_INTEGRATION_ID = "calendar_agenda"
EMAIL_APP_PASSWORD_ENV_KEY = "TWINR_INTEGRATION_EMAIL_APP_PASSWORD"
_MAX_ICS_DOWNLOAD_BYTES = 2 * 1024 * 1024
_ICS_URL_FETCH_TIMEOUT_S = 10.0


@dataclass(frozen=True, slots=True)
class IntegrationReadiness:
    integration_id: str
    label: str
    status: str
    summary: str
    detail: str
    warnings: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True, slots=True)
class ManagedIntegrationsRuntime:
    email_mailbox: EmailMailboxAdapter | None = None
    calendar_agenda: ReadOnlyCalendarAdapter | None = None
    readiness: tuple[IntegrationReadiness, ...] = ()

    def readiness_for(self, integration_id: str) -> IntegrationReadiness | None:
        for item in self.readiness:
            if item.integration_id == integration_id:
                return item
        return None


def build_managed_integrations(
    project_root: str | Path,
    *,
    env_path: str | Path | None = None,
    url_text_loader: Callable[[str], str] | None = None,
) -> ManagedIntegrationsRuntime:
    project_path = Path(project_root).resolve()
    store = TwinrIntegrationStore.from_project_root(project_path)
    env_values = _read_env_values(_resolve_env_path(project_path, env_path))

    email_adapter, email_readiness = _build_email_mailbox_runtime(
        store.get(EMAIL_MAILBOX_INTEGRATION_ID),
        env_values=env_values,
    )
    calendar_adapter, calendar_readiness = _build_calendar_agenda_runtime(
        project_path,
        store.get(CALENDAR_AGENDA_INTEGRATION_ID),
        url_text_loader=url_text_loader or _fetch_ics_url,
    )
    return ManagedIntegrationsRuntime(
        email_mailbox=email_adapter,
        calendar_agenda=calendar_adapter,
        readiness=(email_readiness, calendar_readiness),
    )


def build_email_mailbox_adapter(
    project_root: str | Path,
    *,
    env_path: str | Path | None = None,
) -> EmailMailboxAdapter | None:
    return build_managed_integrations(project_root, env_path=env_path).email_mailbox


def build_calendar_agenda_adapter(
    project_root: str | Path,
    *,
    env_path: str | Path | None = None,
    url_text_loader: Callable[[str], str] | None = None,
) -> ReadOnlyCalendarAdapter | None:
    return build_managed_integrations(
        project_root,
        env_path=env_path,
        url_text_loader=url_text_loader,
    ).calendar_agenda


def validate_calendar_source(*, source_kind: str, source_value: str) -> None:
    if source_kind != "ics_url":
        return

    parts = urlsplit(source_value)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError("Calendar URL must start with http:// or https://.")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(
            "Calendar URL must not include embedded credentials, query tokens, or fragments. "
            "Use a plain feed URL or a local ICS file."
        )


def _build_email_mailbox_runtime(
    record: ManagedIntegrationConfig,
    *,
    env_values: dict[str, str],
) -> tuple[EmailMailboxAdapter | None, IntegrationReadiness]:
    if not record.enabled:
        return None, IntegrationReadiness(
            integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
            label="Email",
            status="muted",
            summary="Disabled",
            detail="No mailbox connection is active. Configure account data and enable it when ready.",
        )

    profile = (record.value("profile", "gmail") or "gmail").strip()
    account_email = record.value("account_email", "").strip()
    from_address = record.value("from_address", "").strip() or account_email
    imap_host = record.value("imap_host", "imap.gmail.com" if profile == "gmail" else "").strip()
    smtp_host = record.value("smtp_host", "smtp.gmail.com" if profile == "gmail" else "").strip()
    secret_value = env_values.get(EMAIL_APP_PASSWORD_ENV_KEY, "").strip()

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
        imap_port = _parse_positive_int(record.value("imap_port", "993"), label="IMAP port")
        smtp_port = _parse_positive_int(record.value("smtp_port", "587"), label="SMTP port")
        contacts, contact_warnings = _parse_known_contacts(record.value("known_contacts_text", ""))
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
        raise RuntimeError("Built-in email integration manifest is missing.")

    adapter = EmailMailboxAdapter(
        manifest=manifest,
        contacts=contacts,
        mailbox_reader=IMAPMailboxReader(
            IMAPMailboxConfig(
                host=imap_host,
                port=imap_port,
                username=normalized_account,
                password=secret_value,
                mailbox=record.value("imap_mailbox", "INBOX").strip() or "INBOX",
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
            unread_only_default=_parse_bool(record.value("unread_only_default", "true"), default=True),
            restrict_reads_to_known_senders=_parse_bool(
                record.value("restrict_reads_to_known_senders", "false"),
                default=False,
            ),
            restrict_recipients_to_known_contacts=_parse_bool(
                record.value("restrict_recipients_to_known_contacts", "false"),
                default=False,
            ),
        ),
    )
    status = "warn" if contact_warnings else "ok"
    summary = "Ready with warnings" if contact_warnings else "Ready"
    detail = (
        f"{normalized_account} via {profile.replace('_', ' ')} · "
        f"IMAP {imap_host}:{imap_port} · SMTP {smtp_host}:{smtp_port} · "
        "credential stored separately in .env"
    )
    if contact_warnings:
        detail = f"{detail} · {contact_warnings[0]}"
    return adapter, IntegrationReadiness(
        integration_id=EMAIL_MAILBOX_INTEGRATION_ID,
        label="Email",
        status=status,
        summary=summary,
        detail=detail,
        warnings=contact_warnings,
    )


def _build_calendar_agenda_runtime(
    project_root: Path,
    record: ManagedIntegrationConfig,
    *,
    url_text_loader: Callable[[str], str],
) -> tuple[ReadOnlyCalendarAdapter | None, IntegrationReadiness]:
    if not record.enabled:
        return None, IntegrationReadiness(
            integration_id=CALENDAR_AGENDA_INTEGRATION_ID,
            label="Calendar",
            status="muted",
            summary="Disabled",
            detail="No agenda source is active. Configure an ICS file or feed when ready.",
        )

    source_kind = (record.value("source_kind", "ics_file") or "ics_file").strip()
    source_value = record.value("source_value", "").strip()
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
        reader, detail, warnings = _build_calendar_reader(
            project_root,
            source_kind=source_kind,
            source_value=source_value,
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
        raise RuntimeError("Built-in calendar integration manifest is missing.")

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
    default_timezone,
    url_text_loader: Callable[[str], str],
) -> tuple[ICSCalendarSource, str, tuple[str, ...]]:
    if source_kind == "ics_file":
        resolved_path = Path(source_value)
        if not resolved_path.is_absolute():
            resolved_path = (project_root / resolved_path).resolve()
        if not resolved_path.exists():
            raise ValueError(f"Configured ICS file does not exist: {resolved_path}")
        return (
            ICSCalendarSource.from_path(resolved_path, default_timezone=default_timezone),
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


def _parse_known_contacts(text: str) -> tuple[ApprovedEmailContacts, tuple[str, ...]]:
    contacts: list[EmailContact] = []
    warnings: list[str] = []
    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        display_name, email_address = parseaddr(line)
        normalized = normalize_email(email_address)
        if not normalized or "@" not in normalized:
            warnings.append(f"Ignored malformed contact line {index}.")
            continue
        contacts.append(
            EmailContact(
                email=normalized,
                display_name=display_name.strip() or normalized,
            )
        )
    return ApprovedEmailContacts(tuple(contacts)), tuple(warnings)


def _validate_email_address(value: str, *, label: str) -> str:
    normalized = normalize_email(value)
    if not normalized or "@" not in normalized:
        raise ValueError(f"{label} must be a valid email address.")
    return normalized


def _parse_positive_int(value: str, *, label: str) -> int:
    try:
        parsed = int(value.strip())
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number.") from None
    if parsed <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return parsed


def _parse_bool(value: str, *, default: bool) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Boolean value is invalid: {value}")


def _resolve_timezone(value: str):
    try:
        return ZoneInfo(value.strip())
    except Exception:
        raise ValueError(f"Timezone is invalid: {value}") from None


def _timezone_label(value) -> str:
    return getattr(value, "key", str(value))


def _resolve_env_path(project_root: Path, env_path: str | Path | None) -> Path:
    if env_path is None:
        return project_root / ".env"
    path = Path(env_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _read_env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _fetch_ics_url(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Twinr/0.1"})
    with urlopen(request, timeout=_ICS_URL_FETCH_TIMEOUT_S) as response:
        payload = response.read(_MAX_ICS_DOWNLOAD_BYTES + 1)
        if len(payload) > _MAX_ICS_DOWNLOAD_BYTES:
            raise RuntimeError("Calendar feed is too large.")
        charset = response.headers.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def _display_url(url: str) -> str:
    parts = urlsplit(url)
    hostname = parts.hostname or ""
    netloc = hostname
    if parts.port is not None:
        netloc = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
