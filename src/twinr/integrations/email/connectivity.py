# CHANGELOG: 2026-03-30
# BUG-1: IMAP abort/readonly failures were misclassified as bad credentials because
#        imaplib.IMAP4.abort and imaplib.IMAP4.readonly are subclasses of IMAP4.error.
# BUG-2: Reader/sender factory construction errors bypassed redaction and could crash
#        the setup wizard before any EmailConnectionTestResult was returned.
# BUG-3: IMAP and SMTP probes ran sequentially and without a wrapper-level deadline,
#        so one slow server could stall the wizard far longer than intended.
# SEC-1: A slow or malicious mail endpoint could hang the setup flow on a Raspberry Pi
#        by holding open TCP/TLS/login phases indefinitely; probes are now deadline-
#        bounded and isolated in daemon threads.
# IMP-1: IMAP and SMTP probes now run in parallel for lower wall-clock latency.
# IMP-2: Failure classification now walks exception causes/contexts and distinguishes
#        DNS, certificate-validation, timeout, auth, and server-disconnect failures.
# BREAKING: EmailTransportProbe gained code and duration_ms fields, and
#           EmailConnectionTestResult gained duration_ms. Update strict serializers/tests.

"""Run bounded mailbox connectivity checks for the managed email wizard.

This module keeps operator setup verification separate from the normal read and
send adapters. It performs one IMAP login/mailbox-open check and one SMTP
greeting/login check, then returns only redacted operator-facing status text.

2026 upgrade goals for a Pi-hosted setup wizard:
- hard wall-clock bounds even when lower-level adapters misconfigure socket timeouts
- concurrent IMAP/SMTP probing to cut setup latency
- stable redacted classifications for common operator-remediable failures
- no raw exception leakage into operator-visible text
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import imaplib
from queue import Empty, Queue
import smtplib
import socket
import ssl
import threading
from time import monotonic
from typing import Callable

from twinr.integrations.email.imap import IMAPMailboxConfig, IMAPMailboxReader
from twinr.integrations.email.smtp import SMTPMailSender, SMTPMailSenderConfig

try:  # Python 3.11+
    _BASE_EXCEPTION_GROUP = BaseExceptionGroup  # type: ignore[name-defined]
except NameError:  # pragma: no cover - compatibility guard for older interpreters
    _BASE_EXCEPTION_GROUP = None


DEFAULT_PROBE_TIMEOUT_SECONDS = 20.0
_TIMEOUT_CONFIG_ATTRS = (
    "timeout_seconds",
    "timeout",
    "connect_timeout_seconds",
    "connect_timeout",
    "socket_timeout_seconds",
    "socket_timeout",
    "read_timeout_seconds",
    "read_timeout",
)


@dataclass(frozen=True, slots=True)
class EmailTransportProbe:
    """Describe one redacted probe result for IMAP or SMTP."""

    status: str
    summary: str
    detail: str
    code: str = "unknown"
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class EmailConnectionTestResult:
    """Describe the full redacted connection-test outcome for one mailbox."""

    status: str
    summary: str
    detail: str
    imap: EmailTransportProbe
    smtp: EmailTransportProbe
    tested_at: str
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class _FailureClassification:
    code: str
    summary: str
    detail: str


def run_email_connectivity_test(
    imap_config: IMAPMailboxConfig,
    smtp_config: SMTPMailSenderConfig,
    *,
    mailbox_reader_factory: Callable[[IMAPMailboxConfig], IMAPMailboxReader] | None = None,
    mail_sender_factory: Callable[[SMTPMailSenderConfig], SMTPMailSender] | None = None,
    timeout_seconds: float | None = None,
) -> EmailConnectionTestResult:
    """Run one bounded IMAP+SMTP connectivity test with redacted status text."""

    started_at = monotonic()
    imap_factory = mailbox_reader_factory or IMAPMailboxReader
    smtp_factory = mail_sender_factory or SMTPMailSender

    imap_timeout_seconds = _resolve_timeout_seconds(timeout_seconds, imap_config)
    smtp_timeout_seconds = _resolve_timeout_seconds(timeout_seconds, smtp_config)

    imap_result, smtp_result = _run_parallel_bounded_probes(
        imap_job=lambda: _run_imap_probe(imap_config, imap_factory),
        smtp_job=lambda: _run_smtp_probe(smtp_config, smtp_factory),
        imap_timeout_seconds=imap_timeout_seconds,
        smtp_timeout_seconds=smtp_timeout_seconds,
    )

    status, summary, detail = _overall_status(imap_result, smtp_result)
    return EmailConnectionTestResult(
        status=status,
        summary=summary,
        detail=detail,
        imap=imap_result,
        smtp=smtp_result,
        tested_at=datetime.now(UTC).isoformat(),
        duration_ms=_elapsed_ms(started_at),
    )


def _run_parallel_bounded_probes(
    *,
    imap_job: Callable[[], EmailTransportProbe],
    smtp_job: Callable[[], EmailTransportProbe],
    imap_timeout_seconds: float,
    smtp_timeout_seconds: float,
) -> tuple[EmailTransportProbe, EmailTransportProbe]:
    """Run both probes concurrently and bound each with its own deadline."""

    result_queue: Queue[tuple[str, EmailTransportProbe]] = Queue(maxsize=2)
    started_at = monotonic()
    deadlines = {
        "imap": started_at + imap_timeout_seconds,
        "smtp": started_at + smtp_timeout_seconds,
    }

    threads = (
        threading.Thread(
            target=_probe_worker,
            args=("imap", imap_job, result_queue),
            name="twinr-imap-probe",
            daemon=True,
        ),
        threading.Thread(
            target=_probe_worker,
            args=("smtp", smtp_job, result_queue),
            name="twinr-smtp-probe",
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()

    results: dict[str, EmailTransportProbe] = {}
    pending = {"imap", "smtp"}

    while pending:
        now = monotonic()
        expired = [name for name in pending if deadlines[name] <= now]
        for name in expired:
            timeout_seconds = max(0.001, deadlines[name] - started_at)
            results[name] = _timeout_probe(name, timeout_seconds)
            pending.remove(name)

        if not pending:
            break

        wait_for_seconds = min(max(0.0, deadlines[name] - monotonic()) for name in pending)
        if wait_for_seconds <= 0:
            continue

        try:
            name, result = result_queue.get(timeout=wait_for_seconds)
        except Empty:
            continue

        if name in pending:
            results[name] = result
            pending.remove(name)

    return results["imap"], results["smtp"]


def _probe_worker(
    name: str,
    job: Callable[[], EmailTransportProbe],
    result_queue: Queue[tuple[str, EmailTransportProbe]],
) -> None:
    """Run one probe job and publish its redacted result."""

    try:
        result = job()
    except Exception:  # pragma: no cover - defensive fallback
        result = EmailTransportProbe(
            status="fail",
            summary="Internal error",
            detail=f"Twinr hit an internal error while running the {name.upper()} connection test.",
            code="internal_error",
            duration_ms=None,
        )

    try:
        result_queue.put_nowait((name, result))
    except Exception:  # pragma: no cover - queue is bounded and local; this should not fail
        return


def _run_imap_probe(
    imap_config: IMAPMailboxConfig,
    mailbox_reader_factory: Callable[[IMAPMailboxConfig], IMAPMailboxReader],
) -> EmailTransportProbe:
    """Run one IMAP construction/login/open probe and convert failures into plain text."""

    started_at = monotonic()
    try:
        reader = mailbox_reader_factory(imap_config)
        reader.probe_connection()
    except Exception as exc:  # pragma: no cover - covered via concrete probe methods/tests
        return _failed_probe(_classify_imap_failure(exc), started_at)

    return EmailTransportProbe(
        status="ok",
        summary="Connected",
        detail="Twinr logged in to IMAP and opened the selected mailbox folder.",
        code="ok",
        duration_ms=_elapsed_ms(started_at),
    )


def _run_smtp_probe(
    smtp_config: SMTPMailSenderConfig,
    mail_sender_factory: Callable[[SMTPMailSenderConfig], SMTPMailSender],
) -> EmailTransportProbe:
    """Run one SMTP construction/greeting/login probe and convert failures into plain text."""

    started_at = monotonic()
    try:
        sender = mail_sender_factory(smtp_config)
        sender.probe_connection()
    except Exception as exc:  # pragma: no cover - covered via concrete probe methods/tests
        return _failed_probe(_classify_smtp_failure(exc), started_at)

    return EmailTransportProbe(
        status="ok",
        summary="Connected",
        detail="Twinr reached SMTP, completed TLS, and logged in to the outgoing mail server.",
        code="ok",
        duration_ms=_elapsed_ms(started_at),
    )


def _failed_probe(classification: _FailureClassification, started_at: float) -> EmailTransportProbe:
    return EmailTransportProbe(
        status="fail",
        summary=classification.summary,
        detail=classification.detail,
        code=classification.code,
        duration_ms=_elapsed_ms(started_at),
    )


def _timeout_probe(transport_name: str, timeout_seconds: float) -> EmailTransportProbe:
    transport_label = transport_name.upper()
    server_label = "mailbox server" if transport_name == "imap" else "outgoing mail server"
    return EmailTransportProbe(
        status="fail",
        summary="Timed out",
        detail=f"Twinr did not finish the {transport_label} probe against the {server_label} in time.",
        code="timeout",
        duration_ms=int(round(timeout_seconds * 1000)),
    )


def _classify_imap_failure(exc: Exception) -> _FailureClassification:
    """Return one redacted operator-facing IMAP failure summary."""

    chain = _exception_chain(exc)

    if _contains_exception(chain, ssl.SSLCertVerificationError):
        return _FailureClassification(
            code="tls_cert",
            summary="Certificate check failed",
            detail="Twinr reached the mailbox server, but its TLS certificate could not be verified.",
        )
    if _contains_exception(chain, socket.gaierror):
        return _FailureClassification(
            code="dns",
            summary="Server name not found",
            detail="Twinr could not resolve the mailbox server name. Check the IMAP host and the Pi network settings.",
        )
    if _contains_exception(chain, TimeoutError):
        return _FailureClassification(
            code="timeout",
            summary="Timed out",
            detail="The mailbox server did not answer in time.",
        )
    if _contains_exception(chain, imaplib.IMAP4.readonly):
        return _FailureClassification(
            code="mailbox_access",
            summary="Mailbox access failed",
            detail="Twinr reached IMAP, but the selected mailbox folder could not be opened with the required access.",
        )
    if _contains_exception(chain, imaplib.IMAP4.abort):
        return _FailureClassification(
            code="server_disconnect",
            summary="Server disconnected",
            detail="The mailbox server closed the session before Twinr could finish the IMAP probe.",
        )
    if _contains_exception(chain, imaplib.IMAP4.error):
        return _FailureClassification(
            code="auth",
            summary="Login rejected",
            detail="The mailbox login was rejected. Check the address, password, and mailbox folder.",
        )
    if _contains_exception(chain, ssl.SSLError):
        return _FailureClassification(
            code="tls",
            summary="TLS failed",
            detail="TLS setup failed while Twinr was reaching the mailbox server.",
        )
    if _contains_exception(chain, (ValueError, TypeError, RuntimeError)):
        return _FailureClassification(
            code="settings",
            summary="Settings invalid",
            detail="The mailbox settings are incomplete or not compatible with this server.",
        )
    if _contains_exception(chain, OSError):
        return _FailureClassification(
            code="network",
            summary="Network connection failed",
            detail="Twinr could not open the mailbox network connection.",
        )
    return _FailureClassification(
        code="unknown",
        summary="Failed",
        detail="Twinr could not finish the mailbox connection test.",
    )


def _classify_smtp_failure(exc: Exception) -> _FailureClassification:
    """Return one redacted operator-facing SMTP failure summary."""

    chain = _exception_chain(exc)

    if _contains_exception(chain, ssl.SSLCertVerificationError):
        return _FailureClassification(
            code="tls_cert",
            summary="Certificate check failed",
            detail="Twinr reached the outgoing mail server, but its TLS certificate could not be verified.",
        )
    if _contains_exception(chain, socket.gaierror):
        return _FailureClassification(
            code="dns",
            summary="Server name not found",
            detail="Twinr could not resolve the outgoing mail server name. Check the SMTP host and the Pi network settings.",
        )
    if _contains_exception(chain, TimeoutError):
        return _FailureClassification(
            code="timeout",
            summary="Timed out",
            detail="The outgoing mail server did not answer in time.",
        )
    if _contains_exception(chain, smtplib.SMTPAuthenticationError):
        return _FailureClassification(
            code="auth",
            summary="Login rejected",
            detail="The outgoing mail login was rejected. Check the address, password, and provider rules.",
        )
    if _contains_exception(chain, smtplib.SMTPNotSupportedError):
        return _FailureClassification(
            code="smtp_capability",
            summary="Server capability mismatch",
            detail="The outgoing mail server did not accept the required encryption or authentication setup.",
        )
    if _contains_exception(chain, smtplib.SMTPServerDisconnected):
        return _FailureClassification(
            code="server_disconnect",
            summary="Server disconnected",
            detail="The outgoing mail server closed the session before Twinr could finish the SMTP probe.",
        )
    if _contains_exception(chain, smtplib.SMTPConnectError):
        return _FailureClassification(
            code="smtp_connect",
            summary="Connection rejected",
            detail="Twinr reached the outgoing mail server, but the SMTP greeting or connect phase failed.",
        )
    if _contains_exception(chain, smtplib.SMTPHeloError):
        return _FailureClassification(
            code="smtp_greeting",
            summary="Greeting rejected",
            detail="The outgoing mail server rejected the SMTP greeting.",
        )
    if _contains_exception(chain, smtplib.SMTPException):
        return _FailureClassification(
            code="smtp",
            summary="SMTP failed",
            detail="Twinr reached SMTP but could not complete the outgoing login.",
        )
    if _contains_exception(chain, ssl.SSLError):
        return _FailureClassification(
            code="tls",
            summary="TLS failed",
            detail="TLS setup failed while Twinr was reaching the outgoing mail server.",
        )
    if _contains_exception(chain, (ValueError, TypeError, RuntimeError)):
        return _FailureClassification(
            code="settings",
            summary="Settings invalid",
            detail="The outgoing mail settings are incomplete or not compatible with this server.",
        )
    if _contains_exception(chain, OSError):
        return _FailureClassification(
            code="network",
            summary="Network connection failed",
            detail="Twinr could not open the outgoing mail network connection.",
        )
    return _FailureClassification(
        code="unknown",
        summary="Failed",
        detail="Twinr could not finish the outgoing mail connection test.",
    )


def _overall_status(
    imap_result: EmailTransportProbe,
    smtp_result: EmailTransportProbe,
) -> tuple[str, str, str]:
    """Build one redacted overall status from both transport probes."""

    if imap_result.status == "ok" and smtp_result.status == "ok":
        return (
            "ok",
            "Connection test passed",
            "Twinr reached both the mailbox login and the outgoing mail server with the saved settings.",
        )

    if imap_result.status != "ok" and smtp_result.status != "ok":
        if imap_result.code == smtp_result.code == "timeout":
            return (
                "fail",
                "Connection test timed out",
                "Neither mail server finished the connection test before the deadline. Check network reachability and timeout settings.",
            )
        if imap_result.code == smtp_result.code == "dns":
            return (
                "fail",
                "Mail server names not found",
                "Twinr could not resolve either saved mail server name. Check the hosts and the Pi network/DNS settings.",
            )
        if imap_result.code == smtp_result.code == "tls_cert":
            return (
                "fail",
                "Mail server certificate check failed",
                "Both mail servers were reached, but their TLS certificates could not be verified by this device.",
            )
        if imap_result.code == smtp_result.code == "auth":
            return (
                "fail",
                "Both mail logins were rejected",
                "Both IMAP and SMTP rejected the saved mailbox credentials or provider-specific login rules.",
            )
        return (
            "fail",
            "Connection test failed",
            "Twinr could not complete either mail login. Check the mailbox address, password, hosts, ports, and network path.",
        )

    if imap_result.status != "ok":
        summary = {
            "auth": "Mailbox login rejected",
            "dns": "Mailbox server name not found",
            "mailbox_access": "Mailbox access failed",
            "server_disconnect": "Mailbox server disconnected",
            "settings": "Mailbox settings invalid",
            "timeout": "Mailbox connection timed out",
            "tls": "Mailbox TLS failed",
            "tls_cert": "Mailbox certificate check failed",
        }.get(imap_result.code, "Mailbox connection failed")
        return (
            "fail",
            summary,
            f"{imap_result.detail} SMTP worked, but Twinr still cannot use this mailbox.",
        )

    summary = {
        "auth": "Outgoing mail login rejected",
        "dns": "Outgoing mail server name not found",
        "server_disconnect": "Outgoing mail server disconnected",
        "settings": "Outgoing mail settings invalid",
        "smtp_capability": "Outgoing mail server capability mismatch",
        "timeout": "Outgoing mail connection timed out",
        "tls": "Outgoing mail TLS failed",
        "tls_cert": "Outgoing mail certificate check failed",
    }.get(smtp_result.code, "Outgoing mail connection failed")
    return (
        "fail",
        summary,
        f"{smtp_result.detail} IMAP worked, but Twinr still cannot send mail from this mailbox.",
    )


def _exception_chain(exc: Exception) -> tuple[Exception, ...]:
    """Flatten explicit/implicit exception chains and ExceptionGroup leaves."""

    flattened: list[Exception] = []
    stack: list[Exception] = [exc]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        flattened.append(current)

        if _BASE_EXCEPTION_GROUP is not None and isinstance(current, _BASE_EXCEPTION_GROUP):
            nested = [nested for nested in current.exceptions if isinstance(nested, Exception)]
            stack.extend(reversed(nested))

        cause = current.__cause__
        if isinstance(cause, Exception):
            stack.append(cause)
            continue

        context = current.__context__
        if isinstance(context, Exception) and not current.__suppress_context__:
            stack.append(context)

    return tuple(flattened)


def _contains_exception(
    chain: tuple[Exception, ...],
    exc_type: type[Exception] | tuple[type[Exception], ...],
) -> bool:
    return any(isinstance(item, exc_type) for item in chain)


def _resolve_timeout_seconds(timeout_seconds: float | None, config: object) -> float:
    explicit = _coerce_positive_timeout_seconds(timeout_seconds)
    if explicit is not None:
        return explicit

    configured = _configured_timeout_seconds(config)
    if configured is not None:
        return configured

    return DEFAULT_PROBE_TIMEOUT_SECONDS


def _configured_timeout_seconds(config: object) -> float | None:
    for attr_name in _TIMEOUT_CONFIG_ATTRS:
        if not hasattr(config, attr_name):
            continue
        value = getattr(config, attr_name)
        timeout_seconds = _coerce_positive_timeout_seconds(value)
        if timeout_seconds is not None:
            return timeout_seconds
    return None


def _coerce_positive_timeout_seconds(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    try:
        timeout_seconds = float(value)
    except (TypeError, ValueError):
        return None

    if timeout_seconds <= 0:
        return None

    return timeout_seconds


def _elapsed_ms(started_at: float) -> int:
    return int(round((monotonic() - started_at) * 1000))


__all__ = [
    "EmailConnectionTestResult",
    "EmailTransportProbe",
    "run_email_connectivity_test",
]