"""Run bounded mailbox connectivity checks for the managed email wizard.

This module keeps operator setup verification separate from the normal read and
send adapters. It performs one IMAP login/mailbox-open check and one SMTP
greeting/login check, then returns only redacted operator-facing status text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import imaplib
import smtplib
import ssl
from typing import Callable

from twinr.integrations.email.imap import IMAPMailboxConfig, IMAPMailboxReader
from twinr.integrations.email.smtp import SMTPMailSender, SMTPMailSenderConfig


@dataclass(frozen=True, slots=True)
class EmailTransportProbe:
    """Describe one redacted probe result for IMAP or SMTP."""

    status: str
    summary: str
    detail: str


@dataclass(frozen=True, slots=True)
class EmailConnectionTestResult:
    """Describe the full redacted connection-test outcome for one mailbox."""

    status: str
    summary: str
    detail: str
    imap: EmailTransportProbe
    smtp: EmailTransportProbe
    tested_at: str


def run_email_connectivity_test(
    imap_config: IMAPMailboxConfig,
    smtp_config: SMTPMailSenderConfig,
    *,
    mailbox_reader_factory: Callable[[IMAPMailboxConfig], IMAPMailboxReader] | None = None,
    mail_sender_factory: Callable[[SMTPMailSenderConfig], SMTPMailSender] | None = None,
) -> EmailConnectionTestResult:
    """Run one bounded IMAP+SMTP connectivity test with redacted status text."""

    reader = (mailbox_reader_factory or IMAPMailboxReader)(imap_config)
    sender = (mail_sender_factory or SMTPMailSender)(smtp_config)

    imap_result = _run_imap_probe(reader)
    smtp_result = _run_smtp_probe(sender)

    if imap_result.status == "ok" and smtp_result.status == "ok":
        status = "ok"
        summary = "Connection test passed"
        detail = "Twinr reached both the mailbox login and the outgoing mail server with the saved settings."
    elif imap_result.status != "ok" and smtp_result.status != "ok":
        status = "fail"
        summary = "Connection test failed"
        detail = "Twinr could not complete either mail login. Check the mailbox address, password, hosts, and ports."
    elif imap_result.status != "ok":
        status = "fail"
        summary = "Mailbox login failed"
        detail = "IMAP did not work even though SMTP did. Twinr still cannot use this mailbox."
    else:
        status = "fail"
        summary = "Outgoing mail login failed"
        detail = "SMTP did not work even though IMAP did. Twinr still cannot use this mailbox."

    return EmailConnectionTestResult(
        status=status,
        summary=summary,
        detail=detail,
        imap=imap_result,
        smtp=smtp_result,
        tested_at=datetime.now(UTC).isoformat(),
    )


def _run_imap_probe(reader: IMAPMailboxReader) -> EmailTransportProbe:
    """Run one IMAP login/open probe and convert failures into plain text."""

    try:
        reader.probe_connection()
    except Exception as exc:  # pragma: no cover - covered via concrete probe methods/tests
        return EmailTransportProbe(
            status="fail",
            summary="Failed",
            detail=_imap_failure_detail(exc),
        )
    return EmailTransportProbe(
        status="ok",
        summary="Connected",
        detail="Twinr logged in to IMAP and opened the selected mailbox folder.",
    )


def _run_smtp_probe(sender: SMTPMailSender) -> EmailTransportProbe:
    """Run one SMTP greeting/login probe and convert failures into plain text."""

    try:
        sender.probe_connection()
    except Exception as exc:  # pragma: no cover - covered via concrete probe methods/tests
        return EmailTransportProbe(
            status="fail",
            summary="Failed",
            detail=_smtp_failure_detail(exc),
        )
    return EmailTransportProbe(
        status="ok",
        summary="Connected",
        detail="Twinr reached SMTP, completed TLS, and logged in to the outgoing mail server.",
    )


def _imap_failure_detail(exc: Exception) -> str:
    """Return one redacted operator-facing IMAP failure summary."""

    root = _root_exception(exc)
    if isinstance(root, imaplib.IMAP4.error):
        return "The mailbox login was rejected. Check the address, password, and mailbox folder."
    if isinstance(root, imaplib.IMAP4.abort):
        return "The mailbox server closed the session before Twinr could finish the login."
    if isinstance(root, ssl.SSLError):
        return "TLS setup failed while Twinr was reaching the mailbox server."
    if isinstance(root, TimeoutError):
        return "The mailbox server did not answer in time."
    if isinstance(root, ValueError | TypeError | RuntimeError):
        return "The mailbox settings are incomplete or not compatible with this server."
    if isinstance(root, OSError):
        return "Twinr could not open the mailbox network connection."
    return "Twinr could not finish the mailbox connection test."


def _smtp_failure_detail(exc: Exception) -> str:
    """Return one redacted operator-facing SMTP failure summary."""

    root = _root_exception(exc)
    if isinstance(root, smtplib.SMTPAuthenticationError):
        return "The outgoing mail login was rejected. Check the address, password, and provider rules."
    if isinstance(root, smtplib.SMTPNotSupportedError):
        return "The outgoing mail server did not accept the required encryption setup."
    if isinstance(root, smtplib.SMTPServerDisconnected):
        return "The outgoing mail server closed the session before Twinr could finish the login."
    if isinstance(root, smtplib.SMTPException):
        return "Twinr reached SMTP but could not complete the outgoing login."
    if isinstance(root, ssl.SSLError):
        return "TLS setup failed while Twinr was reaching the outgoing mail server."
    if isinstance(root, TimeoutError):
        return "The outgoing mail server did not answer in time."
    if isinstance(root, ValueError | TypeError | RuntimeError):
        return "The outgoing mail settings are incomplete or not compatible with this server."
    if isinstance(root, OSError):
        return "Twinr could not open the outgoing mail network connection."
    return "Twinr could not finish the outgoing mail connection test."


def _root_exception(exc: Exception) -> Exception:
    """Return the deepest causal exception for stable failure classification."""

    current = exc
    while current.__cause__ is not None and isinstance(current.__cause__, Exception):
        current = current.__cause__
    return current


__all__ = [
    "EmailConnectionTestResult",
    "EmailTransportProbe",
    "run_email_connectivity_test",
]
