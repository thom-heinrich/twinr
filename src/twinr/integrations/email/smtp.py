"""Send validated email drafts through SMTP.

This module validates SMTP transport settings, builds RFC 5322 messages from
``EmailDraft`` records, and delivers them over STARTTLS or implicit TLS.
"""

from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import make_msgid
import logging
import smtplib
import ssl
from typing import Callable

from twinr.integrations.email.adapter import MailSender
from twinr.integrations.email.models import EmailDraft, normalize_email

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SMTPMailSenderConfig:
    """Hold validated SMTP transport settings for one mailbox account."""

    host: str
    username: str
    password: str
    from_address: str
    port: int = 587
    use_starttls: bool = True
    use_ssl: bool = False
    timeout_s: float = 20.0

    def __post_init__(self) -> None:
        normalized_from = normalize_email(self.from_address)
        host = self.host.strip()

        # AUDIT-FIX(#1): Reject contradictory TLS modes so the transport configuration is unambiguous and secure.
        if self.use_ssl and self.use_starttls:
            raise ValueError("SMTP use_ssl and use_starttls are mutually exclusive")
        # AUDIT-FIX(#1): Enforce encrypted transport because Twinr must never send senior data or credentials over plaintext SMTP.
        if not self.use_ssl and not self.use_starttls:
            raise ValueError("SMTP transport encryption must be enabled")

        # AUDIT-FIX(#7): Fail fast on invalid connection settings instead of discovering them only after opening sockets.
        if not host:
            raise ValueError("SMTP host must not be empty")
        # AUDIT-FIX(#7): Validate port bounds explicitly for deterministic configuration errors.
        if not 1 <= self.port <= 65535:
            raise ValueError("SMTP port must be between 1 and 65535")
        # AUDIT-FIX(#7): Zero/negative timeouts are invalid for a network client and can produce confusing runtime failures.
        if self.timeout_s <= 0:
            raise ValueError("SMTP timeout_s must be greater than 0")

        object.__setattr__(self, "host", host)
        object.__setattr__(self, "from_address", normalized_from)


class SMTPMailSender(MailSender):
    """Send Twinr email drafts through a configured SMTP transport."""

    def __init__(
        self,
        config: SMTPMailSenderConfig,
        *,
        connection_factory: Callable[[SMTPMailSenderConfig], object] | None = None,
    ) -> None:
        self.config = config
        self._connection_factory = connection_factory or self._default_connection

    def send(self, draft: EmailDraft) -> str | None:
        """Send one draft and return the provider-assigned message ID.

        Args:
            draft: Validated outbound email draft to send.

        Returns:
            The generated ``Message-ID`` header, or ``None`` if the transport
            cannot provide one.
        """
        # AUDIT-FIX(#4): Build and validate the message before opening the SMTP socket so malformed drafts cannot leak connections.
        message = self._build_message(draft)
        connection = self._connection_factory(self.config)
        try:
            self._start_session(connection)
            # AUDIT-FIX(#2): Sending is required work; treat a missing send_message implementation as a hard failure instead of silent success.
            refused = self._require_method(connection, "send_message")(message)
            # AUDIT-FIX(#2): send_message can report per-recipient refusal without raising; surface that as a real failure.
            if refused:
                raise smtplib.SMTPRecipientsRefused(refused)
            return message["Message-ID"]
        finally:
            # AUDIT-FIX(#5): Best-effort cleanup must never replace the original SMTP/send exception.
            self._close_connection(connection)

    def _build_message(self, draft: EmailDraft) -> EmailMessage:
        """Convert a validated draft into an RFC 5322 message object."""
        to_recipients = self._normalize_recipients(draft.to)
        cc_recipients = self._normalize_recipients(draft.cc)

        # AUDIT-FIX(#6): Fail fast with a deterministic error when a draft has no valid recipients.
        if not to_recipients and not cc_recipients:
            raise ValueError("Email draft must contain at least one recipient")

        message = EmailMessage()
        message["From"] = self.config.from_address
        if to_recipients:
            # AUDIT-FIX(#6): Do not emit an empty To header when the draft legitimately uses only Cc recipients.
            message["To"] = ", ".join(to_recipients)
        if cc_recipients:
            message["Cc"] = ", ".join(cc_recipients)
        message["Subject"] = draft.subject
        message["Message-ID"] = make_msgid(domain=self.config.from_address.split("@", 1)[-1])
        if draft.in_reply_to:
            message["In-Reply-To"] = draft.in_reply_to
        if draft.references:
            message["References"] = " ".join(draft.references)
        message.set_content(draft.body)
        return message

    def _normalize_recipients(self, recipients) -> list[str]:
        """Normalize recipients at the transport boundary."""
        # AUDIT-FIX(#6): Normalize every recipient at the transport boundary so invalid addresses fail before any network I/O starts.
        if not recipients:
            return []
        return [normalize_email(recipient) for recipient in recipients]

    def _default_connection(self, config: SMTPMailSenderConfig) -> object:
        """Build the default SMTP client from validated settings."""
        if config.use_ssl:
            # AUDIT-FIX(#3): Pass an explicitly verified TLS context; permissive library defaults are not acceptable for senior data.
            return smtplib.SMTP_SSL(
                config.host,
                config.port,
                timeout=config.timeout_s,
                context=self._create_tls_context(),
            )
        return smtplib.SMTP(config.host, config.port, timeout=config.timeout_s)

    def _start_session(self, connection: object) -> None:
        """Run the SMTP greeting, TLS, and login handshake."""
        self._require_method(connection, "ehlo")()
        if self.config.use_starttls and not self.config.use_ssl:
            starttls = self._require_method(connection, "starttls")
            try:
                # AUDIT-FIX(#3): STARTTLS must use a verified TLS context so the server identity is actually checked.
                starttls(context=self._create_tls_context())
            except TypeError:
                # AUDIT-FIX(#3): Keep test doubles/custom factories compatible when they implement starttls() without the context kwarg.
                starttls()
            self._require_method(connection, "ehlo")()
        if self.config.username:
            self._require_method(connection, "login")(self.config.username, self.config.password)

    def _create_tls_context(self) -> ssl.SSLContext:
        """Create the verified TLS context used for SMTP sessions."""
        # AUDIT-FIX(#3): create_default_context enables certificate validation and hostname checks for server authentication.
        return ssl.create_default_context()

    def _require_method(self, connection: object, method_name: str) -> Callable[..., object]:
        """Return a required SMTP connection method or raise ``TypeError``."""
        method = getattr(connection, method_name, None)
        if method is None or not callable(method):
            # AUDIT-FIX(#2): Missing transport methods are programmer/configuration errors and must not be silently ignored.
            raise TypeError(f"SMTP connection does not provide required method '{method_name}'")
        return method

    def _close_connection(self, connection: object) -> None:
        """Close the SMTP connection without masking the original failure."""
        try:
            # AUDIT-FIX(#5): Suppress cleanup-only failures so the original send error is not overwritten.
            quit_method = getattr(connection, "quit", None)
            if quit_method is not None and callable(quit_method):
                quit_method()
        except Exception:
            LOGGER.warning("SMTP QUIT failed during connection cleanup.", exc_info=True)

        try:
            # AUDIT-FIX(#5): Always fall back to close() so failed or missing QUIT does not leave sockets behind.
            close_method = getattr(connection, "close", None)
            if close_method is not None and callable(close_method):
                close_method()
        except Exception:
            LOGGER.warning("SMTP close failed during connection cleanup.", exc_info=True)

    def _call_optional(self, connection: object, method_name: str, *args):
        """Call an optional SMTP method when a test double exposes it."""
        method = getattr(connection, method_name, None)
        if method is None:
            return None
        return method(*args)
