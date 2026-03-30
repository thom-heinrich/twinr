"""Send validated email drafts through SMTP.

# CHANGELOG: 2026-03-30
# BUG-1: Fix partial-delivery handling; partial recipient refusals no longer masquerade as total failure.
# BUG-2: Add RFC 5322 Date header plus explicit SMTP envelope recipients, including optional Bcc support.
# BUG-3: Deduplicate recipients and validate every visible/hidden recipient before network I/O.
# SEC-1: Add optional OAuth2/XOAUTH2 SMTP AUTH support for modern providers that have deprecated basic auth.
# SEC-2: Reuse a hardened TLS client context with explicit TLS >= 1.2 and optional custom CA roots.
# IMP-1: Auto-select the standard submission port (587 for STARTTLS, 465 for implicit TLS) when none is provided.
# IMP-2: Use RFC-correct SMTP message policy, optional Reply-To/HTML support, and more robust EHLO/HELO negotiation.

This module validates SMTP transport settings, builds RFC 5322 messages from
``EmailDraft`` records, and delivers them over STARTTLS or implicit TLS.
"""

from __future__ import annotations

import email.policy
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
import logging
import smtplib
import ssl
from typing import Callable, Iterable, Literal

from twinr.integrations.email.adapter import MailSender
from twinr.integrations.email.models import EmailDraft, normalize_email

LOGGER = logging.getLogger(__name__)

_DEFAULT_STARTTLS_PORT = 587
_DEFAULT_SSL_PORT = 465


class SMTPPartialDeliveryError(smtplib.SMTPException):
    """Raised when some recipients accepted the mail and some were refused."""

    def __init__(
        self,
        *,
        refused_recipients: dict[str, tuple[int, bytes]],
        accepted_recipients: tuple[str, ...],
        message_id: str | None,
    ) -> None:
        super().__init__(
            "SMTP accepted the message for some recipients but refused others. "
            "Inspect 'accepted_recipients' and 'refused_recipients' before retrying."
        )
        self.refused_recipients = refused_recipients
        self.accepted_recipients = accepted_recipients
        self.message_id = message_id
        # Compatibility convenience for code that already expects a recipients mapping.
        self.recipients = refused_recipients


@dataclass(frozen=True, slots=True)
class SMTPMailSenderConfig:
    """Hold validated SMTP transport settings for one mailbox account."""

    host: str
    username: str
    password: str
    from_address: str
    port: int | None = None
    use_starttls: bool = True
    use_ssl: bool = False
    timeout_s: float = 20.0
    auth_mode: Literal["auto", "password", "xoauth2", "none"] = "auto"
    oauth2_access_token_supplier: Callable[[], str] | None = None
    ca_file: str | None = None
    ca_path: str | None = None
    ca_data: str | bytes | None = None
    minimum_tls_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2
    local_hostname: str | None = None
    source_address: tuple[str, int] | None = None

    def __post_init__(self) -> None:
        normalized_from = normalize_email(self.from_address)
        host = self.host.strip()
        username = self.username.strip()
        password = self.password or ""
        auth_mode = self.auth_mode.strip().lower()

        if self.use_ssl and self.use_starttls:
            raise ValueError("SMTP use_ssl and use_starttls are mutually exclusive")
        if not self.use_ssl and not self.use_starttls:
            raise ValueError("SMTP transport encryption must be enabled")
        if not host:
            raise ValueError("SMTP host must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("SMTP timeout_s must be greater than 0")

        if auth_mode not in {"auto", "password", "xoauth2", "none"}:
            raise ValueError("SMTP auth_mode must be one of: auto, password, xoauth2, none")

        resolved_port = self.port
        if resolved_port is None:
            resolved_port = _DEFAULT_SSL_PORT if self.use_ssl else _DEFAULT_STARTTLS_PORT
        if not 1 <= resolved_port <= 65535:
            raise ValueError("SMTP port must be between 1 and 65535")

        if auth_mode == "auto":
            if password and not username:
                raise ValueError("SMTP username must not be empty when a password is provided")
            if self.oauth2_access_token_supplier is not None and not username:
                raise ValueError(
                    "SMTP username must not be empty when an OAuth2 token supplier is provided"
                )

        if auth_mode == "password" and not username:
            raise ValueError("SMTP username must not be empty when auth_mode='password'")
        if auth_mode == "xoauth2":
            if not username:
                raise ValueError("SMTP username must not be empty when auth_mode='xoauth2'")
            if self.oauth2_access_token_supplier is None:
                raise ValueError(
                    "SMTP oauth2_access_token_supplier must be provided when auth_mode='xoauth2'"
                )
        if auth_mode == "none" and (
            username or password or self.oauth2_access_token_supplier is not None
        ):
            raise ValueError(
                "SMTP auth_mode='none' is incompatible with username/password/oauth2 credentials"
            )
        if not isinstance(self.minimum_tls_version, ssl.TLSVersion):
            raise ValueError("SMTP minimum_tls_version must be an ssl.TLSVersion value")
        if self.minimum_tls_version not in {ssl.TLSVersion.TLSv1_2, ssl.TLSVersion.TLSv1_3}:
            raise ValueError("SMTP minimum_tls_version must be TLSv1_2 or TLSv1_3")

        object.__setattr__(self, "host", host)
        object.__setattr__(self, "username", username)
        object.__setattr__(self, "password", password)
        object.__setattr__(self, "from_address", normalized_from)
        object.__setattr__(self, "port", resolved_port)
        object.__setattr__(self, "auth_mode", auth_mode)


@dataclass(frozen=True, slots=True)
class _PreparedMessage:
    """Internal envelope + RFC 5322 message payload."""

    message: EmailMessage
    envelope_sender: str
    envelope_recipients: tuple[str, ...]


class SMTPMailSender(MailSender):
    """Send Twinr email drafts through a configured SMTP transport."""

    def __init__(
        self,
        config: SMTPMailSenderConfig,
        *,
        connection_factory: Callable[[SMTPMailSenderConfig], object] | None = None,
    ) -> None:
        self.config = config
        self._tls_context = self._create_tls_context()
        self._connection_factory = connection_factory or self._default_connection

    def send(self, draft: EmailDraft) -> str | None:
        """Send one draft and return the RFC 5322 Message-ID.

        Args:
            draft: Validated outbound email draft to send.

        Returns:
            The outbound ``Message-ID`` header, or ``None`` if unavailable.

        Raises:
            SMTPPartialDeliveryError:
                # BREAKING: partial recipient refusal is now surfaced explicitly
                # instead of being misreported as "all recipients refused".
                The message was already accepted for at least one recipient, so
                blanket retries may create duplicates.
        """
        prepared = self._build_message(draft)
        connection = self._connection_factory(self.config)
        try:
            self._start_session(connection)
            refused = self._require_method(connection, "send_message")(
                prepared.message,
                from_addr=prepared.envelope_sender,
                to_addrs=list(prepared.envelope_recipients),
            )
            if refused:
                accepted = tuple(
                    recipient
                    for recipient in prepared.envelope_recipients
                    if recipient not in refused
                )
                LOGGER.warning(
                    "SMTP partially refused recipients.",
                    extra={
                        "accepted_recipients": accepted,
                        "refused_recipients": refused,
                        "message_id": prepared.message.get("Message-ID"),
                    },
                )
                raise SMTPPartialDeliveryError(
                    refused_recipients=refused,
                    accepted_recipients=accepted,
                    message_id=prepared.message.get("Message-ID"),
                )
            return prepared.message.get("Message-ID")
        finally:
            self._close_connection(connection)

    def probe_connection(self) -> None:
        """Open one bounded SMTP session and complete the login handshake.

        The setup wizard uses this to verify the saved outgoing-mail transport
        before Twinr allows the managed mailbox integration to be re-enabled.
        """
        connection = self._connection_factory(self.config)
        try:
            self._start_session(connection)
        finally:
            self._close_connection(connection)

    def _build_message(self, draft: EmailDraft) -> _PreparedMessage:
        """Convert a validated draft into an RFC 5322 message + SMTP envelope."""
        to_recipients = self._normalize_recipients(getattr(draft, "to", None))
        cc_recipients = self._normalize_recipients(getattr(draft, "cc", None))
        bcc_recipients = self._normalize_recipients(getattr(draft, "bcc", None))
        envelope_recipients = tuple(
            self._unique_preserve_order([*to_recipients, *cc_recipients, *bcc_recipients])
        )

        if not envelope_recipients:
            raise ValueError("Email draft must contain at least one recipient")

        message = EmailMessage(policy=email.policy.SMTP)
        message["From"] = self.config.from_address
        if to_recipients:
            message["To"] = ", ".join(to_recipients)
        if cc_recipients:
            message["Cc"] = ", ".join(cc_recipients)
        reply_to = self._coerce_optional_address_field(getattr(draft, "reply_to", None))
        if reply_to:
            message["Reply-To"] = reply_to
        message["Subject"] = getattr(draft, "subject")
        message["Date"] = formatdate(localtime=True)
        message["Message-ID"] = make_msgid(domain=self._message_id_domain())

        in_reply_to = getattr(draft, "in_reply_to", None)
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to

        references = getattr(draft, "references", None)
        if references:
            if isinstance(references, str):
                message["References"] = references
            else:
                message["References"] = " ".join(str(reference) for reference in references)

        body = getattr(draft, "body")
        html_body = getattr(draft, "html_body", None)
        if html_body:
            message.set_content(body, charset="utf-8")
            message.add_alternative(html_body, subtype="html", charset="utf-8")
        else:
            message.set_content(body, charset="utf-8")

        return _PreparedMessage(
            message=message,
            envelope_sender=self.config.from_address,
            envelope_recipients=envelope_recipients,
        )

    def _normalize_recipients(self, recipients: Iterable[str] | None) -> list[str]:
        """Normalize and deduplicate recipients at the transport boundary."""
        if not recipients:
            return []
        normalized = [normalize_email(recipient) for recipient in recipients]
        return self._unique_preserve_order(normalized)

    def _coerce_optional_address_field(self, value: object) -> str | None:
        """Normalize an optional single-address or multi-address header field."""
        if not value:
            return None
        if isinstance(value, str):
            return normalize_email(value)
        if isinstance(value, Iterable):
            normalized = self._normalize_recipients(value)
            return ", ".join(normalized) if normalized else None
        raise TypeError("SMTP address header field must be a string or iterable of strings")

    def _unique_preserve_order(self, values: Iterable[str]) -> list[str]:
        """Return values without duplicates while preserving the first occurrence."""
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique.append(value)
        return unique

    def _default_connection(self, config: SMTPMailSenderConfig) -> object:
        """Build the default SMTP client from validated settings."""
        if config.use_ssl:
            return smtplib.SMTP_SSL(
                config.host,
                config.port,
                local_hostname=config.local_hostname,
                timeout=config.timeout_s,
                source_address=config.source_address,
                context=self._tls_context,
            )
        return smtplib.SMTP(
            config.host,
            config.port,
            local_hostname=config.local_hostname,
            timeout=config.timeout_s,
            source_address=config.source_address,
        )

    def _start_session(self, connection: object) -> None:
        """Run the SMTP greeting, TLS, and authentication handshake."""
        self._ehlo_or_helo_if_needed(connection)

        if self.config.use_starttls and not self.config.use_ssl:
            starttls = self._require_method(connection, "starttls")
            try:
                starttls(context=self._tls_context)
            except TypeError:
                starttls()
            self._ehlo_or_helo_if_needed(connection)

        self._authenticate(connection)

    def _authenticate(self, connection: object) -> None:
        """Authenticate using password auth or XOAUTH2, depending on config."""
        auth_mode = self._resolve_auth_mode()
        if auth_mode == "none":
            return
        if auth_mode == "password":
            self._require_method(connection, "login")(self.config.username, self.config.password)
            return
        if auth_mode == "xoauth2":
            auth = self._require_method(connection, "auth")
            username = self.config.username
            token_supplier = self.config.oauth2_access_token_supplier
            if token_supplier is None:
                raise ValueError("SMTP oauth2_access_token_supplier is required for XOAUTH2")
            token = token_supplier().strip()
            if not token:
                raise ValueError("SMTP OAuth2 access token supplier returned an empty token")

            def _xoauth2(challenge: bytes | None = None) -> str:
                if challenge is not None:
                    LOGGER.debug("SMTP XOAUTH2 server challenge: %r", challenge)
                    # Gmail's documented XOAUTH2 failure flow expects an empty response.
                    return ""
                return f"user={username}\x01auth=Bearer {token}\x01\x01"

            auth("XOAUTH2", _xoauth2, initial_response_ok=True)
            return
        raise ValueError(f"Unsupported SMTP auth mode: {auth_mode}")

    def _resolve_auth_mode(self) -> Literal["password", "xoauth2", "none"]:
        """Resolve the effective SMTP authentication mode for this session."""
        if self.config.auth_mode == "password":
            return "password"
        if self.config.auth_mode == "xoauth2":
            return "xoauth2"
        if self.config.auth_mode == "none":
            return "none"

        if self.config.oauth2_access_token_supplier is not None:
            return "xoauth2"
        if self.config.username:
            return "password"
        return "none"

    def _ehlo_or_helo_if_needed(self, connection: object) -> None:
        """Negotiate ESMTP with graceful fallback for non-standard test doubles."""
        method = getattr(connection, "ehlo_or_helo_if_needed", None)
        if callable(method):
            method()
            return

        ehlo = getattr(connection, "ehlo", None)
        if callable(ehlo):
            ehlo()
            return

        self._require_method(connection, "helo")()

    def _create_tls_context(self) -> ssl.SSLContext:
        """Create the verified TLS context used for SMTP sessions."""
        context = ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH,
            cafile=self.config.ca_file,
            capath=self.config.ca_path,
            cadata=self.config.ca_data,
        )
        context.minimum_version = self.config.minimum_tls_version
        return context

    def _message_id_domain(self) -> str:
        """Return an ASCII-safe domain for Message-ID generation."""
        domain = self.config.from_address.rsplit("@", 1)[-1]
        try:
            return domain.encode("idna").decode("ascii")
        except UnicodeError:
            LOGGER.warning("Falling back to localhost for Message-ID domain.", exc_info=True)
            return "localhost"

    def _require_method(self, connection: object, method_name: str) -> Callable[..., object]:
        """Return a required SMTP connection method or raise ``TypeError``."""
        method = getattr(connection, method_name, None)
        if method is None or not callable(method):
            raise TypeError(f"SMTP connection does not provide required method '{method_name}'")
        return method

    def _close_connection(self, connection: object) -> None:
        """Close the SMTP connection without masking the original failure."""
        try:
            quit_method = getattr(connection, "quit", None)
            if quit_method is not None and callable(quit_method):
                quit_method()
        except Exception:
            LOGGER.warning("SMTP QUIT failed during connection cleanup.", exc_info=True)

        try:
            close_method = getattr(connection, "close", None)
            if close_method is not None and callable(close_method):
                close_method()
        except Exception:
            LOGGER.warning("SMTP close failed during connection cleanup.", exc_info=True)