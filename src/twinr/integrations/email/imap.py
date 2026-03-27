"""Read recent email summaries from an IMAP mailbox.

This module validates IMAP connection settings, opens bounded mailbox
sessions, and converts raw RFC822 messages into sanitized
``EmailMessageSummary`` records for Twinr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy
from email.errors import HeaderParseError
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from html.parser import HTMLParser
import imaplib
import logging
import ssl
import time
from typing import Callable

from twinr.integrations.email.adapter import MailboxReader
from twinr.integrations.email.models import EmailMessageSummary, normalize_email
from twinr.text_utils import collapse_whitespace


logger = logging.getLogger(__name__)


class _HTMLTextExtractor(HTMLParser):
    """Extract visible text nodes from a small HTML fragment."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join(self._chunks)


@dataclass(frozen=True, slots=True)
class IMAPMailboxConfig:
    """Hold validated IMAP connection and preview limits."""

    host: str
    username: str
    password: str = field(repr=False)  # AUDIT-FIX(#10): prevent accidental password leakage via dataclass repr/logging.
    mailbox: str = "INBOX"
    port: int = 993
    use_ssl: bool = True
    connect_timeout_seconds: float = 10.0  # AUDIT-FIX(#1): bounded connect timeout for unstable home Wi-Fi.
    operation_timeout_seconds: float = 10.0  # AUDIT-FIX(#1): bounded socket timeout for blocking IMAP operations.
    max_retries: int = 1  # AUDIT-FIX(#1): one bounded retry gives recovery without long stalls.
    retry_backoff_seconds: float = 0.5  # AUDIT-FIX(#1): short backoff avoids immediate hammering on transient failures.
    starttls: bool = True  # AUDIT-FIX(#2): require encrypted transport when plain IMAP socket is used.
    allow_insecure_auth: bool = False  # AUDIT-FIX(#2): explicit escape hatch for legacy servers; insecure by design.
    preview_char_limit: int = 160
    preview_byte_limit: int = 4096

    def __post_init__(self) -> None:
        # AUDIT-FIX(#12): fail fast on invalid configuration instead of failing late during IMAP dialogue.
        if not self.host.strip():
            raise ValueError("IMAP host must not be empty.")
        if not self.username.strip():
            raise ValueError("IMAP username must not be empty.")
        if not self.password:
            raise ValueError("IMAP password must not be empty.")
        if not self.mailbox.strip():
            raise ValueError("IMAP mailbox must not be empty.")
        if any(control in self.mailbox for control in ("\r", "\n", "\x00")):
            raise ValueError("IMAP mailbox contains invalid control characters.")
        if not self.use_ssl and self.port == 993:
            object.__setattr__(self, "port", 143)  # AUDIT-FIX(#13): plain IMAP/STARTTLS should not silently keep the SSL-only default port 993.
        if not (1 <= self.port <= 65535):
            raise ValueError("IMAP port must be between 1 and 65535.")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("IMAP connect timeout must be > 0.")
        if self.operation_timeout_seconds <= 0:
            raise ValueError("IMAP operation timeout must be > 0.")
        if self.max_retries < 0:
            raise ValueError("IMAP max_retries must be >= 0.")
        if self.retry_backoff_seconds < 0:
            raise ValueError("IMAP retry_backoff_seconds must be >= 0.")
        if self.preview_char_limit <= 0:
            raise ValueError("IMAP preview_char_limit must be > 0.")
        if self.preview_byte_limit <= 0:
            raise ValueError("IMAP preview_byte_limit must be > 0.")


class IMAPMailboxReader(MailboxReader):
    """List recent mailbox messages through a bounded IMAP session."""

    def __init__(
        self,
        config: IMAPMailboxConfig,
        *,
        connection_factory: Callable[[IMAPMailboxConfig], object] | None = None,
    ) -> None:
        self.config = config
        self._connection_factory = connection_factory or self._default_connection

    def list_recent(self, *, limit: int, unread_only: bool) -> list[EmailMessageSummary]:
        """Read recent mailbox summaries from the configured IMAP account.

        Args:
            limit: Maximum number of summaries to return.
            unread_only: Whether to restrict the search to unread messages.

        Returns:
            A newest-first list of sanitized message summaries.

        Raises:
            RuntimeError: If the mailbox cannot be read within the configured
                retry budget.
        """
        if limit <= 0:
            return []  # AUDIT-FIX(#9): avoid Python's [-0:] behaviour returning the full mailbox for limit=0.

        attempts = self.config.max_retries + 1  # AUDIT-FIX(#1): bounded retries for transient network/IMAP failures.
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._list_recent_once(limit=limit, unread_only=unread_only)
            except RuntimeError as exc:
                logger.warning("IMAP mailbox read failed: %s", exc)
                raise RuntimeError("Unable to read mailbox.") from exc  # AUDIT-FIX(#6): surface a stable, non-jargon failure to the caller.
            except (imaplib.IMAP4.abort, imaplib.IMAP4.error, OSError, ssl.SSLError, TimeoutError) as exc:
                last_error = exc
                logger.warning(
                    "IMAP mailbox read attempt %s/%s failed: %s",
                    attempt,
                    attempts,
                    exc.__class__.__name__,
                )
                if attempt >= attempts:
                    break
                self._sleep_before_retry(attempt)

        if last_error is not None:
            raise RuntimeError("Unable to read mailbox.") from last_error
        raise RuntimeError("Unable to read mailbox.")

    def probe_connection(self) -> None:
        """Log in once and open the configured mailbox with bounded timeouts.

        This reuses the same transport policy as normal mailbox reads but skips
        message search and fetch work. The method is intended for operator setup
        checks where Twinr must prove that the mailbox login and selected folder
        are reachable before the integration is enabled.
        """

        connection = self._connection_factory(self.config)
        try:
            self._prepare_connection(connection)
            self._expect_ok(self._call(connection, "login", self.config.username, self.config.password))
            self._expect_ok(self._call(connection, "select", self.config.mailbox, True))
        finally:
            self._safe_call(connection, "close")
            self._safe_call(connection, "logout")

    def _list_recent_once(self, *, limit: int, unread_only: bool) -> list[EmailMessageSummary]:
        """Run one IMAP search/fetch pass without retry handling."""
        connection = self._connection_factory(self.config)
        try:
            self._prepare_connection(connection)  # AUDIT-FIX(#1,#2): apply operation timeout and secure transport before login.
            self._expect_ok(self._call(connection, "login", self.config.username, self.config.password))  # AUDIT-FIX(#3): validate login status explicitly.
            self._expect_ok(self._call(connection, "select", self.config.mailbox, True))
            criterion = "UNSEEN" if unread_only else "ALL"
            search_response = self._expect_ok(self._call(connection, "uid", "SEARCH", None, criterion))  # AUDIT-FIX(#4): use stable UIDs instead of mutable sequence numbers.
            message_ids = self._extract_message_ids(search_response)[-limit:]  # AUDIT-FIX(#3): safely extract IDs from possibly empty/malformed responses.

            messages: list[EmailMessageSummary] = []
            for message_id in reversed(message_ids):
                try:
                    fetch_response = self._expect_ok(
                        self._call(connection, "uid", "FETCH", message_id, "(FLAGS RFC822)")  # AUDIT-FIX(#4,#5): fetch flags alongside RFC822 so unread status is per-message accurate.
                    )
                    raw_message = self._first_message_bytes(fetch_response[1])
                    if raw_message is None:
                        logger.warning("Skipping IMAP UID %r because FETCH returned no RFC822 payload.", message_id)
                        continue

                    unread = unread_only or not self._message_has_seen_flag(fetch_response[1])  # AUDIT-FIX(#5): preserve unread state even when listing ALL messages.
                    messages.append(self._parse_message(raw_message, unread=unread))
                except Exception:
                    logger.exception("Skipping IMAP UID %r because parsing failed.", message_id)  # AUDIT-FIX(#6,#11): skip a single broken message instead of failing the full list.
                    continue
            return messages
        finally:
            self._safe_call(connection, "close")
            self._safe_call(connection, "logout")

    def _default_connection(self, config: IMAPMailboxConfig) -> object:
        """Build the default IMAP client from validated settings."""
        ssl_context = ssl.create_default_context()  # AUDIT-FIX(#2): explicit default TLS context with certificate validation.
        if config.use_ssl:
            return imaplib.IMAP4_SSL(
                config.host,
                config.port,
                ssl_context=ssl_context,
                timeout=config.connect_timeout_seconds,  # AUDIT-FIX(#1): bound connection establishment time.
            )
        return imaplib.IMAP4(
            config.host,
            config.port,
            timeout=config.connect_timeout_seconds,  # AUDIT-FIX(#1): bound plain-socket connection establishment time.
        )

    def _prepare_connection(self, connection: object) -> None:
        """Apply socket timeouts and the configured transport policy."""
        self._apply_socket_timeout(connection)  # AUDIT-FIX(#1): avoid indefinite blocking on search/fetch/login calls.
        if self.config.use_ssl:
            return

        if self.config.starttls:
            self._starttls(connection)  # AUDIT-FIX(#2): upgrade plain IMAP connections before credentials are sent.
            self._apply_socket_timeout(connection)
            return

        if not self.config.allow_insecure_auth:
            raise RuntimeError("Plaintext IMAP authentication is disabled.")  # AUDIT-FIX(#2): make insecure transport an explicit opt-in.

        logger.warning("Proceeding with insecure plaintext IMAP authentication because allow_insecure_auth=True.")

    def _starttls(self, connection: object) -> None:
        """Upgrade a plain IMAP connection to STARTTLS."""
        starttls = getattr(connection, "starttls", None)
        if starttls is None or not callable(starttls):
            raise RuntimeError("IMAP STARTTLS is unavailable for this connection.")
        self._expect_ok(starttls(ssl_context=ssl.create_default_context()))

    def _apply_socket_timeout(self, connection: object) -> None:
        """Apply the configured operation timeout to the IMAP socket."""
        sock = getattr(connection, "sock", None)
        if sock is None:
            return
        settimeout = getattr(sock, "settimeout", None)
        if settimeout is None or not callable(settimeout):
            return
        settimeout(self.config.operation_timeout_seconds)

    def _sleep_before_retry(self, attempt: int) -> None:
        """Sleep for the configured linear retry backoff."""
        if self.config.retry_backoff_seconds <= 0:
            return
        time.sleep(self.config.retry_backoff_seconds * attempt)

    def _parse_message(self, raw_message: bytes, *, unread: bool) -> EmailMessageSummary:
        """Parse one raw RFC822 message into a sanitized summary."""
        message = BytesParser(policy=policy.default).parsebytes(raw_message)
        raw_from = self._header_as_text(message.get("From", ""))
        sender_name, sender_email = parseaddr(raw_from)
        received_at = self._parse_datetime(self._header_as_text(message.get("Date")))
        return EmailMessageSummary(
            message_id=self._header_as_text(message.get("Message-ID", "")).strip(),
            sender_email=normalize_email(sender_email),
            sender_name=self._decode_header_value(sender_name).strip(),  # AUDIT-FIX(#6): decode RFC 2047 display names instead of returning raw encoded words.
            subject=self._decode_header_value(self._header_as_text(message.get("Subject", ""))),
            received_at=received_at,
            preview=self._preview_for_message(message),
            unread=unread,
        )

    def _preview_for_message(self, message: EmailMessage) -> str:
        """Choose the best available preview text from a message body."""
        html_fallback: str | None = None

        if message.is_multipart():
            for part in message.walk():
                if part.is_multipart():
                    continue
                if part.get_content_disposition() == "attachment":
                    continue  # AUDIT-FIX(#7): do not use text attachments as the preview body.

                content_type = part.get_content_type()
                if content_type == "text/plain":
                    text = self._decode_part_text(part)
                    if text:
                        return self._normalize_preview(text)
                if content_type == "text/html" and html_fallback is None:
                    html_fallback = self._decode_part_text(part)

            return self._normalize_preview(self._html_to_text(html_fallback or ""))  # AUDIT-FIX(#7): HTML-only messages get a readable fallback preview.

        content_type = message.get_content_type()
        payload = self._decode_part_text(message)
        if content_type == "text/html":
            return self._normalize_preview(self._html_to_text(payload))
        return self._normalize_preview(payload)

    def _decode_part_text(self, part: EmailMessage) -> str:
        """Decode one MIME part into bounded text for previews."""
        payload = part.get_payload(decode=True)
        if payload is None:
            raw_payload = part.get_payload()
            if isinstance(raw_payload, str):
                return raw_payload
            if isinstance(raw_payload, list):
                return ""
            return str(raw_payload)

        preview_bytes = payload[: self.config.preview_byte_limit]  # AUDIT-FIX(#7): cap preview decode work on large messages/attachments.
        charset = part.get_content_charset() or "utf-8"
        for candidate in (charset, "utf-8", "latin-1"):
            try:
                return preview_bytes.decode(candidate, errors="replace")
            except LookupError:
                continue
        return preview_bytes.decode("utf-8", errors="replace")

    def _html_to_text(self, value: str) -> str:
        """Strip HTML markup into plain text for preview generation."""
        if not value:
            return ""
        parser = _HTMLTextExtractor()
        try:
            parser.feed(value)
            parser.close()
        except Exception:
            return value
        return parser.text()

    def _normalize_preview(self, value: str) -> str:
        """Collapse preview whitespace and enforce the configured limit."""
        return collapse_whitespace(value)[: self.config.preview_char_limit]

    def _decode_header_value(self, value: str) -> str:
        """Decode a mail header value while tolerating malformed input."""
        if not value:
            return ""
        try:
            return str(make_header(decode_header(value))).strip()
        except (HeaderParseError, UnicodeError, ValueError):
            return value.strip()  # AUDIT-FIX(#6): malformed headers should degrade to raw text, not abort listing.

    def _parse_datetime(self, value: str | None) -> datetime | None:
        """Parse a message date header into an aware UTC datetime."""
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)  # AUDIT-FIX(#8): normalize naive timestamps to UTC-aware values.
        return parsed.astimezone(timezone.utc)  # AUDIT-FIX(#8): downstream code now gets a single comparable timezone.

    def _header_as_text(self, value: object | None) -> str:
        """Coerce an optional header object into text."""
        if value is None:
            return ""
        return str(value)

    def _call(self, connection: object, method_name: str, *args):
        """Call a required IMAP method or raise a runtime error."""
        method = getattr(connection, method_name, None)
        if method is None or not callable(method):
            raise RuntimeError(f"IMAP connection is missing method {method_name!r}.")  # AUDIT-FIX(#11): fail clearly on bad test doubles or broken connections.
        return method(*args)

    def _safe_call(self, connection: object, method_name: str) -> None:
        """Invoke a best-effort IMAP cleanup method."""
        try:
            method = getattr(connection, method_name, None)
            if method is not None:
                method()
        except Exception:
            logger.debug("Ignoring IMAP cleanup failure during %s.", method_name, exc_info=True)  # AUDIT-FIX(#11): keep cleanup best-effort but preserve observability.

    def _expect_ok(self, response):
        """Require an IMAP response with an ``OK`` status."""
        if not isinstance(response, (tuple, list)) or not response:
            raise RuntimeError(f"Malformed IMAP response: {response!r}.")  # AUDIT-FIX(#3,#11): guard against empty/non-tuple responses.
        status = response[0]
        if isinstance(status, bytes):
            status = status.decode("utf-8", errors="ignore")
        if str(status).upper() != "OK":
            raise RuntimeError(f"IMAP request failed with status {status!r}.")
        return response

    def _extract_message_ids(self, response) -> list[bytes]:
        """Extract message UIDs from an IMAP SEARCH response."""
        if not isinstance(response, (tuple, list)) or len(response) < 2:
            return []
        payload = response[1]
        if not isinstance(payload, list) or not payload:
            return []

        first = payload[0]
        if isinstance(first, bytes):
            return [token for token in first.split() if token]
        if isinstance(first, str):
            return [token.encode("ascii", errors="ignore") for token in first.split() if token]
        return []

    def _message_has_seen_flag(self, payload: object) -> bool:
        """Return whether FETCH metadata marks a message as seen."""
        if not isinstance(payload, list):
            return False
        for item in payload:
            metadata: object
            if isinstance(item, tuple) and item:
                metadata = item[0]
            else:
                metadata = item

            if isinstance(metadata, bytes):
                text = metadata.decode("utf-8", errors="ignore")
            elif isinstance(metadata, str):
                text = metadata
            else:
                continue

            if "\\SEEN" in text.upper():
                return True
        return False

    def _first_message_bytes(self, payload: object) -> bytes | None:
        """Return the first RFC822 byte payload from a FETCH response."""
        if not isinstance(payload, list):
            return None
        for item in payload:
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], bytes):
                return item[1]
        return None
