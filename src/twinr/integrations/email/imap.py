# CHANGELOG: 2026-03-30
# BUG-1: Replaced O(N) mailbox-wide SEARCH + client-side tail slicing with capability-driven recent-ID selection:
#        UID SORT (REVERSE ARRIVAL) when available, otherwise bounded UIDNEXT-window search.
# BUG-2: Replaced full RFC822 downloads with bounded partial FETCHes and server PREVIEW support to avoid
#        latency spikes, RAM pressure, and large-message stalls on Raspberry Pi 4 deployments.
# BUG-3: Fixed sender parsing for RFC 2047 / internationalized From headers by preferring headerregistry
#        addresses over parseaddr-on-raw-header.
# SEC-1: HTML preview extraction now drops script/style/template/hidden content and strips bidi control
#        characters to reduce indirect prompt-injection and hidden-text spoofing into downstream AI prompts.
# SEC-2: Modern OAuth2 IMAP auth (XOAUTH2 / OAUTHBEARER) is supported; password auth remains for legacy
#        servers, but the module no longer forces password-only deployments.
# IMP-1: Added capability-driven PREVIEW (RFC 8970) support, UTF8=ACCEPT enablement, and safer cleanup via
#        unselect() when available.
# IMP-2: Added bounded TLS policy (TLS >= 1.2 where supported) and more resilient per-message fallback logic.
# IMP-3: Optional selectolax acceleration for HTML-to-text extraction with stdlib fallback.

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
import re
import ssl
import time
from typing import Callable

try:
    from selectolax.parser import HTMLParser as _SelectolaxHTMLParser
except Exception:  # pragma: no cover - optional fast path.
    _SelectolaxHTMLParser = None

from twinr.integrations.email.adapter import MailboxReader
from twinr.integrations.email.models import EmailMessageSummary, normalize_email
from twinr.text_utils import collapse_whitespace


logger = logging.getLogger(__name__)

_HIDDEN_STYLE_RE = re.compile(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.IGNORECASE)
_STATUS_UIDNEXT_RE = re.compile(r"\bUIDNEXT\s+(\d+)\b", re.IGNORECASE)
_BIDI_TRANSLATION = str.maketrans(
    {
        "\u202a": "",
        "\u202b": "",
        "\u202c": "",
        "\u202d": "",
        "\u202e": "",
        "\u2066": "",
        "\u2067": "",
        "\u2068": "",
        "\u2069": "",
    }
)
_HEADER_FETCH_FIELDS = (
    "DATE",
    "FROM",
    "SUBJECT",
    "MESSAGE-ID",
    "MIME-VERSION",
    "CONTENT-TYPE",
    "CONTENT-TRANSFER-ENCODING",
)


class _HTMLTextExtractor(HTMLParser):
    """Extract user-visible text from a small HTML fragment."""

    _SKIP_TAGS = {"head", "script", "style", "template", "noscript", "svg", "canvas", "title"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._hidden_stack: list[bool] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        lower_tag = tag.lower()
        attr_map = {str(key).lower(): ("" if value is None else str(value)) for key, value in attrs}

        if lower_tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return

        hidden = self._is_hidden(attr_map)
        self._hidden_stack.append(hidden)
        if hidden:
            return

        if lower_tag in {"br", "p", "div", "li", "tr", "td", "th", "section", "article"}:
            self._chunks.append(" ")

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        hidden = self._hidden_stack.pop() if self._hidden_stack else False
        if hidden:
            return

        if lower_tag in {"br", "p", "div", "li", "tr", "td", "th", "section", "article"}:
            self._chunks.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if any(self._hidden_stack):
            return
        if data:
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join(self._chunks)

    def _is_hidden(self, attrs: dict[str, str]) -> bool:
        if "hidden" in attrs:
            return True
        if attrs.get("aria-hidden", "").strip().lower() == "true":
            return True
        style = attrs.get("style", "")
        if style and _HIDDEN_STYLE_RE.search(style):
            return True
        return False


@dataclass(frozen=True, slots=True)
class IMAPMailboxConfig:
    """Hold validated IMAP connection and preview limits."""

    host: str
    username: str
    password: str = field(default="", repr=False)
    mailbox: str = "INBOX"
    port: int = 993
    use_ssl: bool = True
    connect_timeout_seconds: float = 10.0
    operation_timeout_seconds: float = 10.0
    max_retries: int = 1
    retry_backoff_seconds: float = 0.5
    starttls: bool = True
    allow_insecure_auth: bool = False
    preview_char_limit: int = 160
    preview_byte_limit: int = 4096
    fetch_byte_limit: int = 65536
    prefer_server_preview: bool = True
    enable_utf8: bool = True
    oauth2_access_token: str | None = field(default=None, repr=False)
    oauth2_auth_mechanism: str = "AUTO"

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("IMAP host must not be empty.")
        if not self.username.strip():
            raise ValueError("IMAP username must not be empty.")
        if not (self.password or self.oauth2_access_token):
            raise ValueError("Provide either an IMAP password or an OAuth2 access token.")
        if not self.mailbox.strip():
            raise ValueError("IMAP mailbox must not be empty.")
        if any(control in self.mailbox for control in ("\r", "\n", "\x00")):
            raise ValueError("IMAP mailbox contains invalid control characters.")
        if not self.use_ssl and self.port == 993:
            object.__setattr__(self, "port", 143)
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
        if self.fetch_byte_limit <= 0:
            raise ValueError("IMAP fetch_byte_limit must be > 0.")
        mechanism = self.oauth2_auth_mechanism.upper()
        if mechanism not in {"AUTO", "XOAUTH2", "OAUTHBEARER"}:
            raise ValueError("IMAP oauth2_auth_mechanism must be AUTO, XOAUTH2, or OAUTHBEARER.")


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
        """Read recent mailbox summaries from the configured IMAP account."""
        if limit <= 0:
            return []

        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._list_recent_once(limit=limit, unread_only=unread_only)
            except RuntimeError as exc:
                logger.warning("IMAP mailbox read failed: %s", exc)
                raise RuntimeError("Unable to read mailbox.") from exc
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

        raise RuntimeError("Unable to read mailbox.") from last_error

    def probe_connection(self) -> None:
        """Log in once and open the configured mailbox with bounded timeouts."""
        connection = self._connection_factory(self.config)
        try:
            self._prepare_connection(connection)
            self._authenticate(connection)
            self._post_auth_setup(connection)
            self._expect_ok(self._call(connection, "select", self.config.mailbox, True))
        finally:
            self._cleanup_connection(connection)

    def _list_recent_once(self, *, limit: int, unread_only: bool) -> list[EmailMessageSummary]:
        connection = self._connection_factory(self.config)
        try:
            self._prepare_connection(connection)
            self._authenticate(connection)
            self._post_auth_setup(connection)
            self._expect_ok(self._call(connection, "select", self.config.mailbox, True))

            supports_server_preview = self.config.prefer_server_preview and "PREVIEW" in self._capabilities(connection)
            message_ids = self._select_recent_message_ids(connection, limit=limit, unread_only=unread_only)

            messages: list[EmailMessageSummary] = []
            for message_id in message_ids:
                try:
                    messages.append(
                        self._fetch_message_summary(
                            connection,
                            message_id,
                            supports_server_preview=supports_server_preview,
                        )
                    )
                except Exception:
                    logger.exception("Skipping IMAP UID %r because parsing failed.", message_id)
                    continue
            return messages
        finally:
            self._cleanup_connection(connection)

    def _fetch_message_summary(
        self,
        connection: object,
        message_id: bytes,
        *,
        supports_server_preview: bool,
    ) -> EmailMessageSummary:
        if supports_server_preview:
            header_fetch = self._expect_ok(
                self._call(
                    connection,
                    "uid",
                    "FETCH",
                    message_id,
                    f'(FLAGS BODY.PEEK[HEADER.FIELDS ({self._header_fields_fetch_token()})])',
                )
            )
            header_bytes = self._first_message_bytes(header_fetch[1]) or b""
            message = BytesParser(policy=policy.default).parsebytes(header_bytes)
            preview = ""
            try:
                preview = self._fetch_server_preview(connection, message_id) or ""
            except Exception:
                logger.debug("PREVIEW fetch failed for IMAP UID %r; falling back to bounded BODY.PEEK.", message_id, exc_info=True)
            if not preview:
                preview = self._preview_from_partial_message_fetch(connection, message_id)
            unread = not self._message_has_seen_flag(header_fetch[1])
            return self._summary_from_message(message, preview=preview, unread=unread)

        fetch_response = self._expect_ok(
            self._call(
                connection,
                "uid",
                "FETCH",
                message_id,
                f"(FLAGS BODY.PEEK[]<0.{self.config.fetch_byte_limit}>)",
            )
        )
        raw_message = self._first_message_bytes(fetch_response[1])
        if raw_message is None:
            raise RuntimeError("IMAP FETCH returned no message bytes.")
        message = BytesParser(policy=policy.default).parsebytes(raw_message)
        preview = self._preview_for_message(message)
        unread = not self._message_has_seen_flag(fetch_response[1])
        return self._summary_from_message(message, preview=preview, unread=unread)

    def _summary_from_message(
        self,
        message: EmailMessage,
        *,
        preview: str,
        unread: bool,
    ) -> EmailMessageSummary:
        sender_name, sender_email = self._sender_from_message(message)
        received_at = self._parse_datetime(self._header_as_text(message.get("Date")))
        return EmailMessageSummary(
            message_id=self._sanitize_text(self._header_as_text(message.get("Message-ID", "")).strip()),
            sender_email=normalize_email(sender_email),
            sender_name=sender_name,
            subject=self._sanitize_text(self._decode_header_value(self._header_as_text(message.get("Subject", "")))),
            received_at=received_at,
            preview=self._normalize_preview(preview if preview else self._preview_for_message(message)),
            unread=unread,
        )

    def _sender_from_message(self, message: EmailMessage) -> tuple[str, str]:
        from_header = message.get("From")
        addresses = getattr(from_header, "addresses", None)
        if addresses:
            first = addresses[0]
            display_name = self._sanitize_text(str(getattr(first, "display_name", "") or ""))
            addr_spec = str(getattr(first, "addr_spec", "") or "")
            return display_name, addr_spec

        raw_from = self._header_as_text(from_header)
        sender_name, sender_email = parseaddr(raw_from)
        return self._sanitize_text(self._decode_header_value(sender_name).strip()), sender_email

    def _select_recent_message_ids(self, connection: object, *, limit: int, unread_only: bool) -> list[bytes]:
        criterion = "UNSEEN" if unread_only else "ALL"
        caps = self._capabilities(connection)

        if "SORT" in caps:
            try:
                response = self._expect_ok(
                    self._call(connection, "uid", "SORT", "(REVERSE ARRIVAL)", "UTF-8", criterion)
                )
                ids = self._extract_message_ids(response)
                if ids:
                    return ids[:limit]
            except Exception:
                logger.debug("IMAP SORT failed; falling back to UID-window search.", exc_info=True)

        return self._search_recent_ids_with_uid_windows(connection, limit=limit, unread_only=unread_only)

    def _search_recent_ids_with_uid_windows(
        self,
        connection: object,
        *,
        limit: int,
        unread_only: bool,
    ) -> list[bytes]:
        criterion = "UNSEEN" if unread_only else "ALL"
        uid_next = self._status_uidnext(connection)

        if uid_next is None or uid_next <= 1:
            response = self._expect_ok(self._call(connection, "uid", "SEARCH", None, criterion))
            return list(reversed(self._extract_message_ids(response)[-limit:]))

        collected: list[bytes] = []
        seen: set[bytes] = set()
        upper = uid_next - 1
        window = max(limit * 4, 128)

        while upper >= 1 and len(collected) < limit:
            lower = max(1, upper - window + 1)
            response = self._expect_ok(
                self._call(connection, "uid", "SEARCH", None, "UID", f"{lower}:{upper}", criterion)
            )
            window_ids = self._extract_message_ids(response)
            for uid in reversed(window_ids):
                if uid in seen:
                    continue
                seen.add(uid)
                collected.append(uid)
                if len(collected) >= limit:
                    break

            if lower == 1:
                break

            upper = lower - 1
            window *= 2

        return collected

    def _status_uidnext(self, connection: object) -> int | None:
        try:
            response = self._expect_ok(self._call(connection, "status", self.config.mailbox, "(UIDNEXT)"))
        except Exception:
            return None

        texts: list[str] = []
        payload = response[1]
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, bytes):
                    texts.append(item.decode("utf-8", errors="ignore"))
                elif isinstance(item, str):
                    texts.append(item)

        match = _STATUS_UIDNEXT_RE.search(" ".join(texts))
        if not match:
            return None
        return int(match.group(1))

    def _fetch_server_preview(self, connection: object, message_id: bytes) -> str | None:
        response = self._expect_ok(self._call(connection, "uid", "FETCH", message_id, "(PREVIEW)"))
        return self._extract_preview_value(response[1])

    def _preview_from_partial_message_fetch(self, connection: object, message_id: bytes) -> str:
        response = self._expect_ok(
            self._call(
                connection,
                "uid",
                "FETCH",
                message_id,
                f"(BODY.PEEK[]<0.{self.config.fetch_byte_limit}>)",
            )
        )
        raw_message = self._first_message_bytes(response[1])
        if raw_message is None:
            return ""
        try:
            message = BytesParser(policy=policy.default).parsebytes(raw_message)
        except Exception:
            return ""
        return self._preview_for_message(message)

    def _extract_preview_value(self, payload: object) -> str | None:
        if not isinstance(payload, list):
            return None

        metadata_chunks: list[bytes] = []
        for item in payload:
            if isinstance(item, tuple) and item:
                header = item[0]
                if isinstance(header, bytes):
                    metadata_chunks.append(header)
                elif isinstance(header, str):
                    metadata_chunks.append(header.encode("utf-8", errors="ignore"))

                if isinstance(header, bytes) and b"PREVIEW" in header.upper():
                    if len(item) >= 2 and isinstance(item[1], bytes):
                        return item[1].decode("utf-8", errors="replace")
                    if len(item) >= 2 and isinstance(item[1], str):
                        return item[1]

            elif isinstance(item, bytes):
                metadata_chunks.append(item)
            elif isinstance(item, str):
                metadata_chunks.append(item.encode("utf-8", errors="ignore"))

        metadata = b" ".join(metadata_chunks)
        if re.search(rb"\bPREVIEW\s+NIL\b", metadata, re.IGNORECASE):
            return None

        match = re.search(rb'PREVIEW\s+"((?:[^"\\]|\\.)*)"', metadata, re.IGNORECASE)
        if not match:
            return None

        encoded = match.group(1)
        return encoded.replace(b'\\"', b'"').replace(b"\\\\", b"\\").decode("utf-8", errors="replace")

    def _authenticate(self, connection: object) -> None:
        capabilities = self._capabilities(connection, refresh=True)
        token = self.config.oauth2_access_token

        if token:
            try:
                mechanism = self._choose_oauth_mechanism(capabilities)
            except RuntimeError:
                if self.config.password:
                    self._expect_ok(self._call(connection, "login", self.config.username, self.config.password))
                    return
                raise
            self._expect_ok(self._authenticate_oauth(connection, mechanism, token))
            return

        self._expect_ok(self._call(connection, "login", self.config.username, self.config.password))

    def _post_auth_setup(self, connection: object) -> None:
        capabilities = self._capabilities(connection, refresh=True)
        if self.config.enable_utf8 and "UTF8=ACCEPT" in capabilities:
            enable = getattr(connection, "enable", None)
            if callable(enable):
                try:
                    self._expect_ok(enable("UTF8=ACCEPT"))
                except Exception:
                    logger.debug("IMAP UTF8=ACCEPT enable failed.", exc_info=True)
        self._capabilities(connection, refresh=True)

    def _choose_oauth_mechanism(self, capabilities: set[str]) -> str:
        configured = self.config.oauth2_auth_mechanism.upper()
        if configured != "AUTO":
            expected = f"AUTH={configured}"
            if expected not in capabilities:
                raise RuntimeError(f"IMAP server does not advertise {expected}.")
            return configured
        if "AUTH=OAUTHBEARER" in capabilities:
            return "OAUTHBEARER"
        if "AUTH=XOAUTH2" in capabilities:
            return "XOAUTH2"
        raise RuntimeError("IMAP server does not advertise an OAuth2 SASL mechanism.")

    def _authenticate_oauth(self, connection: object, mechanism: str, access_token: str):
        payload = self._build_oauth_payload(mechanism, access_token)
        sent_payload = False

        def authcallback(_challenge: bytes) -> bytes:
            nonlocal sent_payload
            if sent_payload:
                return b""
            sent_payload = True
            return payload

        return self._call(connection, "authenticate", mechanism, authcallback)

    def _build_oauth_payload(self, mechanism: str, access_token: str) -> bytes:
        username = self.config.username
        if mechanism == "XOAUTH2":
            return f"user={username}\x01auth=Bearer {access_token}\x01\x01".encode("utf-8")
        if mechanism == "OAUTHBEARER":
            return (
                f"n,a={username},\x01"
                f"host={self.config.host}\x01"
                f"port={self.config.port}\x01"
                f"auth=Bearer {access_token}\x01\x01"
            ).encode("utf-8")
        raise RuntimeError(f"Unsupported OAuth mechanism {mechanism!r}.")

    def _default_connection(self, config: IMAPMailboxConfig) -> object:
        ssl_context = self._build_ssl_context()
        if config.use_ssl:
            return imaplib.IMAP4_SSL(
                config.host,
                config.port,
                ssl_context=ssl_context,
                timeout=config.connect_timeout_seconds,
            )
        return imaplib.IMAP4(
            config.host,
            config.port,
            timeout=config.connect_timeout_seconds,
        )

    def _build_ssl_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context()
        if hasattr(ssl, "TLSVersion"):
            try:
                context.minimum_version = ssl.TLSVersion.TLSv1_2
            except Exception:
                pass
        return context

    def _prepare_connection(self, connection: object) -> None:
        self._apply_socket_timeout(connection)
        if self.config.use_ssl:
            return

        if self.config.starttls:
            self._starttls(connection)
            self._apply_socket_timeout(connection)
            return

        if not self.config.allow_insecure_auth:
            raise RuntimeError("Plaintext IMAP authentication is disabled.")

        logger.warning("Proceeding with insecure plaintext IMAP authentication because allow_insecure_auth=True.")

    def _starttls(self, connection: object) -> None:
        starttls = getattr(connection, "starttls", None)
        if starttls is None or not callable(starttls):
            raise RuntimeError("IMAP STARTTLS is unavailable for this connection.")
        self._expect_ok(starttls(ssl_context=self._build_ssl_context()))

    def _capabilities(self, connection: object, *, refresh: bool = False) -> set[str]:
        if refresh:
            try:
                self._expect_ok(self._call(connection, "capability"))
            except Exception:
                logger.debug("IMAP CAPABILITY refresh failed.", exc_info=True)

        raw_capabilities = getattr(connection, "capabilities", ())
        parsed: set[str] = set()
        for item in raw_capabilities or ():
            if isinstance(item, bytes):
                parsed.update(token.upper() for token in item.decode("utf-8", errors="ignore").split())
            elif isinstance(item, str):
                parsed.update(token.upper() for token in item.split())
        return parsed

    def _apply_socket_timeout(self, connection: object) -> None:
        sock = getattr(connection, "sock", None)
        if sock is None:
            return
        settimeout = getattr(sock, "settimeout", None)
        if settimeout is None or not callable(settimeout):
            return
        settimeout(self.config.operation_timeout_seconds)

    def _cleanup_connection(self, connection: object) -> None:
        unselect = getattr(connection, "unselect", None)
        if callable(unselect):
            self._safe_call(connection, "unselect")
        else:
            self._safe_call(connection, "close")
        self._safe_call(connection, "logout")

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.config.retry_backoff_seconds <= 0:
            return
        time.sleep(self.config.retry_backoff_seconds * attempt)

    def _preview_for_message(self, message: EmailMessage) -> str:
        html_fallback: str | None = None

        if message.is_multipart():
            for part in message.walk():
                if part.is_multipart():
                    continue
                if part.get_content_disposition() == "attachment":
                    continue
                if part.get_filename():
                    continue

                content_type = part.get_content_type()
                if content_type == "text/plain":
                    text = self._decode_part_text(part)
                    if text:
                        return self._normalize_preview(text)
                if content_type == "text/html" and html_fallback is None:
                    html_fallback = self._decode_part_text(part)

            return self._normalize_preview(self._html_to_text(html_fallback or ""))

        content_type = message.get_content_type()
        payload = self._decode_part_text(message)
        if content_type == "text/html":
            return self._normalize_preview(self._html_to_text(payload))
        return self._normalize_preview(payload)

    def _decode_part_text(self, part: EmailMessage) -> str:
        payload = part.get_payload(decode=True)
        if payload is None:
            raw_payload = part.get_payload()
            if isinstance(raw_payload, str):
                return raw_payload[: self.config.preview_byte_limit]
            if isinstance(raw_payload, list):
                return ""
            return str(raw_payload)[: self.config.preview_byte_limit]

        preview_bytes = payload[: self.config.preview_byte_limit]
        charset = part.get_content_charset() or "utf-8"
        for candidate in (charset, "utf-8", "latin-1"):
            try:
                return preview_bytes.decode(candidate, errors="replace")
            except LookupError:
                continue
        return preview_bytes.decode("utf-8", errors="replace")

    def _html_to_text(self, value: str) -> str:
        if not value:
            return ""

        if _SelectolaxHTMLParser is not None:
            try:
                tree = _SelectolaxHTMLParser(value)
                tree.strip_tags(["head", "script", "style", "template", "noscript", "svg", "canvas"], recursive=True)
                for selector in (
                    '[hidden]',
                    '[aria-hidden="true"]',
                    '[style*="display:none"]',
                    '[style*="display: none"]',
                    '[style*="visibility:hidden"]',
                    '[style*="visibility: hidden"]',
                ):
                    for node in tree.css(selector):
                        node.remove()
                body = tree.body or tree.root
                body.merge_text_nodes()
                return body.text(deep=True, separator=" ", strip=True)
            except Exception:
                logger.debug("selectolax HTML extraction failed; falling back to html.parser.", exc_info=True)

        parser = _HTMLTextExtractor()
        try:
            parser.feed(value)
            parser.close()
        except Exception:
            return value
        return parser.text()

    def _normalize_preview(self, value: str) -> str:
        return collapse_whitespace(self._sanitize_text(value))[: self.config.preview_char_limit]

    def _sanitize_text(self, value: str) -> str:
        if not value:
            return ""
        cleaned = value.translate(_BIDI_TRANSLATION)
        cleaned = "".join(ch for ch in cleaned if ch in ("\t", "\n") or ord(ch) >= 32)
        return cleaned.strip()

    def _decode_header_value(self, value: str) -> str:
        if not value:
            return ""
        try:
            return str(make_header(decode_header(value))).strip()
        except (HeaderParseError, UnicodeError, ValueError):
            return value.strip()

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _header_as_text(self, value: object | None) -> str:
        if value is None:
            return ""
        return str(value)

    def _header_fields_fetch_token(self) -> str:
        return " ".join(_HEADER_FETCH_FIELDS)

    def _call(self, connection: object, method_name: str, *args):
        method = getattr(connection, method_name, None)
        if method is None or not callable(method):
            raise RuntimeError(f"IMAP connection is missing method {method_name!r}.")
        return method(*args)

    def _safe_call(self, connection: object, method_name: str) -> None:
        try:
            method = getattr(connection, method_name, None)
            if method is not None:
                method()
        except Exception:
            logger.debug("Ignoring IMAP cleanup failure during %s.", method_name, exc_info=True)

    def _expect_ok(self, response):
        if not isinstance(response, (tuple, list)) or not response:
            raise RuntimeError(f"Malformed IMAP response: {response!r}.")
        status = response[0]
        if isinstance(status, bytes):
            status = status.decode("utf-8", errors="ignore")
        if str(status).upper() != "OK":
            raise RuntimeError(f"IMAP request failed with status {status!r}.")
        return response

    def _extract_message_ids(self, response) -> list[bytes]:
        if not isinstance(response, (tuple, list)) or len(response) < 2:
            return []
        payload = response[1]
        if not isinstance(payload, list) or not payload:
            return []

        ids: list[bytes] = []
        for item in payload:
            if isinstance(item, bytes):
                ids.extend(token for token in item.split() if token)
            elif isinstance(item, str):
                ids.extend(token.encode("ascii", errors="ignore") for token in item.split() if token)
        return ids

    def _message_has_seen_flag(self, payload: object) -> bool:
        if not isinstance(payload, list):
            return False
        for item in payload:
            if isinstance(item, tuple) and item:
                metadata: object = item[0]
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
        if not isinstance(payload, list):
            return None
        for item in payload:
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], bytes):
                return item[1]
        return None