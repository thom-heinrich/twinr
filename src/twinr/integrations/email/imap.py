from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
import imaplib
from typing import Callable

from twinr.integrations.email.adapter import MailboxReader
from twinr.integrations.email.models import EmailMessageSummary, normalize_email
from twinr.text_utils import collapse_whitespace


@dataclass(frozen=True, slots=True)
class IMAPMailboxConfig:
    host: str
    username: str
    password: str
    mailbox: str = "INBOX"
    port: int = 993
    use_ssl: bool = True


class IMAPMailboxReader(MailboxReader):
    def __init__(
        self,
        config: IMAPMailboxConfig,
        *,
        connection_factory: Callable[[IMAPMailboxConfig], object] | None = None,
    ) -> None:
        self.config = config
        self._connection_factory = connection_factory or self._default_connection

    def list_recent(self, *, limit: int, unread_only: bool) -> list[EmailMessageSummary]:
        connection = self._connection_factory(self.config)
        try:
            self._call(connection, "login", self.config.username, self.config.password)
            self._expect_ok(self._call(connection, "select", self.config.mailbox, True))
            criterion = "UNSEEN" if unread_only else "ALL"
            search_response = self._expect_ok(self._call(connection, "search", None, criterion))
            message_ids = [token for token in search_response[1][0].split() if token][-limit:]

            messages: list[EmailMessageSummary] = []
            for message_id in reversed(message_ids):
                fetch_response = self._expect_ok(self._call(connection, "fetch", message_id, "(RFC822)"))
                raw_message = self._first_message_bytes(fetch_response[1])
                if raw_message is None:
                    continue
                messages.append(self._parse_message(raw_message, unread=unread_only))
            return messages
        finally:
            self._safe_call(connection, "close")
            self._safe_call(connection, "logout")

    def _default_connection(self, config: IMAPMailboxConfig) -> object:
        if config.use_ssl:
            return imaplib.IMAP4_SSL(config.host, config.port)
        return imaplib.IMAP4(config.host, config.port)

    def _parse_message(self, raw_message: bytes, *, unread: bool) -> EmailMessageSummary:
        message = BytesParser(policy=policy.default).parsebytes(raw_message)
        sender_name, sender_email = parseaddr(message.get("From", ""))
        received_at = self._parse_datetime(message.get("Date"))
        return EmailMessageSummary(
            message_id=str(message.get("Message-ID", "")).strip(),
            sender_email=normalize_email(sender_email),
            sender_name=sender_name.strip(),
            subject=self._decode_header_value(message.get("Subject", "")),
            received_at=received_at,
            preview=self._preview_for_message(message),
            unread=unread,
        )

    def _preview_for_message(self, message) -> str:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() != "text/plain":
                    continue
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                return self._normalize_preview(payload.decode(charset, errors="replace"))
            return ""

        payload = message.get_payload(decode=True)
        if payload is None:
            return self._normalize_preview(str(message.get_payload()))
        charset = message.get_content_charset() or "utf-8"
        return self._normalize_preview(payload.decode(charset, errors="replace"))

    def _normalize_preview(self, value: str) -> str:
        return collapse_whitespace(value)[:160]

    def _decode_header_value(self, value: str) -> str:
        if not value:
            return ""
        return str(make_header(decode_header(value))).strip()

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None

    def _call(self, connection: object, method_name: str, *args):
        method = getattr(connection, method_name)
        return method(*args)

    def _safe_call(self, connection: object, method_name: str) -> None:
        try:
            method = getattr(connection, method_name, None)
            if method is not None:
                method()
        except Exception:
            return

    def _expect_ok(self, response):
        status = response[0]
        if isinstance(status, bytes):
            status = status.decode("utf-8", errors="ignore")
        if str(status).upper() != "OK":
            raise RuntimeError(f"IMAP request failed with status {status!r}.")
        return response

    def _first_message_bytes(self, payload: object) -> bytes | None:
        if not isinstance(payload, list):
            return None
        for item in payload:
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], bytes):
                return item[1]
        return None
