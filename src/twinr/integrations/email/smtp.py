from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import make_msgid
import smtplib
from typing import Callable

from twinr.integrations.email.adapter import MailSender
from twinr.integrations.email.models import EmailDraft, normalize_email


@dataclass(frozen=True, slots=True)
class SMTPMailSenderConfig:
    host: str
    username: str
    password: str
    from_address: str
    port: int = 587
    use_starttls: bool = True
    use_ssl: bool = False
    timeout_s: float = 20.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "from_address", normalize_email(self.from_address))


class SMTPMailSender(MailSender):
    def __init__(
        self,
        config: SMTPMailSenderConfig,
        *,
        connection_factory: Callable[[SMTPMailSenderConfig], object] | None = None,
    ) -> None:
        self.config = config
        self._connection_factory = connection_factory or self._default_connection

    def send(self, draft: EmailDraft) -> str | None:
        connection = self._connection_factory(self.config)
        message = self._build_message(draft)
        try:
            self._start_session(connection)
            self._call_optional(connection, "send_message", message)
            return message["Message-ID"]
        finally:
            self._call_optional(connection, "quit")

    def _build_message(self, draft: EmailDraft) -> EmailMessage:
        message = EmailMessage()
        message["From"] = self.config.from_address
        message["To"] = ", ".join(draft.to)
        if draft.cc:
            message["Cc"] = ", ".join(draft.cc)
        message["Subject"] = draft.subject
        message["Message-ID"] = make_msgid(domain=self.config.from_address.split("@", 1)[-1])
        if draft.in_reply_to:
            message["In-Reply-To"] = draft.in_reply_to
        if draft.references:
            message["References"] = " ".join(draft.references)
        message.set_content(draft.body)
        return message

    def _default_connection(self, config: SMTPMailSenderConfig) -> object:
        if config.use_ssl:
            return smtplib.SMTP_SSL(config.host, config.port, timeout=config.timeout_s)
        return smtplib.SMTP(config.host, config.port, timeout=config.timeout_s)

    def _start_session(self, connection: object) -> None:
        self._call_optional(connection, "ehlo")
        if self.config.use_starttls and not self.config.use_ssl:
            self._call_optional(connection, "starttls")
            self._call_optional(connection, "ehlo")
        if self.config.username:
            self._call_optional(connection, "login", self.config.username, self.config.password)

    def _call_optional(self, connection: object, method_name: str, *args):
        method = getattr(connection, method_name, None)
        if method is None:
            return None
        return method(*args)
