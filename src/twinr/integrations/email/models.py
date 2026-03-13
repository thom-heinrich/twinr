from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


def normalize_email(value: str) -> str:
    return value.strip().lower()


@dataclass(frozen=True, slots=True)
class EmailContact:
    email: str
    display_name: str
    aliases: tuple[str, ...] = ()
    allow_read: bool = True
    allow_send: bool = True

    def __post_init__(self) -> None:
        normalized = normalize_email(self.email)
        if not normalized or "@" not in normalized:
            raise ValueError("EmailContact.email must be a valid email address.")
        object.__setattr__(self, "email", normalized)
        object.__setattr__(self, "aliases", tuple(alias.strip().lower() for alias in self.aliases if alias.strip()))


@dataclass(frozen=True, slots=True)
class EmailMessageSummary:
    message_id: str
    sender_email: str
    sender_name: str
    subject: str
    received_at: datetime | None = None
    preview: str = ""
    unread: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "message_id": self.message_id,
            "sender_email": self.sender_email,
            "sender_name": self.sender_name,
            "subject": self.subject,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "preview": self.preview,
            "unread": self.unread,
        }


@dataclass(frozen=True, slots=True)
class EmailDraft:
    to: tuple[str, ...]
    subject: str
    body: str
    cc: tuple[str, ...] = ()
    in_reply_to: str | None = None
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.to:
            raise ValueError("EmailDraft requires at least one recipient.")
        object.__setattr__(self, "to", tuple(normalize_email(address) for address in self.to))
        object.__setattr__(self, "cc", tuple(normalize_email(address) for address in self.cc))

    def recipients(self) -> tuple[str, ...]:
        return self.to + self.cc


@dataclass(slots=True)
class ApprovedEmailContacts:
    contacts: tuple[EmailContact, ...] = field(default_factory=tuple)
    _by_email: dict[str, EmailContact] = field(init=False, repr=False)
    _by_alias: dict[str, EmailContact] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_email: dict[str, EmailContact] = {}
        self._by_alias: dict[str, EmailContact] = {}
        for contact in self.contacts:
            self._by_email[contact.email] = contact
            self._by_alias[contact.display_name.strip().lower()] = contact
            for alias in contact.aliases:
                self._by_alias[alias] = contact

    def resolve(self, identifier: str) -> EmailContact | None:
        normalized = normalize_email(identifier)
        if normalized in self._by_email:
            return self._by_email[normalized]
        return self._by_alias.get(identifier.strip().lower())

    def can_read_from(self, sender_email: str) -> bool:
        contact = self.resolve(sender_email)
        return bool(contact and contact.allow_read)

    def can_send_to(self, recipient_email: str) -> bool:
        contact = self.resolve(recipient_email)
        return bool(contact and contact.allow_send)

    def require_allowed_recipients(self, recipients: tuple[str, ...]) -> tuple[EmailContact, ...]:
        approved: list[EmailContact] = []
        for recipient in recipients:
            contact = self.resolve(recipient)
            if contact is None or not contact.allow_send:
                raise ValueError(f"Recipient {recipient!r} is not on the approved email allowlist.")
            approved.append(contact)
        return tuple(approved)
