from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from types import MappingProxyType


_EMAIL_LOCAL_PART_RE = re.compile(  # AUDIT-FIX(#1): Enforce conservative, injection-safe email syntax.
    r"(?i)^[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*$"
)
_EMAIL_DOMAIN_RE = re.compile(  # AUDIT-FIX(#1): Validate canonicalized IDNA domains and reject malformed hostnames.
    r"(?i)^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_FORBIDDEN_INLINE_CHARS = frozenset({"\r", "\n", "\x00", "\t"})  # AUDIT-FIX(#1): Block header/log injection primitives.
_FORBIDDEN_BODY_CHARS = frozenset({"\x00"})  # AUDIT-FIX(#1): NUL bytes are never valid in stored message bodies.


def _ensure_str(value: object, field_name: str) -> str:  # AUDIT-FIX(#1): Fail fast with controlled type errors instead of AttributeError.
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    return value


def _contains_forbidden_chars(value: str, forbidden_chars: frozenset[str]) -> bool:  # AUDIT-FIX(#1): Centralize control-character checks.
    return any(char in forbidden_chars for char in value)


def _normalize_alias_key(value: str) -> str:  # AUDIT-FIX(#2): Canonicalize alias/display-name keys and reject control characters.
    text = _ensure_str(value, "identifier")
    if _contains_forbidden_chars(text, _FORBIDDEN_INLINE_CHARS):
        raise ValueError("identifier contains forbidden control characters.")
    return " ".join(text.strip().lower().split())


def _normalize_display_name(value: object, field_name: str, *, allow_empty: bool = False) -> str:  # AUDIT-FIX(#2): Prevent blank names from becoming resolvable aliases.
    text = _ensure_str(value, field_name)
    if _contains_forbidden_chars(text, _FORBIDDEN_INLINE_CHARS):
        raise ValueError(f"{field_name} contains forbidden control characters.")
    normalized = " ".join(text.strip().split())
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _normalize_header_text(value: object, field_name: str, *, allow_empty: bool = False) -> str:  # AUDIT-FIX(#1): Reject CRLF/NUL in header-like fields.
    text = _ensure_str(value, field_name)
    if _contains_forbidden_chars(text, _FORBIDDEN_INLINE_CHARS):
        raise ValueError(f"{field_name} contains forbidden control characters.")
    normalized = text.strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _normalize_preview(value: object) -> str:  # AUDIT-FIX(#1): Keep previews single-line and safe for logs/UI surfaces.
    text = _ensure_str(value, "preview")
    if _contains_forbidden_chars(text, _FORBIDDEN_BODY_CHARS):
        raise ValueError("preview contains forbidden control characters.")
    return " ".join(text.replace("\r", " ").replace("\n", " ").replace("\t", " ").split())


def _normalize_body(value: object) -> str:  # AUDIT-FIX(#1): Allow normal multiline bodies while rejecting unsafe NUL bytes.
    text = _ensure_str(value, "body")
    if _contains_forbidden_chars(text, _FORBIDDEN_BODY_CHARS):
        raise ValueError("body contains forbidden control characters.")
    return text


def _coerce_bool(value: object, field_name: str) -> bool:  # AUDIT-FIX(#4): Avoid truthy string bugs such as "false" evaluating to True.
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise TypeError(f"{field_name} must be a boolean or a supported boolean-like literal.")


def _iter_strings(values: object, field_name: str) -> tuple[str, ...]:  # AUDIT-FIX(#1): Treat single strings as single items and reject mappings/char iteration bugs.
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    if isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a string or an iterable of strings, not a mapping.")
    if not isinstance(values, Iterable):
        raise TypeError(f"{field_name} must be a string or an iterable of strings.")
    items = tuple(values)
    normalized_items: list[str] = []
    for index, item in enumerate(items):
        normalized_items.append(_ensure_str(item, f"{field_name}[{index}]"))
    return tuple(normalized_items)


def _deduplicate_preserving_order(values: Iterable[str]) -> tuple[str, ...]:  # AUDIT-FIX(#7): Prevent duplicate recipients and duplicate aliases from propagating.
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def normalize_email(value: str) -> str:
    candidate = _ensure_str(value, "email").strip()  # AUDIT-FIX(#1): Convert AttributeError into explicit input validation.
    if not candidate:
        raise ValueError("email cannot be empty.")  # AUDIT-FIX(#1): Reject blank addresses before downstream authorization logic.
    if _contains_forbidden_chars(candidate, _FORBIDDEN_INLINE_CHARS):
        raise ValueError("email contains forbidden control characters.")  # AUDIT-FIX(#1): Block CRLF/NUL/TAB injection in address fields.
    if any(character.isspace() for character in candidate):
        raise ValueError("email cannot contain whitespace.")  # AUDIT-FIX(#1): Prevent hidden-address and parser ambiguity issues.
    if candidate.count("@") != 1:
        raise ValueError("email must contain a single '@' character.")  # AUDIT-FIX(#1): Reject malformed multi-@ addresses.
    local_part, domain_part = candidate.split("@", 1)
    if not local_part or not domain_part:
        raise ValueError("email must contain both local and domain parts.")  # AUDIT-FIX(#1): Reject partial addresses.
    if len(local_part) > 64 or len(candidate) > 254:
        raise ValueError("email exceeds RFC-compatible length limits.")  # AUDIT-FIX(#1): Reject pathological inputs early.
    if local_part[0] == "." or local_part[-1] == "." or ".." in local_part:
        raise ValueError("email local part is malformed.")  # AUDIT-FIX(#1): Reject dot-edge cases that often bypass simplistic validators.
    if not _EMAIL_LOCAL_PART_RE.fullmatch(local_part):
        raise ValueError("email local part is invalid.")  # AUDIT-FIX(#1): Enforce a conservative allowlist of safe characters.
    try:
        ascii_domain = domain_part.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("email domain is invalid.") from exc  # AUDIT-FIX(#1): Support IDNA domains without accepting malformed Unicode.
    if not _EMAIL_DOMAIN_RE.fullmatch(ascii_domain):
        raise ValueError("email domain is invalid.")  # AUDIT-FIX(#1): Reject malformed DNS labels and bare hostnames.
    return f"{local_part.lower()}@{ascii_domain.lower()}"  # AUDIT-FIX(#1): Canonicalize addresses for stable allowlist matching.


def _try_normalize_email(value: object) -> str | None:  # AUDIT-FIX(#1): Lookup paths should fail closed instead of raising on bad user input.
    if not isinstance(value, str):
        return None
    try:
        return normalize_email(value)
    except (TypeError, ValueError):
        return None


def _normalize_email_collection(values: object, *, field_name: str, allow_empty: bool) -> tuple[str, ...]:  # AUDIT-FIX(#1): Validate recipient collections and support single-string inputs safely.
    items = _iter_strings(values, field_name)
    normalized = _deduplicate_preserving_order(normalize_email(item) for item in items)
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} requires at least one valid email address.")
    return normalized


def _normalize_header_collection(values: object, *, field_name: str) -> tuple[str, ...]:  # AUDIT-FIX(#1): Normalize Message-ID style collections safely.
    items = _iter_strings(values, field_name)
    normalized = _deduplicate_preserving_order(
        _normalize_header_text(item, f"{field_name}[{index}]") for index, item in enumerate(items)
    )
    return normalized


def _normalize_identifier_collection(values: object, *, field_name: str, allow_empty: bool) -> tuple[str, ...]:  # AUDIT-FIX(#7): Validate alias/email identifier collections without char-by-char iteration.
    items = _iter_strings(values, field_name)
    normalized_items: list[str] = []
    for index, item in enumerate(items):
        if _contains_forbidden_chars(item, _FORBIDDEN_INLINE_CHARS):
            raise ValueError(f"{field_name}[{index}] contains forbidden control characters.")
        normalized = item.strip()
        if not normalized:
            raise ValueError(f"{field_name}[{index}] cannot be empty.")
        normalized_items.append(normalized)
    normalized = _deduplicate_preserving_order(normalized_items)
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} requires at least one identifier.")
    return normalized


def _normalize_received_at(value: datetime | str | None) -> datetime | None:  # AUDIT-FIX(#5): Enforce timezone-aware timestamps and normalize to UTC.
    if value is None:
        return None
    parsed = value
    if isinstance(parsed, str):
        candidate = parsed.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = f"{candidate[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError("received_at must be a valid ISO-8601 datetime.") from exc
    if not isinstance(parsed, datetime):
        raise TypeError("received_at must be a datetime, ISO-8601 string, or None.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("received_at must be timezone-aware.")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class EmailContact:
    email: str
    display_name: str
    aliases: tuple[str, ...] = ()
    allow_read: bool = True
    allow_send: bool = True

    def __post_init__(self) -> None:
        normalized = normalize_email(self.email)  # AUDIT-FIX(#1): Use strict email validation instead of '@' substring checks.
        display_name = _normalize_display_name(self.display_name, "display_name")  # AUDIT-FIX(#2): Reject blank/unsafe display names.
        aliases = _deduplicate_preserving_order(
            normalized_alias
            for alias in _iter_strings(self.aliases, "aliases")
            if (normalized_alias := _normalize_alias_key(alias))
        )  # AUDIT-FIX(#2): Canonicalize aliases and drop blank entries without silent collisions.
        allow_read = _coerce_bool(self.allow_read, "allow_read")  # AUDIT-FIX(#4): Prevent truthy-string permission bugs.
        allow_send = _coerce_bool(self.allow_send, "allow_send")  # AUDIT-FIX(#4): Prevent truthy-string permission bugs.
        object.__setattr__(self, "email", normalized)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "allow_read", allow_read)
        object.__setattr__(self, "allow_send", allow_send)


@dataclass(frozen=True, slots=True)
class EmailMessageSummary:
    message_id: str
    sender_email: str
    sender_name: str
    subject: str
    received_at: datetime | None = None
    preview: str = ""
    unread: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_id", _normalize_header_text(self.message_id, "message_id"))  # AUDIT-FIX(#1): Sanitize identifier headers.
        object.__setattr__(self, "sender_email", normalize_email(self.sender_email))  # AUDIT-FIX(#1): Keep sender addresses canonical and validated.
        object.__setattr__(self, "sender_name", _normalize_display_name(self.sender_name, "sender_name", allow_empty=True))  # AUDIT-FIX(#2): Strip unsafe whitespace/control chars.
        object.__setattr__(self, "subject", _normalize_header_text(self.subject, "subject", allow_empty=True))  # AUDIT-FIX(#1): Block CRLF injection in subject lines.
        object.__setattr__(self, "received_at", _normalize_received_at(self.received_at))  # AUDIT-FIX(#5): Normalize timestamps to timezone-aware UTC.
        object.__setattr__(self, "preview", _normalize_preview(self.preview))  # AUDIT-FIX(#1): Keep previews safe for UI/log surfaces.
        object.__setattr__(self, "unread", _coerce_bool(self.unread, "unread"))  # AUDIT-FIX(#4): Prevent string/untyped unread state bugs.

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
        to_addresses = _normalize_email_collection(self.to, field_name="to", allow_empty=False)  # AUDIT-FIX(#1): Support single-string recipients safely and validate all addresses.
        cc_addresses = _normalize_email_collection(self.cc, field_name="cc", allow_empty=True)  # AUDIT-FIX(#1): Validate CC addresses with the same guarantees as TO.
        to_set = set(to_addresses)
        cc_addresses = tuple(address for address in cc_addresses if address not in to_set)  # AUDIT-FIX(#7): Prevent duplicate recipients across TO and CC.
        subject = _normalize_header_text(self.subject, "subject", allow_empty=True)  # AUDIT-FIX(#1): Reject header injection in outgoing drafts.
        body = _normalize_body(self.body)  # AUDIT-FIX(#1): Keep multiline content but reject unsafe NUL bytes.
        in_reply_to = None
        if self.in_reply_to is not None:
            in_reply_to = _normalize_header_text(self.in_reply_to, "in_reply_to")  # AUDIT-FIX(#1): Sanitize reply threading headers.
        references = _normalize_header_collection(self.references, field_name="references")  # AUDIT-FIX(#1): Sanitize Message-ID reference headers.
        object.__setattr__(self, "to", to_addresses)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "cc", cc_addresses)
        object.__setattr__(self, "in_reply_to", in_reply_to)
        object.__setattr__(self, "references", references)

    def recipients(self) -> tuple[str, ...]:
        return self.to + self.cc


def _register_unique(mapping: dict[str, EmailContact], key: str, contact: EmailContact, *, key_kind: str) -> None:  # AUDIT-FIX(#2): Reject ambiguous allowlist keys instead of silently overwriting.
    existing = mapping.get(key)
    if existing is None:
        mapping[key] = contact
        return
    if existing is contact:
        return
    raise ValueError(f"Duplicate approved contact {key_kind} detected.")


@dataclass(frozen=True, slots=True)  # AUDIT-FIX(#6): Freeze allowlist containers so contact indexes cannot drift after construction.
class ApprovedEmailContacts:
    contacts: tuple[EmailContact, ...] = field(default_factory=tuple)
    _by_email: Mapping[str, EmailContact] = field(init=False, repr=False)
    _by_alias: Mapping[str, EmailContact] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        raw_contacts = self.contacts
        if isinstance(raw_contacts, EmailContact):
            contacts = (raw_contacts,)
        elif isinstance(raw_contacts, Mapping):
            raise TypeError("contacts must be an EmailContact or an iterable of EmailContact instances.")
        elif isinstance(raw_contacts, Iterable):
            contacts = tuple(raw_contacts)
        else:
            raise TypeError("contacts must be an EmailContact or an iterable of EmailContact instances.")
        by_email: dict[str, EmailContact] = {}
        by_alias: dict[str, EmailContact] = {}
        for index, contact in enumerate(contacts):
            if not isinstance(contact, EmailContact):
                raise TypeError(f"contacts[{index}] must be an EmailContact.")  # AUDIT-FIX(#6): Fail fast on malformed allowlist entries.
            _register_unique(by_email, contact.email, contact, key_kind="email")  # AUDIT-FIX(#2): Reject duplicate approved email addresses.
            _register_unique(by_alias, _normalize_alias_key(contact.display_name), contact, key_kind="display_name")  # AUDIT-FIX(#2): Reject duplicate names that would make voice resolution ambiguous.
            for alias in contact.aliases:
                _register_unique(by_alias, alias, contact, key_kind="alias")  # AUDIT-FIX(#2): Reject duplicate aliases across contacts.
        object.__setattr__(self, "contacts", contacts)
        object.__setattr__(self, "_by_email", MappingProxyType(by_email))  # AUDIT-FIX(#6): Expose read-only indexes to prevent accidental mutation.
        object.__setattr__(self, "_by_alias", MappingProxyType(by_alias))  # AUDIT-FIX(#6): Expose read-only indexes to prevent accidental mutation.

    def resolve(self, identifier: str) -> EmailContact | None:
        if not isinstance(identifier, str):
            return None  # AUDIT-FIX(#1): Invalid lookup inputs should fail closed.
        stripped_identifier = identifier.strip()
        if not stripped_identifier:
            return None  # AUDIT-FIX(#2): Blank lookups must never resolve to a contact.
        normalized = _try_normalize_email(stripped_identifier)
        if normalized is not None:
            contact = self._by_email.get(normalized)
            if contact is not None:
                return contact
        try:
            alias_key = _normalize_alias_key(stripped_identifier)
        except ValueError:
            return None  # AUDIT-FIX(#1): Unsafe identifiers should not bubble control-character errors through lookup helpers.
        return self._by_alias.get(alias_key)

    def can_read_from(self, sender_email: str) -> bool:
        normalized = _try_normalize_email(sender_email)  # AUDIT-FIX(#3): Incoming trust decisions must be based on canonical sender email only.
        if normalized is None:
            return False
        contact = self._by_email.get(normalized)
        return bool(contact and contact.allow_read)

    def can_send_to(self, recipient_email: str) -> bool:
        contact = self.resolve(recipient_email)
        return bool(contact and contact.allow_send)

    def require_allowed_recipients(self, recipients: tuple[str, ...]) -> tuple[EmailContact, ...]:
        approved: list[EmailContact] = []
        seen_emails: set[str] = set()
        for recipient in _normalize_identifier_collection(recipients, field_name="recipients", allow_empty=False):
            contact = self.resolve(recipient)
            if contact is None or not contact.allow_send:
                raise ValueError("One or more recipients are not approved for sending.")  # AUDIT-FIX(#7): Avoid echoing raw untrusted input into errors/logs.
            if contact.email in seen_emails:
                continue  # AUDIT-FIX(#7): Prevent duplicate sends when aliases and emails refer to the same contact.
            approved.append(contact)
            seen_emails.add(contact.email)
        return tuple(approved)