"""Define Twinr's canonical email integration data models.

This module normalizes email addresses, approved contacts, mailbox summaries,
and outbound drafts so IMAP, SMTP, and runtime configuration paths share the
same safety guarantees.
"""

# CHANGELOG: 2026-03-30
# BUG-1: Stop lowercasing mailbox local-parts; preserve mailbox identity and use exact matching for inbound trust checks.
# BUG-2: Replace regex/IDNA2003 email parsing with modern email-validator normalization, mailbox-string support, and SMTPUTF8 metadata.
# BUG-3: Fix multilingual lookup failures by normalizing aliases/display names with Unicode NFKC + casefold instead of ASCII lower().
# SEC-1: Add practical resource-exhaustion limits for headers, previews, bodies, aliases, and recipient counts on Raspberry Pi 4 deployments.
# SEC-2: Reject unsafe Unicode control/format characters in display-like fields to reduce spoofing and UI/log confusion.
# IMP-1: Add NormalizedEmailAddress metadata (ASCII wire form + SMTPUTF8 requirement) so runtime senders can make correct transport decisions.
# IMP-2: Add cached mailbox normalization and mailbox-string input support for stricter, faster 2026-style parsing across IMAP/SMTP/runtime paths.
# BREAKING: `email` fields now store the normalized Unicode email form from `email-validator` instead of forced-lowercase ASCII/punycode.
# BREAKING: This module now requires `email-validator>=2.3.0` for standards-compliant normalization and mailbox parsing.

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from types import MappingProxyType
from email.utils import parseaddr
import unicodedata

try:
    from email_validator import EmailNotValidError, validate_email
except ImportError as exc:  # pragma: no cover - dependency error path
    raise RuntimeError(
        "Twinr email models require 'email-validator>=2.3.0'. "
        "Install it with: pip install 'email-validator>=2.3.0'"
    ) from exc


__all__ = [
    "NormalizedEmailAddress",
    "normalize_email",
    "normalize_email_details",
    "EmailContact",
    "EmailMessageSummary",
    "EmailDraft",
    "ApprovedEmailContacts",
]


# Edge-safe size ceilings chosen for Raspberry Pi 4 deployments.
_MAX_MAILBOX_INPUT_LENGTH = 4096
_MAX_DISPLAY_NAME_LENGTH = 256
_MAX_ALIAS_LENGTH = 256
_MAX_IDENTIFIER_LENGTH = 512
_MAX_HEADER_LENGTH = 998  # RFC 5322 hard line limit excluding CRLF.
_MAX_PREVIEW_LENGTH = 512
_MAX_BODY_LENGTH = 64 * 1024
_MAX_REFERENCES = 64
_MAX_ALIASES_PER_CONTACT = 64
_MAX_APPROVED_CONTACTS = 4096
_MAX_DRAFT_RECIPIENTS = 32

_FORBIDDEN_INLINE_CHARS = frozenset({"\r", "\n", "\x00", "\t"})
_FORBIDDEN_BODY_CHARS = frozenset({"\x00"})

# Reject control/format/private/unassigned characters in display-like fields.
_UNSAFE_TEXT_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})
_ALLOWED_FORMAT_CHARS = frozenset({"‌", "‍"})


def _ensure_str(value: object, field_name: str) -> str:
    """Require a string input for one named field."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    return value


def _contains_forbidden_chars(value: str, forbidden_chars: frozenset[str]) -> bool:
    """Return whether text contains any forbidden control characters."""
    return any(char in forbidden_chars for char in value)


def _contains_unsafe_text_chars(value: str) -> bool:
    """Return whether text contains invisible/control Unicode unsafe for UI/log fields."""
    return any(
        unicodedata.category(char) in _UNSAFE_TEXT_CATEGORIES and char not in _ALLOWED_FORMAT_CHARS
        for char in value
    )


def _normalize_unicode_text(value: str, *, form: str) -> str:
    """Apply one Unicode normalization form."""
    return unicodedata.normalize(form, value)


def _enforce_max_length(
    value: str,
    field_name: str,
    *,
    max_length: int,
    truncate: bool = False,
) -> str:
    """Enforce or clamp a maximum code-point length."""
    if len(value) <= max_length:
        return value
    if not truncate:
        raise ValueError(f"{field_name} exceeds the maximum length of {max_length} characters.")
    if max_length <= 1:
        return "…"
    truncated = value[: max_length - 1].rstrip()
    return f"{truncated}…" if truncated else "…"


def _normalize_alias_key(value: str) -> str:
    """Normalize one alias or display-name lookup key."""
    text = _ensure_str(value, "identifier")
    if _contains_forbidden_chars(text, _FORBIDDEN_INLINE_CHARS):
        raise ValueError("identifier contains forbidden control characters.")
    if _contains_unsafe_text_chars(text):
        raise ValueError("identifier contains unsafe Unicode control or format characters.")
    normalized = _normalize_unicode_text(text, form="NFKC")
    normalized = " ".join(normalized.strip().split())
    normalized = _enforce_max_length(normalized, "identifier", max_length=_MAX_ALIAS_LENGTH)
    return normalized.casefold()


def _normalize_display_name(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
    truncate: bool = False,
) -> str:
    """Normalize a human display name used in voice and allowlist flows."""
    text = _ensure_str(value, field_name)
    if _contains_forbidden_chars(text, _FORBIDDEN_INLINE_CHARS):
        raise ValueError(f"{field_name} contains forbidden control characters.")
    normalized = _normalize_unicode_text(text, form="NFC")
    if _contains_unsafe_text_chars(normalized):
        raise ValueError(f"{field_name} contains unsafe Unicode control or format characters.")
    normalized = " ".join(normalized.strip().split())
    normalized = _enforce_max_length(
        normalized,
        field_name,
        max_length=_MAX_DISPLAY_NAME_LENGTH,
        truncate=truncate,
    )
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _normalize_header_text(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
    truncate: bool = False,
) -> str:
    """Normalize a single-line mail header value."""
    text = _ensure_str(value, field_name)
    if _contains_forbidden_chars(text, _FORBIDDEN_INLINE_CHARS):
        raise ValueError(f"{field_name} contains forbidden control characters.")
    normalized = _normalize_unicode_text(text, form="NFC").strip()
    if _contains_unsafe_text_chars(normalized):
        raise ValueError(f"{field_name} contains unsafe Unicode control or format characters.")
    normalized = _enforce_max_length(
        normalized,
        field_name,
        max_length=_MAX_HEADER_LENGTH,
        truncate=truncate,
    )
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _normalize_preview(value: object) -> str:
    """Normalize a preview string for safe UI and logging use."""
    text = _ensure_str(value, "preview")
    if _contains_forbidden_chars(text, _FORBIDDEN_BODY_CHARS):
        raise ValueError("preview contains forbidden control characters.")
    normalized = _normalize_unicode_text(text, form="NFC")
    normalized = " ".join(
        normalized.replace("\r", " ").replace("\n", " ").replace("\t", " ").split()
    )
    if _contains_unsafe_text_chars(normalized):
        raise ValueError("preview contains unsafe Unicode control or format characters.")
    return _enforce_max_length(
        normalized,
        "preview",
        max_length=_MAX_PREVIEW_LENGTH,
        truncate=True,
    )


def _normalize_body(value: object) -> str:
    """Normalize a multiline mail body while rejecting unsafe bytes."""
    text = _ensure_str(value, "body")
    if _contains_forbidden_chars(text, _FORBIDDEN_BODY_CHARS):
        raise ValueError("body contains forbidden control characters.")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return _enforce_max_length(normalized, "body", max_length=_MAX_BODY_LENGTH)


def _coerce_bool(value: object, field_name: str) -> bool:
    """Parse a strict boolean or supported boolean-like literal."""
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


def _iter_strings(values: object, field_name: str) -> tuple[str, ...]:
    """Normalize a string or iterable of strings into a tuple."""
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


def _deduplicate_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    """Drop duplicates while keeping the first-seen item order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class NormalizedEmailAddress:
    """One validated mailbox address in normalized form."""

    normalized: str
    ascii_email: str | None
    requires_smtputf8: bool
    display_name: str = ""

    def wire_email(self) -> str:
        """Return the best transport-safe address representation."""
        return self.ascii_email or self.normalized


def _parse_mailbox_parts(value: str) -> tuple[str, str] | None:
    """Parse a mailbox string into display name and addr-spec when possible."""
    try:
        display_name, addr_spec = parseaddr(value, strict=True)
    except TypeError:  # Python < 3.13 fallback
        display_name, addr_spec = parseaddr(value)
    addr_spec = addr_spec.strip()
    if not addr_spec:
        return None
    return display_name, addr_spec


@lru_cache(maxsize=4096)
def _normalize_email_details_cached(value: str, allow_display_name: bool) -> NormalizedEmailAddress:
    """Normalize one mailbox string using a modern 2026 email parser."""
    candidate = _ensure_str(value, "email").strip()
    if not candidate:
        raise ValueError("email cannot be empty.")
    if _contains_forbidden_chars(candidate, _FORBIDDEN_INLINE_CHARS):
        raise ValueError("email contains forbidden control characters.")
    candidate = _normalize_unicode_text(candidate, form="NFC")
    candidate = _enforce_max_length(
        candidate,
        "email",
        max_length=_MAX_MAILBOX_INPUT_LENGTH,
    )
    try:
        info = validate_email(
            candidate,
            check_deliverability=False,
            allow_display_name=allow_display_name,
            allow_smtputf8=True,
            allow_quoted_local=False,
            allow_domain_literal=False,
            strict=True,
        )
        display_name = ""
        raw_display_name = getattr(info, "display_name", None)
        if raw_display_name:
            try:
                display_name = _normalize_display_name(
                    raw_display_name,
                    "display_name",
                    allow_empty=True,
                    truncate=True,
                )
            except ValueError:
                display_name = ""
        return NormalizedEmailAddress(
            normalized=info.normalized,
            ascii_email=info.ascii_email,
            requires_smtputf8=bool(info.smtputf8),
            display_name=display_name,
        )
    except EmailNotValidError as exc:
        if allow_display_name:
            parsed_parts = _parse_mailbox_parts(candidate)
            if parsed_parts is not None:
                parsed_display_name, addr_spec = parsed_parts
                try:
                    parsed_info = validate_email(
                        addr_spec,
                        check_deliverability=False,
                        allow_display_name=False,
                        allow_smtputf8=True,
                        allow_quoted_local=False,
                        allow_domain_literal=False,
                        strict=True,
                    )
                except EmailNotValidError:
                    pass
                else:
                    try:
                        display_name = _normalize_display_name(
                            parsed_display_name,
                            "display_name",
                            allow_empty=True,
                            truncate=True,
                        )
                    except ValueError:
                        display_name = ""
                    return NormalizedEmailAddress(
                        normalized=parsed_info.normalized,
                        ascii_email=parsed_info.ascii_email,
                        requires_smtputf8=bool(parsed_info.smtputf8),
                        display_name=display_name,
                    )
        raise ValueError(str(exc)) from exc


def normalize_email_details(value: str) -> NormalizedEmailAddress:
    """Return the normalized mailbox metadata for an address or mailbox string."""
    return _normalize_email_details_cached(value, True)


def normalize_email(value: str) -> str:
    """Normalize an email address or mailbox string into Twinr's canonical safe form."""
    return normalize_email_details(value).normalized


def _try_normalize_email_details(value: object) -> NormalizedEmailAddress | None:
    """Normalize a mailbox input or return ``None`` for invalid input."""
    if not isinstance(value, str):
        return None
    try:
        return normalize_email_details(value)
    except (TypeError, ValueError):
        return None


def _normalize_email_collection(
    values: object,
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[NormalizedEmailAddress, ...]:
    """Normalize one or more email addresses into a unique tuple."""
    items = _iter_strings(values, field_name)
    normalized: list[NormalizedEmailAddress] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        try:
            details = normalize_email_details(item)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{field_name}[{index}] {exc}") from exc
        if details.normalized in seen:
            continue
        seen.add(details.normalized)
        normalized.append(details)
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} requires at least one valid email address.")
    return tuple(normalized)


def _normalize_header_collection(values: object, *, field_name: str) -> tuple[str, ...]:
    """Normalize a unique collection of single-line header values."""
    items = _iter_strings(values, field_name)
    if len(items) > _MAX_REFERENCES:
        raise ValueError(f"{field_name} cannot contain more than {_MAX_REFERENCES} items.")
    normalized = _deduplicate_preserving_order(
        _normalize_header_text(item, f"{field_name}[{index}]") for index, item in enumerate(items)
    )
    return normalized


def _normalize_identifier_collection(
    values: object,
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Normalize a tuple of recipient or alias lookup identifiers."""
    items = _iter_strings(values, field_name)
    normalized_items: list[str] = []
    for index, item in enumerate(items):
        if _contains_forbidden_chars(item, _FORBIDDEN_INLINE_CHARS):
            raise ValueError(f"{field_name}[{index}] contains forbidden control characters.")
        normalized = _normalize_unicode_text(item, form="NFC").strip()
        normalized = _enforce_max_length(
            normalized,
            f"{field_name}[{index}]",
            max_length=_MAX_IDENTIFIER_LENGTH,
        )
        if not normalized:
            raise ValueError(f"{field_name}[{index}] cannot be empty.")
        normalized_items.append(normalized)
    normalized = _deduplicate_preserving_order(normalized_items)
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} requires at least one identifier.")
    return normalized


def _normalize_received_at(value: datetime | str | None) -> datetime | None:
    """Normalize a received-at value into an aware UTC datetime."""
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
    """Describe one approved email contact for reading and sending."""

    email: str
    display_name: str
    aliases: tuple[str, ...] = ()
    allow_read: bool = True
    allow_send: bool = True
    ascii_email: str | None = field(init=False)
    requires_smtputf8: bool = field(init=False)

    def __post_init__(self) -> None:
        details = normalize_email_details(self.email)
        raw_display_name = self.display_name or details.display_name
        display_name = _normalize_display_name(raw_display_name, "display_name")
        aliases = _deduplicate_preserving_order(
            normalized_alias
            for alias in _iter_strings(self.aliases, "aliases")
            if (normalized_alias := _normalize_alias_key(alias))
        )
        if len(aliases) > _MAX_ALIASES_PER_CONTACT:
            raise ValueError(f"aliases cannot contain more than {_MAX_ALIASES_PER_CONTACT} items.")
        allow_read = _coerce_bool(self.allow_read, "allow_read")
        allow_send = _coerce_bool(self.allow_send, "allow_send")
        object.__setattr__(self, "email", details.normalized)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "allow_read", allow_read)
        object.__setattr__(self, "allow_send", allow_send)
        object.__setattr__(self, "ascii_email", details.ascii_email)
        object.__setattr__(self, "requires_smtputf8", details.requires_smtputf8)

    def wire_email(self) -> str:
        """Return the best transport-safe representation of the contact email."""
        return self.ascii_email or self.email

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the contact."""
        return {
            "email": self.email,
            "display_name": self.display_name,
            "aliases": self.aliases,
            "allow_read": self.allow_read,
            "allow_send": self.allow_send,
            "ascii_email": self.ascii_email,
            "requires_smtputf8": self.requires_smtputf8,
        }


@dataclass(frozen=True, slots=True)
class EmailMessageSummary:
    """Represent one sanitized mailbox message preview."""

    message_id: str
    sender_email: str
    sender_name: str
    subject: str
    received_at: datetime | None = None
    preview: str = ""
    unread: bool = True
    sender_ascii_email: str | None = field(init=False)
    sender_requires_smtputf8: bool = field(init=False)

    def __post_init__(self) -> None:
        sender_details = normalize_email_details(self.sender_email)
        normalized_sender_name = self.sender_name
        if not normalized_sender_name and sender_details.display_name:
            normalized_sender_name = sender_details.display_name
        object.__setattr__(self, "message_id", _normalize_header_text(self.message_id, "message_id"))
        object.__setattr__(self, "sender_email", sender_details.normalized)
        object.__setattr__(
            self,
            "sender_name",
            _normalize_display_name(
                normalized_sender_name,
                "sender_name",
                allow_empty=True,
                truncate=True,
            ),
        )
        object.__setattr__(
            self,
            "subject",
            _normalize_header_text(self.subject, "subject", allow_empty=True, truncate=True),
        )
        object.__setattr__(self, "received_at", _normalize_received_at(self.received_at))
        object.__setattr__(self, "preview", _normalize_preview(self.preview))
        object.__setattr__(self, "unread", _coerce_bool(self.unread, "unread"))
        object.__setattr__(self, "sender_ascii_email", sender_details.ascii_email)
        object.__setattr__(self, "sender_requires_smtputf8", sender_details.requires_smtputf8)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the summary."""
        return {
            "message_id": self.message_id,
            "sender_email": self.sender_email,
            "sender_name": self.sender_name,
            "subject": self.subject,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "preview": self.preview,
            "unread": self.unread,
            "sender_ascii_email": self.sender_ascii_email,
            "sender_requires_smtputf8": self.sender_requires_smtputf8,
        }


@dataclass(frozen=True, slots=True)
class EmailDraft:
    """Represent one validated outbound email draft."""

    to: tuple[str, ...]
    subject: str
    body: str
    cc: tuple[str, ...] = ()
    in_reply_to: str | None = None
    references: tuple[str, ...] = ()
    to_ascii: tuple[str | None, ...] = field(init=False)
    cc_ascii: tuple[str | None, ...] = field(init=False)
    requires_smtputf8: bool = field(init=False)

    def __post_init__(self) -> None:
        to_details = _normalize_email_collection(self.to, field_name="to", allow_empty=False)
        cc_details = _normalize_email_collection(self.cc, field_name="cc", allow_empty=True)
        to_set = {details.normalized for details in to_details}
        cc_details = tuple(details for details in cc_details if details.normalized not in to_set)
        if len(to_details) + len(cc_details) > _MAX_DRAFT_RECIPIENTS:
            raise ValueError(f"draft cannot contain more than {_MAX_DRAFT_RECIPIENTS} recipients.")
        subject = _normalize_header_text(self.subject, "subject", allow_empty=True)
        body = _normalize_body(self.body)
        in_reply_to = None
        if self.in_reply_to is not None:
            in_reply_to = _normalize_header_text(self.in_reply_to, "in_reply_to")
        references = _normalize_header_collection(self.references, field_name="references")
        object.__setattr__(self, "to", tuple(details.normalized for details in to_details))
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "cc", tuple(details.normalized for details in cc_details))
        object.__setattr__(self, "in_reply_to", in_reply_to)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "to_ascii", tuple(details.ascii_email for details in to_details))
        object.__setattr__(self, "cc_ascii", tuple(details.ascii_email for details in cc_details))
        object.__setattr__(
            self,
            "requires_smtputf8",
            any(details.requires_smtputf8 for details in (*to_details, *cc_details)),
        )

    def recipients(self) -> tuple[str, ...]:
        """Return all recipients in send order."""
        return self.to + self.cc

    def wire_recipients(self) -> tuple[str, ...]:
        """Return the best transport-safe representation of all recipients."""
        return tuple(
            ascii_address if ascii_address is not None else normalized
            for normalized, ascii_address in zip(self.to + self.cc, self.to_ascii + self.cc_ascii)
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the draft."""
        return {
            "to": self.to,
            "subject": self.subject,
            "body": self.body,
            "cc": self.cc,
            "in_reply_to": self.in_reply_to,
            "references": self.references,
            "to_ascii": self.to_ascii,
            "cc_ascii": self.cc_ascii,
            "requires_smtputf8": self.requires_smtputf8,
        }


def _register_unique(mapping: dict[str, EmailContact], key: str, contact: EmailContact, *, key_kind: str) -> None:
    """Insert one contact into an index unless the key is ambiguous."""
    existing = mapping.get(key)
    if existing is None:
        mapping[key] = contact
        return
    if existing is contact:
        return
    raise ValueError(f"Duplicate approved contact {key_kind} detected.")


def _register_unique_optional_casefold(
    mapping: dict[str, EmailContact | None],
    key: str,
    contact: EmailContact,
) -> None:
    """Insert one compatibility lookup key, marking collisions as ambiguous."""
    existing = mapping.get(key)
    if existing is None:
        mapping[key] = contact
        return
    if existing is contact:
        return
    mapping[key] = None


@dataclass(frozen=True, slots=True)
class ApprovedEmailContacts:
    """Index approved email contacts for safe read and send decisions."""

    contacts: tuple[EmailContact, ...] = field(default_factory=tuple)
    _by_email: Mapping[str, EmailContact] = field(init=False, repr=False)
    _by_email_compat: Mapping[str, EmailContact | None] = field(init=False, repr=False)
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
        if len(contacts) > _MAX_APPROVED_CONTACTS:
            raise ValueError(f"contacts cannot contain more than {_MAX_APPROVED_CONTACTS} items.")
        by_email: dict[str, EmailContact] = {}
        by_email_compat: dict[str, EmailContact | None] = {}
        by_alias: dict[str, EmailContact] = {}
        for index, contact in enumerate(contacts):
            if not isinstance(contact, EmailContact):
                raise TypeError(f"contacts[{index}] must be an EmailContact.")
            _register_unique(by_email, contact.email, contact, key_kind="email")
            _register_unique_optional_casefold(by_email_compat, contact.email.casefold(), contact)
            _register_unique(
                by_alias,
                _normalize_alias_key(contact.display_name),
                contact,
                key_kind="display_name",
            )
            for alias in contact.aliases:
                _register_unique(by_alias, alias, contact, key_kind="alias")
        object.__setattr__(self, "contacts", contacts)
        object.__setattr__(self, "_by_email", MappingProxyType(by_email))
        object.__setattr__(self, "_by_email_compat", MappingProxyType(by_email_compat))
        object.__setattr__(self, "_by_alias", MappingProxyType(by_alias))

    def resolve(self, identifier: str) -> EmailContact | None:
        """Resolve an email, mailbox string, display name, or alias to a contact."""
        if not isinstance(identifier, str):
            return None
        stripped_identifier = identifier.strip()
        if not stripped_identifier:
            return None
        normalized_email = _try_normalize_email_details(stripped_identifier)
        if normalized_email is not None:
            contact = self._by_email.get(normalized_email.normalized)
            if contact is not None:
                return contact
            compat_contact = self._by_email_compat.get(normalized_email.normalized.casefold())
            if compat_contact is not None:
                return compat_contact
        try:
            alias_key = _normalize_alias_key(stripped_identifier)
        except ValueError:
            return None
        return self._by_alias.get(alias_key)

    def can_read_from(self, sender_email: str) -> bool:
        """Return whether a sender is approved for mailbox reads.

        Trust decisions intentionally require an exact normalized address match.
        """
        normalized = _try_normalize_email_details(sender_email)
        if normalized is None:
            return False
        contact = self._by_email.get(normalized.normalized)
        return bool(contact and contact.allow_read)

    def can_send_to(self, recipient_email: str) -> bool:
        """Return whether a recipient is approved for outbound mail."""
        contact = self.resolve(recipient_email)
        return bool(contact and contact.allow_send)

    def require_allowed_recipients(self, recipients: tuple[str, ...] | str) -> tuple[EmailContact, ...]:
        """Resolve recipients and raise if any are not approved for sending."""
        approved: list[EmailContact] = []
        seen_emails: set[str] = set()
        for recipient in _normalize_identifier_collection(recipients, field_name="recipients", allow_empty=False):
            contact = self.resolve(recipient)
            if contact is None or not contact.allow_send:
                raise ValueError("One or more recipients are not approved for sending.")
            if contact.email in seen_emails:
                continue
            approved.append(contact)
            seen_emails.add(contact.email)
        return tuple(approved)