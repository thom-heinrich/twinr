# CHANGELOG: 2026-03-30
# BUG-1: Bounded mailbox-provider iteration and enforced post-provider caps so ignored limits or infinite iterables cannot exhaust memory.
# BUG-2: Replaced delimiter splitting with RFC-aware address parsing; display-name recipients such as "Doe, Jane <jane@example.com>" now parse correctly.
# BUG-3: Stopped trusting provider summary payloads wholesale; mailbox summaries are now sanitized, truncated, and explicitly labeled as untrusted external content.
# SEC-1: # BREAKING: read_recent requires explicit user/caregiver approval by default to prevent privacy leaks on voice-triggered elder-care deployments.
# SEC-2: # BREAKING: outbound recipients are restricted to approved contacts by default unless settings explicitly disable it.
# IMP-1: Added schema-first operation specifications and typed internal command parsing aligned with 2026 structured tool-call patterns.
# IMP-2: Added RFC-aware email normalization, approval provenance, provider latency metrics, and downstream trust labels for mailbox content.

"""Translate Twinr email integration requests into mailbox actions.

This module defines the adapter-level settings, mailbox and sender protocols,
and the orchestration layer that validates integration requests before reading,
drafting, or sending email.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from email.utils import getaddresses
from enum import Enum
from itertools import islice
from time import monotonic
from typing import Final, Protocol, runtime_checkable

from twinr.integrations.adapter import IntegrationAdapter
from twinr.integrations.email.models import ApprovedEmailContacts, EmailDraft, EmailMessageSummary, normalize_email
from twinr.integrations.models import IntegrationManifest, IntegrationRequest, IntegrationResult

try:
    from email_validator import EmailNotValidError, validate_email as _validate_email_address
except ImportError:  # pragma: no cover - optional frontier dependency
    EmailNotValidError = ValueError
    _validate_email_address = None

logger = logging.getLogger(__name__)

_TOOL_SCHEMA_VERSION: Final[str] = "2026-03-30"
_DEFAULT_READ_LIMIT: Final[int] = 5
_FORBIDDEN_SUMMARY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "body_html",
        "body_preview",
        "body_text",
        "full_text",
        "html",
        "html_body",
        "mime",
        "raw",
        "raw_body",
        "raw_headers",
        "raw_message",
        "raw_mime",
        "raw_source",
        "source",
        "text",
        "text_body",
        "attachments",
        "attachment_ids",
        "attachment_names",
    }
)
_COMMON_SUMMARY_ATTRS: Final[tuple[str, ...]] = (
    "id",
    "message_id",
    "thread_id",
    "subject",
    "snippet",
    "preview",
    "sender_email",
    "sender_name",
    "received_at",
    "timestamp",
    "unread",
    "is_unread",
    "has_attachments",
)
_CONTROL_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1F\x7F]")
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_MESSAGE_ID_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^<[^<>\s]{1,318}>$")


class EmailOperation(str, Enum):
    """Supported outward-facing operation IDs."""

    READ_RECENT = "read_recent"
    DRAFT_REPLY = "draft_reply"
    DRAFT_MESSAGE = "draft_message"
    SEND_MESSAGE = "send_message"
    SEND_EMAIL = "send_email"


_DRAFT_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "to": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}, "minItems": 1},
            ]
        },
        "cc": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ]
        },
        "subject": {"type": "string", "minLength": 1, "maxLength": 160},
        "body": {"type": "string", "minLength": 1, "maxLength": 2000},
        "in_reply_to": {"type": "string", "maxLength": 320},
        "references": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}, "maxItems": 20},
            ]
        },
    },
    "required": ["to", "subject", "body"],
    "x-policy": {
        "requires_explicit_approval": True,
        "recipient_scope": "approved_contacts_by_default",
        "redacted_fields": ["body"],
    },
}

EMAIL_OPERATION_SCHEMAS: Final[dict[str, dict[str, object]]] = {
    EmailOperation.READ_RECENT.value: {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
            "unread_only": {"type": "boolean", "default": True},
        },
        "required": [],
        "x-policy": {
            "requires_explicit_approval_by_default": True,
            "output_trust": "untrusted_external",
        },
    },
    EmailOperation.DRAFT_REPLY.value: deepcopy(_DRAFT_SCHEMA),
    EmailOperation.DRAFT_MESSAGE.value: deepcopy(_DRAFT_SCHEMA),
    EmailOperation.SEND_MESSAGE.value: deepcopy(_DRAFT_SCHEMA),
    EmailOperation.SEND_EMAIL.value: deepcopy(_DRAFT_SCHEMA),
}


def _parse_strict_bool(value: object, *, field_name: str) -> bool:
    """Parse a strict boolean or supported boolean-like literal."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean value.")


def _parse_positive_int(value: object, *, field_name: str) -> int:
    """Parse a positive integer from config or request input."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive whole number.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive whole number.") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be a positive whole number.")
    return parsed


@runtime_checkable
class MailboxReader(Protocol):
    """Define the mailbox read contract required by the email adapter."""

    def list_recent(self, *, limit: int, unread_only: bool) -> list[EmailMessageSummary]:
        """Return the newest mailbox summaries that match the filter."""
        ...


@runtime_checkable
class MailSender(Protocol):
    """Define the outbound mail transport contract required by the adapter."""

    def send(self, draft: EmailDraft) -> str | None:
        """Send one validated draft and optionally return a provider message ID."""
        ...


@dataclass(frozen=True, slots=True)
class EmailAdapterSettings:
    """Hold bounded safety and presentation limits for email operations."""

    max_read_results: int = 8
    max_subject_chars: int = 160
    max_body_chars: int = 2000
    unread_only_default: bool = True
    # BREAKING: Reads now require explicit approval by default because mailbox summaries are sensitive data in elder-care voice deployments.
    require_explicit_approval_for_reads: bool = True
    restrict_reads_to_known_senders: bool = False
    # BREAKING: Outbound mail is now default-deny to approved contacts unless explicitly relaxed by configuration.
    restrict_recipients_to_known_contacts: bool = True
    max_total_recipients: int = 10
    max_reference_ids: int = 20
    max_header_value_chars: int = 320
    max_provider_scan_results: int = 32
    read_scan_multiplier: int = 4
    max_summary_fields: int = 12
    max_summary_collection_items: int = 8
    max_summary_string_chars: int = 240
    max_summary_key_chars: int = 64
    mark_read_results_as_untrusted: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_read_results", _parse_positive_int(self.max_read_results, field_name="max_read_results"))
        object.__setattr__(self, "max_subject_chars", _parse_positive_int(self.max_subject_chars, field_name="max_subject_chars"))
        object.__setattr__(self, "max_body_chars", _parse_positive_int(self.max_body_chars, field_name="max_body_chars"))
        object.__setattr__(self, "max_total_recipients", _parse_positive_int(self.max_total_recipients, field_name="max_total_recipients"))
        object.__setattr__(self, "max_reference_ids", _parse_positive_int(self.max_reference_ids, field_name="max_reference_ids"))
        object.__setattr__(self, "max_header_value_chars", _parse_positive_int(self.max_header_value_chars, field_name="max_header_value_chars"))
        object.__setattr__(self, "max_provider_scan_results", _parse_positive_int(self.max_provider_scan_results, field_name="max_provider_scan_results"))
        object.__setattr__(self, "read_scan_multiplier", _parse_positive_int(self.read_scan_multiplier, field_name="read_scan_multiplier"))
        object.__setattr__(self, "max_summary_fields", _parse_positive_int(self.max_summary_fields, field_name="max_summary_fields"))
        object.__setattr__(
            self,
            "max_summary_collection_items",
            _parse_positive_int(self.max_summary_collection_items, field_name="max_summary_collection_items"),
        )
        object.__setattr__(self, "max_summary_string_chars", _parse_positive_int(self.max_summary_string_chars, field_name="max_summary_string_chars"))
        object.__setattr__(self, "max_summary_key_chars", _parse_positive_int(self.max_summary_key_chars, field_name="max_summary_key_chars"))
        object.__setattr__(
            self,
            "unread_only_default",
            _parse_strict_bool(self.unread_only_default, field_name="unread_only_default"),
        )
        object.__setattr__(
            self,
            "require_explicit_approval_for_reads",
            _parse_strict_bool(self.require_explicit_approval_for_reads, field_name="require_explicit_approval_for_reads"),
        )
        object.__setattr__(
            self,
            "restrict_reads_to_known_senders",
            _parse_strict_bool(self.restrict_reads_to_known_senders, field_name="restrict_reads_to_known_senders"),
        )
        object.__setattr__(
            self,
            "restrict_recipients_to_known_contacts",
            _parse_strict_bool(
                self.restrict_recipients_to_known_contacts,
                field_name="restrict_recipients_to_known_contacts",
            ),
        )
        object.__setattr__(
            self,
            "mark_read_results_as_untrusted",
            _parse_strict_bool(self.mark_read_results_as_untrusted, field_name="mark_read_results_as_untrusted"),
        )


@dataclass(frozen=True, slots=True)
class ReadRecentCommand:
    """Validated parameters for reading mailbox summaries."""

    limit: int
    unread_only: bool
    approval_source: str | None


@dataclass(frozen=True, slots=True)
class DraftCommand:
    """Validated parameters for drafting or sending one outbound email."""

    to: tuple[str, ...]
    cc: tuple[str, ...]
    subject: str
    body: str
    in_reply_to: str | None
    references: tuple[str, ...]
    approval_source: str
    smtputf8_required: bool = False


@dataclass(slots=True)
class EmailMailboxAdapter(IntegrationAdapter):
    """Execute Twinr email integration requests against mailbox providers.

    The adapter keeps request validation, contact policy enforcement, mailbox
    reads, and SMTP sends within a single bounded integration surface.
    """

    manifest: IntegrationManifest
    contacts: ApprovedEmailContacts
    mailbox_reader: MailboxReader
    mail_sender: MailSender | None = None
    settings: EmailAdapterSettings = field(default_factory=EmailAdapterSettings)

    def execute(self, request: IntegrationRequest) -> IntegrationResult:
        """Dispatch one email integration request into a bounded result."""
        raw_operation_id = getattr(request, "operation_id", None)
        operation_id = raw_operation_id.strip() if isinstance(raw_operation_id, str) else raw_operation_id
        started_at = monotonic()
        try:
            if operation_id == EmailOperation.READ_RECENT.value:
                result = self._read_recent(request)
            elif operation_id in {EmailOperation.DRAFT_REPLY.value, EmailOperation.DRAFT_MESSAGE.value}:
                result = self._draft_message(request)
            elif operation_id in {EmailOperation.SEND_MESSAGE.value, EmailOperation.SEND_EMAIL.value}:
                result = self._send_message(request)
            else:
                result = self._failure_result(
                    summary="I couldn't understand that email action.",
                    error_code="unsupported_operation",
                    operation_id=operation_id,
                )
        except ValueError as exc:
            result = self._failure_result(
                summary=str(exc),
                error_code="invalid_request",
                operation_id=operation_id,
            )
        except RuntimeError as exc:
            logger.warning("Email operation %r failed cleanly.", operation_id, exc_info=True)
            result = self._failure_result(
                summary=str(exc),
                error_code="service_unavailable",
                operation_id=operation_id,
            )
        except Exception:
            logger.exception("Unexpected email adapter failure for operation %r.", operation_id)
            result = self._failure_result(
                summary="I couldn't complete the email action right now. Please try again.",
                error_code="unexpected_error",
                operation_id=operation_id,
            )

        self._attach_result_meta(result, operation_id=operation_id, started_at=started_at)
        return result

    @classmethod
    def operation_schemas(cls) -> Mapping[str, Mapping[str, object]]:
        """Expose schema-first operation specs for tool-call builders."""
        return deepcopy(EMAIL_OPERATION_SCHEMAS)

    def _read_recent(self, request: IntegrationRequest) -> IntegrationResult:
        """Build a structured result for recent mailbox summaries."""
        command = self._parse_read_recent_command(request)

        scan_limit = min(
            self.settings.max_provider_scan_results,
            max(command.limit, command.limit * self.settings.read_scan_multiplier),
        )
        provider_started_at = monotonic()
        try:
            raw_messages = self.mailbox_reader.list_recent(  # pylint: disable=assignment-from-no-return
                limit=scan_limit,
                unread_only=command.unread_only,
            )
            message_dicts = self._collect_recent_messages(raw_messages, requested_limit=command.limit)
        except Exception as exc:
            raise RuntimeError("I couldn't read recent email right now.") from exc

        details: dict[str, object] = {
            "messages": message_dicts,
            "count": len(message_dicts),
            "approval_source": command.approval_source,
            "provider_latency_ms": int((monotonic() - provider_started_at) * 1000),
        }
        if self.settings.mark_read_results_as_untrusted:
            details["content_trust"] = "untrusted_external"
            details["treat_content_as_data_only"] = True

        return IntegrationResult(
            ok=True,
            summary=f"{len(message_dicts)} email summaries ready.",
            details=details,
        )

    def _draft_message(self, request: IntegrationRequest) -> IntegrationResult:
        """Validate and stage an outbound email draft without sending it."""
        command = self._build_draft_command(request)
        draft = self._command_to_draft(command)
        return IntegrationResult(
            ok=True,
            summary=f"Draft prepared for {', '.join(draft.to)}.",
            details={
                "draft": {
                    "to": list(draft.to),
                    "cc": list(draft.cc),
                    "subject": draft.subject,
                    "body_preview": "[redacted]",
                    "body_chars": len(draft.body),
                    "in_reply_to": draft.in_reply_to,
                    "references": list(draft.references),
                    "approval_source": command.approval_source,
                    "smtputf8_required": command.smtputf8_required,
                }
            },
            redacted_fields=("body", "body_preview"),
        )

    def _send_message(self, request: IntegrationRequest) -> IntegrationResult:
        """Validate and send one outbound email draft."""
        if self.mail_sender is None:
            raise RuntimeError("I couldn't send email because the mail service is not available.")

        command = self._build_draft_command(request)
        draft = self._command_to_draft(command)

        provider_started_at = monotonic()
        try:
            provider_message_id = self.mail_sender.send(draft)
        except Exception as exc:
            raise RuntimeError("I couldn't send that email right now.") from exc

        if provider_message_id is not None:
            if not isinstance(provider_message_id, str):
                raise RuntimeError("I couldn't confirm that the email was sent.")
            provider_message_id = provider_message_id.strip() or None

        return IntegrationResult(
            ok=True,
            summary=f"Email sent to {', '.join(draft.to)}.",
            details={
                "to": list(draft.to),
                "cc": list(draft.cc),
                "subject": draft.subject,
                "provider_message_id": provider_message_id,
                "provider_latency_ms": int((monotonic() - provider_started_at) * 1000),
                "approval_source": command.approval_source,
                "smtputf8_required": command.smtputf8_required,
            },
            redacted_fields=("body",),
        )

    def _parse_read_recent_command(self, request: IntegrationRequest) -> ReadRecentCommand:
        """Parse and validate one read_recent request."""
        params = self._parameters(request)
        limit = min(
            self.settings.max_read_results,
            self._parse_limit(params.get("limit", _DEFAULT_READ_LIMIT)),
        )
        unread_only = self._parse_optional_bool(
            params.get("unread_only"),
            default=self.settings.unread_only_default,
            field_name="unread_only",
        )
        approval_source = self._approval_source(request)
        if self.settings.require_explicit_approval_for_reads:
            approval_source = self._ensure_explicit_approval(request)
        return ReadRecentCommand(
            limit=limit,
            unread_only=unread_only,
            approval_source=approval_source,
        )

    def _build_draft_command(self, request: IntegrationRequest) -> DraftCommand:
        """Construct a validated outbound-email command from request parameters."""
        approval_source = self._ensure_explicit_approval(request)
        params = self._parameters(request)

        recipients, recipient_requires_smtputf8 = self._parse_recipients(params.get("to"))
        cc, cc_requires_smtputf8 = self._parse_recipients(params.get("cc", ()), optional=True)
        recipients, cc = self._deduplicate_recipients(recipients, cc)

        total_recipients = len(recipients) + len(cc)
        if total_recipients > self.settings.max_total_recipients:
            raise ValueError("Too many email recipients were provided.")

        if self.settings.restrict_recipients_to_known_contacts:
            self.contacts.require_allowed_recipients(list(recipients) + list(cc))

        subject = self._parse_text_field(
            params.get("subject"),
            field_name="subject",
            max_chars=self.settings.max_subject_chars,
            allow_newlines=False,
            required=True,
        )
        body = self._parse_text_field(
            params.get("body"),
            field_name="body",
            max_chars=self.settings.max_body_chars,
            allow_newlines=True,
            required=True,
        )

        in_reply_to = self._parse_message_id(params.get("in_reply_to"), field_name="in_reply_to")
        references = self._parse_message_id_collection(params.get("references", ()), field_name="references")

        return DraftCommand(
            to=recipients,
            cc=cc,
            subject=subject,
            body=body,
            in_reply_to=in_reply_to,
            references=references,
            approval_source=approval_source,
            smtputf8_required=recipient_requires_smtputf8 or cc_requires_smtputf8,
        )

    def _command_to_draft(self, command: DraftCommand) -> EmailDraft:
        """Convert a validated command into the provider-facing EmailDraft."""
        return EmailDraft(
            to=command.to,
            cc=command.cc,
            subject=command.subject,
            body=command.body,
            in_reply_to=command.in_reply_to,
            references=command.references,
        )

    def _approval_source(self, request: IntegrationRequest) -> str | None:
        """Return the approval source, if any explicit approval flag is set."""
        if self._parse_request_confirmation(getattr(request, "explicit_user_confirmation", False)):
            return "user"
        if self._parse_request_confirmation(getattr(request, "explicit_caregiver_confirmation", False)):
            return "caregiver"
        return None

    def _ensure_explicit_approval(self, request: IntegrationRequest) -> str:
        """Require an explicit user or caregiver approval signal."""
        approval_source = self._approval_source(request)
        if approval_source is not None:
            return approval_source
        raise ValueError("Please confirm before I prepare, read, or send email.")

    def _parse_recipients(self, raw_value: object, *, optional: bool = False) -> tuple[tuple[str, ...], bool]:
        """Normalize one or more recipient inputs into unique email addresses."""
        if raw_value in (None, ""):
            if optional:
                return (), False
            raise ValueError("Please choose at least one email recipient.")

        field_values = self._coerce_recipient_field_values(raw_value)
        parsed_pairs = self._parse_address_fields(field_values)

        recipients: list[str] = []
        seen: set[str] = set()
        smtputf8_required = False
        for display_name, address in parsed_pairs:
            if not address:
                if display_name:
                    raise ValueError("One email address is not valid.")
                continue
            try:
                normalized, candidate_requires_smtputf8 = self._normalize_recipient_candidate(address)
            except Exception as exc:
                raise ValueError("One email address is not valid.") from exc
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            recipients.append(normalized)
            smtputf8_required = smtputf8_required or candidate_requires_smtputf8

        if not recipients and not optional:
            raise ValueError("Please choose at least one email recipient.")
        return tuple(recipients), smtputf8_required

    def _coerce_recipient_field_values(self, raw_value: object) -> list[str]:
        """Coerce raw recipient parameters into address-header strings."""
        if isinstance(raw_value, str):
            return [raw_value]
        if isinstance(raw_value, (list, tuple)):
            values: list[str] = []
            for value in raw_value:
                if value in (None, ""):
                    continue
                if not isinstance(value, str):
                    raise ValueError("Each email recipient must be plain text.")
                values.append(value)
            return values
        raise ValueError("Email recipients must be plain text.")

    def _parse_address_fields(self, field_values: Sequence[str]) -> list[tuple[str, str]]:
        """Parse one or more address-header values with the stdlib RFC parser."""
        try:
            return getaddresses(field_values, strict=True)  # Python 3.13+
        except TypeError:  # pragma: no cover - compatibility path for older Python builds
            return getaddresses(field_values)

    def _normalize_recipient_candidate(self, candidate: str) -> tuple[str, bool]:
        """Normalize one recipient email address into canonical wire-safe form."""
        candidate = candidate.strip()
        if not candidate:
            raise ValueError("Email recipient is empty.")

        if _validate_email_address is not None:
            try:
                validated = _validate_email_address(
                    candidate,
                    allow_display_name=False,
                    check_deliverability=False,
                    strict=False,
                )
            except EmailNotValidError as exc:  # pragma: no cover - depends on optional dependency
                raise ValueError("One email address is not valid.") from exc

            normalized = getattr(validated, "ascii_email", None) or getattr(validated, "normalized", None)
            if not isinstance(normalized, str) or not normalized:
                raise ValueError("One email address is not valid.")
            return normalized, bool(getattr(validated, "smtputf8", False))

        normalized = normalize_email(candidate)
        return normalized, self._requires_smtputf8(normalized)

    def _parameters(self, request: IntegrationRequest) -> Mapping[str, object]:
        """Return request parameters as a mapping or raise a validation error."""
        parameters = getattr(request, "parameters", None)
        if parameters is None:
            return {}
        if not isinstance(parameters, Mapping):
            raise ValueError("I couldn't understand the email request.")
        return parameters

    def _parse_limit(self, raw_value: object) -> int:
        """Parse the requested mailbox read limit with a safe default."""
        if raw_value in (None, ""):
            return _DEFAULT_READ_LIMIT
        return _parse_positive_int(raw_value, field_name="limit")

    def _parse_optional_bool(self, raw_value: object, *, default: bool, field_name: str) -> bool:
        """Parse an optional request flag without falling back to truthiness."""
        if raw_value in (None, ""):
            return default
        try:
            return _parse_strict_bool(raw_value, field_name=field_name)
        except ValueError as exc:
            raise ValueError(f"I couldn't understand the {field_name} setting.") from exc

    def _parse_request_confirmation(self, raw_value: object) -> bool:
        """Parse a confirmation flag and reject ambiguous approval values."""
        if raw_value in (None, ""):
            return False
        try:
            return _parse_strict_bool(raw_value, field_name="confirmation")
        except ValueError as exc:
            raise ValueError("Email approval is not clear. Please confirm again.") from exc

    def _parse_text_field(
        self,
        raw_value: object,
        *,
        field_name: str,
        max_chars: int,
        allow_newlines: bool,
        required: bool,
    ) -> str:
        """Normalize and bound a subject/body text field from the request."""
        if raw_value is None:
            text = ""
        elif isinstance(raw_value, str):
            text = raw_value
        else:
            raise ValueError(f"Email {field_name} must be plain text.")

        normalized = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
        candidate = normalized.strip()

        if not allow_newlines:
            candidate = " ".join(candidate.split())
            normalized = candidate

        if not candidate:
            if required:
                raise ValueError(f"Please provide an email {field_name}.")
            return ""

        if self._contains_disallowed_control_chars(normalized, allow_newlines=allow_newlines):
            raise ValueError(f"Email {field_name} contains unsupported characters.")
        if len(normalized) > max_chars:
            raise ValueError(f"Email {field_name} is too long.")
        return normalized

    def _parse_message_id(self, raw_value: object, *, field_name: str) -> str | None:
        """Normalize one message-ID style header value from the request."""
        if raw_value in (None, ""):
            return None
        if not isinstance(raw_value, str):
            raise ValueError("Email reply information is not valid.")

        normalized = raw_value.strip()
        if not normalized:
            return None
        if self._contains_disallowed_control_chars(normalized, allow_newlines=False):
            raise ValueError(f"{field_name} contains unsupported characters.")
        if len(normalized) > self.settings.max_header_value_chars:
            raise ValueError(f"{field_name} is too long.")
        if normalized.startswith("<") and normalized.endswith(">") and not _MESSAGE_ID_TOKEN_RE.fullmatch(normalized):
            raise ValueError(f"{field_name} is not a valid message ID.")
        return normalized

    def _parse_message_id_collection(self, raw_value: object, *, field_name: str) -> tuple[str, ...]:
        """Normalize a bounded collection of message-ID reference values."""
        if raw_value in (None, "", ()):
            return ()

        tokens: list[str] = []
        if isinstance(raw_value, str):
            tokens.extend(raw_value.split())
        elif isinstance(raw_value, (list, tuple)):
            for value in raw_value:
                if value in (None, ""):
                    continue
                if not isinstance(value, str):
                    raise ValueError("Email reply information is not valid.")
                tokens.extend(value.split())
        else:
            raise ValueError("Email reply information is not valid.")

        references: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            normalized = self._parse_message_id(token, field_name=field_name)
            if normalized is None:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            references.append(normalized)

        if len(references) > self.settings.max_reference_ids:
            raise ValueError("Too many email reply references were provided.")
        return tuple(references)

    def _collect_recent_messages(self, raw_messages: object, *, requested_limit: int) -> list[dict[str, object]]:
        """Collect at most requested_limit sanitized recent-message summaries."""
        messages: list[dict[str, object]] = []
        for index, message in enumerate(self._iter_bounded_messages(raw_messages)):
            if self.settings.restrict_reads_to_known_senders and not self._is_allowed_sender(message, log_missing=True):
                continue
            try:
                messages.append(self._summary_to_safe_dict(message))
            except Exception:
                logger.warning("Skipping malformed email summary at index %s.", index, exc_info=True)
                continue
            if len(messages) >= requested_limit:
                break
        return messages

    def _iter_bounded_messages(self, raw_messages: object) -> Iterable[EmailMessageSummary]:
        """Iterate over provider results without trusting provider-side limits."""
        if isinstance(raw_messages, list):
            iterator = iter(raw_messages)
        elif isinstance(raw_messages, Iterable) and not isinstance(raw_messages, (str, bytes, bytearray)):
            iterator = iter(raw_messages)
        else:
            raise RuntimeError("I couldn't read recent email right now.")
        return islice(iterator, self.settings.max_provider_scan_results)

    def _summary_to_safe_dict(self, message: EmailMessageSummary) -> dict[str, object]:
        """Serialize one provider summary into a bounded, downstream-safe payload."""
        payload: dict[str, object] = {}

        raw_dict: Mapping[str, object] | None = None
        as_dict = getattr(message, "as_dict", None)
        if callable(as_dict):
            candidate = as_dict()
            if isinstance(candidate, Mapping):
                raw_dict = candidate

        if raw_dict is not None:
            payload.update(self._sanitize_mapping(raw_dict, depth=0))

        for attr_name in _COMMON_SUMMARY_ATTRS:
            if attr_name in payload:
                continue
            value = getattr(message, attr_name, None)
            if value is None:
                continue
            payload[attr_name] = self._sanitize_json_like(value, depth=0)

        sender_allowed = self._is_allowed_sender(message, log_missing=False)
        if self.settings.mark_read_results_as_untrusted:
            payload["content_trust"] = "untrusted_external"
            payload["treat_as_data_only"] = True
        payload["sender_allowed"] = sender_allowed

        return payload

    def _sanitize_mapping(self, value: Mapping[str, object], *, depth: int) -> dict[str, object]:
        """Sanitize one mapping into bounded JSON-like values."""
        sanitized: dict[str, object] = {}
        for index, (raw_key, raw_item) in enumerate(value.items()):
            if index >= self.settings.max_summary_fields:
                break
            key = self._normalize_summary_text(str(raw_key), max_chars=self.settings.max_summary_key_chars)
            if not key or key.casefold() in _FORBIDDEN_SUMMARY_KEYS:
                continue
            sanitized[key] = self._sanitize_json_like(raw_item, depth=depth + 1)
        return sanitized

    def _sanitize_json_like(self, value: object, *, depth: int) -> object:
        """Convert arbitrary provider values into bounded, JSON-friendly values."""
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._normalize_summary_text(value, max_chars=self.settings.max_summary_string_chars)
        if depth >= 2:
            return self._normalize_summary_text(str(value), max_chars=self.settings.max_summary_string_chars)
        if isinstance(value, Mapping):
            return self._sanitize_mapping(value, depth=depth + 1)
        if isinstance(value, (list, tuple)):
            return [
                self._sanitize_json_like(item, depth=depth + 1)
                for item in islice(value, self.settings.max_summary_collection_items)
            ]
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return [
                self._sanitize_json_like(item, depth=depth + 1)
                for item in islice(value, self.settings.max_summary_collection_items)
            ]
        return self._normalize_summary_text(str(value), max_chars=self.settings.max_summary_string_chars)

    def _normalize_summary_text(self, value: str, *, max_chars: int) -> str:
        """Collapse unsafe whitespace and truncate summary text."""
        cleaned = value.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        cleaned = _CONTROL_CHAR_RE.sub(" ", cleaned)
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1].rstrip() + "…"

    def _extract_sender_email(self, message: EmailMessageSummary) -> str | None:
        """Best-effort extraction of a sender address from provider summary data."""
        sender_email = getattr(message, "sender_email", None)
        if not isinstance(sender_email, str) or not sender_email.strip():
            return None

        parsed_pairs = self._parse_address_fields([sender_email])
        if len(parsed_pairs) == 1 and parsed_pairs[0][1]:
            return parsed_pairs[0][1]
        return sender_email

    def _is_allowed_sender(self, message: EmailMessageSummary, *, log_missing: bool) -> bool:
        """Return whether a summary's sender passes the contact allow list."""
        sender_email = self._extract_sender_email(message)
        if sender_email is None:
            if log_missing:
                logger.warning("Skipping email summary with missing sender_email.")
            return False
        try:
            normalized_email, _ = self._normalize_recipient_candidate(sender_email)
            return self.contacts.can_read_from(normalized_email)
        except Exception:
            logger.warning("Failed sender allow-list check.", exc_info=True)
            return False

    def _deduplicate_recipients(
        self,
        recipients: tuple[str, ...],
        cc: tuple[str, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Remove recipients from cc that already appear in to."""
        seen = {value.casefold() for value in recipients}
        deduplicated_cc: list[str] = []
        for value in cc:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduplicated_cc.append(value)
        return recipients, tuple(deduplicated_cc)

    def _attach_result_meta(self, result: IntegrationResult, *, operation_id: object, started_at: float) -> None:
        """Attach non-sensitive adapter metadata to every result."""
        details = getattr(result, "details", None)
        if not isinstance(details, dict):
            return
        details.setdefault("operation_id", str(operation_id) if operation_id is not None else None)
        details.setdefault("schema_version", _TOOL_SCHEMA_VERSION)
        details.setdefault("adapter_latency_ms", int((monotonic() - started_at) * 1000))

    def _failure_result(self, *, summary: str, error_code: str, operation_id: object) -> IntegrationResult:
        """Build a normalized integration failure payload."""
        return IntegrationResult(
            ok=False,
            summary=summary,
            details={
                "error_code": error_code,
                "operation_id": str(operation_id) if operation_id is not None else None,
                "schema_version": _TOOL_SCHEMA_VERSION,
            },
        )

    @staticmethod
    def _requires_smtputf8(email_address: str) -> bool:
        """Return whether the normalized address requires SMTPUTF8 transport."""
        local_part, _, _ = email_address.partition("@")
        return any(ord(char) > 127 for char in local_part)

    @staticmethod
    def _contains_disallowed_control_chars(value: str, *, allow_newlines: bool) -> bool:
        """Return whether text contains control characters this field forbids."""
        for char in value:
            codepoint = ord(char)
            if allow_newlines and char == "\n":
                continue
            if codepoint < 32 or codepoint == 127:
                return True
        return False