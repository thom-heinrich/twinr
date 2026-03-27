"""Translate Twinr email integration requests into mailbox actions.

This module defines the adapter-level settings, mailbox and sender protocols,
and the orchestration layer that validates integration requests before reading,
drafting, or sending email.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from twinr.integrations.adapter import IntegrationAdapter
from twinr.integrations.email.models import ApprovedEmailContacts, EmailDraft, EmailMessageSummary, normalize_email
from twinr.integrations.models import IntegrationManifest, IntegrationRequest, IntegrationResult

logger = logging.getLogger(__name__)


def _parse_strict_bool(value: object, *, field_name: str) -> bool:
    """Parse a strict boolean or supported boolean-like literal."""
    # AUDIT-FIX(2): Parse approval booleans strictly so strings like "false" do not evaluate to True.
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
    # AUDIT-FIX(4): Parse numeric input explicitly instead of relying on int(...) over arbitrary values.
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
    restrict_reads_to_known_senders: bool = False
    restrict_recipients_to_known_contacts: bool = False
    max_total_recipients: int = 10
    max_reference_ids: int = 20
    max_header_value_chars: int = 320

    def __post_init__(self) -> None:
        # AUDIT-FIX(8): Validate env/config-derived settings early so safety limits cannot be disabled by bad values.
        object.__setattr__(self, "max_read_results", _parse_positive_int(self.max_read_results, field_name="max_read_results"))
        object.__setattr__(self, "max_subject_chars", _parse_positive_int(self.max_subject_chars, field_name="max_subject_chars"))
        object.__setattr__(self, "max_body_chars", _parse_positive_int(self.max_body_chars, field_name="max_body_chars"))
        object.__setattr__(self, "max_total_recipients", _parse_positive_int(self.max_total_recipients, field_name="max_total_recipients"))
        object.__setattr__(self, "max_reference_ids", _parse_positive_int(self.max_reference_ids, field_name="max_reference_ids"))
        object.__setattr__(self, "max_header_value_chars", _parse_positive_int(self.max_header_value_chars, field_name="max_header_value_chars"))
        object.__setattr__(
            self,
            "unread_only_default",
            _parse_strict_bool(self.unread_only_default, field_name="unread_only_default"),
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
        """Dispatch one email integration request into a bounded result.

        Args:
            request: Integration request carrying the operation ID,
                parameters, and explicit approval flags.

        Returns:
            A structured ``IntegrationResult`` describing success or the
            normalized failure mode.
        """
        operation_id = getattr(request, "operation_id", None)
        try:
            # AUDIT-FIX(3): Convert validation/provider failures into structured results so the voice flow does not crash.
            if operation_id == "read_recent":
                return self._read_recent(request)
            if operation_id == "draft_reply":
                return self._draft_message(request)
            if operation_id == "send_message":
                return self._send_message(request)
            return self._failure_result(
                summary="I couldn't understand that email action.",
                error_code="unsupported_operation",
                operation_id=operation_id,
            )
        except ValueError as exc:
            return self._failure_result(
                summary=str(exc),
                error_code="invalid_request",
                operation_id=operation_id,
            )
        except RuntimeError as exc:
            logger.warning("Email operation %r failed cleanly.", operation_id, exc_info=True)
            return self._failure_result(
                summary=str(exc),
                error_code="service_unavailable",
                operation_id=operation_id,
            )
        except Exception:
            logger.exception("Unexpected email adapter failure for operation %r.", operation_id)
            return self._failure_result(
                summary="I couldn't complete the email action right now. Please try again.",
                error_code="unexpected_error",
                operation_id=operation_id,
            )

    def _read_recent(self, request: IntegrationRequest) -> IntegrationResult:
        """Build a structured result for recent mailbox summaries."""
        params = self._parameters(request)
        limit = min(
            self.settings.max_read_results,
            self._parse_limit(params.get("limit", 5)),
        )
        unread_only = self._parse_optional_bool(
            params.get("unread_only"),
            default=self.settings.unread_only_default,
            field_name="unread_only",
        )

        try:
            # AUDIT-FIX(3): Isolate mailbox provider failures behind a user-safe RuntimeError.
            raw_messages = self.mailbox_reader.list_recent(  # pylint: disable=assignment-from-no-return
                limit=limit,
                unread_only=unread_only,
            )
        except Exception as exc:
            raise RuntimeError("I couldn't read recent email right now.") from exc

        messages = self._coerce_message_list(raw_messages)
        message_dicts: list[dict[str, object]] = []
        for index, message in enumerate(messages):
            if self.settings.restrict_reads_to_known_senders and not self._is_allowed_sender(message):
                continue
            try:
                # AUDIT-FIX(9): Skip malformed provider records instead of failing the entire read operation.
                message_dicts.append(message.as_dict())
            except Exception:
                logger.warning("Skipping malformed email summary at index %s.", index, exc_info=True)

        details = {"messages": message_dicts, "count": len(message_dicts)}
        return IntegrationResult(
            ok=True,
            summary=f"{len(message_dicts)} email summaries ready.",
            details=details,
        )

    def _draft_message(self, request: IntegrationRequest) -> IntegrationResult:
        """Validate and stage an outbound email draft without sending it."""
        self._ensure_explicit_approval(request)
        draft = self._build_draft(request)
        return IntegrationResult(
            ok=True,
            summary=f"Draft prepared for {', '.join(draft.to)}.",
            details={
                "draft": {
                    "to": list(draft.to),
                    "cc": list(draft.cc),
                    "subject": draft.subject,
                    # AUDIT-FIX(5): Do not leak dictated body content via previews in integration details.
                    "body_preview": "[redacted]",
                    "body_chars": len(draft.body),
                    "in_reply_to": draft.in_reply_to,
                }
            },
            redacted_fields=("body", "body_preview"),
        )

    def _send_message(self, request: IntegrationRequest) -> IntegrationResult:
        """Validate and send one outbound email draft."""
        self._ensure_explicit_approval(request)
        if self.mail_sender is None:
            raise RuntimeError("I couldn't send email because the mail service is not available.")

        draft = self._build_draft(request)
        try:
            # AUDIT-FIX(3): Translate provider send failures into a recoverable user-facing error.
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
            },
            redacted_fields=("body",),
        )

    def _build_draft(self, request: IntegrationRequest) -> EmailDraft:
        """Construct a validated ``EmailDraft`` from integration parameters."""
        params = self._parameters(request)
        recipients = self._parse_recipients(params.get("to"))
        cc = self._parse_recipients(params.get("cc", ()), optional=True)
        recipients, cc = self._deduplicate_recipients(recipients, cc)

        total_recipients = len(recipients) + len(cc)
        if total_recipients > self.settings.max_total_recipients:
            raise ValueError("Too many email recipients were provided.")

        if self.settings.restrict_recipients_to_known_contacts:
            self.contacts.require_allowed_recipients(recipients + cc)

        # AUDIT-FIX(1): Sanitize single-line header text to prevent CRLF/header injection.
        subject = self._parse_text_field(
            params.get("subject"),
            field_name="subject",
            max_chars=self.settings.max_subject_chars,
            allow_newlines=False,
            required=True,
        )
        # AUDIT-FIX(6): Reject None and non-text body values instead of silently turning them into strings like "None".
        body = self._parse_text_field(
            params.get("body"),
            field_name="body",
            max_chars=self.settings.max_body_chars,
            allow_newlines=True,
            required=True,
        )

        # AUDIT-FIX(7): Validate threading headers as bounded text collections instead of accepting arbitrary iterables.
        in_reply_to = self._parse_message_id(params.get("in_reply_to"), field_name="in_reply_to")
        references = self._parse_message_id_collection(params.get("references", ()), field_name="references")

        return EmailDraft(
            to=recipients,
            cc=cc,
            subject=subject,
            body=body,
            in_reply_to=in_reply_to,
            references=references,
        )

    def _ensure_explicit_approval(self, request: IntegrationRequest) -> None:
        """Require an explicit user or caregiver approval signal."""
        # AUDIT-FIX(2): Require an actual boolean confirmation rather than generic truthiness.
        if self._parse_request_confirmation(getattr(request, "explicit_user_confirmation", False)):
            return
        if self._parse_request_confirmation(getattr(request, "explicit_caregiver_confirmation", False)):
            return
        raise ValueError("Please confirm before I prepare or send an email.")

    def _parse_recipients(self, raw_value: object, *, optional: bool = False) -> tuple[str, ...]:
        """Normalize one or more recipient inputs into unique email addresses."""
        if raw_value in (None, ""):
            if optional:
                return ()
            raise ValueError("Please choose at least one email recipient.")

        # AUDIT-FIX(6): Reject None/non-text recipient entries instead of silently coercing them with str(...).
        values: list[str] = []
        if isinstance(raw_value, str):
            values.extend(self._split_recipient_tokens(raw_value))
        elif isinstance(raw_value, (list, tuple)):
            for value in raw_value:
                if value in (None, ""):
                    continue
                if not isinstance(value, str):
                    raise ValueError("Each email recipient must be plain text.")
                values.extend(self._split_recipient_tokens(value))
        else:
            raise ValueError("Email recipients must be plain text.")

        recipients: list[str] = []
        seen: set[str] = set()
        for value in values:
            candidate = value.strip()
            if not candidate:
                continue
            try:
                normalized = normalize_email(candidate)
            except Exception as exc:
                raise ValueError("One email address is not valid.") from exc
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            recipients.append(normalized)

        if not recipients and not optional:
            raise ValueError("Please choose at least one email recipient.")
        return tuple(recipients)

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
            return 5
        return _parse_positive_int(raw_value, field_name="limit")

    def _parse_optional_bool(self, raw_value: object, *, default: bool, field_name: str) -> bool:
        """Parse an optional request flag without falling back to truthiness."""
        # AUDIT-FIX(4): Avoid bool("false") and other Python truthiness traps for request flags.
        if raw_value in (None, ""):
            return default
        try:
            return _parse_strict_bool(raw_value, field_name=field_name)
        except ValueError as exc:
            raise ValueError(f"I couldn't understand the {field_name} setting.") from exc

    def _parse_request_confirmation(self, raw_value: object) -> bool:
        """Parse a confirmation flag and reject ambiguous approval values."""
        # AUDIT-FIX(2): Treat ambiguous approval values as no approval instead of accepting them implicitly.
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
        # AUDIT-FIX(1): Normalize and validate user-provided header/body text before it reaches the mail provider.
        if raw_value is None:
            text = ""
        elif isinstance(raw_value, str):
            text = raw_value
        else:
            raise ValueError(f"Email {field_name} must be plain text.")

        normalized = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4).strip()
        if not allow_newlines:
            normalized = " ".join(part for part in normalized.split())

        if not normalized:
            if required:
                raise ValueError(f"Please provide an email {field_name}.")
            return ""

        if self._contains_disallowed_control_chars(normalized, allow_newlines=allow_newlines):
            raise ValueError(f"Email {field_name} contains unsupported characters.")
        if len(normalized) > max_chars:
            raise ValueError(f"Email {field_name} is too long.")
        return normalized

    def _parse_message_id(self, raw_value: object, *, field_name: str) -> str | None:
        """Normalize one message ID style header value from the request."""
        # AUDIT-FIX(1): Reject control characters in threading headers to close CRLF/header injection paths.
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
        return normalized

    def _parse_message_id_collection(self, raw_value: object, *, field_name: str) -> tuple[str, ...]:
        """Normalize a bounded collection of message ID reference values."""
        # AUDIT-FIX(7): Bound and validate message-id collections instead of iterating arbitrary caller-controlled objects.
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

    def _coerce_message_list(self, raw_messages: object) -> list[EmailMessageSummary]:
        """Coerce mailbox provider output into a concrete summary list."""
        if isinstance(raw_messages, list):
            return raw_messages
        if isinstance(raw_messages, Iterable) and not isinstance(raw_messages, (str, bytes, bytearray)):
            return list(raw_messages)
        raise RuntimeError("I couldn't read recent email right now.")

    def _is_allowed_sender(self, message: EmailMessageSummary) -> bool:
        """Return whether a summary's sender passes the contact allow list."""
        sender_email = getattr(message, "sender_email", None)
        if not isinstance(sender_email, str) or not sender_email.strip():
            logger.warning("Skipping email summary with missing sender_email.")
            return False
        try:
            return self.contacts.can_read_from(sender_email)
        except Exception:
            logger.warning("Failed sender allow-list check.", exc_info=True)
            return False

    def _deduplicate_recipients(
        self,
        recipients: tuple[str, ...],
        cc: tuple[str, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Remove recipients from ``cc`` that already appear in ``to``."""
        # AUDIT-FIX(10): Split common recipient separators and remove duplicates across To/Cc.
        seen = {value.casefold() for value in recipients}
        deduplicated_cc: list[str] = []
        for value in cc:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduplicated_cc.append(value)
        return recipients, tuple(deduplicated_cc)

    def _failure_result(self, *, summary: str, error_code: str, operation_id: object) -> IntegrationResult:
        """Build a normalized integration failure payload."""
        return IntegrationResult(
            ok=False,
            summary=summary,
            details={
                "error_code": error_code,
                "operation_id": str(operation_id) if operation_id is not None else None,
            },
        )

    @staticmethod
    def _split_recipient_tokens(raw_value: str) -> list[str]:
        """Split one recipient string on the separators Twinr accepts."""
        normalized = raw_value.replace(";", ",").replace("\r", ",").replace("\n", ",")
        return [token.strip() for token in normalized.split(",")]

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
