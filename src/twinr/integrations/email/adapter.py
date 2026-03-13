from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from twinr.integrations.adapter import IntegrationAdapter
from twinr.integrations.email.models import ApprovedEmailContacts, EmailDraft, EmailMessageSummary, normalize_email
from twinr.integrations.models import IntegrationManifest, IntegrationRequest, IntegrationResult


@runtime_checkable
class MailboxReader(Protocol):
    def list_recent(self, *, limit: int, unread_only: bool) -> list[EmailMessageSummary]:
        ...


@runtime_checkable
class MailSender(Protocol):
    def send(self, draft: EmailDraft) -> str | None:
        ...


@dataclass(frozen=True, slots=True)
class EmailAdapterSettings:
    max_read_results: int = 8
    max_subject_chars: int = 160
    max_body_chars: int = 2000
    unread_only_default: bool = True
    restrict_reads_to_known_senders: bool = False
    restrict_recipients_to_known_contacts: bool = False


@dataclass(slots=True)
class EmailMailboxAdapter(IntegrationAdapter):
    manifest: IntegrationManifest
    contacts: ApprovedEmailContacts
    mailbox_reader: MailboxReader
    mail_sender: MailSender | None = None
    settings: EmailAdapterSettings = field(default_factory=EmailAdapterSettings)

    def execute(self, request: IntegrationRequest) -> IntegrationResult:
        if request.operation_id == "read_recent":
            return self._read_recent(request)
        if request.operation_id == "draft_reply":
            return self._draft_message(request)
        if request.operation_id == "send_message":
            return self._send_message(request)
        raise ValueError(f"Unsupported email operation: {request.operation_id}")

    def _read_recent(self, request: IntegrationRequest) -> IntegrationResult:
        limit = min(self.settings.max_read_results, max(1, int(request.parameters.get("limit", 5))))
        unread_only = bool(request.parameters.get("unread_only", self.settings.unread_only_default))
        messages = self.mailbox_reader.list_recent(limit=limit, unread_only=unread_only)
        if self.settings.restrict_reads_to_known_senders:
            messages = [
                message
                for message in messages
                if self.contacts.can_read_from(message.sender_email)
            ]

        details = {"messages": [message.as_dict() for message in messages], "count": len(messages)}
        return IntegrationResult(
            ok=True,
            summary=f"{len(messages)} email summaries ready.",
            details=details,
        )

    def _draft_message(self, request: IntegrationRequest) -> IntegrationResult:
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
                    "body_preview": draft.body[:160],
                    "in_reply_to": draft.in_reply_to,
                }
            },
            redacted_fields=("body",),
        )

    def _send_message(self, request: IntegrationRequest) -> IntegrationResult:
        self._ensure_explicit_approval(request)
        if self.mail_sender is None:
            raise RuntimeError("No MailSender is configured for send_message.")

        draft = self._build_draft(request)
        provider_message_id = self.mail_sender.send(draft)
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
        recipients = self._parse_recipients(request.parameters.get("to"))
        cc = self._parse_recipients(request.parameters.get("cc", ()), optional=True)
        if self.settings.restrict_recipients_to_known_contacts:
            self.contacts.require_allowed_recipients(recipients + cc)

        subject = str(request.parameters.get("subject", "")).strip()
        body = str(request.parameters.get("body", "")).strip()
        if not subject:
            raise ValueError("Email subject must not be empty.")
        if not body:
            raise ValueError("Email body must not be empty.")
        if len(subject) > self.settings.max_subject_chars:
            raise ValueError("Email subject exceeds the allowed length.")
        if len(body) > self.settings.max_body_chars:
            raise ValueError("Email body exceeds the allowed length.")

        in_reply_to = request.parameters.get("in_reply_to")
        references = request.parameters.get("references", ())
        if isinstance(references, str):
            references = (references,)

        return EmailDraft(
            to=recipients,
            cc=cc,
            subject=subject,
            body=body,
            in_reply_to=(str(in_reply_to).strip() or None) if in_reply_to is not None else None,
            references=tuple(str(value).strip() for value in references if str(value).strip()),
        )

    def _ensure_explicit_approval(self, request: IntegrationRequest) -> None:
        if request.explicit_user_confirmation or request.explicit_caregiver_confirmation:
            return
        raise ValueError("Email draft/send requires explicit approval.")

    def _parse_recipients(self, raw_value: object, *, optional: bool = False) -> tuple[str, ...]:
        if raw_value in (None, ""):
            if optional:
                return ()
            raise ValueError("At least one email recipient is required.")
        if isinstance(raw_value, str):
            values = [raw_value]
        elif isinstance(raw_value, (list, tuple)):
            values = [str(value) for value in raw_value]
        else:
            raise ValueError("Email recipients must be a string or a list of strings.")

        recipients = tuple(normalize_email(value) for value in values if str(value).strip())
        if not recipients and not optional:
            raise ValueError("At least one email recipient is required.")
        return recipients
