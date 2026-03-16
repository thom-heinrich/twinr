"""Expose Twinr's email integration building blocks.

Import the bounded email adapter, approved-contact models, IMAP mailbox
reader, and SMTP sender from this package root.
"""

from twinr.integrations.email.adapter import EmailAdapterSettings, EmailMailboxAdapter, MailboxReader, MailSender
from twinr.integrations.email.imap import IMAPMailboxConfig, IMAPMailboxReader
from twinr.integrations.email.models import ApprovedEmailContacts, EmailContact, EmailDraft, EmailMessageSummary, normalize_email
from twinr.integrations.email.smtp import SMTPMailSender, SMTPMailSenderConfig

__all__ = [
    "ApprovedEmailContacts",
    "EmailAdapterSettings",
    "EmailContact",
    "EmailDraft",
    "EmailMailboxAdapter",
    "EmailMessageSummary",
    "IMAPMailboxConfig",
    "IMAPMailboxReader",
    "MailboxReader",
    "MailSender",
    "SMTPMailSender",
    "SMTPMailSenderConfig",
    "normalize_email",
]
