"""Expose Twinr's email integration building blocks.

Import the bounded email adapter, approved-contact models, IMAP mailbox
reader, SMTP sender, and mailbox connectivity probe helpers from this package
root.
"""

from twinr.integrations.email.adapter import EmailAdapterSettings, EmailMailboxAdapter, MailboxReader, MailSender
from twinr.integrations.email.connectivity import EmailConnectionTestResult, EmailTransportProbe, run_email_connectivity_test
from twinr.integrations.email.imap import IMAPMailboxConfig, IMAPMailboxReader
from twinr.integrations.email.models import ApprovedEmailContacts, EmailContact, EmailDraft, EmailMessageSummary, normalize_email
from twinr.integrations.email.smtp import SMTPMailSender, SMTPMailSenderConfig

__all__ = [
    "ApprovedEmailContacts",
    "EmailAdapterSettings",
    "EmailConnectionTestResult",
    "EmailContact",
    "EmailDraft",
    "EmailMailboxAdapter",
    "EmailMessageSummary",
    "EmailTransportProbe",
    "IMAPMailboxConfig",
    "IMAPMailboxReader",
    "MailboxReader",
    "MailSender",
    "SMTPMailSender",
    "SMTPMailSenderConfig",
    "normalize_email",
    "run_email_connectivity_test",
]
