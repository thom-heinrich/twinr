from pathlib import Path
import imaplib
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.integrations import EmailAdapterSettings, IntegrationRequest, manifest_for_id
from twinr.integrations.email import (
    ApprovedEmailContacts,
    EmailContact,
    EmailDraft,
    EmailMailboxAdapter,
    EmailMessageSummary,
    IMAPMailboxConfig,
    IMAPMailboxReader,
    SMTPMailSender,
    SMTPMailSenderConfig,
    run_email_connectivity_test,
)


class _FakeMailboxReader:
    def __init__(self, messages: list[EmailMessageSummary]) -> None:
        self.messages = messages

    def list_recent(self, *, limit: int, unread_only: bool) -> list[EmailMessageSummary]:
        if unread_only:
            return [message for message in self.messages if message.unread][:limit]
        return self.messages[:limit]


class _FakeMailSender:
    def __init__(self) -> None:
        self.sent: list[EmailDraft] = []

    def send(self, draft: EmailDraft) -> str:
        self.sent.append(draft)
        return "smtp-123"


class _FakeSMTPConnection:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.message = None

    def ehlo(self) -> None:
        self.calls.append("ehlo")

    def starttls(self) -> None:
        self.calls.append("starttls")

    def login(self, username: str, password: str) -> None:
        self.calls.append(f"login:{username}:{password}")

    def send_message(self, message) -> None:
        self.calls.append("send_message")
        self.message = message

    def quit(self) -> None:
        self.calls.append("quit")


class _FakeSocket:
    def __init__(self) -> None:
        self.timeout = None

    def settimeout(self, value: float) -> None:
        self.timeout = value


class _FakeIMAPConnection:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.sock = _FakeSocket()

    def login(self, username: str, password: str) -> tuple[str, list[bytes]]:
        self.calls.append(f"login:{username}:{password}")
        return ("OK", [b"logged in"])

    def select(self, mailbox: str, readonly: bool) -> tuple[str, list[bytes]]:
        self.calls.append(f"select:{mailbox}:{readonly}")
        return ("OK", [b"1"])

    def close(self) -> None:
        self.calls.append("close")

    def logout(self) -> None:
        self.calls.append("logout")


class _FakeIMAPAuthFailureConnection(_FakeIMAPConnection):
    def login(self, username: str, password: str) -> tuple[str, list[bytes]]:
        raise imaplib.IMAP4.error("bad credentials")


class EmailAllowlistTests(unittest.TestCase):
    def test_resolves_aliases_and_display_name(self) -> None:
        contacts = ApprovedEmailContacts(
            contacts=(
                EmailContact(
                    email="anna@example.com",
                    display_name="Anna",
                    aliases=("daughter", "family"),
                ),
            )
        )

        self.assertEqual(contacts.resolve("Anna").email, "anna@example.com")
        self.assertEqual(contacts.resolve("daughter").email, "anna@example.com")
        self.assertTrue(contacts.can_send_to("anna@example.com"))


class EmailMailboxAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = manifest_for_id("email_mailbox")
        assert self.manifest is not None
        self.contacts = ApprovedEmailContacts(
            contacts=(
                EmailContact(email="anna@example.com", display_name="Anna"),
                EmailContact(email="doctor@example.com", display_name="Dr. Weber", allow_send=False),
            )
        )

    def test_read_recent_keeps_unknown_senders_by_default(self) -> None:
        mailbox_reader = _FakeMailboxReader(
            [
                EmailMessageSummary("m1", "anna@example.com", "Anna", "Hallo", preview="Hi", unread=True),
                EmailMessageSummary("m2", "unknown@example.com", "Unknown", "Spam", preview="Nope", unread=True),
            ]
        )
        adapter = EmailMailboxAdapter(
            manifest=self.manifest,
            contacts=self.contacts,
            mailbox_reader=mailbox_reader,
            settings=EmailAdapterSettings(max_read_results=5),
        )

        result = adapter.execute(IntegrationRequest(integration_id="email_mailbox", operation_id="read_recent"))

        self.assertEqual(result.details["count"], 2)

    def test_read_recent_can_restrict_to_known_senders(self) -> None:
        mailbox_reader = _FakeMailboxReader(
            [
                EmailMessageSummary("m1", "anna@example.com", "Anna", "Hallo", preview="Hi", unread=True),
                EmailMessageSummary("m2", "unknown@example.com", "Unknown", "Spam", preview="Nope", unread=True),
            ]
        )
        adapter = EmailMailboxAdapter(
            manifest=self.manifest,
            contacts=self.contacts,
            mailbox_reader=mailbox_reader,
            settings=EmailAdapterSettings(
                max_read_results=5,
                restrict_reads_to_known_senders=True,
            ),
        )

        result = adapter.execute(IntegrationRequest(integration_id="email_mailbox", operation_id="read_recent"))

        self.assertEqual(result.details["count"], 1)
        self.assertEqual(result.details["messages"][0]["sender_email"], "anna@example.com")

    def test_draft_requires_explicit_permission(self) -> None:
        adapter = EmailMailboxAdapter(
            manifest=self.manifest,
            contacts=self.contacts,
            mailbox_reader=_FakeMailboxReader([]),
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="email_mailbox",
                operation_id="draft_reply",
                parameters={"to": "unknown@example.com", "subject": "Hallo", "body": "Text"},
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.details["error_code"], "invalid_request")
        self.assertEqual(result.summary, "Please confirm before I prepare or send an email.")

    def test_draft_allows_unknown_recipient_after_permission(self) -> None:
        adapter = EmailMailboxAdapter(
            manifest=self.manifest,
            contacts=self.contacts,
            mailbox_reader=_FakeMailboxReader([]),
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="email_mailbox",
                operation_id="draft_reply",
                parameters={"to": "unknown@example.com", "subject": "Hallo", "body": "Text"},
                explicit_user_confirmation=True,
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.details["draft"]["to"], ["unknown@example.com"])

    def test_send_can_still_be_restricted_to_known_contacts_if_enabled(self) -> None:
        fake_sender = _FakeMailSender()
        adapter = EmailMailboxAdapter(
            manifest=self.manifest,
            contacts=self.contacts,
            mailbox_reader=_FakeMailboxReader([]),
            mail_sender=fake_sender,
            settings=EmailAdapterSettings(restrict_recipients_to_known_contacts=True),
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="email_mailbox",
                operation_id="send_message",
                parameters={"to": "unknown@example.com", "subject": "Hallo", "body": "Ich bin zuhause."},
                explicit_user_confirmation=True,
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.details["error_code"], "invalid_request")
        self.assertEqual(result.summary, "One or more recipients are not approved for sending.")
        self.assertEqual(fake_sender.sent, [])

    def test_send_allows_unknown_recipient_after_permission_by_default(self) -> None:
        fake_sender = _FakeMailSender()
        adapter = EmailMailboxAdapter(
            manifest=self.manifest,
            contacts=self.contacts,
            mailbox_reader=_FakeMailboxReader([]),
            mail_sender=fake_sender,
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="email_mailbox",
                operation_id="send_message",
                parameters={"to": "unknown@example.com", "subject": "Hallo", "body": "Ich bin zuhause."},
                explicit_user_confirmation=True,
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(fake_sender.sent[0].to, ("unknown@example.com",))

    def test_send_message_records_provider_message_id(self) -> None:
        fake_sender = _FakeMailSender()
        adapter = EmailMailboxAdapter(
            manifest=self.manifest,
            contacts=self.contacts,
            mailbox_reader=_FakeMailboxReader([]),
            mail_sender=fake_sender,
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="email_mailbox",
                operation_id="send_message",
                parameters={"to": "anna@example.com", "subject": "Hallo", "body": "Ich bin zuhause."},
                explicit_user_confirmation=True,
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.details["provider_message_id"], "smtp-123")
        self.assertEqual(fake_sender.sent[0].to, ("anna@example.com",))


class SMTPMailSenderTests(unittest.TestCase):
    def test_smtp_sender_builds_message_and_starts_session(self) -> None:
        connection = _FakeSMTPConnection()
        sender = SMTPMailSender(
            SMTPMailSenderConfig(
                host="smtp.example.com",
                username="anna",
                password="secret",
                from_address="twinr@example.com",
            ),
            connection_factory=lambda config: connection,
        )

        message_id = sender.send(
            EmailDraft(
                to=("anna@example.com",),
                subject="Check-in",
                body="Ich bin zuhause.",
            )
        )

        self.assertIn("starttls", connection.calls)
        self.assertIn("send_message", connection.calls)
        self.assertEqual(connection.message["To"], "anna@example.com")
        self.assertTrue(str(message_id).startswith("<"))

    def test_probe_connection_runs_greeting_and_login(self) -> None:
        connection = _FakeSMTPConnection()
        sender = SMTPMailSender(
            SMTPMailSenderConfig(
                host="smtp.example.com",
                username="anna",
                password="secret",
                from_address="twinr@example.com",
            ),
            connection_factory=lambda config: connection,
        )

        sender.probe_connection()

        self.assertIn("ehlo", connection.calls)
        self.assertIn("starttls", connection.calls)
        self.assertIn("login:anna:secret", connection.calls)


class IMAPMailboxReaderTests(unittest.TestCase):
    def test_probe_connection_logs_in_and_selects_mailbox(self) -> None:
        connection = _FakeIMAPConnection()
        reader = IMAPMailboxReader(
            IMAPMailboxConfig(
                host="imap.example.com",
                username="anna@example.com",
                password="secret",
                mailbox="INBOX",
            ),
            connection_factory=lambda config: connection,
        )

        reader.probe_connection()

        self.assertIn("login:anna@example.com:secret", connection.calls)
        self.assertIn("select:INBOX:True", connection.calls)
        self.assertEqual(connection.sock.timeout, 10.0)


class EmailConnectivityTestTests(unittest.TestCase):
    def test_connectivity_test_reports_success_when_both_transports_work(self) -> None:
        result = run_email_connectivity_test(
            IMAPMailboxConfig(
                host="imap.example.com",
                username="anna@example.com",
                password="secret",
                mailbox="INBOX",
            ),
            SMTPMailSenderConfig(
                host="smtp.example.com",
                username="anna@example.com",
                password="secret",
                from_address="anna@example.com",
            ),
            mailbox_reader_factory=lambda config: IMAPMailboxReader(config, connection_factory=lambda _config: _FakeIMAPConnection()),
            mail_sender_factory=lambda config: SMTPMailSender(config, connection_factory=lambda _config: _FakeSMTPConnection()),
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.summary, "Connection test passed")
        self.assertEqual(result.imap.status, "ok")
        self.assertEqual(result.smtp.status, "ok")

    def test_connectivity_test_redacts_imap_auth_failure(self) -> None:
        result = run_email_connectivity_test(
            IMAPMailboxConfig(
                host="imap.example.com",
                username="anna@example.com",
                password="secret",
                mailbox="INBOX",
            ),
            SMTPMailSenderConfig(
                host="smtp.example.com",
                username="anna@example.com",
                password="secret",
                from_address="anna@example.com",
            ),
            mailbox_reader_factory=lambda config: IMAPMailboxReader(
                config,
                connection_factory=lambda _config: _FakeIMAPAuthFailureConnection(),
            ),
            mail_sender_factory=lambda config: SMTPMailSender(config, connection_factory=lambda _config: _FakeSMTPConnection()),
        )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.imap.status, "fail")
        self.assertEqual(result.smtp.status, "ok")
        self.assertIn("mailbox login was rejected", result.imap.detail.lower())


if __name__ == "__main__":
    unittest.main()
