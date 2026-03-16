from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.integrations import (
    EMAIL_APP_PASSWORD_ENV_KEY,
    ManagedIntegrationConfig,
    TwinrIntegrationStore,
    build_managed_integrations,
)


class ManagedIntegrationRuntimeTests(unittest.TestCase):
    def test_builds_email_adapter_from_store_and_env_without_secret_echo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f'{EMAIL_APP_PASSWORD_ENV_KEY}="abcd efgh ijkl mnop"\n', encoding="utf-8")
            store = TwinrIntegrationStore.from_project_root(root)
            store.save(
                ManagedIntegrationConfig(
                    integration_id="email_mailbox",
                    enabled=True,
                    settings={
                        "profile": "gmail",
                        "account_email": "anna@gmail.com",
                        "from_address": "anna@gmail.com",
                        "imap_host": "imap.gmail.com",
                        "imap_port": "993",
                        "imap_mailbox": "INBOX",
                        "smtp_host": "smtp.gmail.com",
                        "smtp_port": "587",
                        "unread_only_default": "true",
                        "restrict_reads_to_known_senders": "false",
                        "restrict_recipients_to_known_contacts": "false",
                        "known_contacts_text": "Anna <anna@gmail.com>",
                    },
                )
            )

            runtime = build_managed_integrations(root, env_path=env_path)

        self.assertIsNotNone(runtime.email_mailbox)
        assert runtime.email_mailbox is not None
        self.assertEqual(runtime.email_mailbox.mailbox_reader.config.password, "abcd efgh ijkl mnop")
        assert runtime.email_mailbox.mail_sender is not None
        self.assertEqual(runtime.email_mailbox.mail_sender.config.password, "abcd efgh ijkl mnop")
        readiness = runtime.readiness_for("email_mailbox")
        assert readiness is not None
        self.assertEqual(readiness.status, "warn")
        self.assertEqual(readiness.summary, "Ready with warnings")
        self.assertIn("credential stored separately in .env", readiness.detail)
        self.assertIn("Outbound email is not restricted to approved contacts.", readiness.detail)
        self.assertIn(
            "Outbound email is not restricted to approved contacts. This increases the risk of misaddressed or unsafe sends.",
            readiness.warnings,
        )
        self.assertNotIn("abcd", readiness.detail)
        self.assertNotIn("mnop", readiness.detail)
        self.assertNotIn("abcd efgh ijkl mnop", repr(readiness))

    def test_email_adapter_needs_secret_before_it_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            store = TwinrIntegrationStore.from_project_root(root)
            store.save(
                ManagedIntegrationConfig(
                    integration_id="email_mailbox",
                    enabled=True,
                    settings={
                        "profile": "gmail",
                        "account_email": "anna@gmail.com",
                        "from_address": "anna@gmail.com",
                        "imap_host": "imap.gmail.com",
                        "imap_port": "993",
                        "imap_mailbox": "INBOX",
                        "smtp_host": "smtp.gmail.com",
                        "smtp_port": "587",
                    },
                )
            )

            runtime = build_managed_integrations(root, env_path=env_path)

        self.assertIsNone(runtime.email_mailbox)
        readiness = runtime.readiness_for("email_mailbox")
        assert readiness is not None
        self.assertEqual(readiness.status, "warn")
        self.assertIn("credential", readiness.detail)

    def test_calendar_runtime_blocks_tokenized_url_before_it_can_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            store = TwinrIntegrationStore.from_project_root(root)
            store.save(
                ManagedIntegrationConfig(
                    integration_id="calendar_agenda",
                    enabled=True,
                    settings={
                        "source_kind": "ics_url",
                        "source_value": "https://calendar.example.com/feed.ics?token=super-secret",
                        "timezone": "Europe/Berlin",
                        "default_upcoming_days": "7",
                        "max_events": "12",
                    },
                )
            )

            runtime = build_managed_integrations(root, env_path=env_path)

        self.assertIsNone(runtime.calendar_agenda)
        readiness = runtime.readiness_for("calendar_agenda")
        assert readiness is not None
        self.assertEqual(readiness.status, "warn")
        self.assertIn("must not include embedded credentials, query tokens, or fragments", readiness.detail)
        self.assertNotIn("token=super-secret", readiness.detail)

    def test_calendar_runtime_builds_plain_url_source_without_fetching_on_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            store = TwinrIntegrationStore.from_project_root(root)
            store.save(
                ManagedIntegrationConfig(
                    integration_id="calendar_agenda",
                    enabled=True,
                    settings={
                        "source_kind": "ics_url",
                        "source_value": "https://calendar.example.com/feed.ics",
                        "timezone": "Europe/Berlin",
                        "default_upcoming_days": "7",
                        "max_events": "12",
                    },
                )
            )

            runtime = build_managed_integrations(root, env_path=env_path)

        self.assertIsNotNone(runtime.calendar_agenda)
        readiness = runtime.readiness_for("calendar_agenda")
        assert readiness is not None
        self.assertEqual(readiness.status, "ok")
        self.assertIn("https://calendar.example.com/feed.ics", readiness.detail)


if __name__ == "__main__":
    unittest.main()
