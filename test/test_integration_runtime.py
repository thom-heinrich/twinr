from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.integrations import (
    AggregatedSmartHomeProvider,
    EMAIL_APP_PASSWORD_ENV_KEY,
    HUE_ADDITIONAL_BRIDGE_HOSTS_SETTING_KEY,
    HUE_APPLICATION_KEY_ENV_KEY,
    ManagedIntegrationConfig,
    TwinrIntegrationStore,
    build_managed_integrations,
    build_smart_home_hub_adapter,
    hue_application_key_env_key_for_host,
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
        self.assertIn("credential stored outside config", readiness.detail)
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

    def test_email_runtime_uses_united_domains_profile_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f'{EMAIL_APP_PASSWORD_ENV_KEY}="mailbox-secret"\n', encoding="utf-8")
            store = TwinrIntegrationStore.from_project_root(root)
            store.save(
                ManagedIntegrationConfig(
                    integration_id="email_mailbox",
                    enabled=True,
                    settings={
                        "profile": "united_domains",
                        "account_email": "anna@example.de",
                        "from_address": "anna@example.de",
                    },
                )
            )

            runtime = build_managed_integrations(root, env_path=env_path)

        self.assertIsNotNone(runtime.email_mailbox)
        assert runtime.email_mailbox is not None
        self.assertEqual(runtime.email_mailbox.mailbox_reader.config.host, "imaps.udag.de")
        self.assertEqual(runtime.email_mailbox.mail_sender.config.host, "smtps.udag.de")
        readiness = runtime.readiness_for("email_mailbox")
        assert readiness is not None
        self.assertIn("imaps.udag.de:993", readiness.detail)
        self.assertIn("smtps.udag.de:587", readiness.detail)

    def test_email_runtime_warns_for_oauth_only_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f'{EMAIL_APP_PASSWORD_ENV_KEY}="mailbox-secret"\n', encoding="utf-8")
            store = TwinrIntegrationStore.from_project_root(root)
            store.save(
                ManagedIntegrationConfig(
                    integration_id="email_mailbox",
                    enabled=True,
                    settings={
                        "profile": "outlook_oauth",
                        "account_email": "anna@outlook.com",
                        "from_address": "anna@outlook.com",
                    },
                )
            )

            runtime = build_managed_integrations(root, env_path=env_path)

        self.assertIsNone(runtime.email_mailbox)
        readiness = runtime.readiness_for("email_mailbox")
        assert readiness is not None
        self.assertEqual(readiness.status, "warn")
        self.assertEqual(readiness.summary, "Needs OAuth2")
        self.assertIn("OAuth2 / Modern Auth", readiness.detail)

    def test_calendar_runtime_hides_tokenized_url_query_in_readiness_text(self) -> None:
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

        self.assertIsNotNone(runtime.calendar_agenda)
        readiness = runtime.readiness_for("calendar_agenda")
        assert readiness is not None
        self.assertEqual(readiness.status, "ok")
        self.assertIn("https://calendar.example.com/feed.ics", readiness.detail)
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

    def test_builds_smart_home_hub_from_store_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f'{HUE_APPLICATION_KEY_ENV_KEY}="local-hue-key"\n', encoding="utf-8")
            store = TwinrIntegrationStore.from_project_root(root)
            store.save(
                ManagedIntegrationConfig(
                    integration_id="smart_home_hub",
                    enabled=True,
                    settings={
                        "provider": "hue",
                        "bridge_host": "192.168.1.20",
                        "verify_tls": "true",
                        "request_timeout_s": "10",
                        "event_timeout_s": "2",
                    },
                )
            )

            runtime = build_managed_integrations(root, env_path=env_path)
            adapter = build_smart_home_hub_adapter(root, env_path=env_path)

        self.assertIsNotNone(runtime.smart_home_hub)
        self.assertIsNotNone(adapter)
        readiness = runtime.readiness_for("smart_home_hub")
        assert readiness is not None
        self.assertEqual(readiness.status, "ok")
        self.assertIn("192.168.1.20", readiness.detail)
        self.assertIn("local app key stored outside config", readiness.detail)
        self.assertIn("TLS verify on", readiness.detail)
        self.assertNotIn("local-hue-key", readiness.detail)

    def test_builds_multi_bridge_smart_home_hub_from_store_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            secondary_env_key = hue_application_key_env_key_for_host("192.168.1.21")
            env_path.write_text(
                "\n".join(
                    (
                        f'{HUE_APPLICATION_KEY_ENV_KEY}="primary-hue-key"',
                        f'{secondary_env_key}="secondary-hue-key"',
                        "",
                    )
                ),
                encoding="utf-8",
            )
            store = TwinrIntegrationStore.from_project_root(root)
            store.save(
                ManagedIntegrationConfig(
                    integration_id="smart_home_hub",
                    enabled=True,
                    settings={
                        "provider": "hue",
                        "bridge_host": "192.168.1.20",
                        HUE_ADDITIONAL_BRIDGE_HOSTS_SETTING_KEY: "192.168.1.21",
                        "verify_tls": "true",
                        "request_timeout_s": "10",
                        "event_timeout_s": "2",
                    },
                )
            )

            runtime = build_managed_integrations(root, env_path=env_path)
            adapter = build_smart_home_hub_adapter(root, env_path=env_path)

        self.assertIsNotNone(runtime.smart_home_hub)
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertIsInstance(adapter.entity_provider, AggregatedSmartHomeProvider)
        self.assertEqual(
            tuple(provider.route_id for provider in adapter.entity_provider.providers),
            ("192.168.1.20", "192.168.1.21"),
        )
        readiness = runtime.readiness_for("smart_home_hub")
        assert readiness is not None
        self.assertEqual(readiness.status, "ok")
        self.assertIn("2/2 bridges ready", readiness.detail)
        self.assertIn("192.168.1.20", readiness.detail)
        self.assertIn("192.168.1.21", readiness.detail)

    def test_multi_bridge_runtime_stays_available_when_one_secondary_key_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f'{HUE_APPLICATION_KEY_ENV_KEY}="primary-hue-key"\n', encoding="utf-8")
            store = TwinrIntegrationStore.from_project_root(root)
            store.save(
                ManagedIntegrationConfig(
                    integration_id="smart_home_hub",
                    enabled=True,
                    settings={
                        "provider": "hue",
                        "bridge_host": "192.168.1.20",
                        HUE_ADDITIONAL_BRIDGE_HOSTS_SETTING_KEY: "192.168.1.21",
                        "verify_tls": "true",
                        "request_timeout_s": "10",
                        "event_timeout_s": "2",
                    },
                )
            )

            runtime = build_managed_integrations(root, env_path=env_path)
            adapter = build_smart_home_hub_adapter(root, env_path=env_path)

        self.assertIsNotNone(runtime.smart_home_hub)
        self.assertIsNotNone(adapter)
        assert adapter is not None
        self.assertIsInstance(adapter.entity_provider, AggregatedSmartHomeProvider)
        self.assertEqual(
            tuple(provider.route_id for provider in adapter.entity_provider.providers),
            ("192.168.1.20",),
        )
        readiness = runtime.readiness_for("smart_home_hub")
        assert readiness is not None
        self.assertEqual(readiness.status, "warn")
        self.assertIn("1/2 bridges ready", readiness.detail)
        self.assertIn("192.168.1.21", readiness.detail)

    def test_smart_home_hub_needs_application_key_before_it_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            store = TwinrIntegrationStore.from_project_root(root)
            store.save(
                ManagedIntegrationConfig(
                    integration_id="smart_home_hub",
                    enabled=True,
                    settings={
                        "provider": "hue",
                        "bridge_host": "192.168.1.20",
                        "verify_tls": "true",
                        "request_timeout_s": "10",
                        "event_timeout_s": "2",
                    },
                )
            )

            runtime = build_managed_integrations(root, env_path=env_path)

        self.assertIsNone(runtime.smart_home_hub)
        readiness = runtime.readiness_for("smart_home_hub")
        assert readiness is not None
        self.assertEqual(readiness.status, "warn")
        self.assertIn("Hue application key", readiness.detail)


if __name__ == "__main__":
    unittest.main()
