from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from twinr.integrations import ManagedIntegrationConfig, TwinrIntegrationStore
from twinr.integrations.email.connectivity import (
    EmailConnectionTestResult,
    EmailTransportProbe,
)
from twinr.web.app import create_app
from twinr.web.app_impl.main import create_app as create_internal_app

_EXPECTED_GOLDEN_DIGESTS = {
    "home": "20faa9822c426c68a44b554df3381b9b210b8249c42ebdbbefc9539f58e4eb88",
    "advanced": "88b7ee62e7344185272a9aa59dc5d3ef616078c27277cb7ebca0d55a3961b384",
    "ops_config": "c750e911d62195617d06004a51d83871bdb5318fb480ded6cd37575b3927b30e",
    "email_wizard": "e5c2e35bea8a004720ff37c47ab752e3830739519892f17e64e3ce54c50c9798",
    "voice_profile": "6fae999887049195c0bdcd5510dc6416390906b4af581e4c01fc75a9906095d7",
}
_TEST_WHATSAPP_ALLOW_FROM = "+15555554567"


class _WarningQuietTestClient(TestClient):
    """Suppress known task-cancel deprecation warnings during requests."""

    def request(self, *args, **kwargs):  # type: ignore[override]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Passing 'msg' argument to Task\.cancel\(\) is deprecated since Python 3\.11, and scheduled for removal in Python 3\.14\.",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Passing 'msg' argument to Future\.cancel\(\) is deprecated since Python 3\.11, and scheduled for removal in Python 3\.14\.",
                category=DeprecationWarning,
            )
            return super().request(*args, **kwargs)


def _normalize_response_text(text: str, root: Path) -> str:
    """Strip temp-root variance from rendered HTML before hashing/comparison."""

    normalized = text.replace(str(root), "<ROOT>")
    return normalized.replace(str(root).replace("/", "&#47;"), "<ROOT_ESC>")


def _response_digest(text: str, root: Path) -> str:
    """Hash one normalized response body."""

    return hashlib.sha256(
        _normalize_response_text(text, root).encode("utf-8")
    ).hexdigest()


class WebAppRefactorParityTests(unittest.TestCase):
    def _make_client(
        self,
        app_factory,
        *,
        extra_env: dict[str, str] | None = None,
        base_url: str = "http://localhost",
        client_host: str = "127.0.0.1",
    ) -> tuple[TestClient, Path]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        env_path = root / ".env"
        personality_dir = root / "personality"
        personality_dir.mkdir(parents=True, exist_ok=True)
        env_lines = [
            "OPENAI_MODEL=gpt-5.4-mini",
            "OPENAI_API_KEY=sk-test-1234",
            "TWINR_WEB_HOST=0.0.0.0",
            "TWINR_WEB_PORT=1337",
            f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
        ]
        if extra_env:
            env_lines.extend(f"{key}={value}" for key, value in extra_env.items())
        env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        (personality_dir / "SYSTEM.md").write_text("System text\n", encoding="utf-8")
        (personality_dir / "PERSONALITY.md").write_text(
            "Personality text\n",
            encoding="utf-8",
        )
        (personality_dir / "USER.md").write_text("User text\n", encoding="utf-8")
        client = _WarningQuietTestClient(
            app_factory(env_path),
            base_url=base_url,
            client=(client_host, 50000),
        )
        return client, root

    def _write_whatsapp_worker_package(self, project_root: Path) -> None:
        worker_root = (
            project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
        )
        worker_root.mkdir(parents=True, exist_ok=True)
        (worker_root / "package.json").write_text(
            '{"name": "worker"}\n',
            encoding="utf-8",
        )
        (worker_root / "index.mjs").write_text(
            "console.log('worker');\n",
            encoding="utf-8",
        )

    def _seed_email_record(self, project_root: Path) -> TwinrIntegrationStore:
        store = TwinrIntegrationStore.from_project_root(project_root)
        store.save(
            ManagedIntegrationConfig(
                integration_id="email_mailbox",
                enabled=False,
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
        return store

    def test_public_wrapper_preserves_create_app_module(self) -> None:
        self.assertEqual(create_app.__module__, "twinr.web.app")

    def test_golden_master_hashes_remain_stable(self) -> None:
        client, root = self._make_client(create_app)
        cases = (
            ("home", "/"),
            ("advanced", "/advanced"),
            ("ops_config", "/ops/config"),
            ("email_wizard", "/integrations/email"),
            ("voice_profile", "/voice-profile"),
        )
        for name, path in cases:
            with self.subTest(case=name):
                response = client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    _response_digest(response.text, root),
                    _EXPECTED_GOLDEN_DIGESTS[name],
                )

    def test_wrapper_and_internal_match_representative_pages(self) -> None:
        cases = (
            ("home", "/"),
            ("advanced", "/advanced"),
            ("ops_config", "/ops/config"),
            ("email_wizard", "/integrations/email"),
            ("voice_profile", "/voice-profile"),
        )
        for name, path in cases:
            with self.subTest(case=name):
                wrapped_client, wrapped_root = self._make_client(create_app)
                internal_client, internal_root = self._make_client(create_internal_app)
                wrapped_response = wrapped_client.get(path)
                internal_response = internal_client.get(path)
                self.assertEqual(wrapped_response.status_code, internal_response.status_code)
                self.assertEqual(
                    _normalize_response_text(wrapped_response.text, wrapped_root),
                    _normalize_response_text(internal_response.text, internal_root),
                )

    def test_wrapper_and_internal_match_conversation_lab_redirect(self) -> None:
        wrapped_client, _wrapped_root = self._make_client(create_app)
        internal_client, _internal_root = self._make_client(create_internal_app)

        with patch(
            "twinr.web.app.create_conversation_lab_session",
            return_value="session_fixed",
        ):
            wrapped_response = wrapped_client.post(
                "/ops/debug/conversation-lab/new",
                data={},
                follow_redirects=False,
            )
            internal_response = internal_client.post(
                "/ops/debug/conversation-lab/new",
                data={},
                follow_redirects=False,
            )

        self.assertEqual(wrapped_response.status_code, 303)
        self.assertEqual(internal_response.status_code, 303)
        self.assertEqual(
            wrapped_response.headers["location"],
            "/ops/debug?tab=conversation_lab&lab_session=session_fixed",
        )
        self.assertEqual(
            wrapped_response.headers["location"],
            internal_response.headers["location"],
        )

    def test_wrapper_and_internal_match_whatsapp_pairing_redirect(self) -> None:
        extra_env = {
            "TWINR_WHATSAPP_ALLOW_FROM": _TEST_WHATSAPP_ALLOW_FROM,
            "TWINR_WHATSAPP_SELF_CHAT_MODE": "1",
            "TWINR_WHATSAPP_GROUPS_ENABLED": "0",
        }
        wrapped_client, wrapped_root = self._make_client(create_app, extra_env=extra_env)
        internal_client, internal_root = self._make_client(
            create_internal_app,
            extra_env=extra_env,
        )
        self._write_whatsapp_worker_package(wrapped_root)
        self._write_whatsapp_worker_package(internal_root)

        with patch(
            "twinr.web.support.whatsapp.subprocess.run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="v20.20.1\n",
                stderr="",
            ),
        ), patch(
            "twinr.web.app.WhatsAppPairingCoordinator.start_pairing",
            return_value=True,
        ):
            wrapped_response = wrapped_client.post(
                "/connect/whatsapp",
                data={"_action": "start_pairing"},
                follow_redirects=False,
            )
            internal_response = internal_client.post(
                "/connect/whatsapp",
                data={"_action": "start_pairing"},
                follow_redirects=False,
            )

        self.assertEqual(wrapped_response.status_code, 303)
        self.assertEqual(internal_response.status_code, 303)
        self.assertEqual(
            wrapped_response.headers["location"],
            "/connect/whatsapp?step=pairing",
        )
        self.assertEqual(
            wrapped_response.headers["location"],
            internal_response.headers["location"],
        )

    def test_wrapper_and_internal_match_email_connection_test_flow(self) -> None:
        extra_env = {"TWINR_INTEGRATION_EMAIL_APP_PASSWORD": "abcdefghijklmnop"}
        wrapped_client, wrapped_root = self._make_client(create_app, extra_env=extra_env)
        internal_client, internal_root = self._make_client(
            create_internal_app,
            extra_env=extra_env,
        )
        wrapped_store = self._seed_email_record(wrapped_root)
        internal_store = self._seed_email_record(internal_root)

        with patch(
            "twinr.web.app.run_email_connectivity_test",
            return_value=EmailConnectionTestResult(
                status="ok",
                summary="Connection test passed",
                detail="Twinr reached both servers.",
                imap=EmailTransportProbe(
                    status="ok",
                    summary="Connected",
                    detail="IMAP worked.",
                ),
                smtp=EmailTransportProbe(
                    status="ok",
                    summary="Connected",
                    detail="SMTP worked.",
                ),
                tested_at="2026-03-26T18:00:00+00:00",
            ),
        ):
            wrapped_response = wrapped_client.post(
                "/integrations/email",
                data={
                    "_action": "run_connection_test",
                    "enabled": "false",
                    "unread_only_default": "true",
                    "restrict_reads_to_known_senders": "false",
                    "restrict_recipients_to_known_contacts": "true",
                    "known_contacts_text": "Anna <anna@gmail.com>",
                },
                follow_redirects=False,
            )
            internal_response = internal_client.post(
                "/integrations/email",
                data={
                    "_action": "run_connection_test",
                    "enabled": "false",
                    "unread_only_default": "true",
                    "restrict_reads_to_known_senders": "false",
                    "restrict_recipients_to_known_contacts": "true",
                    "known_contacts_text": "Anna <anna@gmail.com>",
                },
                follow_redirects=False,
            )

        self.assertEqual(wrapped_response.status_code, 303)
        self.assertEqual(internal_response.status_code, 303)
        self.assertEqual(
            wrapped_response.headers["location"],
            "/integrations/email?saved=1&step=guardrails",
        )
        self.assertEqual(
            wrapped_response.headers["location"],
            internal_response.headers["location"],
        )

        wrapped_record = wrapped_store.get("email_mailbox")
        internal_record = internal_store.get("email_mailbox")
        self.assertEqual(
            wrapped_record.value("connection_test_status"),
            internal_record.value("connection_test_status"),
        )
        self.assertEqual(
            wrapped_record.value("connection_test_imap_summary"),
            internal_record.value("connection_test_imap_summary"),
        )
        self.assertEqual(
            wrapped_record.value("connection_test_smtp_summary"),
            internal_record.value("connection_test_smtp_summary"),
        )
        self.assertEqual(
            wrapped_record.value("restrict_recipients_to_known_contacts"),
            internal_record.value("restrict_recipients_to_known_contacts"),
        )


if __name__ == "__main__":
    unittest.main()
