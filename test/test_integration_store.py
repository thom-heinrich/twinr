import sqlite3
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.integrations import ManagedIntegrationConfig, TwinrIntegrationStore, integration_store_path


class TwinrIntegrationStoreTests(unittest.TestCase):
    def test_save_and_reload_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TwinrIntegrationStore.from_project_root(temp_dir)
            saved = store.save(
                ManagedIntegrationConfig(
                    integration_id="email_mailbox",
                    enabled=True,
                    settings={"profile": "gmail", "account_email": "anna@example.com"},
                )
            )

            reloaded = store.get("email_mailbox")

        self.assertTrue(saved.updated_at)
        self.assertTrue(reloaded.enabled)
        self.assertEqual(reloaded.value("profile"), "gmail")
        self.assertEqual(reloaded.value("account_email"), "anna@example.com")

    def test_integration_store_path_points_to_artifacts_store(self) -> None:
        path = integration_store_path("/tmp/twinr")
        self.assertEqual(path.as_posix(), "/tmp/twinr/artifacts/stores/integrations/integrations.sqlite3")

    def test_legacy_json_check_schema_is_rebuilt_into_portable_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = TwinrIntegrationStore.from_project_root(root)
            store.path.parent.mkdir(parents=True, exist_ok=True)

            connection = sqlite3.connect(store.path)
            try:
                connection.execute("PRAGMA application_id=0x54574E52")
                connection.execute("PRAGMA user_version=2")
                connection.execute(
                    """
                    CREATE TABLE integration_configs (
                        integration_id TEXT PRIMARY KEY,
                        enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                        settings_json TEXT NOT NULL CHECK (
                            json_valid(settings_json)
                            AND json_type(settings_json) = 'object'
                        ),
                        updated_at TEXT NOT NULL
                    ) STRICT
                    """
                )
                connection.execute(
                    """
                    INSERT INTO integration_configs (integration_id, enabled, settings_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        "calendar_agenda",
                        1,
                        '{"source_kind":"ics_url","source_value":"https://calendar.example.com/feed.ics?token=abc"}',
                        "2026-03-30T12:00:00+00:00",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            rebuilt = store.get("calendar_agenda")

            self.assertTrue(rebuilt.enabled)
            self.assertEqual(rebuilt.value("source_kind"), "ics_url")
            self.assertEqual(
                rebuilt.value("source_value"),
                "https://calendar.example.com/feed.ics?token=abc",
            )

            connection = sqlite3.connect(store.path)
            try:
                user_version = connection.execute("PRAGMA user_version").fetchone()[0]
                schema_sql = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='integration_configs'"
                ).fetchone()[0]
            finally:
                connection.close()

        self.assertEqual(user_version, 3)
        self.assertNotIn("json_valid(", schema_sql.casefold())
        self.assertNotIn("json_type(", schema_sql.casefold())


if __name__ == "__main__":
    unittest.main()
