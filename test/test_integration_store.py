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
        self.assertEqual(path.as_posix(), "/tmp/twinr/artifacts/stores/integrations/integrations.json")


if __name__ == "__main__":
    unittest.main()
