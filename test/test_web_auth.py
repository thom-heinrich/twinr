from pathlib import Path
import tempfile
import unittest

from twinr.web.support.auth import (
    FileBackedWebAuthStore,
    build_web_auth_session_cookie,
    default_web_auth_username,
    load_authenticated_web_session,
    verify_web_auth_password,
)


class WebAuthStoreTests(unittest.TestCase):
    def make_store(self) -> FileBackedWebAuthStore:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return FileBackedWebAuthStore.from_project_root(Path(temp_dir.name))

    def test_load_or_bootstrap_creates_admin_admin_state(self) -> None:
        store = self.make_store()

        state = store.load_or_bootstrap()

        self.assertEqual(state.username, default_web_auth_username())
        self.assertTrue(state.must_change_password)
        self.assertTrue(store.path.exists())
        self.assertTrue(verify_web_auth_password(state, username="admin", password="admin"))

    def test_update_password_replaces_bootstrap_credential_and_invalidates_old_cookie(self) -> None:
        store = self.make_store()
        state = store.load_or_bootstrap()
        bootstrap_cookie = build_web_auth_session_cookie(state, username=state.username, issued_at=1_700_000_000)

        updated = store.update_password(
            current_password="admin",
            new_password="better-pass-123",
            confirm_password="better-pass-123",
        )

        self.assertFalse(updated.must_change_password)
        self.assertTrue(verify_web_auth_password(updated, username="admin", password="better-pass-123"))
        self.assertFalse(verify_web_auth_password(updated, username="admin", password="admin"))
        self.assertIsNone(load_authenticated_web_session(updated, bootstrap_cookie, now=1_700_000_100))

    def test_update_password_rejects_too_short_passwords(self) -> None:
        store = self.make_store()
        store.load_or_bootstrap()

        with self.assertRaisesRegex(ValueError, "at least"):
            store.update_password(
                current_password="admin",
                new_password="short",
                confirm_password="short",
            )


if __name__ == "__main__":
    unittest.main()
