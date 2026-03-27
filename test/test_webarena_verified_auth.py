# ruff: noqa: E402
"""Regression coverage for official WebArena Verified auth/state derivation."""

from pathlib import Path
from types import SimpleNamespace
from typing import Literal
import sys
import tempfile
import unittest
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from test.browser_benchmarks.webarena_verified_adapter import build_webarena_verified_config
import test.browser_benchmarks.webarena_verified_auth_bootstrap as auth_bootstrap
from test.browser_benchmarks.webarena_verified_auth import derive_task_auth_context
from test.browser_benchmarks.webarena_verified_auth_bootstrap import (
    default_auth_state_root,
    ensure_task_auth_context,
    official_reference_credentials_for_site,
)


class WebArenaVerifiedAuthTests(unittest.TestCase):
    def test_official_reference_credentials_cover_supported_sites(self) -> None:
        shopping = official_reference_credentials_for_site("shopping")
        gitlab = official_reference_credentials_for_site("gitlab")
        reddit = official_reference_credentials_for_site("reddit")
        self.assertIsNotNone(shopping)
        self.assertIsNotNone(gitlab)
        self.assertIsNotNone(reddit)
        assert shopping is not None
        assert gitlab is not None
        assert reddit is not None
        self.assertEqual(
            shopping.username,
            "emma.lopez@gmail.com",
        )
        self.assertEqual(
            gitlab.username,
            "byteblaze",
        )
        self.assertEqual(
            reddit.password,
            "test1234",
        )
        self.assertIsNone(official_reference_credentials_for_site("wikipedia"))

    def test_derive_task_auth_context_resolves_storage_state_from_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            auth_root = Path(temp_dir)
            state_path = auth_root / ".auth" / "shopping_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text('{"cookies":[],"origins":[]}', encoding="utf-8")
            config = build_webarena_verified_config(shopping_url="http://localhost:7770")
            task = SimpleNamespace(
                task_id=21,
                sites=(SimpleNamespace(value="shopping", url_name_template="__SHOPPING__"),),
            )

            context = derive_task_auth_context(
                task=task,
                config=config,
                auth_state_root=auth_root,
            )

        self.assertTrue(context.require_login)
        self.assertEqual(context.storage_state_path, str(state_path.resolve()))

    def test_derive_task_auth_context_builds_header_login_metadata(self) -> None:
        config = build_webarena_verified_config(
            shopping_admin_url="http://localhost:7780",
            shopping_admin_credentials={"username": "admin"},
            shopping_admin_use_header_login=True,
        )
        task = SimpleNamespace(
            task_id=0,
            sites=(SimpleNamespace(value="shopping_admin", url_name_template="__SHOPPING_ADMIN__"),),
        )
        benchmark = SimpleNamespace(get_custom_auth_header_name=lambda _site_name: "X-M2-Admin-Auto-Login")

        context = derive_task_auth_context(
            task=task,
            config=config,
            benchmark=benchmark,
            auth_state_root=None,
        )

        self.assertTrue(context.require_login)
        self.assertEqual(context.extra_http_headers, {"X-M2-Admin-Auto-Login": "admin"})

    def test_default_auth_state_root_is_created(self) -> None:
        root = default_auth_state_root()
        self.assertTrue(root.is_dir())

    def test_ensure_task_auth_context_materializes_missing_storage_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            auth_root = Path(temp_dir)
            expected_state_path = auth_root / ".auth" / "shopping_state.json"
            config = build_webarena_verified_config(shopping_url="http://localhost:7770")
            task = SimpleNamespace(
                task_id=21,
                sites=(SimpleNamespace(value="shopping", url_name_template="__SHOPPING__"),),
            )

            original_materialize = ensure_task_auth_context.__globals__["_materialize_site_auth_state"]

            def _fake_materialize(**kwargs: object) -> None:
                target_state_path = Path(str(kwargs["target_state_path"]))
                target_state_path.parent.mkdir(parents=True, exist_ok=True)
                target_state_path.write_text('{"cookies":[],"origins":[]}', encoding="utf-8")

            ensure_task_auth_context.__globals__["_materialize_site_auth_state"] = _fake_materialize
            try:
                context = ensure_task_auth_context(
                    task=task,
                    config=config,
                    benchmark=None,
                    auth_state_root=auth_root,
                )
            finally:
                ensure_task_auth_context.__globals__["_materialize_site_auth_state"] = original_materialize

        self.assertTrue(context.require_login)
        self.assertEqual(context.storage_state_path, str(expected_state_path.resolve()))

    def test_ensure_task_auth_context_keeps_fresh_storage_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            auth_root = Path(temp_dir)
            state_path = auth_root / ".auth" / "shopping_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text('{"cookies":[],"origins":[]}', encoding="utf-8")
            config = build_webarena_verified_config(shopping_url="http://localhost:7770")
            task = SimpleNamespace(
                task_id=21,
                sites=(SimpleNamespace(value="shopping", url_name_template="__SHOPPING__"),),
            )

            with (
                mock.patch.object(
                    auth_bootstrap,
                    "_storage_state_grants_authenticated_access",
                    return_value=True,
                ) as validator,
                mock.patch.object(auth_bootstrap, "_materialize_site_auth_state") as materialize,
            ):
                context = ensure_task_auth_context(
                    task=task,
                    config=config,
                    benchmark=None,
                    auth_state_root=auth_root,
                )

        self.assertTrue(context.require_login)
        self.assertEqual(context.storage_state_path, str(state_path.resolve()))
        validator.assert_called_once()
        materialize.assert_not_called()

    def test_ensure_task_auth_context_refreshes_stale_storage_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            auth_root = Path(temp_dir)
            state_path = auth_root / ".auth" / "shopping_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text("stale-state", encoding="utf-8")
            config = build_webarena_verified_config(shopping_url="http://localhost:7770")
            task = SimpleNamespace(
                task_id=21,
                sites=(SimpleNamespace(value="shopping", url_name_template="__SHOPPING__"),),
            )

            def _fake_materialize(**kwargs: object) -> None:
                target_state_path = Path(str(kwargs["target_state_path"]))
                target_state_path.parent.mkdir(parents=True, exist_ok=True)
                target_state_path.write_text('{"cookies":[{"name":"fresh"}],"origins":[]}', encoding="utf-8")

            with (
                mock.patch.object(
                    auth_bootstrap,
                    "_storage_state_grants_authenticated_access",
                    return_value=False,
                ) as validator,
                mock.patch.object(
                    auth_bootstrap,
                    "_materialize_site_auth_state",
                    side_effect=_fake_materialize,
                ) as materialize,
            ):
                context = ensure_task_auth_context(
                    task=task,
                    config=config,
                    benchmark=None,
                    auth_state_root=auth_root,
                )
                refreshed_state = state_path.read_text(encoding="utf-8")

        self.assertTrue(context.require_login)
        self.assertEqual(context.storage_state_path, str(state_path.resolve()))
        self.assertIn('"fresh"', refreshed_state)
        validator.assert_called_once()
        materialize.assert_called_once()

    def test_storage_state_grants_authenticated_access_revalidates_live_shopping_account_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "shopping_state.json"
            state_path.write_text('{"cookies":[],"origins":[]}', encoding="utf-8")
            config = build_webarena_verified_config(shopping_url="http://localhost:7770")
            urls: list[str] = []
            cache = getattr(auth_bootstrap, "_VALIDATED_STORAGE_STATE_CACHE", None)
            if isinstance(cache, dict):
                cache.clear()

            class _FakePage:
                url = "http://localhost:7770/customer/account/"

                def goto(self, url: str, **_: object) -> None:
                    urls.append(url)

                def content(self) -> str:
                    return "<body>My Account</body>"

            class _FakeContext:
                def new_page(self) -> _FakePage:
                    return _FakePage()

                def close(self) -> None:
                    return None

            class _FakeBrowser:
                def new_context(self, **_: object) -> _FakeContext:
                    return _FakeContext()

                def close(self) -> None:
                    return None

            class _FakeChromium:
                def launch(self, **_: object) -> _FakeBrowser:
                    return _FakeBrowser()

            class _FakePlaywrightManager:
                chromium = _FakeChromium()

                def __enter__(self) -> "_FakePlaywrightManager":
                    return self

                def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
                    return False

            with (
                mock.patch.object(auth_bootstrap, "sync_playwright", return_value=_FakePlaywrightManager()),
                mock.patch.object(auth_bootstrap, "_wait_for_network_idle", return_value=None),
                mock.patch.object(
                    auth_bootstrap,
                    "_page_matches_logged_in_target",
                    side_effect=[True, False],
                ) as matcher,
            ):
                first = auth_bootstrap._storage_state_grants_authenticated_access(
                    site_name="shopping",
                    config=config,
                    storage_state_path=str(state_path),
                )
                second = auth_bootstrap._storage_state_grants_authenticated_access(
                    site_name="shopping",
                    config=config,
                    storage_state_path=str(state_path),
                )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(
            urls,
            [
                "http://localhost:7770/customer/account/",
                "http://localhost:7770/customer/account/",
            ],
        )
        self.assertEqual(matcher.call_count, 2)


if __name__ == "__main__":
    unittest.main()
