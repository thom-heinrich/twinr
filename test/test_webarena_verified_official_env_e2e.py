# ruff: noqa: E402
"""Live official-env E2E proofs for WebArena Verified benchmark integration.

These tests stay below the LLM layer on purpose. They verify that Twinr's
official benchmark bridge can materialize authenticated browser state for the
cached WebArena Verified environments and that the dense-reader evidence layer
captures the long-page/table evidence the benchmark failures exposed.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from typing import ClassVar

from playwright.sync_api import sync_playwright

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from browser_automation.dense_reader import DensePageReader, _content_block_packets_from_evidence
from test.browser_benchmarks.webarena_verified_adapter import (
    build_webarena_task_run,
    build_webarena_verified_config,
    evaluate_browser_result,
    load_webarena_verified_task,
    run_twinr_task,
)
from test.browser_benchmarks.webarena_verified_auth_bootstrap import ensure_task_auth_context
from twinr.agent.base_agent.config import TwinrConfig


def _url_is_reachable(url: str) -> bool:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:
            return int(getattr(response, "status", 200)) < 500
    except urllib.error.HTTPError as exc:
        return int(getattr(exc, "code", 500) or 500) < 500
    except (urllib.error.URLError, TimeoutError):
        return False


class WebArenaVerifiedOfficialEnvE2ETests(unittest.TestCase):
    _config: ClassVar[object]
    _twinr_config: ClassVar[TwinrConfig]
    _reader: ClassVar[DensePageReader]
    _auth_root: ClassVar[Path]

    @classmethod
    def setUpClass(cls) -> None:
        cls._config = build_webarena_verified_config(
            shopping_url="http://localhost:7770",
            shopping_admin_url="http://localhost:7780",
            reddit_url="http://localhost:9999",
            gitlab_url="http://localhost:8023",
            shopping_admin_use_header_login=True,
        )
        required_urls = [
            "http://localhost:7770",
            "http://localhost:7780",
            "http://localhost:9999",
            "http://localhost:8023",
        ]
        if not all(_url_is_reachable(url) for url in required_urls):
            raise unittest.SkipTest("Official WebArena Verified envs are not reachable on localhost.")
        cls._twinr_config = TwinrConfig.from_env(_REPO_ROOT / ".env")
        cls._reader = DensePageReader(
            config=cls._twinr_config,
            workspace_root=_REPO_ROOT / "browser_automation",
            allowed_domains=("localhost", "127.0.0.1"),
            model="gpt-5.4-mini",
            max_runtime_s=60.0,
        )
        cls._auth_root = Path("/tmp/twinr_webarena_verified_auth")

    def test_live_auth_context_materializes_for_multisite_cached_envs(self) -> None:
        cases = [
            ("shopping", 21),
            ("shopping_admin", 11),
            ("reddit", 28),
            ("gitlab", 170),
        ]
        for site_name, task_id in cases:
            with self.subTest(site=site_name, task_id=task_id):
                benchmark, task = load_webarena_verified_task(task_id=task_id, config=self._config)
                context = ensure_task_auth_context(
                    task=task,
                    config=self._config,
                    benchmark=benchmark,
                    auth_state_root=self._auth_root,
                )
                self.assertTrue(context.require_login)
                if site_name == "shopping_admin":
                    self.assertTrue(context.extra_http_headers or context.storage_state_path)
                else:
                    self.assertTrue(context.storage_state_path)
                    self.assertTrue(Path(str(context.storage_state_path)).is_file())

    def test_shopping_reviews_e2e_exposes_all_small_ear_cup_reviewers(self) -> None:
        benchmark, task = load_webarena_verified_task(task_id=21, config=self._config)
        auth_context = ensure_task_auth_context(
            task=task,
            config=self._config,
            benchmark=benchmark,
            auth_state_root=self._auth_root,
        )
        reviewers: set[str] = set()
        with sync_playwright() as manager:
            browser = manager.chromium.launch(headless=True)
            context = browser.new_context(storage_state=auth_context.storage_state_path)
            page = context.new_page()
            page.goto(
                "http://localhost:7770/6s-wireless-headphones-over-ear-noise-canceling-hi-fi-bass-foldable-stereo-wireless-kid-headsets-earbuds-with-built-in-mic-micro-sd-tf-fm-for-iphone-samsung-ipad-pc-black-gold.html",
                wait_until="domcontentloaded",
                timeout=60_000,
            )
            self._reader._settle_page(page=page, timeout_ms=60_000)
            packets = [
                self._reader._snapshot_page(
                    page=page,
                    task_token="test-task21",
                    capture_screenshot=False,
                    capture_html=False,
                    artifacts=[],
                )
            ]
            packets.extend(self._reader._activate_section_queries(page=page, url=page.url, queries=("Reviews", "Review")))
            packets.extend(
                self._reader._scroll_probe_packets(
                    page=page,
                    url=page.url,
                    anchor_queries=("Reviews", "Customer Reviews", "ear cups", "small"),
                )
            )
            flattened = _content_block_packets_from_evidence(
                evidence_packets=packets,
                priority_terms=("ear cups", "small", "reviews"),
                limit=20,
            )
            browser.close()

        for packet in flattened:
            text = json.dumps(packet, ensure_ascii=False)
            for reviewer in ("Catso", "Dibbins", "Anglebert Dinkherhump", "Michelle Davis"):
                if reviewer in text:
                    reviewers.add(reviewer)
        self.assertEqual(
            reviewers,
            {"Catso", "Dibbins", "Anglebert Dinkherhump", "Michelle Davis"},
        )

    def test_shopping_order_history_e2e_exposes_structured_order_rows(self) -> None:
        benchmark, task = load_webarena_verified_task(task_id=96, config=self._config)
        auth_context = ensure_task_auth_context(
            task=task,
            config=self._config,
            benchmark=benchmark,
            auth_state_root=self._auth_root,
        )
        flattened: list[dict[str, object]]
        with sync_playwright() as manager:
            browser = manager.chromium.launch(headless=True)
            context = browser.new_context(storage_state=auth_context.storage_state_path)
            page = context.new_page()
            page.goto("http://localhost:7770/sales/order/history/", wait_until="domcontentloaded", timeout=60_000)
            self._reader._settle_page(page=page, timeout_ms=60_000)
            packets = [
                self._reader._snapshot_page(
                    page=page,
                    task_token="test-order-history",
                    capture_screenshot=False,
                    capture_html=False,
                    artifacts=[],
                )
            ]
            packets.extend(
                self._reader._scroll_probe_packets(
                    page=page,
                    url=page.url,
                    anchor_queries=("order", "status", "date"),
                )
            )
            flattened = _content_block_packets_from_evidence(
                evidence_packets=packets,
                priority_terms=("order", "status", "date"),
                limit=20,
            )
            browser.close()

        flattened_text = "\n".join(json.dumps(packet, ensure_ascii=False) for packet in flattened)
        self.assertIn("000000170", flattened_text)
        self.assertIn("Canceled", flattened_text)
        self.assertIn("000000189", flattened_text)
        self.assertIn("Pending", flattened_text)

    def test_stale_shopping_storage_state_is_refreshed_before_order_history_read(self) -> None:
        benchmark, task = load_webarena_verified_task(task_id=96, config=self._config)
        with tempfile.TemporaryDirectory() as temp_dir:
            auth_root = Path(temp_dir)
            stale_path = auth_root / ".auth" / "shopping_state.json"
            stale_path.parent.mkdir(parents=True, exist_ok=True)
            stale_path.write_text('{"cookies":[],"origins":[]}', encoding="utf-8")

            auth_context = ensure_task_auth_context(
                task=task,
                config=self._config,
                benchmark=benchmark,
                auth_state_root=auth_root,
            )

            with sync_playwright() as manager:
                browser = manager.chromium.launch(headless=True)
                context = browser.new_context(storage_state=auth_context.storage_state_path)
                page = context.new_page()
                page.goto("http://localhost:7770/sales/order/history/", wait_until="domcontentloaded", timeout=60_000)
                self._reader._settle_page(page=page, timeout_ms=60_000)
                page_text = page.locator("body").inner_text(timeout=5_000)
                final_url = page.url
                browser.close()

        self.assertNotIn("/customer/account/login", final_url)
        self.assertIn("My Orders", page_text)
        self.assertIn("000000170", page_text)

    def test_shopping_review_reader_e2e_follows_additional_review_pages(self) -> None:
        benchmark, task = load_webarena_verified_task(task_id=388, config=self._config)
        auth_context = ensure_task_auth_context(
            task=task,
            config=self._config,
            benchmark=benchmark,
            auth_state_root=self._auth_root,
        )
        task_run = build_webarena_task_run(task=task, config=self._config)
        task_run = task_run.__class__(
            task_id=task_run.task_id,
            site_name=task_run.site_name,
            intent=task_run.intent,
            start_url=task_run.start_url,
            goal=task_run.goal,
            results_schema=task_run.results_schema,
            browser_context_storage_state_path=auth_context.storage_state_path,
            browser_context_extra_http_headers=dict(auth_context.extra_http_headers or {}),
        )

        browser_result = run_twinr_task(
            env_file=_REPO_ROOT / ".env",
            task_run=task_run,
            project_root=_REPO_ROOT,
            max_steps=12,
            max_runtime_s=120.0,
        )

        self.assertTrue(browser_result.ok, browser_result.summary)
        answer_markdown = str(browser_result.data.get("answer_markdown") or "")
        self.assertIn("Evelyn Kurver", answer_markdown)
        self.assertIn("N Randall", answer_markdown)

    def test_shopping_under_delivery_absence_returns_official_not_found(self) -> None:
        benchmark, task = load_webarena_verified_task(task_id=235, config=self._config)
        auth_context = ensure_task_auth_context(
            task=task,
            config=self._config,
            benchmark=benchmark,
            auth_state_root=self._auth_root,
        )
        task_run = build_webarena_task_run(task=task, config=self._config)
        task_run = task_run.__class__(
            task_id=task_run.task_id,
            site_name=task_run.site_name,
            intent=task_run.intent,
            start_url=task_run.start_url,
            goal=task_run.goal,
            results_schema=task_run.results_schema,
            browser_context_storage_state_path=auth_context.storage_state_path,
            browser_context_extra_http_headers=dict(auth_context.extra_http_headers or {}),
        )

        browser_result = run_twinr_task(
            env_file=_REPO_ROOT / ".env",
            task_run=task_run,
            project_root=_REPO_ROOT,
            max_steps=12,
            max_runtime_s=120.0,
        )
        evaluation = evaluate_browser_result(
            benchmark=benchmark,
            task=task,
            browser_result=browser_result,
        )

        self.assertTrue(browser_result.ok or str(browser_result.error_code or "") == "not_found")
        self.assertEqual(
            str(evaluation["agent_response"]["status"] or ""),
            "NOT_FOUND_ERROR",
        )
        self.assertTrue(bool(evaluation["pass"]))


if __name__ == "__main__":
    unittest.main()
