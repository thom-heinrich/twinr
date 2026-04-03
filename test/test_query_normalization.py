from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.query_normalization import LongTermQueryRewriter
from twinr.providers.openai.core import client as openai_client_module


class _BackgroundTestRewriter(LongTermQueryRewriter):
    gate: threading.Event
    calls: list[str]

    def _canonicalize_query(self, query_text: str) -> str | None:
        self.calls.append(query_text)
        self.gate.wait(timeout=1.0)
        return "Ask whether everything is okay."


class LongTermQueryRewriterTests(unittest.TestCase):
    def setUp(self) -> None:
        openai_client_module.close_cached_openai_clients()

    def tearDown(self) -> None:
        openai_client_module.close_cached_openai_clients()

    def test_profile_returns_fallback_immediately_and_updates_cache_after_background_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rewriter = _BackgroundTestRewriter(
                config=TwinrConfig(project_root=temp_dir, personality_dir="personality"),
                backend=object(),
            )
            rewriter.gate = threading.Event()
            rewriter.calls = []
            executor = rewriter._rewrite_executor
            self.assertIsNotNone(executor)
            try:
                started = time.perf_counter()
                first = rewriter.profile("na alles klar")
                elapsed_s = time.perf_counter() - started

                self.assertLess(elapsed_s, 0.2)
                self.assertEqual(first.original_text, "na alles klar")
                self.assertIsNone(first.canonical_english_text)
                self.assertEqual(first.retrieval_text, "na alles klar")

                rewriter.gate.set()
                resolved = first
                for _ in range(100):
                    time.sleep(0.01)
                    resolved = rewriter.profile("na alles klar")
                    if resolved.canonical_english_text is not None:
                        break

                self.assertEqual(resolved.canonical_english_text, "Ask whether everything is okay.")
                self.assertEqual(resolved.retrieval_text, "Ask whether everything is okay.")
                self.assertEqual(rewriter.calls, ["na alles klar"])
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

    def test_profile_can_wait_briefly_for_background_rewrite_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rewriter = _BackgroundTestRewriter(
                config=TwinrConfig(project_root=temp_dir, personality_dir="personality"),
                backend=object(),
            )
            rewriter.gate = threading.Event()
            rewriter.calls = []
            executor = rewriter._rewrite_executor
            self.assertIsNotNone(executor)
            timer = threading.Timer(0.05, rewriter.gate.set)
            timer.start()
            try:
                started = time.perf_counter()
                profile = rewriter.profile("na alles klar", wait_for_rewrite_s=0.5)
                elapsed_s = time.perf_counter() - started

                self.assertGreaterEqual(elapsed_s, 0.04)
                self.assertLess(elapsed_s, 0.5)
                self.assertEqual(profile.canonical_english_text, "Ask whether everything is okay.")
                self.assertEqual(profile.retrieval_text, "Ask whether everything is okay.")
                self.assertEqual(rewriter.calls, ["na alles klar"])
            finally:
                timer.cancel()
                executor.shutdown(wait=True, cancel_futures=True)

    def test_profile_waits_for_inflight_rewrite_after_fallback_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rewriter = _BackgroundTestRewriter(
                config=TwinrConfig(project_root=temp_dir, personality_dir="personality"),
                backend=object(),
            )
            rewriter.gate = threading.Event()
            rewriter.calls = []
            executor = rewriter._rewrite_executor
            self.assertIsNotNone(executor)
            timer = threading.Timer(0.05, rewriter.gate.set)
            try:
                first = rewriter.profile("na alles klar")
                self.assertIsNone(first.canonical_english_text)

                timer.start()
                started = time.perf_counter()
                profile = rewriter.profile("na alles klar", wait_for_rewrite_s=0.5)
                elapsed_s = time.perf_counter() - started

                self.assertGreaterEqual(elapsed_s, 0.04)
                self.assertLess(elapsed_s, 0.5)
                self.assertEqual(profile.canonical_english_text, "Ask whether everything is okay.")
                self.assertEqual(profile.retrieval_text, "Ask whether everything is okay.")
                self.assertEqual(rewriter.calls, ["na alles klar"])
            finally:
                timer.cancel()
                executor.shutdown(wait=True, cancel_futures=True)

    def test_shutdown_wait_true_closes_executor_threads_and_backend_client(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            close_calls: list[str] = []

            class _Client:
                def close(self) -> None:
                    close_calls.append("closed")

            backend = type("BackendStub", (), {"_client": _Client()})()
            rewriter = _BackgroundTestRewriter(
                config=TwinrConfig(project_root=temp_dir, personality_dir="personality"),
                backend=backend,
            )
            rewriter.gate = threading.Event()
            rewriter.calls = []

            rewriter.profile("na alles klar")
            executor = rewriter._rewrite_executor
            self.assertIsNotNone(executor)
            for _ in range(100):
                if executor._threads:  # type: ignore[attr-defined]
                    break
                time.sleep(0.01)
            worker_threads = tuple(executor._threads)  # type: ignore[attr-defined]
            self.assertTrue(worker_threads)

            rewriter.shutdown(wait=True)

            self.assertIsNone(rewriter._rewrite_executor)
            self.assertEqual(rewriter._pending_rewrites, {})
            self.assertIsNone(rewriter.backend)
            self.assertEqual(close_calls, ["closed"])
            self.assertTrue(all(not thread.is_alive() for thread in worker_threads))

    def test_shutdown_wait_true_does_not_close_borrowed_cached_client(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            class _SharedHttpClient:
                def __init__(self) -> None:
                    self.closed = False

                def close(self) -> None:
                    self.closed = True

            class _OpenAIClientStub:
                def __init__(self, *, http_client: _SharedHttpClient | None = None) -> None:
                    self._client = http_client if http_client is not None else _SharedHttpClient()
                    self.close_calls = 0

                def close(self) -> None:
                    self.close_calls += 1
                    self._client.close()

            shared_client = _OpenAIClientStub()
            borrowed_client = _OpenAIClientStub(http_client=shared_client._client)
            setattr(borrowed_client, "_twinr_borrowed_http_client", True)

            backend = type("BackendStub", (), {"_client": borrowed_client})()
            rewriter = _BackgroundTestRewriter(
                config=TwinrConfig(project_root=temp_dir, personality_dir="personality"),
                backend=backend,
            )
            rewriter.gate = threading.Event()
            rewriter.calls = []

            rewriter.shutdown(wait=True)

            self.assertFalse(shared_client._client.closed)
            self.assertEqual(shared_client.close_calls, 0)

    def test_profile_after_nonblocking_shutdown_skips_late_rewrite_submit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rewriter = LongTermQueryRewriter(
                config=TwinrConfig(project_root=temp_dir, personality_dir="personality"),
                backend=object(),
            )

            rewriter.shutdown(wait=False)
            profile = rewriter.profile("Wie geht es Janina?")
            rewriter.shutdown(wait=True)

            self.assertEqual(profile.retrieval_text, "Wie geht es Janina?")
            self.assertFalse(rewriter._pending_rewrites)


if __name__ == "__main__":
    unittest.main()
