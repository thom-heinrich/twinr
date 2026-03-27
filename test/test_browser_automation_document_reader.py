# ruff: noqa: E402
"""Regression coverage for PDF/document browser automation helpers."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, cast
import httpx
import sys
import tempfile
import threading
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.browser_automation import BrowserAutomationRequest
from twinr.browser_automation.runtime import PdfDocumentReader, call_openai_with_retry

from browser_automation.dense_reader import DensePageReader, DensePageReaderResult
from browser_automation.tessairact_bridge import TessairactVendoredDriver
from openai import APIConnectionError


class _QuietSimpleHandler(SimpleHTTPRequestHandler):
    """Serve local test fixtures without noisy access logs."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        del format, args


@contextmanager
def _serve_directory(*, root: Path) -> Iterator[str]:
    handler = partial(_QuietSimpleHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, name="document-test-server", daemon=True)
    thread.start()
    try:
        host = str(server.server_address[0])
        port = int(server.server_address[1])
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


def _write_minimal_pdf(*, path: Path, text: str) -> None:
    """Write one tiny text PDF that ``pdftotext`` can read in tests."""

    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    stream = f"BT\n/F1 12 Tf\n72 100 Td\n({text}) Tj\nET\n"
    objects.append(
        f"5 0 obj\n<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}endstream\nendobj\n"
    )
    parts = ["%PDF-1.4\n"]
    offsets = [0]
    current_offset = len(parts[0].encode("latin-1"))
    for object_text in objects:
        offsets.append(current_offset)
        parts.append(object_text)
        current_offset += len(object_text.encode("latin-1"))
    xref_start = current_offset
    xref = ["xref\n0 6\n", "0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n")
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"
    path.write_bytes(("".join(parts) + "".join(xref) + trailer).encode("latin-1"))


class PdfDocumentReaderTests(unittest.TestCase):
    def test_reader_downloads_pdf_and_extracts_query_packets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "schedule.pdf"
            _write_minimal_pdf(path=pdf_path, text="ZOB Bussteig 5 06:34")
            with _serve_directory(root=root) as base_url:
                reader = PdfDocumentReader(
                    artifacts_root=root / "artifacts",
                    allowed_domains=("127.0.0.1",),
                )
                result = reader.read_pdf(
                    url=f"{base_url}/schedule.pdf",
                    task_token="schedule",
                    query_terms=("ZOB Bussteig 5", "06:34"),
                )

        self.assertEqual(result.page_count, 1)
        self.assertIn("ZOB Bussteig 5 06:34", result.text)
        packets = result.to_evidence_packets(query_terms=("ZOB Bussteig 5", "06:34"), source_label="City-Bus")
        self.assertEqual(packets[0]["kind"], "document_snapshot")
        self.assertTrue(any(packet["kind"] == "document_match" for packet in packets))
        self.assertTrue(any(artifact.kind == "document" for artifact in result.artifacts))
        self.assertTrue(any(artifact.kind == "document_text" for artifact in result.artifacts))


class ProviderRetryTests(unittest.TestCase):
    def test_call_openai_with_retry_recovers_after_transient_connection_error(self) -> None:
        attempts = {"count": 0}

        def _call() -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
            return "ok"

        result = call_openai_with_retry(label="document_reader_retry", func=_call, max_attempts=3, base_delay_s=0.0)

        self.assertEqual(result, "ok")
        self.assertEqual(attempts["count"], 2)


class DenseReaderDocumentTests(unittest.TestCase):
    def test_dense_reader_uses_visible_pdf_links_as_document_evidence(self) -> None:
        class _FakeDenseReader(DensePageReader):
            def _plan_reader(self, *, goal, snapshot, navigator_data):  # type: ignore[override]
                del goal, snapshot, navigator_data
                return {
                    "query_terms": ["ZOB Bussteig 5", "06:34"],
                    "heading_queries": ["City-Bus"],
                }

            def _plan_document(self, *, goal, navigator_data):  # type: ignore[override]
                del goal, navigator_data
                return {
                    "query_terms": ["ZOB Bussteig 5", "06:34"],
                    "heading_queries": ["City-Bus"],
                    "reason": "Read visible PDF links directly.",
                }

            def _verify_freeform(self, *, goal, navigator_data, evidence_packets, expects_null_result):  # type: ignore[override]
                del goal, navigator_data, expects_null_result
                for packet in evidence_packets:
                    if str(packet.get("kind") or "") == "document_match" and "06:34" in str(packet.get("text") or ""):
                        return {
                            "supported": True,
                            "not_found": False,
                            "answer_markdown": "06:34",
                            "key_points": ["The PDF match packet contains ZOB Bussteig 5 06:34."],
                            "reason": "Visible PDF link was downloaded and read successfully.",
                            "observational_page_check": False,
                            "answer_scoped_to_current_page": False,
                        }
                return {
                    "supported": False,
                    "not_found": True,
                    "answer_markdown": "",
                    "key_points": [],
                    "reason": "No document evidence packet contained the requested time.",
                    "observational_page_check": False,
                    "answer_scoped_to_current_page": False,
                }

            def _classify_negative_observational_support(self, *, goal, evidence_packets, verification):  # type: ignore[override]
                del goal, evidence_packets, verification
                return {
                    "observational_page_check": False,
                    "supported_negative_observation": False,
                    "reason": "",
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            pdf_path = root / "schedule.pdf"
            html_path = root / "index.html"
            _write_minimal_pdf(path=pdf_path, text="ZOB Bussteig 5 06:34")
            html_path.write_text(
                """
                <html>
                  <body>
                    <main>
                      <article>
                        <h1>Busfahrplaene</h1>
                        <p><a href="/schedule.pdf">City-Busfahrplan seit 01.12.2025</a></p>
                      </article>
                    </main>
                  </body>
                </html>
                """,
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            with _serve_directory(root=root) as base_url:
                reader = _FakeDenseReader(
                    config=config,
                    workspace_root=root / "browser_automation",
                    allowed_domains=("127.0.0.1",),
                    model="gpt-5.4-mini",
                    max_runtime_s=30.0,
                )
                result = reader.read(
                    task_token="passau_local",
                    goal="Determine the first Monday-Friday departure time from ZOB Bussteig 5.",
                    url=f"{base_url}/index.html",
                    navigator_data={},
                    capture_screenshot=False,
                    capture_html=False,
                )

        self.assertTrue(result.supported)
        self.assertEqual(result.answer_markdown, "06:34")
        self.assertTrue(any(packet["kind"] == "document_match" for packet in result.evidence_packets))
        self.assertTrue(any(artifact.kind == "document" for artifact in result.artifacts))

    def test_dense_reader_prefers_document_fast_path_before_large_page_plan(self) -> None:
        class _FastPathReader(DensePageReader):
            def _plan_reader(self, *, goal, snapshot, navigator_data):  # type: ignore[override]
                del goal, snapshot, navigator_data
                raise AssertionError("large page planner should not run when visible PDF links are already present")

            def _plan_document(self, *, goal, navigator_data):  # type: ignore[override]
                del goal, navigator_data
                return {
                    "query_terms": ["ZOB Bussteig 5", "06:34"],
                    "heading_queries": ["Montag-Freitag"],
                    "reason": "Read visible PDF links first.",
                }

            def _verify_freeform(self, *, goal, navigator_data, evidence_packets, expects_null_result):  # type: ignore[override]
                del goal, navigator_data, expects_null_result
                for packet in evidence_packets:
                    if str(packet.get("kind") or "") == "document_match" and "06:34" in str(packet.get("text") or ""):
                        return {
                            "supported": True,
                            "not_found": False,
                            "answer_markdown": "06:34",
                            "key_points": ["Document fast path found ZOB Bussteig 5 06:34."],
                            "reason": "Visible PDF link was read before the large page planner ran.",
                            "observational_page_check": False,
                            "answer_scoped_to_current_page": False,
                        }
                return {
                    "supported": False,
                    "not_found": True,
                    "answer_markdown": "",
                    "key_points": [],
                    "reason": "No document evidence packet contained the requested time.",
                    "observational_page_check": False,
                    "answer_scoped_to_current_page": False,
                }

            def _classify_negative_observational_support(self, *, goal, evidence_packets, verification):  # type: ignore[override]
                del goal, evidence_packets, verification
                return {
                    "observational_page_check": False,
                    "supported_negative_observation": False,
                    "reason": "",
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            pdf_path = root / "schedule.pdf"
            html_path = root / "index.html"
            _write_minimal_pdf(path=pdf_path, text="ZOB Bussteig 5 06:34")
            html_path.write_text(
                """
                <html>
                  <body>
                    <main>
                      <section>
                        <table>
                          <tr>
                            <td>ZOB Bussteig 5 City-Busfahrplan seit 01.12.2025</td>
                            <td><a href="/schedule.pdf">Download</a></td>
                          </tr>
                        </table>
                      </section>
                    </main>
                  </body>
                </html>
                """,
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            with _serve_directory(root=root) as base_url:
                reader = _FastPathReader(
                    config=config,
                    workspace_root=root / "browser_automation",
                    allowed_domains=("127.0.0.1",),
                    model="gpt-5.4-mini",
                    max_runtime_s=30.0,
                )
                result = reader.read(
                    task_token="passau_fast_path",
                    goal="Determine the first Monday-Friday departure time from ZOB Bussteig 5.",
                    url=f"{base_url}/index.html",
                    navigator_data={},
                    capture_screenshot=False,
                    capture_html=False,
                )

        self.assertTrue(result.supported)
        self.assertEqual(result.answer_markdown, "06:34")
        self.assertTrue(any(packet["kind"] == "document_match" for packet in result.evidence_packets))

    def test_dense_reader_scans_document_links_beyond_leading_navigation_noise(self) -> None:
        class _NavigationHeavyReader(DensePageReader):
            def _plan_reader(self, *, goal, snapshot, navigator_data):  # type: ignore[override]
                del goal, snapshot, navigator_data
                raise AssertionError("large page planner should not run when the page contains a matching PDF link")

            def _plan_document(self, *, goal, navigator_data):  # type: ignore[override]
                del goal, navigator_data
                return {
                    "query_terms": ["ZOB Bussteig 5", "06:34", "Citybus Parkhaus Bahnhofstr."],
                    "heading_queries": ["Montag-Freitag", "Citybus Parkhaus Bahnhofstr. - ZOB - Römerplatz - Ilzbrücke"],
                    "reason": "Prefer the matching PDF link even when many unrelated navigation links come first.",
                }

            def _verify_freeform(self, *, goal, navigator_data, evidence_packets, expects_null_result):  # type: ignore[override]
                del goal, navigator_data, expects_null_result
                for packet in evidence_packets:
                    if str(packet.get("kind") or "") != "document_match":
                        continue
                    text = str(packet.get("text") or "")
                    if "ZOB Bussteig 5" in text and "06:34" in text:
                        return {
                            "supported": True,
                            "not_found": False,
                            "answer_markdown": "06:34",
                            "key_points": ["The matching PDF link behind the navigation noise contains ZOB Bussteig 5 06:34."],
                            "reason": "The document fast path scanned past the leading navigation links and read the timetable PDF.",
                            "observational_page_check": False,
                            "answer_scoped_to_current_page": False,
                        }
                return {
                    "supported": False,
                    "not_found": True,
                    "answer_markdown": "",
                    "key_points": [],
                    "reason": "No document evidence packet contained the requested time.",
                    "observational_page_check": False,
                    "answer_scoped_to_current_page": False,
                }

            def _classify_negative_observational_support(self, *, goal, evidence_packets, verification):  # type: ignore[override]
                del goal, evidence_packets, verification
                return {
                    "observational_page_check": False,
                    "supported_negative_observation": False,
                    "reason": "",
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            pdf_path = root / "schedule.pdf"
            html_path = root / "index.html"
            _write_minimal_pdf(path=pdf_path, text="ZOB Bussteig 5 06:34")
            navigation = "\n".join(
                f'<a href="/nav-{index}.html">Navigation {index}</a>'
                for index in range(1, 80)
            )
            html_path.write_text(
                f"""
                <html>
                  <body>
                    <nav>{navigation}</nav>
                    <main>
                      <section>
                        <table>
                          <tr>
                            <td>ZOB Bussteig 5</td>
                            <td>Citybus Parkhaus Bahnhofstr. - ZOB - Römerplatz - Ilzbrücke</td>
                            <td><a href="/schedule.pdf">Download</a></td>
                          </tr>
                        </table>
                      </section>
                    </main>
                  </body>
                </html>
                """,
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            with _serve_directory(root=root) as base_url:
                reader = _NavigationHeavyReader(
                    config=config,
                    workspace_root=root / "browser_automation",
                    allowed_domains=("127.0.0.1",),
                    model="gpt-5.4-mini",
                    max_runtime_s=30.0,
                )
                result = reader.read(
                    task_token="passau_nav_noise",
                    goal=(
                        "Open the PDF timetable for Citybus Parkhaus Bahnhofstr. - ZOB - Römerplatz - "
                        "Ilzbrücke and return the first Monday-Friday departure from ZOB Bussteig 5."
                    ),
                    url=f"{base_url}/index.html",
                    navigator_data={},
                    capture_screenshot=False,
                    capture_html=False,
                )

        self.assertTrue(result.supported)
        self.assertEqual(result.answer_markdown, "06:34")
        self.assertTrue(any(packet["kind"] == "document_match" for packet in result.evidence_packets))


class DirectPdfDriverTests(unittest.TestCase):
    def test_driver_reads_direct_pdf_urls_without_vendored_browser_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")

            class _FakeReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **_: object) -> DensePageReaderResult:
                    return DensePageReaderResult(
                        supported=True,
                        answer_markdown="06:34",
                        key_points=("Document reader extracted 06:34 from the PDF.",),
                        reason="Direct PDF reading succeeded.",
                        evidence_packets=(
                            {
                                "kind": "document_match",
                                "url": "https://example.org/schedule.pdf",
                                "query": "ZOB Bussteig 5",
                                "text": "ZOB Bussteig 5 06:34",
                            },
                        ),
                        artifacts=(),
                        typed_result={
                            "status": "success",
                            "results": None,
                            "answer_text": "06:34",
                            "reason": "Direct PDF reading succeeded.",
                            "key_points": ["Document reader extracted 06:34 from the PDF."],
                            "used_capabilities": ["pdf_document"],
                        },
                    )

            class _ExplodingLoader:
                def load(self) -> object:
                    raise AssertionError("Vendored specialist should not be used for direct PDF URLs")

            driver._dense_reader_factory = cast(Any, _FakeReader)
            driver._specialist_loader = cast(Any, _ExplodingLoader())

            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="direct_pdf",
                    goal="Read the timetable PDF and return the first departure time.",
                    start_url="https://example.org/schedule.pdf",
                    allowed_domains=("example.org",),
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_url, "https://example.org/schedule.pdf")
        self.assertEqual(result.data["mode"], "tessairact_direct_document_reader")

    def test_driver_reads_page_level_pdf_links_before_vendored_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")
            captured: dict[str, object] = {}

            class _FakeReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **kwargs: object) -> DensePageReaderResult:
                    captured.update(kwargs)
                    return DensePageReaderResult(
                        supported=True,
                        answer_markdown="06:34",
                        key_points=("Page-level PDF fast path extracted 06:34.",),
                        reason="Visible PDF link on the page was enough to answer without the vendored loop.",
                        evidence_packets=(
                            {
                                "kind": "document_match",
                                "url": "https://example.org/schedule.pdf",
                                "query": "ZOB Bussteig 5",
                                "text": "ZOB Bussteig 5 06:34",
                            },
                        ),
                        artifacts=(),
                    )

            class _ExplodingLoader:
                def load(self) -> object:
                    raise AssertionError("Vendored specialist should not run when page-level document fast path succeeds")

            driver._dense_reader_factory = cast(Any, _FakeReader)
            driver._specialist_loader = cast(Any, _ExplodingLoader())

            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="page_document",
                    goal="Read the page-level City-Bus PDF and return the first departure time.",
                    start_url="https://example.org/busfahrplan.html",
                    allowed_domains=("example.org",),
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_url, "https://example.org/busfahrplan.html")
        self.assertEqual(result.data["mode"], "tessairact_page_document_reader")
        self.assertTrue(captured["document_fast_path_only"])


class ExceptionRescueDriverTests(unittest.TestCase):
    def test_driver_recovers_vendored_exception_with_dense_reader_after_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            html_path = root / "index.html"
            html_path.write_text(
                """
                <html>
                  <body>
                    <main>
                      <article>
                        <h1>Busfahrplaene</h1>
                        <p><a href="/schedule.pdf">City-Busfahrplan seit 01.12.2025</a></p>
                      </article>
                    </main>
                  </body>
                </html>
                """,
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            captured: dict[str, object] = {}

            class _FakeSpecialist:
                BrowserLoopConfig = staticmethod(lambda **kwargs: dict(kwargs))

                @staticmethod
                def run_browser_loop(**kwargs: object) -> dict[str, object]:
                    runtime = kwargs["runtime"]
                    runtime._browser_session_act(  # type: ignore[attr-defined]
                        arguments={
                            "session_id": None,
                            "start_url": kwargs["start_url"],
                            "actions": [],
                            "take_snapshot": True,
                            "snapshot_include_elements": True,
                            "snapshot_include_a11y": False,
                            "snapshot_screenshot": False,
                            "snapshot_save_html": False,
                            "snapshot_include_content_blocks": True,
                            "snapshot_max_content_blocks": 16,
                            "timeout_s": 10.0,
                            "wait_until": "domcontentloaded",
                            "trace": False,
                            "user_agent": "",
                        }
                    )
                    raise RuntimeError("provider exploded after snapshot")

            class _FakeLoader:
                def load(self) -> object:
                    return _FakeSpecialist()

            class _FakeReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **kwargs: object) -> DensePageReaderResult:
                    if bool(kwargs.get("document_fast_path_only")):
                        return DensePageReaderResult(
                            supported=False,
                            answer_markdown="",
                            key_points=(),
                            reason="Preflight intentionally skipped so the exception-rescue path can be exercised.",
                            evidence_packets=(),
                            artifacts=(),
                        )
                    captured.update(kwargs)
                    return DensePageReaderResult(
                        supported=True,
                        answer_markdown="06:34",
                        key_points=("Dense reader recovered the answer from the latest snapshot URL.",),
                        reason="Exception rescue re-read the latest visible page state.",
                        evidence_packets=(
                            {
                                "kind": "document_match",
                                "url": str(kwargs.get("url") or ""),
                                "query": "ZOB Bussteig 5",
                                "text": "ZOB Bussteig 5 06:34",
                            },
                        ),
                        artifacts=(),
                    )

            with _serve_directory(root=root) as base_url:
                driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")
                driver._specialist_loader = cast(Any, _FakeLoader())
                driver._dense_reader_factory = cast(Any, _FakeReader)

                result = driver.execute(
                    BrowserAutomationRequest(
                        task_id="exception_rescue",
                        goal="Read the City-Bus timetable and return the first departure time.",
                        start_url=f"{base_url}/index.html",
                        allowed_domains=("127.0.0.1",),
                    )
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_url, captured["url"])
        self.assertEqual(result.data["mode"], "tessairact_exception_plus_dense_reader")
        self.assertEqual(captured["url"], result.final_url)
        self.assertTrue(result.data["verifier_decision"]["supported"])
        self.assertEqual(result.data["answer_markdown"], "06:34")


if __name__ == "__main__":
    unittest.main()
