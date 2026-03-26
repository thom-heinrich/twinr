#!/usr/bin/env python3
"""Run a local fixture-based browser benchmark against one Twinr runtime.

This benchmark serves local browser-automation fixtures over HTTP and exercises
the driver against repeatable SPA, hydration, virtualized list, infinite
scroll, popup, iframe, and transactional sandbox tasks. It is intended for
both local and Pi acceptance.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
import argparse
import json
import threading
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.browser_automation import BrowserAutomationRequest, load_browser_automation_driver


@dataclass(frozen=True, slots=True)
class FixtureBenchmarkCase:
    """Describe one fixture-backed browser task plus its acceptance markers."""

    id: str
    goal: str
    path: str
    expected_url_substring: str
    expected_answer_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FixtureServer:
    base_url: str
    server: ThreadingHTTPServer
    thread: threading.Thread


class _QuietSimpleHandler(SimpleHTTPRequestHandler):
    """Serve fixture files without noisy access logs during benchmark runs."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        del format, args


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    haystack = str(text or "").lower()
    return any(marker.lower() in haystack for marker in markers)


def _classify(*, ok: bool, url_ok: bool, answer_ok: bool) -> str:
    if ok and url_ok and answer_ok:
        return "pass"
    if ok and (not url_ok or not answer_ok):
        return "false_completion"
    if (not ok) and url_ok:
        return "fail_closed_after_navigation"
    return "fail"


@contextmanager
def _serve_fixture_directory(*, fixture_root: Path) -> Iterator[_FixtureServer]:
    handler = partial(_QuietSimpleHandler, directory=str(fixture_root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, name="browser-fixture-server", daemon=True)
    thread.start()
    try:
        host = str(server.server_address[0])
        port = int(server.server_address[1])
        yield _FixtureServer(base_url=f"http://{host}:{port}", server=server, thread=thread)
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr environment file.")
    parser.add_argument(
        "--environment-label",
        default="local_fixture",
        help="Label written into the result payload, for example local_fixture or pi_fixture.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit JSON output path. Defaults to artifacts/reports/browser_automation/browser_fixture_suite_<label>.json",
    )
    parser.add_argument("--max-steps", type=int, default=10, help="Per-case browser step budget.")
    parser.add_argument("--max-runtime-s", type=float, default=60.0, help="Per-case browser runtime budget.")
    parser.add_argument("--capture-html", action="store_true", help="Capture final HTML artifacts during the run.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    env_path = (repo_root / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)
    config = TwinrConfig.from_env(env_path)
    driver = load_browser_automation_driver(config=config, project_root=repo_root)
    fixture_root = repo_root / "test" / "fixtures" / "browser_automation"

    with _serve_fixture_directory(fixture_root=fixture_root) as fixture_server:
        base_url = fixture_server.base_url
        cases = (
            FixtureBenchmarkCase(
                id="fixture_same_tab_spa",
                goal="On the fixture docs page, open the Dark mode page and tell me which utility variant is used for manual dark mode styling.",
                path="/same_tab_spa_index.html",
                expected_url_substring="/same_tab_spa_dark_mode.html",
                expected_answer_markers=("dark:",),
            ),
            FixtureBenchmarkCase(
                id="fixture_hydration",
                goal="On the hydration dashboard, tell me the hydrated card title after the page finishes loading.",
                path="/hydration_dashboard.html",
                expected_url_substring="/hydration_dashboard.html",
                expected_answer_markers=("hydrated status card",),
            ),
            FixtureBenchmarkCase(
                id="fixture_virtualized_catalog",
                goal="On the virtualized catalog page, use the search box to find Helios 37 and tell me its status.",
                path="/virtualized_catalog.html",
                expected_url_substring="/virtualized_catalog.html",
                expected_answer_markers=("standby",),
            ),
            FixtureBenchmarkCase(
                id="fixture_infinite_feed",
                goal="On the infinite feed page, scroll until Orbital Archive appears and tell me its priority.",
                path="/infinite_feed.html",
                expected_url_substring="/infinite_feed.html",
                expected_answer_markers=("amber",),
            ),
            FixtureBenchmarkCase(
                id="fixture_popup",
                goal="On the popup launchpad, open the popup article and tell me the access phrase.",
                path="/popup_index.html",
                expected_url_substring="/popup_target.html",
                expected_answer_markers=("saffron sky",),
            ),
            FixtureBenchmarkCase(
                id="fixture_frame",
                goal="On the frame host page, read the embedded passphrase from the iframe and tell me the phrase.",
                path="/frame_index.html",
                expected_url_substring="/frame_inner.html",
                expected_answer_markers=("cedar compass",),
            ),
            FixtureBenchmarkCase(
                id="fixture_transaction",
                goal="On the transaction sandbox, fill customer name Marta, choose Express delivery, enable gift wrap, submit the order, and tell me the confirmation code.",
                path="/transaction_form.html",
                expected_url_substring="/transaction_result.html",
                expected_answer_markers=("tx-2048",),
            ),
        )

        results: list[dict[str, Any]] = []
        counts = {
            "pass": 0,
            "false_completion": 0,
            "fail_closed_after_navigation": 0,
            "fail": 0,
        }
        for case in cases:
            request = BrowserAutomationRequest(
                task_id=case.id,
                goal=case.goal,
                start_url=f"{base_url}{case.path}",
                allowed_domains=("127.0.0.1",),
                max_steps=int(args.max_steps),
                max_runtime_s=float(args.max_runtime_s),
                capture_screenshot=True,
                capture_html=bool(args.capture_html),
                metadata={"task_kind": "read", "eval_case_id": case.id, "eval_suite": "fixture_browser_suite"},
            )
            result = driver.execute(request)
            answer_markdown = str(result.data.get("answer_markdown") or "")
            final_url = str(result.final_url or "")
            url_ok = case.expected_url_substring in final_url
            answer_ok = _contains_any_marker(answer_markdown, case.expected_answer_markers)
            verdict = _classify(ok=bool(result.ok), url_ok=url_ok, answer_ok=answer_ok)
            counts[verdict] += 1
            results.append(
                {
                    "id": case.id,
                    "goal": case.goal,
                    "start_url": request.start_url,
                    "status": result.status,
                    "ok": result.ok,
                    "verdict": verdict,
                    "summary": result.summary,
                    "final_url": result.final_url,
                    "error_code": result.error_code,
                    "answer_markdown": answer_markdown,
                    "key_points": list(result.data.get("key_points") or []),
                    "visited_urls": list(result.data.get("visited_urls") or []),
                    "trace_path": result.data.get("trace_path"),
                    "step_count": result.data.get("step_count"),
                    "expected_url_substring": case.expected_url_substring,
                    "expected_answer_markers": list(case.expected_answer_markers),
                    "url_ok": url_ok,
                    "answer_ok": answer_ok,
                    "artifacts": [asdict(artifact) for artifact in result.artifacts],
                }
            )

    output_dir = repo_root / "artifacts" / "reports" / "browser_automation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output)
        if args.output
        else output_dir / f"browser_fixture_suite_{args.environment_label}.json"
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": str(args.environment_label),
        "driver_mode": "tessairact_vendored_browser_loop",
        "fixture_root": str(fixture_root),
        "counts": counts,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output_path), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
