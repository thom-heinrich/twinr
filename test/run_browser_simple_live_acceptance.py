#!/usr/bin/env python3
"""Run a simple live-web browser acceptance suite against one Twinr runtime.

This suite exercises the baseline browser capability class that should be
close to deterministic for Twinr: public pages, direct PDF reading,
page-to-PDF retrieval, a simple search/navigation task, and a simple public
form submission flow. It is intended for both local and Pi acceptance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.browser_automation import BrowserAutomationRequest, load_browser_automation_driver


@dataclass(frozen=True, slots=True)
class SimpleLiveCase:
    """Describe one simple live-web browser task plus its acceptance markers."""

    id: str
    goal: str
    start_url: str
    allowed_domains: tuple[str, ...]
    expected_answer_markers: tuple[str, ...]
    expected_url_substring: str = ""
    max_steps: int = 10


_CASES: tuple[SimpleLiveCase, ...] = (
    SimpleLiveCase(
        id="passau_page_to_pdf_time",
        goal=(
            "Öffne die offizielle Stadtwerke-Passau-Downloadcenter-Seite für "
            "ZOB Bussteig 5, öffne den PDF-Fahrplan für Citybus Parkhaus "
            "Bahnhofstr. - ZOB - Römerplatz - Ilzbrücke und nenne daraus die "
            "erste Abfahrt Montag-Freitag ab ZOB Bussteig 5."
        ),
        start_url="https://www.stadtwerke-passau.de/service/downloadcenter.html?linie=&haltestelle=ZOB+Bussteig+5",
        allowed_domains=("www.stadtwerke-passau.de",),
        expected_answer_markers=("06:34", "6:34"),
    ),
    SimpleLiveCase(
        id="passau_direct_pdf_time",
        goal=(
            "Lies dieses PDF für Citybus Parkhaus Bahnhofstr. - ZOB - Römerplatz "
            "- Ilzbrücke und nenne die erste Abfahrt Montag-Freitag ab ZOB "
            "Bussteig 5."
        ),
        start_url="https://www.stadtwerke-passau.de/files/dateien/halteplaene/Linie_citybus_H/ZOB%20Bussteig%205.pdf",
        allowed_domains=("www.stadtwerke-passau.de",),
        expected_url_substring="Linie_citybus_H/ZOB%20Bussteig%205.pdf",
        expected_answer_markers=("06:34", "6:34"),
    ),
    SimpleLiveCase(
        id="w3c_dummy_pdf",
        goal="Lies dieses PDF und nenne den Titel oder die ersten sichtbaren Worte.",
        start_url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        allowed_domains=("www.w3.org",),
        expected_url_substring=".pdf",
        expected_answer_markers=("Dummy PDF file",),
    ),
    SimpleLiveCase(
        id="wikipedia_selenium_software",
        goal=(
            "Gehe auf Wikipedia, suche nach Selenium software und öffne die Seite "
            "Selenium (software). Antworte kurz mit dem Seitentitel."
        ),
        start_url="https://www.wikipedia.org/",
        allowed_domains=("wikipedia.org", "www.wikipedia.org", "en.wikipedia.org"),
        expected_url_substring="Selenium_(software)",
        expected_answer_markers=("Selenium",),
    ),
    SimpleLiveCase(
        id="selenium_simple_form",
        goal=(
            "Fülle auf der Selenium Test-Form Name Marta ein, lasse die anderen "
            "Felder unverändert, sende das Formular ab und nenne den "
            "Bestätigungstext."
        ),
        start_url="https://www.selenium.dev/selenium/web/web-form.html",
        allowed_domains=("www.selenium.dev", "selenium.dev"),
        expected_url_substring="submitted-form.html",
        expected_answer_markers=("Received",),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr environment file.")
    parser.add_argument(
        "--environment-label",
        default="local_simple_live",
        help="Label written into the result payload, for example local_simple_live or pi_simple_live.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Optional explicit JSON output path. Defaults to "
            "artifacts/reports/browser_automation/simple_live_web_acceptance_<label>.json"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=10, help="Per-case browser step budget.")
    parser.add_argument("--max-runtime-s", type=float, default=120.0, help="Per-case browser runtime budget.")
    parser.add_argument("--capture-html", action="store_true", help="Capture final HTML artifacts during the run.")
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Optional case id to run. Repeat to execute a subset of the suite.",
    )
    return parser.parse_args()


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


def _selected_cases(case_ids: list[str]) -> tuple[SimpleLiveCase, ...]:
    """Return the selected suite cases or raise on unknown ids."""

    requested = tuple(str(item or "").strip() for item in case_ids if str(item or "").strip())
    if not requested:
        return _CASES
    requested_set = set(requested)
    selected = tuple(case for case in _CASES if case.id in requested_set)
    missing = sorted(requested_set.difference(case.id for case in selected))
    if missing:
        raise ValueError(f"Unknown simple-live case ids: {', '.join(missing)}")
    return selected


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        cases = _selected_cases(list(args.case_id))
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    env_path = (repo_root / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)
    config = TwinrConfig.from_env(env_path)
    if not bool(getattr(config, "browser_automation_enabled", False)):
        config = replace(config, browser_automation_enabled=True)
    driver = load_browser_automation_driver(config=config, project_root=repo_root)

    results: list[dict[str, Any]] = []
    counts = {
        "pass": 0,
        "false_completion": 0,
        "fail_closed_after_navigation": 0,
        "fail": 0,
    }
    for case in cases:
        result = driver.execute(
            BrowserAutomationRequest(
                task_id=case.id,
                goal=case.goal,
                start_url=case.start_url,
                allowed_domains=case.allowed_domains,
                max_steps=min(int(args.max_steps), int(case.max_steps)),
                max_runtime_s=float(args.max_runtime_s),
                capture_screenshot=False,
                capture_html=bool(args.capture_html),
                metadata={"task_kind": "read", "eval_case_id": case.id, "eval_suite": "simple_live_web_acceptance"},
            )
        )
        answer_markdown = str(result.data.get("answer_markdown") or "")
        final_url = str(result.final_url or "")
        url_ok = case.expected_url_substring in final_url if case.expected_url_substring else True
        answer_ok = _contains_any_marker(answer_markdown, case.expected_answer_markers)
        verdict = _classify(ok=bool(result.ok), url_ok=url_ok, answer_ok=answer_ok)
        counts[verdict] += 1
        results.append(
            {
                "id": case.id,
                "goal": case.goal,
                "start_url": case.start_url,
                "status": result.status,
                "ok": result.ok,
                "verdict": verdict,
                "summary": result.summary,
                "final_url": result.final_url,
                "error_code": result.error_code,
                "answer_markdown": answer_markdown,
                "key_points": list(result.data.get("key_points") or []),
                "used_api_sources": list(result.data.get("used_api_sources") or []),
                "visited_urls": list(result.data.get("visited_urls") or []),
                "trace_path": result.data.get("trace_path"),
                "typed_result": result.data.get("typed_result"),
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
        else output_dir / f"simple_live_web_acceptance_{args.environment_label}.json"
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": str(args.environment_label),
        "driver_mode": "tessairact_vendored_browser_loop_with_document_reader",
        "counts": counts,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output_path), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
