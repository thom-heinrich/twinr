#!/usr/bin/env python3
"""Run the modern public-site browser benchmark against one Twinr runtime.

This script exercises the current browser automation driver on the same
five-site public benchmark matrix used to track dense modern page progress:
React docs, Next.js docs, Tailwind docs, MUI docs, and Apple's iPhone compare
page. It writes a machine-readable JSON report under
``artifacts/reports/browser_automation/`` by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
class BenchmarkCase:
    """Describe one real-site browser task plus its acceptance markers."""

    id: str
    site: str
    complexity: str
    goal: str
    start_url: str
    allowed_domains: tuple[str, ...]
    expected_url_substring: str
    expected_answer_markers: tuple[str, ...]


_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        id="react_useeffectevent",
        site="react.dev",
        complexity="modern_docs_spa",
        goal="On react.dev, navigate from the docs to the useEffectEvent reference page and tell me the short introductory summary shown for useEffectEvent.",
        start_url="https://react.dev/",
        allowed_domains=("react.dev",),
        expected_url_substring="/reference/react/useEffectEvent",
        expected_answer_markers=("separate events from effects",),
    ),
    BenchmarkCase(
        id="next_redirect",
        site="nextjs.org",
        complexity="modern_docs_spa",
        goal="On nextjs.org docs, navigate to the redirect function documentation and tell me which module the redirect function is imported from.",
        start_url="https://nextjs.org/docs",
        allowed_domains=("nextjs.org",),
        expected_url_substring="/docs/app/api-reference/functions/redirect",
        expected_answer_markers=("next/navigation",),
    ),
    BenchmarkCase(
        id="tailwind_darkmode",
        site="tailwindcss.com",
        complexity="modern_docs_spa",
        goal="On tailwindcss.com docs, navigate from the docs home to the Dark mode page and tell me the utility variant or class name used to style elements in dark mode manually.",
        start_url="https://tailwindcss.com/docs/installation/using-vite",
        allowed_domains=("tailwindcss.com",),
        expected_url_substring="/docs/dark-mode",
        expected_answer_markers=("dark:",),
    ),
    BenchmarkCase(
        id="mui_autocomplete",
        site="mui.com",
        complexity="modern_docs_dense",
        goal="On mui.com Material UI docs, navigate to the Autocomplete component page and tell me what the docs mean by free solo mode.",
        start_url="https://mui.com/material-ui/getting-started/",
        allowed_domains=("mui.com",),
        expected_url_substring="/material-ui/react-autocomplete/",
        expected_answer_markers=("not bound to provided options", "the user input is not bound", "any arbitrary value"),
    ),
    BenchmarkCase(
        id="apple_iphone_compare",
        site="apple.com",
        complexity="consumer_product_navigation",
        goal="On apple.com, navigate from the iPhone section to the iPhone comparison page and tell me whether iPhone 16e appears among the compared models.",
        start_url="https://www.apple.com/iphone/",
        allowed_domains=("apple.com", "www.apple.com"),
        expected_url_substring="/iphone/compare/",
        expected_answer_markers=("iphone 16e",),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr environment file.")
    parser.add_argument(
        "--environment-label",
        default="local",
        help="Label written into the result payload, for example local or pi.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit JSON output path. Defaults to artifacts/reports/browser_automation/modern_public_sites_<label>.json",
    )
    parser.add_argument("--max-steps", type=int, default=8, help="Per-case browser step budget.")
    parser.add_argument("--max-runtime-s", type=float, default=60.0, help="Per-case browser runtime budget.")
    parser.add_argument("--capture-html", action="store_true", help="Capture final HTML artifacts during the run.")
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


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    env_path = (repo_root / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)
    config = TwinrConfig.from_env(env_path)
    driver = load_browser_automation_driver(config=config, project_root=repo_root)

    results: list[dict[str, Any]] = []
    counts = {
        "pass": 0,
        "false_completion": 0,
        "fail_closed_after_navigation": 0,
        "fail": 0,
    }
    for case in _CASES:
        request = BrowserAutomationRequest(
            task_id=case.id,
            goal=case.goal,
            start_url=case.start_url,
            allowed_domains=case.allowed_domains,
            max_steps=int(args.max_steps),
            max_runtime_s=float(args.max_runtime_s),
            capture_screenshot=True,
            capture_html=bool(args.capture_html),
            metadata={"task_kind": "read", "eval_case_id": case.id},
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
                "site": case.site,
                "complexity": case.complexity,
                "goal": case.goal,
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
    output_path = Path(args.output) if args.output else output_dir / f"modern_public_sites_{args.environment_label}.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": str(args.environment_label),
        "driver_mode": "tessairact_vendored_browser_loop",
        "counts": counts,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output_path), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
