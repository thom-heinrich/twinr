#!/usr/bin/env python3
"""Run the real-target browser matrix against one Twinr runtime.

This benchmark replays the three browser tasks derived from actual Twinr
conversation-lab sessions: Café Luise through the live search-to-browser
handoff, Café Luise on the corrected official site, and Praxis am Schlump on
the official contact page. The first case intentionally starts with the old bad
domain and relies on the pending search follow-up hint to prove the repaired
live handoff path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.tools.handlers.browser import handle_browser_automation
from twinr.agent.tools.runtime.browser_follow_up import remember_pending_browser_follow_up_hint


@dataclass(frozen=True, slots=True)
class _PendingHintSpec:
    """Describe one pending search follow-up hint to seed before a browser run."""

    question: str
    follow_up_url: str
    follow_up_domain: str
    reason: str


@dataclass(frozen=True, slots=True)
class RealTargetCase:
    """Describe one real browser task plus its acceptance markers."""

    id: str
    category: str
    goal: str
    requested_start_url: str
    requested_allowed_domains: tuple[str, ...]
    expected_url_substring: str
    expected_answer_markers: tuple[str, ...]
    max_steps: int = 8
    pending_hint: _PendingHintSpec | None = None


_CASES: tuple[RealTargetCase, ...] = (
    RealTargetCase(
        id="cafe_luise_live_bad_domain",
        category="live_handoff_exact",
        goal=(
            "Prüfe auf der Website von Café Luise, ob für heute ein Mittagsmenü oder Mittagstisch "
            "veröffentlicht wurde, und gib den aktuellen Fund kurz wieder."
        ),
        requested_start_url="https://www.cafe-luise.de/",
        requested_allowed_domains=("cafe-luise.de",),
        expected_url_substring="cafe-luise-baeckerei.de",
        expected_answer_markers=(
            "kein sichtbares mittagsmenü",
            "kein sichtbarer mittagstisch",
            "kein aktuelles mittagsmenü",
            "keinen aktuell veröffentlichten hinweis",
            "keinen veröffentlichten hinweis",
            "kein heutiges mittagsmenü",
            "kein mittagstisch sichtbar",
            "kein mittagsmenü oder mittagstisch",
            "kein mittagsmenü oder mittagstisch für heute erkennbar",
            "kein mittagsmenü/mittagstisch",
            "kein mittagsmenü/mittagstisch für heute sichtbar",
            "kein mittagsmenü/mittagstisch für heute erkennbar",
            "kein heutiger mittagstisch",
            "kein mittagsmenü sichtbar",
        ),
        max_steps=2,
        pending_hint=_PendingHintSpec(
            question="Hat Café Luise heute online ein Mittagsmenü veröffentlicht?",
            follow_up_url="https://www.cafe-luise-baeckerei.de/",
            follow_up_domain="cafe-luise-baeckerei.de",
            reason="Die offizielle Website sollte geprüft werden.",
        ),
    ),
    RealTargetCase(
        id="cafe_luise_official_menu",
        category="corrected_official_site",
        goal=(
            "Prüfe auf der offiziellen Website von Café Luise in Hamburg, ob für heute ein "
            "Mittagsmenü oder Mittagstisch veröffentlicht wurde. Wenn nichts Aktuelles sichtbar "
            "ist, sage das klar und kurz."
        ),
        requested_start_url="https://www.cafe-luise-baeckerei.de/",
        requested_allowed_domains=(
            "cafe-luise-baeckerei.de",
            "www.cafe-luise-baeckerei.de",
            "cafeluise.com",
            "www.cafeluise.com",
        ),
        expected_url_substring="cafe-luise-baeckerei.de",
        expected_answer_markers=(
            "kein sichtbares mittagsmenü",
            "kein sichtbarer mittagstisch",
            "kein aktuelles mittagsmenü",
            "keinen aktuell veröffentlichten hinweis",
            "keinen veröffentlichten hinweis",
            "kein heutiges mittagsmenü",
            "kein mittagstisch sichtbar",
            "kein mittagsmenü oder mittagstisch",
            "kein mittagsmenü oder mittagstisch für heute erkennbar",
            "kein mittagsmenü/mittagstisch",
            "kein mittagsmenü/mittagstisch für heute sichtbar",
            "kein mittagsmenü/mittagstisch für heute erkennbar",
            "kein heutiger mittagstisch",
            "kein mittagsmenü sichtbar",
        ),
        max_steps=2,
    ),
    RealTargetCase(
        id="praxis_am_schlump_slots",
        category="live_handoff_exact",
        goal=(
            "Auf der offiziellen Termin- bzw. Kontaktseite der Praxis am Schlump prüfen, ob heute "
            "freie Termine, Kalendertage oder Buchungsslots sichtbar sind. Wenn keine Slots sichtbar "
            "sind, sage das klar."
        ),
        requested_start_url="https://www.beim-schlump.de/psychotherapie/",
        requested_allowed_domains=("beim-schlump.de", "www.beim-schlump.de"),
        expected_url_substring="beim-schlump.de/psychotherapie",
        expected_answer_markers=("keine freien termine", "keine buchungsslots", "kein sichtbarer kalender"),
        max_steps=2,
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
        help="Optional explicit JSON output path. Defaults to artifacts/reports/browser_automation/real_target_matrix_<label>.json",
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


def _make_owner(*, config: TwinrConfig) -> SimpleNamespace:
    """Return one minimal owner exposing the browser handler runtime surface."""

    return SimpleNamespace(
        config=config,
        runtime=SimpleNamespace(),
        emit=lambda _message: None,
        _record_event=lambda *_args, **_kwargs: None,
        _record_usage=lambda **_kwargs: None,
    )


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    env_path = (repo_root / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)
    config = TwinrConfig.from_env(env_path)
    if not bool(getattr(config, "browser_automation_enabled", False)):
        config = replace(config, browser_automation_enabled=True)

    results: list[dict[str, Any]] = []
    counts = {
        "pass": 0,
        "false_completion": 0,
        "fail_closed_after_navigation": 0,
        "fail": 0,
    }
    for case in _CASES:
        owner = _make_owner(config=config)
        if case.pending_hint is not None:
            remember_pending_browser_follow_up_hint(
                owner.runtime,
                question=case.pending_hint.question,
                follow_up_url=case.pending_hint.follow_up_url,
                follow_up_domain=case.pending_hint.follow_up_domain,
                site_follow_up_recommended=True,
                question_resolved=False,
                verification_status="partial",
                reason=case.pending_hint.reason,
                sources=(case.pending_hint.follow_up_url,),
            )
        result = handle_browser_automation(
            owner,
            {
                "goal": case.goal,
                "start_url": case.requested_start_url,
                "allowed_domains": list(case.requested_allowed_domains),
                "max_steps": int(case.max_steps),
                "max_runtime_s": float(args.max_runtime_s),
                "capture_html": bool(args.capture_html),
            },
        )
        payload_data = dict(result.get("data") or {})
        answer_markdown = str(payload_data.get("answer_markdown") or "")
        final_url = str(result.get("final_url") or "")
        url_ok = case.expected_url_substring in final_url
        answer_ok = _contains_any_marker(answer_markdown, case.expected_answer_markers)
        verdict = _classify(ok=bool(result.get("ok", False)), url_ok=url_ok, answer_ok=answer_ok)
        counts[verdict] += 1
        results.append(
            {
                "case_id": case.id,
                "category": case.category,
                "goal": case.goal,
                "requested_start_url": case.requested_start_url,
                "requested_allowed_domains": list(case.requested_allowed_domains),
                "status": result.get("status"),
                "ok": result.get("ok"),
                "verdict": verdict,
                "summary": result.get("summary"),
                "final_url": result.get("final_url"),
                "error_code": result.get("error_code"),
                "answer_markdown": answer_markdown,
                "key_points": list(payload_data.get("key_points") or ()),
                "visited_urls": list(payload_data.get("visited_urls") or ()),
                "trace_path": payload_data.get("trace_path"),
                "step_count": payload_data.get("step_count"),
                "request_repair": payload_data.get("request_repair"),
                "same_url_completion_verifier": payload_data.get("same_url_completion_verifier"),
                "expected_url_substring": case.expected_url_substring,
                "expected_answer_markers": list(case.expected_answer_markers),
                "url_ok": url_ok,
                "answer_ok": answer_ok,
                "artifacts": list(result.get("artifacts") or ()),
                "pending_hint": None if case.pending_hint is None else asdict(case.pending_hint),
            }
        )

    output_dir = repo_root / "artifacts" / "reports" / "browser_automation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output)
        if args.output
        else output_dir / f"real_target_matrix_{args.environment_label}.json"
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": str(args.environment_label),
        "driver_mode": "browser_handler_with_pending_search_hint",
        "counts": counts,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output_path), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
