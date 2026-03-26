"""Run a bounded eval for Twinr's reserve-card writer/judge copy stack.

Purpose
-------
Exercise the live reserve-card writer/judge pipeline against two bounded input
sets without mutating runtime stores:

- representative fixture candidates across world, memory, discovery, and reflection
- optional live raw candidates from the same reserve-companion flow used for day plans

The script reports final cards, available writer variants, stage retries, and
basic family counts so prompt/copy changes can be compared on the Pi without
having to wait for organic card rotation.

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src python3 test/run_display_reserve_copy_eval.py --env-file .env
    PYTHONPATH=src python3 test/run_display_reserve_copy_eval.py --env-file /twinr/.env --output /twinr/artifacts/reports/display_reserve_copy_eval_latest.json
    PYTHONPATH=src python3 test/run_display_reserve_copy_eval.py --env-file /twinr/.env --skip-representative --live-max-items 12

Inputs
------
- ``--env-file``: Twinr env file used to construct the runtime config.
- ``--output``: Optional JSON report path.
- ``--skip-live``: Evaluate only the representative fixture set.
- ``--skip-representative``: Evaluate only the live raw reserve candidates.
- ``--live-max-items``: Max raw live candidates pulled from the reserve flow.

Outputs
-------
- JSON summary written to stdout.
- Optional JSON artifact written to ``--output``.

Notes
-----
The live path uses ``DisplayReserveCompanionFlow.load_raw_candidates()`` so it
reads the same upstream reserve sources as the day-plan builder without writing
plan state or publishing cards. Reserve-copy generation itself still performs
real provider calls and therefore requires a valid OpenAI config.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.proactive.runtime.display_reserve_copy_contract import resolve_reserve_copy_family
from twinr.proactive.runtime.display_reserve_diversity import display_reserve_seed_profile
from twinr.proactive.runtime.display_reserve_flow import DisplayReserveCompanionFlow
from twinr.proactive.runtime.display_reserve_generation import (
    DisplayReserveCopyGenerator,
    DisplayReserveGenerationError,
)


def _iso_utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _representative_candidates() -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Return one bounded cross-family goldset for reserve-copy evaluation."""

    return (
        AmbientDisplayImpulseCandidate(
            topic_key="ai companions",
            title="AI companions",
            source="world",
            action="invite_follow_up",
            attention_state="shared_thread",
            salience=0.88,
            eyebrow="",
            headline="AI companions",
            body="Da ist gerade etwas spannend.",
            symbol="sparkles",
            accent="info",
            reason="world_subscription",
            candidate_family="world_subscription",
            generation_context={
                "display_anchor": "KI-Begleiter",
                "hook_hint": "Heute gab es mehrere Meldungen zu persoenlichen KI-Begleitern.",
                "topic_summary": "oeffentliches Thema rund um persoenliche KI-Begleiter",
                "card_intent": {
                    "topic_semantics": "oeffentliches Thema zu KI-Begleitern",
                    "statement_intent": "Twinr soll eine konkrete Beobachtung zu den heutigen Meldungen machen.",
                    "cta_intent": "Zu einer echten Meinung oder Einordnung einladen.",
                    "relationship_stance": "ruhig beobachtend mit leichter Haltung",
                },
            },
        ),
        AmbientDisplayImpulseCandidate(
            topic_key="repair policy",
            title="Repair policy",
            source="world",
            action="brief_update",
            attention_state="growing",
            salience=0.76,
            eyebrow="",
            headline="Repair policy",
            body="Da bewegt sich etwas.",
            symbol="sparkles",
            accent="info",
            reason="world_awareness",
            candidate_family="world_awareness",
            generation_context={
                "display_anchor": "Reparaturen",
                "hook_hint": "Heute wird wieder ueber leichtere Reparaturen diskutiert.",
                "topic_summary": "oeffentliche Debatte ueber Reparaturen und Recht auf Reparatur",
                "card_intent": {
                    "topic_semantics": "oeffentliche Debatte ueber Reparaturen",
                    "statement_intent": "Twinr soll die aktuelle Debatte knapp benennen.",
                    "cta_intent": "Zu einer kleinen Einordnung oder Meinung einladen.",
                    "relationship_stance": "aufmerksam und ruhig",
                },
            },
        ),
        AmbientDisplayImpulseCandidate(
            topic_key="world politics",
            title="World politics",
            source="world",
            action="brief_update",
            attention_state="growing",
            salience=0.72,
            eyebrow="",
            headline="World politics",
            body="Da ist heute etwas los.",
            symbol="sparkles",
            accent="info",
            reason="world_subscription",
            candidate_family="mindshare",
            generation_context={
                "display_anchor": "Weltpolitik",
                "hook_hint": "Heute gab es mehrere politische Meldungen mit internationalem Gewicht.",
                "topic_summary": "oeffentliche Nachrichtenlage mit mehreren grossen Themen",
                "card_intent": {
                    "topic_semantics": "oeffentliches Thema Weltpolitik",
                    "statement_intent": "Twinr soll einen klaren Anlass aus der heutigen Nachrichtenlage benennen.",
                    "cta_intent": "Zu einer Meinung oder Einordnung einladen.",
                    "relationship_stance": "ruhig beobachtend",
                },
            },
        ),
        AmbientDisplayImpulseCandidate(
            topic_key="doctor appointment yesterday",
            title="Arzttermin gestern",
            source="memory_follow_up",
            action="ask_one",
            attention_state="shared_thread",
            salience=0.84,
            eyebrow="",
            headline="Arzttermin gestern",
            body="Da fehlt mir noch etwas.",
            symbol="question",
            accent="warm",
            reason="memory_follow_up",
            candidate_family="memory_follow_up",
            generation_context={
                "display_anchor": "Arzttermin gestern",
                "hook_hint": "Du meintest, dass dort noch etwas offen ist.",
                "display_goal": "open_positive_conversation",
                "card_intent": {
                    "topic_semantics": "frueherer Gespraechsfaden zum Arzttermin von gestern",
                    "statement_intent": "Twinr soll ruhig an den Termin anknuepfen und sagen, dass noch etwas offen ist.",
                    "cta_intent": "Zu einem kurzen Update einladen.",
                    "relationship_stance": "ruhiger Rueckbezug statt Diagnose",
                },
            },
        ),
        AmbientDisplayImpulseCandidate(
            topic_key="two versions sunday",
            title="Sonntag",
            source="memory_conflict",
            action="ask_one",
            attention_state="forming",
            salience=0.79,
            eyebrow="",
            headline="Sonntag",
            body="Da habe ich zwei Versionen.",
            symbol="question",
            accent="warm",
            reason="memory_conflict",
            candidate_family="memory_conflict",
            generation_context={
                "display_anchor": "Sonntag",
                "hook_hint": "Es gibt zwei moegliche Versionen von dem, was am Sonntag passiert ist.",
                "display_goal": "open_positive_conversation",
                "card_intent": {
                    "topic_semantics": "sanfte Klaerung eines persoenlichen Widerspruchs",
                    "statement_intent": "Twinr soll sagen, dass noch zwei Versionen im Kopf sind.",
                    "cta_intent": "Zu einer kurzen Klaerung einladen.",
                    "relationship_stance": "vorsichtig und alltagsnah",
                },
            },
        ),
        AmbientDisplayImpulseCandidate(
            topic_key="visit last week",
            title="Besuch letzte Woche",
            source="relationship",
            action="ask_one",
            attention_state="forming",
            salience=0.66,
            eyebrow="",
            headline="Besuch letzte Woche",
            body="Da ist fuer mich noch etwas offen.",
            symbol="question",
            accent="warm",
            reason="memory_thread",
            candidate_family="memory_thread",
            generation_context={
                "display_anchor": "Besuch letzte Woche",
                "hook_hint": "Der persoenliche Faden dazu wirkt noch offen.",
                "display_goal": "open_positive_conversation",
                "card_intent": {
                    "topic_semantics": "offener persoenlicher Rueckbezug auf einen Besuch",
                    "statement_intent": "Twinr soll ruhig sagen, dass dazu noch etwas offen ist.",
                    "cta_intent": "Zu einem kurzen Weitererzaehlen einladen.",
                    "relationship_stance": "warm und unaufgeregt",
                },
            },
        ),
        AmbientDisplayImpulseCandidate(
            topic_key="preferred name",
            title="Ansprache",
            source="user_discovery",
            action="ask_one",
            attention_state="forming",
            salience=0.83,
            eyebrow="",
            headline="Ansprache",
            body="Das wuerde ich gern wissen.",
            symbol="question",
            accent="warm",
            reason="invite_user_discovery",
            candidate_family="user_discovery",
            generation_context={
                "display_goal": "invite_user_discovery",
                "display_anchor": "Ansprache",
                "hook_hint": "Twinr soll wissen, wie die Person angesprochen werden moechte.",
                "card_intent": {
                    "topic_semantics": "richtige Ansprache oder Name lernen",
                    "statement_intent": "Twinr soll sagen, dass er wissen moechte, wie er die Person ansprechen soll.",
                    "cta_intent": "Zu einer klaren Antwort auf Name oder Ansprache einladen.",
                    "relationship_stance": "freundlich und direkt",
                },
            },
        ),
        AmbientDisplayImpulseCandidate(
            topic_key="morning routine",
            title="Morgenroutine",
            source="user_discovery",
            action="ask_one",
            attention_state="forming",
            salience=0.69,
            eyebrow="",
            headline="Morgenroutine",
            body="Das interessiert mich.",
            symbol="question",
            accent="warm",
            reason="invite_user_discovery",
            candidate_family="user_discovery",
            generation_context={
                "display_goal": "invite_user_discovery",
                "display_anchor": "dein Morgen",
                "hook_hint": "Twinr soll lernen, was morgens gut tut oder wichtig ist.",
                "card_intent": {
                    "topic_semantics": "Alltagsgewohnheit am Morgen",
                    "statement_intent": "Twinr soll freundlich sagen, dass ihn interessiert, was morgens gut tut.",
                    "cta_intent": "Zu einer kleinen Beschreibung der Morgenroutine einladen.",
                    "relationship_stance": "warm und neugierig ohne UI-Ton",
                },
            },
        ),
        AmbientDisplayImpulseCandidate(
            topic_key="twinr boundaries",
            title="Grenzen",
            source="user_discovery",
            action="ask_one",
            attention_state="forming",
            salience=0.61,
            eyebrow="",
            headline="Grenzen",
            body="Das sollte ich wissen.",
            symbol="question",
            accent="warm",
            reason="invite_user_discovery",
            candidate_family="user_discovery",
            generation_context={
                "display_goal": "invite_user_discovery",
                "display_anchor": "No-Gos",
                "hook_hint": "Twinr soll lernen, was er besser lassen sollte.",
                "card_intent": {
                    "topic_semantics": "Grenzen oder Dinge, die Twinr vermeiden sollte",
                    "statement_intent": "Twinr soll sagen, dass er wissen moechte, was er besser lassen sollte.",
                    "cta_intent": "Zu einer klaren Grenze oder Vorliebe einladen.",
                    "relationship_stance": "ruhig und respektvoll",
                },
            },
        ),
        AmbientDisplayImpulseCandidate(
            topic_key="conversation about twinr",
            title="Twinr",
            source="reflection_midterm",
            action="ask_one",
            attention_state="shared_thread",
            salience=0.74,
            eyebrow="",
            headline="Twinr",
            body="Da denke ich noch dran.",
            symbol="question",
            accent="warm",
            reason="reflection_thread",
            candidate_family="reflection_thread",
            generation_context={
                "display_goal": "call_back_to_earlier_conversation",
                "display_anchor": "unser Gespraech ueber Twinr",
                "hook_hint": "Der Faden von vorhin wirkt noch offen.",
                "card_intent": {
                    "topic_semantics": "Rueckbezug auf ein frueheres Gespraech ueber Twinr",
                    "statement_intent": "Twinr soll ruhig sagen, dass ihm das Gespraech ueber Twinr geblieben ist.",
                    "cta_intent": "Zu einem kurzen Weiterreden oder einer Erinnerung einladen.",
                    "relationship_stance": "ruhiger Rueckbezug",
                },
            },
        ),
        AmbientDisplayImpulseCandidate(
            topic_key="doctor appointment from earlier",
            title="Arzttermin",
            source="reflection_midterm",
            action="ask_one",
            attention_state="forming",
            salience=0.58,
            eyebrow="",
            headline="Arzttermin",
            body="Da will ich anknuepfen.",
            symbol="question",
            accent="warm",
            reason="reflection_thread",
            candidate_family="reflection_thread",
            generation_context={
                "display_goal": "call_back_to_earlier_conversation",
                "display_anchor": "dein Gedanke zum Arzttermin",
                "hook_hint": "Der Punkt mit dem Arzttermin von vorhin kann ein kleiner Anschluss sein.",
                "card_intent": {
                    "topic_semantics": "kleiner Rueckbezug auf den frueheren Punkt zum Arzttermin",
                    "statement_intent": "Twinr soll sagen, dass ihm der Punkt zum Arzttermin geblieben ist.",
                    "cta_intent": "Zu einem kleinen Weiterreden einladen.",
                    "relationship_stance": "leicht und ruhig",
                },
            },
        ),
        AmbientDisplayImpulseCandidate(
            topic_key="topic from yesterday",
            title="gestern",
            source="reflection_midterm",
            action="ask_one",
            attention_state="shared_thread",
            salience=0.64,
            eyebrow="",
            headline="gestern",
            body="Das ist noch bei mir.",
            symbol="question",
            accent="warm",
            reason="reflection_thread",
            candidate_family="reflection_preference",
            generation_context={
                "display_goal": "call_back_to_earlier_conversation",
                "display_anchor": "dein Thema von gestern",
                "hook_hint": "Der fruehere Faden koennte heute noch einmal aufgegriffen werden.",
                "card_intent": {
                    "topic_semantics": "Rueckbezug auf ein offenes Thema von gestern",
                    "statement_intent": "Twinr soll ruhig sagen, dass ihm das Thema von gestern noch im Kopf ist.",
                    "cta_intent": "Zu einem kurzen Update einladen.",
                    "relationship_stance": "ruhig und menschlich",
                },
            },
        ),
    )


def _run_one_set(
    *,
    name: str,
    config: TwinrConfig,
    snapshot: PersonalitySnapshot | None,
    candidates: Sequence[AmbientDisplayImpulseCandidate],
    local_now: datetime,
) -> dict[str, object]:
    """Run one bounded rewrite/eval pass for the provided candidate set."""

    generator = DisplayReserveCopyGenerator()
    rewritten, trace = generator.rewrite_candidates_with_trace(
        config=config,
        snapshot=snapshot,
        candidates=candidates,
        local_now=local_now,
    )
    if trace.bypassed:
        raise RuntimeError(f"display reserve eval bypassed: {trace.bypass_reason}")
    selections_by_topic = {selection.topic_key.casefold(): selection for selection in trace.selections}
    retry_stage_counts = Counter(
        attempt.stage_name
        for attempt in trace.stage_attempts
        if attempt.attempt_index > 1
    )
    seed_profiles = {
        candidate.topic_key.casefold(): display_reserve_seed_profile(candidate)
        for candidate in rewritten
    }
    return {
        "name": name,
        "item_count": len(rewritten),
        "duration_s": round(trace.duration_seconds, 3),
        "retry_count": sum(1 for attempt in trace.stage_attempts if attempt.attempt_index > 1),
        "retry_stage_counts": dict(sorted(retry_stage_counts.items())),
        "family_counts": dict(
            sorted(
                Counter(resolve_reserve_copy_family(candidate) for candidate in rewritten).items(),
            )
        ),
        "seed_family_counts": dict(
            sorted(Counter(profile.family for profile in seed_profiles.values()).items())
        ),
        "seed_axis_counts": dict(
            sorted(Counter(profile.axis for profile in seed_profiles.values()).items())
        ),
        "items": [
            {
                "topic_key": candidate.topic_key,
                "candidate_family": candidate.candidate_family,
                "copy_family": resolve_reserve_copy_family(candidate),
                "seed_family": seed_profiles[candidate.topic_key.casefold()].family,
                "seed_axis": seed_profiles[candidate.topic_key.casefold()].axis,
                "headline": candidate.headline,
                "body": candidate.body,
                "variants": (
                    [
                        {
                            "headline": variant.headline,
                            "body": variant.body,
                        }
                        for variant in selection.variants
                    ]
                    if selection is not None
                    else []
                ),
            }
            for candidate in rewritten
            for selection in (selections_by_topic.get(candidate.topic_key.casefold()),)
        ],
        "stage_attempts": [
            {
                "batch_index": attempt.batch_index,
                "stage_name": attempt.stage_name,
                "attempt_index": attempt.attempt_index,
                "topic_keys": list(attempt.topic_keys),
                "max_output_tokens": attempt.max_output_tokens,
                "reasoning_effort": attempt.reasoning_effort,
                "duration_seconds": round(attempt.duration_seconds, 3),
                "succeeded": attempt.succeeded,
                "incomplete_reason": attempt.incomplete_reason,
            }
            for attempt in trace.stage_attempts
        ],
    }


def _load_live_raw_candidates(
    *,
    config: TwinrConfig,
    local_now: datetime,
    max_items: int,
) -> tuple[PersonalitySnapshot | None, tuple[AmbientDisplayImpulseCandidate, ...]]:
    """Load live raw reserve candidates without running copy rewrite twice."""

    flow = DisplayReserveCompanionFlow()
    return flow.load_raw_candidates(
        config,
        local_now=local_now,
        max_items=max_items,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the reserve-copy eval harness."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, help="Twinr env file.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--skip-live", action="store_true", help="Skip live raw reserve candidates.")
    parser.add_argument(
        "--skip-representative",
        action="store_true",
        help="Skip the representative fixture candidate set.",
    )
    parser.add_argument(
        "--live-max-items",
        type=int,
        default=8,
        help="Maximum live raw reserve candidates to evaluate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reserve-copy eval and emit one JSON report."""

    args = _parse_args(argv)
    if args.skip_live and args.skip_representative:
        raise SystemExit("At least one input set must remain enabled.")

    config = TwinrConfig.from_env(Path(args.env_file))
    local_now = datetime.now().astimezone()
    runs: list[dict[str, object]] = []

    if not args.skip_representative:
        runs.append(
            _run_one_set(
                name="representative_fixtures",
                config=config,
                snapshot=None,
                candidates=_representative_candidates(),
                local_now=local_now,
            )
        )

    if not args.skip_live:
        live_snapshot, live_candidates = _load_live_raw_candidates(
            config=config,
            local_now=local_now,
            max_items=max(1, int(args.live_max_items)),
        )
        runs.append(
            _run_one_set(
                name="live_raw_candidates",
                config=config,
                snapshot=live_snapshot,
                candidates=live_candidates,
                local_now=local_now,
            )
        )

    report = {
        "generated_at_utc": _iso_utc_now(),
        "config": {
            "display_reserve_generation_model": (
                config.display_reserve_generation_model or config.default_model
            ),
            "display_reserve_generation_reasoning_effort": config.display_reserve_generation_reasoning_effort,
            "display_reserve_generation_timeout_seconds": config.display_reserve_generation_timeout_seconds,
            "display_reserve_generation_batch_size": config.display_reserve_generation_batch_size,
            "display_reserve_generation_variants_per_candidate": config.display_reserve_generation_variants_per_candidate,
            "display_reserve_generation_max_output_tokens": config.display_reserve_generation_max_output_tokens,
        },
        "runs": runs,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    sys.stdout.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DisplayReserveGenerationError as exc:
        raise SystemExit(str(exc)) from exc
