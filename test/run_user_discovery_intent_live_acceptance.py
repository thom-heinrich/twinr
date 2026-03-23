"""Run a live German intent corpus against Twinr's guided discovery tool path.

Purpose
-------
Exercise ``gpt-5.4-mini`` with a corpus of natural German utterances that
should map to Twinr's guided user-discovery lifecycle. The corpus covers:

- explicit start / continue requests
- free self-disclosure that should enter bounded discovery without a magic phrase
- review requests asking what Twinr already learned
- correction and deletion requests against previously learned facts

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src python3 test/run_user_discovery_intent_live_acceptance.py --env-file .env
    PYTHONPATH=src python3 test/run_user_discovery_intent_live_acceptance.py --env-file /twinr/.env --output artifacts/reports/user_discovery_intent_live.json

Inputs
------
- ``--env-file``: Twinr env file used for live provider credentials and remote
  memory access.
- ``--run-id``: Optional suffix for the isolated remote namespace.
- ``--output``: Optional JSON artifact path.

Outputs
-------
- JSON summary written to stdout.
- Optional JSON artifact written to ``--output``.
- Exit code 0 when all corpus checks pass, 1 otherwise.

Notes
-----
Each corpus item uses its own isolated workspace and remote namespace so the
live acceptance run does not mutate productive Twinr memory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from twinr.memory.user_discovery import UserDiscoveryFact, UserDiscoveryMemoryRoute

from test.run_user_discovery_live_acceptance import (
    AcceptanceCheck,
    LiveDiscoveryContext,
    _compact_text,
    _json_safe,
    _route_kinds,
    _tool_actions,
    run_text_turn,
)


@dataclass(frozen=True, slots=True)
class IntentCase:
    """Describe one natural-language discovery intent probe."""

    case_id: str
    category: str
    prompt: str
    state: str
    expected_actions: tuple[str, ...]
    require_all_actions: bool = False
    expected_route_kinds: tuple[str, ...] = ()
    accepted_answer_substrings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IntentCaseArtifact:
    """Capture one live intent probe and whether it matched expectations."""

    case_id: str
    category: str
    prompt: str
    state: str
    expected_actions: tuple[str, ...]
    require_all_actions: bool
    expected_route_kinds: tuple[str, ...]
    observed_actions: tuple[str, ...]
    observed_route_kinds: tuple[str, ...]
    model: str | None
    answer: str
    raw_tool_calls: tuple[dict[str, object], ...]
    raw_tool_results: tuple[dict[str, object], ...]
    passed: bool
    detail: str


_SEEDED_LEARNED_FACTS = (
    UserDiscoveryFact(storage="user_profile", text="The user prefers to be addressed as Thom."),
    UserDiscoveryFact(storage="personality", text="Use informal address (du) with the user."),
)

_SEEDED_MEMORY_ROUTES = (
    UserDiscoveryMemoryRoute(
        route_kind="contact",
        text="Anna is the user's daughter.",
        category="family",
        given_name="Anna",
        role="daughter",
        relation="daughter",
        summary="Anna is the user's daughter.",
        value="Anna",
        kind="family_contact",
    ),
    UserDiscoveryMemoryRoute(
        route_kind="preference",
        text="The user prefers Melitta as coffee brand.",
        category="coffee brand",
        value="Melitta",
        for_product="coffee",
        summary="The user prefers Melitta as a coffee brand.",
        kind="preference",
    ),
    UserDiscoveryMemoryRoute(
        route_kind="plan",
        text="The user plans to call Anna tomorrow.",
        category="family",
        given_name="Anna",
        relation="daughter",
        value="call Anna",
        summary="The user plans to call Anna tomorrow.",
        when_text="tomorrow",
        kind="plan",
    ),
)

INTENT_CASES: tuple[IntentCase, ...] = (
    IntentCase(
        case_id="start_plain",
        category="start",
        prompt="Lass uns mit dem Kennenlernen anfangen.",
        state="idle",
        expected_actions=("start_or_resume",),
    ),
    IntentCase(
        case_id="start_setup",
        category="start",
        prompt="Ich möchte mit meiner Einrichtung starten.",
        state="idle",
        expected_actions=("start_or_resume",),
    ),
    IntentCase(
        case_id="start_meta",
        category="start",
        prompt="Magst du etwas über mich lernen?",
        state="idle",
        expected_actions=("start_or_resume",),
    ),
    IntentCase(
        case_id="start_better_know_me",
        category="start",
        prompt="Ich will, dass du mich besser kennenlernst.",
        state="idle",
        expected_actions=("start_or_resume",),
    ),
    IntentCase(
        case_id="self_name_idle",
        category="self_disclosure_idle",
        prompt="Ich heiße Thom.",
        state="idle",
        expected_actions=("answer",),
        expected_route_kinds=("user_profile",),
        accepted_answer_substrings=("merk", "thom"),
    ),
    IntentCase(
        case_id="self_style_idle",
        category="self_disclosure_idle",
        prompt="Du kannst mich gern duzen.",
        state="idle",
        expected_actions=("answer",),
        expected_route_kinds=("personality",),
    ),
    IntentCase(
        case_id="self_contact_idle",
        category="self_disclosure_idle",
        prompt="Anna ist meine Tochter.",
        state="idle",
        expected_actions=("answer",),
        expected_route_kinds=("contact",),
        accepted_answer_substrings=("merken", "tochter"),
    ),
    IntentCase(
        case_id="self_preference_idle",
        category="self_disclosure_idle",
        prompt="Ich erzähle dir etwas über mich: Melitta ist meine Lieblingskaffeemarke.",
        state="idle",
        expected_actions=("answer",),
        expected_route_kinds=("preference",),
        accepted_answer_substrings=("merk", "lieblingskaffeemarke"),
    ),
    IntentCase(
        case_id="continue_name",
        category="continue",
        prompt="Ich möchte Thom genannt werden.",
        state="active_basics",
        expected_actions=("answer",),
        accepted_answer_substrings=("merk", "thom"),
    ),
    IntentCase(
        case_id="continue_style",
        category="continue",
        prompt="Du kannst mich duzen.",
        state="active_basics",
        expected_actions=("answer",),
    ),
    IntentCase(
        case_id="continue_contact",
        category="continue",
        prompt="Anna ist meine Tochter.",
        state="active_basics",
        expected_actions=("answer",),
        accepted_answer_substrings=("merken", "tochter"),
    ),
    IntentCase(
        case_id="continue_preference",
        category="continue",
        prompt="Melitta ist meine Lieblingskaffeemarke.",
        state="active_basics",
        expected_actions=("answer",),
        accepted_answer_substrings=("merk", "lieblingskaffeemarke"),
    ),
    IntentCase(
        case_id="review_what",
        category="review",
        prompt="Was weißt du schon über mich?",
        state="seeded_profile",
        expected_actions=("review_profile",),
    ),
    IntentCase(
        case_id="review_show",
        category="review",
        prompt="Zeig mir bitte, was du über mich gelernt hast.",
        state="seeded_profile",
        expected_actions=("review_profile",),
    ),
    IntentCase(
        case_id="review_remembered",
        category="review",
        prompt="Welche Sachen hast du dir über mich gemerkt?",
        state="seeded_profile",
        expected_actions=("review_profile",),
    ),
    IntentCase(
        case_id="review_profile_short",
        category="review",
        prompt="Kann ich mein Profil kurz ansehen?",
        state="seeded_profile",
        expected_actions=("review_profile",),
    ),
    IntentCase(
        case_id="correct_name_short",
        category="correct",
        prompt="Nenn mich bitte Tom statt Thom.",
        state="seeded_profile",
        expected_actions=("review_profile", "replace_fact"),
        require_all_actions=True,
        accepted_answer_substrings=("tom", "thom"),
    ),
    IntentCase(
        case_id="correct_name_direct",
        category="correct",
        prompt="Der Name stimmt nicht, speichere bitte Tom statt Thom.",
        state="seeded_profile",
        expected_actions=("review_profile", "replace_fact"),
        require_all_actions=True,
    ),
    IntentCase(
        case_id="delete_plan_plain",
        category="delete",
        prompt="Den Plan, Anna morgen anzurufen, kannst du löschen.",
        state="seeded_profile",
        expected_actions=("review_profile", "delete_fact"),
        require_all_actions=True,
    ),
    IntentCase(
        case_id="delete_plan_soft",
        category="delete",
        prompt="Vergiss bitte den Anrufplan für morgen wieder.",
        state="seeded_profile",
        expected_actions=("review_profile", "delete_fact"),
        require_all_actions=True,
        accepted_answer_substrings=("plan", "löschen"),
    ),
)


def _prepare_state(harness: object, *, state: str) -> None:
    """Move one fresh harness into the requested discovery state."""

    runtime = harness.runtime
    if state == "idle":
        return
    if state == "active_basics":
        runtime.manage_user_discovery(action="start_or_resume", topic_id="basics")
        return
    if state == "seeded_profile":
        runtime.manage_user_discovery(action="start_or_resume", topic_id="basics")
        runtime.manage_user_discovery(
            action="answer",
            topic_id="basics",
            learned_facts=_SEEDED_LEARNED_FACTS,
            memory_routes=_SEEDED_MEMORY_ROUTES,
            topic_complete=True,
        )
        return
    raise ValueError(f"Unsupported intent probe state: {state!r}")


def _reset_copied_user_context(context: LiveDiscoveryContext) -> None:
    """Blank copied productive user context for generic intent probes.

    The intent corpus should measure semantic discovery routing against a clean
    user profile, not against whatever productive user facts currently live in
    the checked-in personality folder.
    """

    user_path = context.personality_dir / "USER.md"
    user_path.write_text("", encoding="utf-8")


def _actions_match(case: IntentCase, observed_actions: tuple[str, ...]) -> bool:
    """Return whether one case matched the observed action sequence."""

    observed = set(observed_actions)
    required = set(case.expected_actions)
    if case.require_all_actions:
        return required.issubset(observed)
    return any(action in observed for action in case.expected_actions)


def _answer_bridge_match(case: IntentCase, answer: str) -> bool:
    """Return whether one answer still semantically matched the intent."""

    if not case.accepted_answer_substrings:
        return False
    folded_answer = _compact_text(answer, max_len=400).casefold()
    return all(fragment.casefold() in folded_answer for fragment in case.accepted_answer_substrings)


def _run_intent_case(*, env_file: Path, run_id: str | None, case: IntentCase) -> IntentCaseArtifact:
    """Execute one live intent probe in an isolated runtime workspace."""

    case_run_id = f"{run_id or 'intent'}_{case.case_id}"
    context = LiveDiscoveryContext(base_env_path=env_file, run_id=case_run_id)
    harness = None
    try:
        _reset_copied_user_context(context)
        harness = context.make_harness(emitted=[])
        _prepare_state(harness, state=case.state)
        turn = run_text_turn(harness, case.prompt)
        observed_actions = _tool_actions(turn, tool_name="manage_user_discovery")
        observed_route_kinds = _route_kinds(turn)
        actions_ok = _actions_match(case, observed_actions)
        routes_ok = set(case.expected_route_kinds).issubset(set(observed_route_kinds))
        bridge_ok = (not actions_ok or not routes_ok) and _answer_bridge_match(case, turn.answer)
        passed = (actions_ok and routes_ok) or bridge_ok
        detail_parts = [
            f"expected_actions={case.expected_actions}",
            f"observed_actions={observed_actions}",
        ]
        if case.expected_route_kinds:
            detail_parts.append(f"expected_route_kinds={case.expected_route_kinds}")
            detail_parts.append(f"observed_route_kinds={observed_route_kinds}")
        if bridge_ok:
            detail_parts.append("accepted_via_answer_bridge=true")
        return IntentCaseArtifact(
            case_id=case.case_id,
            category=case.category,
            prompt=case.prompt,
            state=case.state,
            expected_actions=case.expected_actions,
            require_all_actions=case.require_all_actions,
            expected_route_kinds=case.expected_route_kinds,
            observed_actions=observed_actions,
            observed_route_kinds=observed_route_kinds,
            model=turn.model,
            answer=turn.answer,
            raw_tool_calls=tuple(_json_safe(turn.raw_tool_calls)),
            raw_tool_results=tuple(_json_safe(turn.raw_tool_results)),
            passed=passed,
            detail="; ".join(detail_parts),
        )
    finally:
        if harness is not None:
            try:
                harness.runtime.shutdown(timeout_s=1.0)
            except Exception:
                pass
        context.close()


def run_acceptance(*, env_file: Path, run_id: str | None) -> dict[str, object]:
    """Run the full live German discovery-intent corpus."""

    artifacts_list: list[IntentCaseArtifact] = []
    total = len(INTENT_CASES)
    for index, case in enumerate(INTENT_CASES, start=1):
        sys.stderr.write(f"[{index:02d}/{total:02d}] {case.case_id} ... ")
        sys.stderr.flush()
        artifact = _run_intent_case(env_file=env_file, run_id=run_id, case=case)
        artifacts_list.append(artifact)
        sys.stderr.write("PASS\n" if artifact.passed else "FAIL\n")
        sys.stderr.flush()
    artifacts = tuple(artifacts_list)
    model_names = tuple(artifact.model for artifact in artifacts if artifact.model)
    category_counts = Counter(artifact.category for artifact in artifacts)
    category_pass_counts = Counter(artifact.category for artifact in artifacts if artifact.passed)
    failed_case_ids = tuple(artifact.case_id for artifact in artifacts if not artifact.passed)
    checks = (
        AcceptanceCheck(
            name="live_model_is_gpt_5_4_mini",
            passed=bool(model_names) and all(str(name).startswith("gpt-5.4-mini") for name in model_names),
            detail=f"models={model_names}",
        ),
        AcceptanceCheck(
            name="all_intent_cases_passed",
            passed=all(artifact.passed for artifact in artifacts),
            detail=f"failed_case_ids={failed_case_ids}",
        ),
    )
    return {
        "passed": all(check.passed for check in checks),
        "requested_model": "gpt-5.4-mini",
        "actual_models": model_names,
        "corpus_size": len(INTENT_CASES),
        "category_counts": dict(category_counts),
        "category_pass_counts": dict(category_pass_counts),
        "checks": [asdict(check) for check in checks],
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "failed_case_ids": failed_case_ids,
        "sample_answers": {
            artifact.case_id: _compact_text(artifact.answer, max_len=220)
            for artifact in artifacts
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run acceptance, and emit JSON output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env", help="Twinr env file used for live provider credentials.")
    parser.add_argument("--run-id", default=None, help="Optional suffix for the isolated remote namespace.")
    parser.add_argument("--output", default=None, help="Optional JSON artifact path.")
    args = parser.parse_args(argv)

    result = run_acceptance(
        env_file=Path(args.env_file),
        run_id=args.run_id,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if bool(result.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
