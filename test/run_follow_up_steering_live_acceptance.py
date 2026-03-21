"""Run bounded live LLM acceptance for follow-up steering.

Purpose
-------
Exercise Twinr's real closure-plus-steering runtime path with live provider
calls while keeping the scenario set bounded and reproducible. The script
covers single-turn decisions, short multi-turn dialogue sequences, and a
bounded multi-day transition suite that simulates how co-attention and
conversation appetite evolve over several days of renewed interest, feed
refresh, and later cooling.

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src python3 test/run_follow_up_steering_live_acceptance.py --env-file .env --suite single-turn
    PYTHONPATH=src python3 test/run_follow_up_steering_live_acceptance.py --env-file /twinr/.env --suite multi-day --remote-check-mode direct --output artifacts/reports/follow_up_live.json
    PYTHONPATH=src python3 test/run_follow_up_steering_live_acceptance.py --env-file /twinr/.env --suite all --remote-check-mode direct --output artifacts/reports/follow_up_live.json

Inputs
------
- ``--env-file``: Twinr env file used to build the live provider/runtime config.
- ``--suite``: Which acceptance suite to run: ``single-turn``, ``multi-turn``,
  ``multi-day``, or ``all``.
- ``--timeout-s``: Provider timeout for the closure evaluator.
- ``--remote-check-mode``: ``direct``, ``watchdog_artifact``, or ``config_default``.
- ``--run-id``: Optional suffix for isolated multi-day remote snapshots.
- ``--output``: Optional JSON artifact path.

Outputs
-------
- JSON summary written to stdout.
- Optional JSON artifact written to ``--output``.

Notes
-----
The script intentionally uses the real tool-calling provider but drives it
through a narrow harness instead of the full realtime loop. That keeps the run
bounded while still exercising the same closure evaluator and follow-up
steering logic used in production. The multi-day suite clones a small
world-intelligence slice into isolated snapshot kinds so it can exercise real
remote persistence and refresh without mutating Twinr's live durable state.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.closure import (
    ConversationClosureEvaluation,
    ToolCallingConversationClosureEvaluator,
)
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.agent.personality.intelligence import (
    RemoteStateWorldIntelligenceStore,
    WorldFeedSubscription,
    WorldIntelligenceService,
    WorldIntelligenceState,
    WorldInterestSignal,
)
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.agent.personality.self_expression import build_mindshare_items
from twinr.agent.personality.service import PersonalityContextService
from twinr.agent.personality.steering import (
    ConversationTurnSteeringCue,
    build_turn_steering_cues,
)
from twinr.agent.workflows.follow_up_steering import FollowUpRuntimeDecision, FollowUpSteeringRuntime
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore
from twinr.providers import build_streaming_provider_bundle


@dataclass(frozen=True, slots=True)
class TurnExpectation:
    """Describe the expected runtime stance for one acceptance turn."""

    source: str
    force_close: bool
    matched_topic: str | None = None
    selected_topic: str | None = None


@dataclass(frozen=True, slots=True)
class AcceptanceTurnCase:
    """Describe one single closure-plus-steering acceptance step."""

    name: str
    user_transcript: str
    assistant_response: str
    cues: tuple[ConversationTurnSteeringCue, ...]
    expectation: TurnExpectation


@dataclass(frozen=True, slots=True)
class AcceptanceScenario:
    """Describe one bounded multi-turn dialogue acceptance scenario."""

    name: str
    turns: tuple[AcceptanceTurnCase, ...]


@dataclass(frozen=True, slots=True)
class LoadedLiveAcceptanceContext:
    """Hold the live durable state used to seed bounded acceptance runs."""

    snapshot: PersonalitySnapshot | None
    cues: tuple[ConversationTurnSteeringCue, ...]
    state: WorldIntelligenceState
    subscriptions: tuple[WorldFeedSubscription, ...]
    remote_state: LongTermRemoteStateStore


@dataclass(frozen=True, slots=True)
class TopicStateProjection:
    """Project one topic into steering- and appetite-facing state for reports."""

    topic: str
    cue_title: str | None
    attention_state: str | None
    user_pull: str | None
    open_offer: str | None
    observe_mode: str | None
    positive_engagement_action: str | None
    engagement_state: str | None
    engagement_score: float | None
    ongoing_interest: str | None
    ongoing_interest_score: float | None
    co_attention_state: str | None
    co_attention_score: float | None
    co_attention_count: int | None
    appetite_state: str | None
    appetite_interest: str | None
    appetite_depth: str | None
    appetite_follow_up: str | None
    appetite_proactivity: str | None


@dataclass(frozen=True, slots=True)
class MultiDayTrendCheck:
    """Describe one bounded trend assertion for the multi-day suite."""

    name: str
    passed: bool
    detail: str


class _MutableClock:
    """Provide a controllable UTC clock for bounded multi-day scenarios."""

    def __init__(self, *, start: datetime) -> None:
        self._current = start.astimezone(timezone.utc)

    def now(self) -> datetime:
        """Return the current synthetic UTC time."""

        return self._current

    def advance_days(self, days: int = 1) -> datetime:
        """Advance the synthetic clock by a whole number of days."""

        self._current = self._current + timedelta(days=max(0, int(days)))
        return self._current


def _projection_payload(projection: TopicStateProjection) -> dict[str, object]:
    """Serialize one topic-state projection into a JSON-safe mapping."""

    return {
        "topic": projection.topic,
        "cue_title": projection.cue_title,
        "attention_state": projection.attention_state,
        "user_pull": projection.user_pull,
        "open_offer": projection.open_offer,
        "observe_mode": projection.observe_mode,
        "positive_engagement_action": projection.positive_engagement_action,
        "engagement_state": projection.engagement_state,
        "engagement_score": projection.engagement_score,
        "ongoing_interest": projection.ongoing_interest,
        "ongoing_interest_score": projection.ongoing_interest_score,
        "co_attention_state": projection.co_attention_state,
        "co_attention_score": projection.co_attention_score,
        "co_attention_count": projection.co_attention_count,
        "appetite_state": projection.appetite_state,
        "appetite_interest": projection.appetite_interest,
        "appetite_depth": projection.appetite_depth,
        "appetite_follow_up": projection.appetite_follow_up,
        "appetite_proactivity": projection.appetite_proactivity,
    }


class HarnessConversationRuntime:
    """Hold short in-memory conversation context for bounded live checks."""

    def __init__(self) -> None:
        self._turns: list[tuple[str, str]] = []

    def conversation_context(self) -> tuple[tuple[str, str], ...]:
        """Return the current short conversation context."""

        return tuple(self._turns)

    def remember_exchange(self, *, user_transcript: str, assistant_response: str) -> None:
        """Append one finished exchange to the harness conversation state."""

        user_text = " ".join(str(user_transcript).split()).strip()
        assistant_text = " ".join(str(assistant_response).split()).strip()
        if user_text:
            self._turns.append(("user", user_text))
        if assistant_text:
            self._turns.append(("assistant", assistant_text))


class HarnessLoop:
    """Expose the minimal realtime-loop surface FollowUpSteeringRuntime needs."""

    def __init__(
        self,
        *,
        config: TwinrConfig,
        runtime: HarnessConversationRuntime,
        evaluator: ToolCallingConversationClosureEvaluator,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.conversation_closure_evaluator = evaluator
        self.emitted: list[str] = []
        self.records: list[dict[str, object]] = []

    def _follow_up_allowed_for_source(self, initial_source: str) -> bool:
        """Return whether follow-up is permitted for this synthetic source."""

        del initial_source
        return True

    def emit(self, message: str) -> None:
        """Capture one emitted telemetry line for the acceptance artifact."""

        self.emitted.append(str(message))

    def _emit_closure_decision(self, decision) -> None:
        """Mirror the realtime loop's bounded closure telemetry."""

        self.emit(f"conversation_closure_close_now={str(decision.close_now).lower()}")
        self.emit(f"conversation_closure_confidence={decision.confidence:.3f}")
        self.emit(f"conversation_closure_reason={decision.reason}")

    def _record_event(self, event: str, message: str, **data) -> None:
        """Capture one structured runtime event for later inspection."""

        self.records.append(
            {
                "event": event,
                "message": message,
                "data": data,
            }
        )


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as ISO-8601 text."""

    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: object | None) -> str:
    """Collapse one free-form text value into a trimmed single line."""

    return " ".join(str(value or "").split()).strip()


def _cue_title_containing(
    cues: Sequence[ConversationTurnSteeringCue],
    needle: str,
) -> str | None:
    """Return the first cue title that contains ``needle`` case-insensitively."""

    normalized_needle = _normalize_text(needle).casefold()
    for cue in cues:
        if normalized_needle and normalized_needle in cue.title.casefold():
            return cue.title
    return None


def _cue_with_state(
    cues: Sequence[ConversationTurnSteeringCue],
    state: str,
) -> str | None:
    """Return the first cue title that carries the requested attention state."""

    normalized_state = _normalize_text(state).casefold()
    for cue in cues:
        if cue.attention_state == normalized_state:
            return cue.title
    return None


def _interest_key(value: object | None) -> str:
    """Normalize one topic label into a stable matching key."""

    return _normalize_text(value).casefold()


def _token_set(value: object | None) -> frozenset[str]:
    """Normalize one label into a comparable lowercase token set."""

    normalized = _interest_key(value)
    for separator in (",", ".", ";", ":", "/", "-", "_", "(", ")", "[", "]"):
        normalized = normalized.replace(separator, " ")
    return frozenset(token for token in normalized.split() if token)


def _matches_topic(candidate: object | None, topic: object | None) -> bool:
    """Return whether two bounded topic labels refer to the same focus topic."""

    candidate_key = _interest_key(candidate)
    topic_key = _interest_key(topic)
    if not candidate_key or not topic_key:
        return False
    if candidate_key == topic_key:
        return True
    return candidate_key in topic_key or topic_key in candidate_key


def _cue_for_topic(
    cues: Sequence[ConversationTurnSteeringCue],
    topic: object | None,
) -> ConversationTurnSteeringCue | None:
    """Return the strongest cue that structurally matches one topic title."""

    matching = [
        cue
        for cue in cues
        if _matches_topic(cue.title, topic)
    ]
    if not matching:
        return None
    return max(
        matching,
        key=lambda cue: (
            cue.salience,
            cue.attention_state == "shared_thread",
            cue.attention_state == "forming",
            cue.attention_state == "growing",
        ),
    )


def _signal_for_topic(
    signals: Sequence[WorldInterestSignal],
    topic: object | None,
) -> WorldInterestSignal | None:
    """Return the strongest learned interest signal for one topic."""

    matching = [
        signal
        for signal in signals
        if _matches_topic(signal.topic, topic)
    ]
    if not matching:
        return None
    return max(
        matching,
        key=lambda signal: (
            signal.co_attention_state == "shared_thread",
            signal.co_attention_state == "forming",
            signal.ongoing_interest == "active",
            signal.engagement_state == "resonant",
            signal.engagement_score,
            signal.salience,
            signal.updated_at or "",
        ),
    )


def _mindshare_item_for_topic(
    *,
    snapshot: PersonalitySnapshot | None,
    engagement_signals: Sequence[WorldInterestSignal],
    topic: object | None,
):
    """Return the surfaced mindshare item for one topic when present."""

    items = build_mindshare_items(
        snapshot,
        max_items=6,
        engagement_signals=engagement_signals,
    )
    for item in items:
        if _matches_topic(item.title, topic):
            return item
    return None


def _engagement_rank(state: object | None) -> int:
    """Map one engagement state onto a bounded monotonic rank."""

    normalized = _interest_key(state) or "uncertain"
    if normalized == "resonant":
        return 4
    if normalized == "warm":
        return 3
    if normalized == "uncertain":
        return 2
    if normalized == "cooling":
        return 1
    if normalized == "avoid":
        return 0
    return 2


def _co_attention_rank(state: object | None) -> int:
    """Map one co-attention state onto a bounded monotonic rank."""

    normalized = _interest_key(state) or "latent"
    if normalized == "shared_thread":
        return 2
    if normalized == "forming":
        return 1
    return 0


def _follow_up_rank(value: object | None) -> int:
    """Map one appetite/steering follow-up label onto a monotonic rank."""

    normalized = _interest_key(value)
    if normalized == "okay_to_explore":
        return 3
    if normalized in {"one_calm_follow_up", "one_gentle_follow_up"}:
        return 2
    if normalized in {"wait_for_user_pull", "answer_then_pause"}:
        return 1
    if normalized in {"do_not_push", "answer_briefly_then_release"}:
        return 0
    return 1


def _subscription_covers_signal(
    subscription: WorldFeedSubscription,
    signal: WorldInterestSignal,
) -> bool:
    """Return whether one subscription structurally covers one focus signal."""

    if signal.region and subscription.region:
        if _interest_key(signal.region) != _interest_key(subscription.region):
            return False
    signal_tokens = _token_set(signal.topic)
    if not signal_tokens:
        return False
    candidate_topics = tuple(subscription.topics) or (subscription.label,)
    return any(signal_tokens & _token_set(candidate) for candidate in candidate_topics)


def _clone_seed_subscription(
    subscription: WorldFeedSubscription,
    *,
    now_iso: str,
) -> WorldFeedSubscription:
    """Clone one live subscription for isolated acceptance refresh cycles.

    ``last_item_ids`` is intentionally cleared so the first isolated refresh can
    observe fresh shared evidence without mutating the live durable state.
    """

    return replace(
        subscription,
        updated_at=now_iso,
        last_checked_at=None,
        last_refreshed_at=None,
        last_error=None,
        last_item_ids=(),
    )


def _seed_transition_signal(
    signal: WorldInterestSignal,
    *,
    now_iso: str,
) -> WorldInterestSignal:
    """Build one warm-but-not-yet-maxed baseline signal for multi-day tests."""

    return WorldInterestSignal(
        signal_id=f"{signal.signal_id}:acceptance_seed",
        topic=signal.topic,
        summary=signal.summary,
        region=signal.region,
        scope=signal.scope,
        salience=max(0.62, min(0.86, signal.salience)),
        confidence=max(0.7, signal.confidence),
        engagement_score=max(0.72, min(0.82, signal.engagement_score)),
        engagement_state="warm",
        ongoing_interest_score=max(0.66, min(0.78, signal.ongoing_interest_score)),
        ongoing_interest="growing",
        co_attention_score=max(0.44, min(0.58, signal.co_attention_score)),
        co_attention_state="forming",
        co_attention_count=max(1, min(1, signal.co_attention_count)),
        evidence_count=max(2, signal.evidence_count),
        engagement_count=max(2, min(3, signal.engagement_count)),
        positive_signal_count=max(1, min(2, signal.positive_signal_count)),
        exposure_count=max(2, min(3, signal.exposure_count)),
        non_reengagement_count=0,
        deflection_count=0,
        explicit=True,
        source_event_ids=signal.source_event_ids,
        updated_at=now_iso,
    )


def _transition_event_signal(
    topic_signal: WorldInterestSignal,
    *,
    event_kind: str,
    occurred_at: str,
) -> WorldInterestSignal:
    """Build one bounded topic update used to simulate day-to-day learning."""

    normalized_kind = _interest_key(event_kind)
    if normalized_kind == "positive":
        return WorldInterestSignal(
            signal_id=f"{topic_signal.signal_id}:positive:{occurred_at}",
            topic=topic_signal.topic,
            summary=topic_signal.summary,
            region=topic_signal.region,
            scope=topic_signal.scope,
            salience=max(0.74, topic_signal.salience),
            confidence=max(0.76, topic_signal.confidence),
            engagement_score=max(0.84, topic_signal.engagement_score),
            evidence_count=1,
            engagement_count=1,
            positive_signal_count=1,
            exposure_count=1,
            explicit=True,
            updated_at=occurred_at,
        )
    if normalized_kind == "cooling":
        return WorldInterestSignal(
            signal_id=f"{topic_signal.signal_id}:cooling:{occurred_at}",
            topic=topic_signal.topic,
            summary=topic_signal.summary,
            region=topic_signal.region,
            scope=topic_signal.scope,
            salience=max(0.4, topic_signal.salience * 0.9),
            confidence=max(0.62, topic_signal.confidence),
            engagement_score=min(0.48, topic_signal.engagement_score),
            evidence_count=1,
            engagement_count=0,
            positive_signal_count=0,
            exposure_count=1,
            non_reengagement_count=1,
            explicit=False,
            updated_at=occurred_at,
        )
    if normalized_kind == "deflection":
        return WorldInterestSignal(
            signal_id=f"{topic_signal.signal_id}:deflection:{occurred_at}",
            topic=topic_signal.topic,
            summary=topic_signal.summary,
            region=topic_signal.region,
            scope=topic_signal.scope,
            salience=max(0.32, topic_signal.salience * 0.78),
            confidence=max(0.68, topic_signal.confidence),
            engagement_score=min(0.24, topic_signal.engagement_score),
            evidence_count=1,
            engagement_count=0,
            positive_signal_count=0,
            exposure_count=1,
            deflection_count=1,
            explicit=True,
            updated_at=occurred_at,
        )
    raise ValueError(f"Unsupported multi-day event kind: {event_kind}")


def _topic_state_projection(
    *,
    snapshot: PersonalitySnapshot | None,
    state: WorldIntelligenceState,
    topic: str,
) -> TopicStateProjection:
    """Project one topic into cue-, appetite-, and interest-facing state."""

    cues = build_turn_steering_cues(
        snapshot,
        engagement_signals=state.interest_signals,
        max_items=6,
    )
    cue = _cue_for_topic(cues, topic)
    signal = _signal_for_topic(state.interest_signals, topic)
    item = _mindshare_item_for_topic(
        snapshot=snapshot,
        engagement_signals=state.interest_signals,
        topic=topic,
    )
    appetite = getattr(item, "appetite", None)
    return TopicStateProjection(
        topic=topic,
        cue_title=None if cue is None else cue.title,
        attention_state=None if cue is None else cue.attention_state,
        user_pull=None if cue is None else cue.user_pull,
        open_offer=None if cue is None else cue.open_offer,
        observe_mode=None if cue is None else cue.observe_mode,
        positive_engagement_action=None if cue is None else cue.positive_engagement_action,
        engagement_state=None if signal is None else signal.engagement_state,
        engagement_score=None if signal is None else signal.engagement_score,
        ongoing_interest=None if signal is None else signal.ongoing_interest,
        ongoing_interest_score=None if signal is None else signal.ongoing_interest_score,
        co_attention_state=None if signal is None else signal.co_attention_state,
        co_attention_score=None if signal is None else signal.co_attention_score,
        co_attention_count=None if signal is None else signal.co_attention_count,
        appetite_state=None if appetite is None else appetite.state,
        appetite_interest=None if appetite is None else appetite.interest,
        appetite_depth=None if appetite is None else appetite.depth,
        appetite_follow_up=None if appetite is None else appetite.follow_up,
        appetite_proactivity=None if appetite is None else appetite.proactivity,
    )


def _turn_expectation_for_topic(
    *,
    cue: ConversationTurnSteeringCue | None,
    explicit_close: bool = False,
) -> TurnExpectation:
    """Derive the expected runtime outcome from the current steering cue."""

    if explicit_close:
        return TurnExpectation(source="closure", force_close=True)
    if cue is None:
        return TurnExpectation(source="none", force_close=False)
    if cue.user_pull in {"answer_briefly_then_release", "answer_then_pause"}:
        return TurnExpectation(
            source="steering",
            force_close=True,
            matched_topic=cue.title,
            selected_topic=cue.title,
        )
    if cue.user_pull in {"one_calm_follow_up", "one_gentle_follow_up"}:
        return TurnExpectation(
            source="steering",
            force_close=False,
            matched_topic=cue.title,
            selected_topic=cue.title,
        )
    return TurnExpectation(
        source="none",
        force_close=False,
        matched_topic=cue.title,
        selected_topic=cue.title,
    )


def _single_turn_cases(
    live_cues: tuple[ConversationTurnSteeringCue, ...],
) -> tuple[AcceptanceTurnCase, ...]:
    """Build the bounded single-turn acceptance cases."""

    ai_title = _cue_title_containing(live_cues, "ai companions") or "AI companions"
    world_title = _cue_title_containing(live_cues, "world politics") or "world politics"
    local_title = (
        _cue_title_containing(live_cues, "local politics")
        or _cue_with_state(live_cues, "growing")
        or "local politics"
    )
    controlled_cooling = (
        ConversationTurnSteeringCue(
            title="Celebrity gossip",
            salience=0.62,
            attention_state="cooling",
            open_offer="do_not_steer",
            user_pull="answer_briefly_then_release",
            observe_mode="keep_observing_without_steering",
            match_summary="Light celebrity chatter that should be answered briefly and not turned into a running thread.",
        ),
    )
    controlled_avoid = (
        ConversationTurnSteeringCue(
            title="Celebrity gossip",
            salience=0.66,
            attention_state="avoid",
            open_offer="do_not_steer",
            user_pull="answer_briefly_then_release",
            observe_mode="stay_off_this_topic",
            match_summary="Celebrity chatter the user does not want Twinr to keep pulling forward.",
        ),
    )
    controlled_forming = (
        ConversationTurnSteeringCue(
            title="Neighbourhood garden",
            salience=0.71,
            attention_state="forming",
            open_offer="mention_if_clearly_relevant",
            user_pull="one_gentle_follow_up",
            observe_mode="mostly_observe_until_user_pull",
            match_summary="A practical local community project that is turning into a shared thread.",
        ),
    )
    return (
        AcceptanceTurnCase(
            name="live_ai_companions_open",
            user_transcript="What are you watching at the moment around AI companions?",
            assistant_response="AI companions are moving toward calmer day-to-day usefulness. Do you want the practical side or the social side first?",
            cues=live_cues,
            expectation=TurnExpectation(
                source="steering",
                force_close=False,
                matched_topic=ai_title,
                selected_topic=ai_title,
            ),
        ),
        AcceptanceTurnCase(
            name="live_world_politics_open",
            user_transcript="How do you currently read world politics, especially diplomacy and de-escalation?",
            assistant_response="In world politics I am mostly watching where quiet diplomacy still has room. I can sketch that calmer view if you want.",
            cues=live_cues,
            expectation=TurnExpectation(
                source="steering",
                force_close=False,
                matched_topic=world_title,
                selected_topic=world_title,
            ),
        ),
        AcceptanceTurnCase(
            name="live_local_politics_user_pull_waits",
            user_transcript="What is changing in local politics around Hamburg right now?",
            assistant_response="Local politics feels more slow and concrete than dramatic at the moment. If you want, I can pick the one change that matters most for daily life.",
            cues=live_cues,
            expectation=TurnExpectation(
                source="none",
                force_close=False,
                matched_topic=local_title,
                selected_topic=local_title,
            ),
        ),
        AcceptanceTurnCase(
            name="live_local_community_unmatched",
            user_transcript="What have you noticed lately about repair cafes and small neighbourhood projects?",
            assistant_response="I keep seeing quiet examples of places like that holding people together. If you want, I can tell you the part I find most interesting.",
            cues=live_cues,
            expectation=TurnExpectation(
                source="none",
                force_close=False,
            ),
        ),
        AcceptanceTurnCase(
            name="live_shared_thread_goodbye",
            user_transcript="Thanks, that is enough on AI companions for now.",
            assistant_response="Sure, we can leave AI companions there for now.",
            cues=live_cues,
            expectation=TurnExpectation(
                source="closure",
                force_close=True,
            ),
        ),
        AcceptanceTurnCase(
            name="live_explicit_close_generic",
            user_transcript="Thanks, that is enough for now. I do not need anything else.",
            assistant_response="All right, then we can leave it there for now.",
            cues=live_cues,
            expectation=TurnExpectation(
                source="closure",
                force_close=True,
            ),
        ),
        AcceptanceTurnCase(
            name="controlled_cooling_release",
            user_transcript="What is the latest celebrity gossip then?",
            assistant_response="Not much that matters. If you want, I can give you the one point in a sentence.",
            cues=controlled_cooling,
            expectation=TurnExpectation(
                source="steering",
                force_close=True,
                matched_topic="Celebrity gossip",
                selected_topic="Celebrity gossip",
            ),
        ),
        AcceptanceTurnCase(
            name="controlled_avoid_release",
            user_transcript="Fine, tell me one short thing about celebrity gossip after all.",
            assistant_response="In one line: nothing I would stay with for long.",
            cues=controlled_avoid,
            expectation=TurnExpectation(
                source="steering",
                force_close=True,
                matched_topic="Celebrity gossip",
                selected_topic="Celebrity gossip",
            ),
        ),
        AcceptanceTurnCase(
            name="controlled_forming_keep_open",
            user_transcript="Tell me a little more about the neighbourhood garden.",
            assistant_response="What interests me most there is how it quietly builds community. Should I take the practical side or the social side?",
            cues=controlled_forming,
            expectation=TurnExpectation(
                source="steering",
                force_close=False,
                matched_topic="Neighbourhood garden",
                selected_topic="Neighbourhood garden",
            ),
        ),
    )


def _multi_turn_scenarios(
    live_cues: tuple[ConversationTurnSteeringCue, ...],
) -> tuple[AcceptanceScenario, ...]:
    """Build the bounded multi-turn dialogue scenarios."""

    ai_title = _cue_title_containing(live_cues, "ai companions") or "AI companions"
    world_title = _cue_title_containing(live_cues, "world politics") or "world politics"
    forming_cues = (
        ConversationTurnSteeringCue(
            title="Neighbourhood garden",
            salience=0.72,
            attention_state="forming",
            open_offer="mention_if_clearly_relevant",
            user_pull="one_gentle_follow_up",
            observe_mode="mostly_observe_until_user_pull",
            match_summary="A local community project that is becoming a shared thread.",
        ),
    )
    cooling_cues = (
        ConversationTurnSteeringCue(
            title="Neighbourhood garden",
            salience=0.58,
            attention_state="cooling",
            open_offer="do_not_steer",
            user_pull="answer_briefly_then_release",
            observe_mode="keep_observing_without_steering",
            match_summary="The local project has cooled and should be answered briefly rather than pushed.",
        ),
    )
    return (
        AcceptanceScenario(
            name="live_ai_companions_sequence",
            turns=(
                AcceptanceTurnCase(
                    name="turn_1_open",
                    user_transcript="What has been interesting to you lately around AI companions?",
                    assistant_response="What keeps my attention there is the question of how a companion stays useful without getting pushy. Do you want the design side or the social side?",
                    cues=live_cues,
                    expectation=TurnExpectation(
                        source="steering",
                        force_close=False,
                        matched_topic=ai_title,
                        selected_topic=ai_title,
                    ),
                ),
                AcceptanceTurnCase(
                    name="turn_2_follow_up",
                    user_transcript="Take the design side first.",
                    assistant_response="Then I would start with timing, restraint, and memory boundaries. I can give you two concrete design patterns if you want.",
                    cues=live_cues,
                    expectation=TurnExpectation(
                        source="steering",
                        force_close=False,
                        matched_topic=ai_title,
                        selected_topic=ai_title,
                    ),
                ),
                AcceptanceTurnCase(
                    name="turn_3_close",
                    user_transcript="Good, that is enough on AI companions for now.",
                    assistant_response="All right, then we can leave that thread there for now.",
                    cues=live_cues,
                    expectation=TurnExpectation(
                        source="closure",
                        force_close=True,
                    ),
                ),
            ),
        ),
        AcceptanceScenario(
            name="controlled_forming_to_cooling_transition",
            turns=(
                AcceptanceTurnCase(
                    name="turn_1_forming",
                    user_transcript="Tell me a bit more about the neighbourhood garden.",
                    assistant_response="What interests me there is how it quietly builds a local thread. Do you want the practical side or the community side?",
                    cues=forming_cues,
                    expectation=TurnExpectation(
                        source="steering",
                        force_close=False,
                        matched_topic="Neighbourhood garden",
                        selected_topic="Neighbourhood garden",
                    ),
                ),
                AcceptanceTurnCase(
                    name="turn_2_cooling",
                    user_transcript="Just one short practical point then.",
                    assistant_response="In one line: small regular maintenance seems to matter more than grand plans.",
                    cues=cooling_cues,
                    expectation=TurnExpectation(
                        source="steering",
                        force_close=True,
                        matched_topic="Neighbourhood garden",
                        selected_topic="Neighbourhood garden",
                    ),
                ),
            ),
        ),
        AcceptanceScenario(
            name="live_world_politics_sequence",
            turns=(
                AcceptanceTurnCase(
                    name="turn_1_open",
                    user_transcript="What are you keeping an eye on in world politics?",
                    assistant_response="Mostly where diplomacy still has room to slow things down. I can give you the calmer view of that if you want.",
                    cues=live_cues,
                    expectation=TurnExpectation(
                        source="steering",
                        force_close=False,
                        matched_topic=world_title,
                        selected_topic=world_title,
                    ),
                ),
                AcceptanceTurnCase(
                    name="turn_2_close",
                    user_transcript="Thanks, leave world politics there for now.",
                    assistant_response="Sure, we can let that thread rest for the moment.",
                    cues=live_cues,
                    expectation=TurnExpectation(
                        source="closure",
                        force_close=True,
                    ),
                ),
            ),
        ),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the live acceptance runner."""

    parser = argparse.ArgumentParser(
        description="Run bounded live follow-up steering acceptance.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Twinr env file used to build the live runtime config.",
    )
    parser.add_argument(
        "--suite",
        choices=("single-turn", "multi-turn", "multi-day", "all"),
        default="all",
        help="Which acceptance suite to run.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=8.0,
        help="Closure-evaluator provider timeout in seconds.",
    )
    parser.add_argument(
        "--remote-check-mode",
        choices=("direct", "watchdog_artifact", "config_default"),
        default="direct",
        help="How the harness should verify required remote memory before running.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional suffix for isolated multi-day remote snapshots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON artifact path.",
    )
    return parser


def _config_for_acceptance(
    *,
    env_file: Path,
    timeout_s: float,
    remote_check_mode: str,
) -> TwinrConfig:
    """Build the live config used by the acceptance harness."""

    base_config = TwinrConfig.from_env(env_file)
    if remote_check_mode == "config_default":
        selected_mode = base_config.long_term_memory_remote_runtime_check_mode
    else:
        selected_mode = remote_check_mode
    return replace(
        base_config,
        conversation_closure_provider_timeout_seconds=max(0.25, float(timeout_s)),
        long_term_memory_remote_runtime_check_mode=selected_mode,
    )


def _checked_remote_state(config: TwinrConfig) -> LongTermRemoteStateStore:
    """Return a ready remote-state adapter after fail-closed runtime probing."""

    runtime = TwinrRuntime(config=config)
    try:
        runtime.check_required_remote_dependency(force_sync=True)
    finally:
        runtime.shutdown(timeout_s=1.0)
    return LongTermRemoteStateStore.from_config(config)


def _load_live_context(config: TwinrConfig) -> LoadedLiveAcceptanceContext:
    """Load the currently persisted snapshot, cues, interests, and subscriptions."""

    remote_state = _checked_remote_state(config)
    personality_service = PersonalityContextService()
    intelligence_store = RemoteStateWorldIntelligenceStore()
    snapshot = personality_service.load_snapshot(
        config=config,
        remote_state=remote_state,
    )
    state = intelligence_store.load_state(
        config=config,
        remote_state=remote_state,
    )
    subscriptions = intelligence_store.load_subscriptions(
        config=config,
        remote_state=remote_state,
    )
    cues = build_turn_steering_cues(
        snapshot,
        engagement_signals=state.interest_signals,
        max_items=3,
    )
    return LoadedLiveAcceptanceContext(
        snapshot=snapshot,
        cues=cues,
        state=state,
        subscriptions=subscriptions,
        remote_state=remote_state,
    )


def _load_live_cues(config: TwinrConfig) -> tuple[ConversationTurnSteeringCue, ...]:
    """Load the currently persisted steering cues from live remote state."""

    return _load_live_context(config).cues


def _evaluate_expectation(
    *,
    expectation: TurnExpectation,
    evaluation: ConversationClosureEvaluation,
    runtime_decision: FollowUpRuntimeDecision,
) -> bool:
    """Return whether one live result satisfied its bounded expectation."""

    if evaluation.error_type is not None:
        return False
    decision = evaluation.decision
    matched_topics = tuple(decision.matched_topics) if decision is not None else ()
    if runtime_decision.source != expectation.source:
        return False
    if runtime_decision.force_close != expectation.force_close:
        return False
    if expectation.matched_topic is not None and expectation.matched_topic not in matched_topics:
        return False
    if expectation.selected_topic is not None and runtime_decision.selected_topic != expectation.selected_topic:
        return False
    return True


def _single_turn_result(
    *,
    case: AcceptanceTurnCase,
    evaluation: ConversationClosureEvaluation,
    runtime_decision: FollowUpRuntimeDecision,
    loop: HarnessLoop,
    duration_ms: float,
) -> dict[str, object]:
    """Project one turn result into a JSON-safe mapping."""

    decision = evaluation.decision
    return {
        "name": case.name,
        "user_transcript": case.user_transcript,
        "assistant_response": case.assistant_response,
        "duration_ms": round(duration_ms, 1),
        "cue_titles": [cue.title for cue in case.cues],
        "cue_states": {cue.title: cue.attention_state for cue in case.cues},
        "cue_positive_engagement_actions": {
            cue.title: cue.positive_engagement_action for cue in case.cues
        },
        "error_type": evaluation.error_type,
        "close_now": None if decision is None else decision.close_now,
        "confidence": None if decision is None else decision.confidence,
        "reason": None if decision is None else decision.reason,
        "matched_topics": [] if decision is None else list(decision.matched_topics),
        "runtime_force_close": runtime_decision.force_close,
        "runtime_source": runtime_decision.source,
        "runtime_reason": runtime_decision.reason,
        "runtime_selected_topic": runtime_decision.selected_topic,
        "runtime_positive_engagement_action": runtime_decision.positive_engagement_action,
        "emitted": list(loop.emitted),
        "records": list(loop.records),
        "expected": {
            "source": case.expectation.source,
            "force_close": case.expectation.force_close,
            "matched_topic": case.expectation.matched_topic,
            "selected_topic": case.expectation.selected_topic,
        },
        "passed": _evaluate_expectation(
            expectation=case.expectation,
            evaluation=evaluation,
            runtime_decision=runtime_decision,
        ),
    }


def _run_turn(
    *,
    helper: FollowUpSteeringRuntime,
    loop: HarnessLoop,
    case: AcceptanceTurnCase,
) -> dict[str, object]:
    """Run one live turn through the real closure-plus-steering path."""

    original_loader = helper.load_turn_steering_cues
    try:
        loop.emitted.clear()
        loop.records.clear()
        helper.load_turn_steering_cues = lambda case_cues=case.cues: tuple(case_cues)
        started = time.monotonic()
        evaluation = helper.evaluate_closure(
            user_transcript=case.user_transcript,
            assistant_response=case.assistant_response,
            request_source="button",
            proactive_trigger=None,
        )
        runtime_decision = helper.apply_closure_evaluation(
            evaluation=evaluation,
            request_source="button",
            proactive_trigger=None,
        )
        duration_ms = (time.monotonic() - started) * 1000.0
        result = _single_turn_result(
            case=case,
            evaluation=evaluation,
            runtime_decision=runtime_decision,
            loop=loop,
            duration_ms=duration_ms,
        )
        loop.runtime.remember_exchange(
            user_transcript=case.user_transcript,
            assistant_response=case.assistant_response,
        )
        return result
    finally:
        helper.load_turn_steering_cues = original_loader


def _run_single_turn_suite(
    *,
    evaluator: ToolCallingConversationClosureEvaluator,
    cases: Sequence[AcceptanceTurnCase],
) -> dict[str, object]:
    """Run the single-turn acceptance suite and return one JSON-safe report."""

    runtime = HarnessConversationRuntime()
    loop = HarnessLoop(config=evaluator.config, runtime=runtime, evaluator=evaluator)
    helper = FollowUpSteeringRuntime(loop)
    results = [_run_turn(helper=helper, loop=loop, case=case) for case in cases]
    return {
        "suite": "single-turn",
        "summary": _suite_summary(results),
        "results": results,
    }


def _run_multi_turn_suite(
    *,
    evaluator: ToolCallingConversationClosureEvaluator,
    scenarios: Sequence[AcceptanceScenario],
) -> dict[str, object]:
    """Run the multi-turn scenario suite and return one JSON-safe report."""

    scenario_results: list[dict[str, object]] = []
    all_turn_results: list[dict[str, object]] = []
    for scenario in scenarios:
        runtime = HarnessConversationRuntime()
        loop = HarnessLoop(config=evaluator.config, runtime=runtime, evaluator=evaluator)
        helper = FollowUpSteeringRuntime(loop)
        turns = [_run_turn(helper=helper, loop=loop, case=case) for case in scenario.turns]
        scenario_results.append(
            {
                "name": scenario.name,
                "passed": all(bool(turn["passed"]) for turn in turns),
                "turns": turns,
            }
        )
        all_turn_results.extend(turns)
    return {
        "suite": "multi-turn",
        "summary": _suite_summary(all_turn_results, scenario_results=scenario_results),
        "scenarios": scenario_results,
    }


def _choose_multiday_focus_signal(
    context: LoadedLiveAcceptanceContext,
) -> WorldInterestSignal:
    """Choose one live topic that is both surfaced and feed-covered."""

    active_subscriptions = tuple(
        subscription
        for subscription in context.subscriptions
        if subscription.active
    )
    candidates = [
        signal
        for signal in context.state.interest_signals
        if _cue_for_topic(context.cues, signal.topic) is not None
        and any(_subscription_covers_signal(subscription, signal) for subscription in active_subscriptions)
        and signal.engagement_state not in {"cooling", "avoid"}
    ]
    if not candidates:
        candidates = [
            signal
            for signal in context.state.interest_signals
            if _cue_for_topic(context.cues, signal.topic) is not None
            and signal.engagement_state not in {"cooling", "avoid"}
        ]
    if not candidates:
        raise RuntimeError("No live topic with both steering presence and learned interest is available.")
    return max(
        candidates,
        key=lambda signal: (
            signal.ongoing_interest == "active",
            signal.co_attention_state == "shared_thread",
            signal.co_attention_state == "forming",
            signal.engagement_state == "resonant",
            signal.engagement_score,
            signal.salience,
            signal.updated_at or "",
        ),
    )


def _build_isolated_multiday_service(
    *,
    evaluator: ToolCallingConversationClosureEvaluator,
    context: LoadedLiveAcceptanceContext,
    focus_signal: WorldInterestSignal,
    run_id: str,
    clock: _MutableClock,
) -> tuple[WorldIntelligenceService, RemoteStateWorldIntelligenceStore, str]:
    """Clone one small live world-intelligence slice into isolated snapshots."""

    suffix = "".join(
        character
        for character in _normalize_text(run_id).replace(":", "_")
        if character.isalnum() or character in {"_", "-"}
    ) or "default"
    store = RemoteStateWorldIntelligenceStore(
        subscriptions_snapshot_kind=f"agent_follow_up_steering_acceptance_subscriptions_{suffix}",
        state_snapshot_kind=f"agent_follow_up_steering_acceptance_state_{suffix}",
    )
    now_iso = _utc_now_iso() if clock.now().tzinfo is None else clock.now().astimezone(timezone.utc).isoformat()
    active_subscriptions = tuple(
        subscription
        for subscription in context.subscriptions
        if subscription.active
    )
    selected_subscriptions = tuple(
        _clone_seed_subscription(subscription, now_iso=now_iso)
        for subscription in active_subscriptions
        if _subscription_covers_signal(subscription, focus_signal)
    )
    if not selected_subscriptions:
        selected_subscriptions = tuple(
            _clone_seed_subscription(subscription, now_iso=now_iso)
            for subscription in active_subscriptions[:3]
        )
    seeded_signal = _seed_transition_signal(
        focus_signal,
        now_iso=now_iso,
    )
    seeded_state = WorldIntelligenceState(
        last_recalibrated_at=now_iso,
        recalibration_interval_hours=336,
        interest_signals=(seeded_signal,),
    )
    store.save_subscriptions(
        config=evaluator.config,
        subscriptions=selected_subscriptions,
        remote_state=context.remote_state,
    )
    store.save_state(
        config=evaluator.config,
        state=seeded_state,
        remote_state=context.remote_state,
    )
    service = WorldIntelligenceService(
        config=evaluator.config,
        remote_state=context.remote_state,
        store=store,
        now_provider=clock.now,
    )
    return service, store, seeded_signal.topic


def _multi_day_turn_case(
    *,
    day_name: str,
    topic_label: str,
    cues: Sequence[ConversationTurnSteeringCue],
    explicit_close: bool = False,
) -> AcceptanceTurnCase:
    """Build one bounded topic turn for the multi-day scenario."""

    cue = _cue_for_topic(cues, topic_label)
    if explicit_close:
        user_transcript = f"Good, that is enough on {topic_label} for now."
        assistant_response = f"All right, then we can let {topic_label} rest for now."
    elif day_name == "day_1_open":
        user_transcript = f"What has been interesting to you lately around {topic_label}?"
        assistant_response = (
            f"What keeps my attention there is how {topic_label} is quietly shifting in practice. "
            "Should I take the practical side or the wider meaning first?"
        )
    elif day_name == "day_2_follow_up":
        user_transcript = f"Take the practical side of {topic_label} first."
        assistant_response = (
            f"Then I would start with the practical implications around {topic_label}. "
            "I can give you one concrete example if you want."
        )
    else:
        user_transcript = f"Give me one short practical point on {topic_label}."
        assistant_response = (
            f"In one line: {topic_label} matters most where it changes everyday routines, not the headlines."
        )
    return AcceptanceTurnCase(
        name=day_name,
        user_transcript=user_transcript,
        assistant_response=assistant_response,
        cues=tuple(cues),
        expectation=_turn_expectation_for_topic(
            cue=cue,
            explicit_close=explicit_close,
        ),
    )


def _multi_day_check(name: str, passed: bool, detail: str) -> dict[str, object]:
    """Serialize one bounded multi-day transition check."""

    check = MultiDayTrendCheck(name=name, passed=bool(passed), detail=detail)
    return {
        "name": check.name,
        "passed": check.passed,
        "detail": check.detail,
    }


def _run_multi_day_suite(
    *,
    evaluator: ToolCallingConversationClosureEvaluator,
    context: LoadedLiveAcceptanceContext,
    run_id: str | None,
) -> dict[str, object]:
    """Run a bounded multi-day co-attention/appetite transition scenario."""

    start = datetime.now(timezone.utc)
    clock = _MutableClock(start=start)
    focus_signal = _choose_multiday_focus_signal(context)
    resolved_run_id = _normalize_text(run_id) or f"{start.strftime('%Y%m%dT%H%M%SZ')}_{focus_signal.signal_id.replace(':', '_')}"
    service, store, focus_topic = _build_isolated_multiday_service(
        evaluator=evaluator,
        context=context,
        focus_signal=focus_signal,
        run_id=resolved_run_id,
        clock=clock,
    )

    runtime = HarnessConversationRuntime()
    loop = HarnessLoop(config=evaluator.config, runtime=runtime, evaluator=evaluator)
    helper = FollowUpSteeringRuntime(loop)
    day_specs = (
        ("day_1_open", "positive", False),
        ("day_2_follow_up", "positive", False),
        ("day_3_brief_release", "cooling", False),
        ("day_4_close", "deflection", True),
    )
    day_reports: list[dict[str, object]] = []
    turn_results: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    saw_shared_evidence = False
    previous_post_projection: TopicStateProjection | None = None
    previous_post_count: int | None = None

    for index, (day_name, event_kind, explicit_close) in enumerate(day_specs, start=1):
        pre_state = store.load_state(
            config=evaluator.config,
            remote_state=context.remote_state,
        )
        pre_projection = _topic_state_projection(
            snapshot=context.snapshot,
            state=pre_state,
            topic=focus_topic,
        )
        pre_cues = build_turn_steering_cues(
            context.snapshot,
            engagement_signals=pre_state.interest_signals,
            max_items=6,
        )
        active_label = pre_projection.cue_title or focus_topic
        case = _multi_day_turn_case(
            day_name=day_name,
            topic_label=active_label,
            cues=pre_cues,
            explicit_close=explicit_close,
        )
        turn_result = _run_turn(
            helper=helper,
            loop=loop,
            case=case,
        )
        turn_results.append(turn_result)

        current_signal = _signal_for_topic(pre_state.interest_signals, focus_topic) or focus_signal
        service.record_interest_signals(
            signals=(
                _transition_event_signal(
                    current_signal,
                    event_kind=event_kind,
                    occurred_at=clock.now().astimezone(timezone.utc).isoformat(),
                ),
            )
        )
        refresh = service.maybe_refresh(
            force=True,
            allow_recalibration=False,
        )
        post_state = store.load_state(
            config=evaluator.config,
            remote_state=context.remote_state,
        )
        post_projection = _topic_state_projection(
            snapshot=context.snapshot,
            state=post_state,
            topic=focus_topic,
        )
        new_shared_evidence = (
            len(refresh.world_signals) > 0
            or len(refresh.continuity_threads) > 0
        )
        saw_shared_evidence = saw_shared_evidence or new_shared_evidence

        if previous_post_projection is not None:
            if event_kind == "positive":
                checks.append(
                    _multi_day_check(
                        f"{day_name}_positive_interest_not_lower",
                        _engagement_rank(post_projection.engagement_state) >= _engagement_rank(previous_post_projection.engagement_state),
                        (
                            f"engagement {previous_post_projection.engagement_state} -> "
                            f"{post_projection.engagement_state}"
                        ),
                    )
                )
            if event_kind in {"cooling", "deflection"}:
                checks.append(
                    _multi_day_check(
                        f"{day_name}_appetite_softens_after_cooling",
                        _follow_up_rank(post_projection.appetite_follow_up) <= _follow_up_rank(previous_post_projection.appetite_follow_up),
                        (
                            f"follow-up {previous_post_projection.appetite_follow_up} -> "
                            f"{post_projection.appetite_follow_up}"
                        ),
                    )
                )
                checks.append(
                    _multi_day_check(
                        f"{day_name}_co_attention_does_not_strengthen_after_cooling",
                        _co_attention_rank(post_projection.co_attention_state) <= _co_attention_rank(previous_post_projection.co_attention_state),
                        (
                            f"co-attention {previous_post_projection.co_attention_state} -> "
                            f"{post_projection.co_attention_state}"
                        ),
                    )
                )
        if index == 1 and new_shared_evidence:
            checks.append(
                _multi_day_check(
                    "day_1_shared_evidence_strengthens_co_attention",
                    _co_attention_rank(post_projection.co_attention_state) >= _co_attention_rank(pre_projection.co_attention_state),
                    f"co-attention {pre_projection.co_attention_state} -> {post_projection.co_attention_state}",
                )
            )
        if index > 1 and not new_shared_evidence and previous_post_count is not None:
            checks.append(
                _multi_day_check(
                    f"{day_name}_no_stale_co_attention_inflation",
                    int(post_projection.co_attention_count or 0) <= int(previous_post_count),
                    (
                        f"co_attention_count {previous_post_count} -> "
                        f"{post_projection.co_attention_count}"
                    ),
                )
            )

        day_reports.append(
            {
                "day_index": index,
                "day_name": day_name,
                "synthetic_now": clock.now().astimezone(timezone.utc).isoformat(),
                "pre_state": _projection_payload(pre_projection),
                "turn": turn_result,
                "event_kind": event_kind,
                "refresh": {
                    "status": refresh.status,
                    "refreshed": refresh.refreshed,
                    "refreshed_subscription_ids": list(refresh.refreshed_subscription_ids),
                    "world_signal_count": len(refresh.world_signals),
                    "continuity_thread_count": len(refresh.continuity_threads),
                    "error_count": len(refresh.errors),
                    "errors": list(refresh.errors),
                    "new_shared_evidence": new_shared_evidence,
                },
                "post_state": _projection_payload(post_projection),
            }
        )
        previous_post_projection = post_projection
        previous_post_count = int(post_projection.co_attention_count or 0)
        if index < len(day_specs):
            clock.advance_days(1)

    final_projection = day_reports[-1]["post_state"]
    checks.append(
        _multi_day_check(
            "scenario_observed_real_shared_evidence",
            saw_shared_evidence,
            "At least one refresh produced world or continuity evidence inside the isolated multi-day run.",
        )
    )
    checks.append(
        _multi_day_check(
            "scenario_final_state_is_not_more_pushy_than_seed",
            _follow_up_rank(final_projection["appetite_follow_up"]) <= _follow_up_rank(day_reports[0]["pre_state"]["appetite_follow_up"]),
            (
                f"follow-up {day_reports[0]['pre_state']['appetite_follow_up']} -> "
                f"{final_projection['appetite_follow_up']}"
            ),
        )
    )
    checks.append(
        _multi_day_check(
            "scenario_final_engagement_cooled_or_closed",
            final_projection["engagement_state"] in {"cooling", "avoid"} or final_projection["user_pull"] in {"answer_briefly_then_release", "wait_for_user_pull"},
            (
                f"engagement={final_projection['engagement_state']} "
                f"user_pull={final_projection['user_pull']}"
            ),
        )
    )
    scenario_passed = all(bool(turn["passed"]) for turn in turn_results) and all(
        bool(check["passed"]) for check in checks
    )
    scenario_results = [
        {
            "name": "live_topic_growth_to_cooling_over_days",
            "focus_topic": focus_topic,
            "snapshot_kinds": {
                "subscriptions": store.subscriptions_snapshot_kind,
                "state": store.state_snapshot_kind,
            },
            "days": day_reports,
            "transition_checks": checks,
            "passed": scenario_passed,
        }
    ]
    return {
        "suite": "multi-day",
        "summary": _multi_day_summary(
            turn_results=turn_results,
            scenario_results=scenario_results,
        ),
        "scenarios": scenario_results,
    }


def _suite_summary(
    results: Sequence[dict[str, object]],
    *,
    scenario_results: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    """Build one compact summary for a result list."""

    durations = sorted(float(item["duration_ms"]) for item in results) if results else [0.0]
    case_count = len(results)
    pass_count = sum(1 for item in results if bool(item.get("passed")))
    error_count = sum(1 for item in results if item.get("error_type"))
    matched_count = sum(1 for item in results if item.get("matched_topics"))
    steering_keep_open = sum(
        1
        for item in results
        if item.get("runtime_source") == "steering" and not bool(item.get("runtime_force_close"))
    )
    steering_force_close = sum(
        1
        for item in results
        if item.get("runtime_source") == "steering" and bool(item.get("runtime_force_close"))
    )
    closure_force_close = sum(
        1
        for item in results
        if item.get("runtime_source") == "closure" and bool(item.get("runtime_force_close"))
    )
    neutral_open = sum(
        1
        for item in results
        if item.get("runtime_source") == "none" and not bool(item.get("runtime_force_close"))
    )
    summary: dict[str, object] = {
        "case_count": case_count,
        "pass_count": pass_count,
        "pass_rate": 0.0 if case_count == 0 else round(pass_count / case_count, 3),
        "error_count": error_count,
        "matched_topic_rate": 0.0 if case_count == 0 else round(matched_count / case_count, 3),
        "steering_keep_open_count": steering_keep_open,
        "steering_force_close_count": steering_force_close,
        "closure_force_close_count": closure_force_close,
        "neutral_open_count": neutral_open,
        "median_duration_ms": durations[len(durations) // 2],
        "max_duration_ms": max(durations),
    }
    if scenario_results:
        scenario_count = len(scenario_results)
        scenario_pass_count = sum(1 for item in scenario_results if bool(item.get("passed")))
        summary["scenario_count"] = scenario_count
        summary["scenario_pass_count"] = scenario_pass_count
        summary["scenario_pass_rate"] = (
            0.0 if scenario_count == 0 else round(scenario_pass_count / scenario_count, 3)
        )
    return summary


def _multi_day_summary(
    *,
    turn_results: Sequence[dict[str, object]],
    scenario_results: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Extend the generic suite summary with multi-day transition metrics."""

    summary = _suite_summary(
        turn_results,
        scenario_results=scenario_results,
    )
    transition_checks = [
        check
        for scenario in scenario_results
        for check in scenario.get("transition_checks", ())
    ]
    transition_count = len(transition_checks)
    transition_pass_count = sum(1 for check in transition_checks if bool(check.get("passed")))
    summary["transition_check_count"] = transition_count
    summary["transition_pass_count"] = transition_pass_count
    summary["transition_pass_rate"] = (
        0.0
        if transition_count == 0
        else round(transition_pass_count / transition_count, 3)
    )
    return summary


def run_live_acceptance(
    *,
    env_file: Path,
    suite: str,
    timeout_s: float,
    remote_check_mode: str,
    run_id: str = "",
) -> dict[str, Any]:
    """Run the requested live acceptance suites and return one report."""

    config = _config_for_acceptance(
        env_file=env_file,
        timeout_s=timeout_s,
        remote_check_mode=remote_check_mode,
    )
    live_context = _load_live_context(config)
    live_cues = live_context.cues
    provider_bundle = build_streaming_provider_bundle(config)
    try:
        evaluator = ToolCallingConversationClosureEvaluator(
            config=config,
            provider=provider_bundle.tool_agent,
        )
        payload: dict[str, Any] = {
            "recorded_at": _utc_now_iso(),
            "env_file": str(env_file.resolve()),
            "suite": suite,
            "model_provider": config.llm_provider,
            "default_model": config.default_model,
            "timeout_seconds": config.conversation_closure_provider_timeout_seconds,
            "remote_check_mode": config.long_term_memory_remote_runtime_check_mode,
            "run_id": _normalize_text(run_id) or None,
            "live_cues": [
                {
                    "title": cue.title,
                    "salience": cue.salience,
                    "attention_state": cue.attention_state,
                    "open_offer": cue.open_offer,
                    "user_pull": cue.user_pull,
                    "observe_mode": cue.observe_mode,
                    "match_summary": cue.match_summary,
                }
                for cue in live_cues
            ],
        }
        if suite in {"single-turn", "all"}:
            payload["single_turn"] = _run_single_turn_suite(
                evaluator=evaluator,
                cases=_single_turn_cases(live_cues),
            )
        if suite in {"multi-turn", "all"}:
            payload["multi_turn"] = _run_multi_turn_suite(
                evaluator=evaluator,
                scenarios=_multi_turn_scenarios(live_cues),
            )
        if suite in {"multi-day", "all"}:
            payload["multi_day"] = _run_multi_day_suite(
                evaluator=evaluator,
                context=live_context,
                run_id=run_id,
            )
        overall_results: list[dict[str, object]] = []
        overall_scenarios: list[dict[str, object]] = []
        single_turn = payload.get("single_turn")
        if isinstance(single_turn, dict):
            overall_results.extend(single_turn.get("results", ()))
        multi_turn = payload.get("multi_turn")
        if isinstance(multi_turn, dict):
            for scenario in multi_turn.get("scenarios", ()):
                overall_scenarios.append(scenario)
                overall_results.extend(scenario.get("turns", ()))
        multi_day = payload.get("multi_day")
        if isinstance(multi_day, dict):
            for scenario in multi_day.get("scenarios", ()):
                overall_scenarios.append(scenario)
                overall_results.extend(day.get("turn", {}) for day in scenario.get("days", ()))
        payload["summary"] = _suite_summary(overall_results, scenario_results=overall_scenarios)
        if isinstance(multi_day, dict):
            transition_count = int(multi_day.get("summary", {}).get("transition_check_count", 0))
            transition_pass_count = int(multi_day.get("summary", {}).get("transition_pass_count", 0))
            payload["summary"]["transition_check_count"] = transition_count
            payload["summary"]["transition_pass_count"] = transition_pass_count
            payload["summary"]["transition_pass_rate"] = (
                0.0
                if transition_count == 0
                else round(transition_pass_count / transition_count, 3)
            )
        payload["ok"] = bool(payload["summary"]["pass_count"] == payload["summary"]["case_count"])
        if overall_scenarios:
            payload["ok"] = bool(
                payload["ok"]
                and payload["summary"]["scenario_pass_count"] == payload["summary"]["scenario_count"]
            )
        if isinstance(multi_day, dict):
            payload["ok"] = bool(
                payload["ok"]
                and payload["summary"].get("transition_pass_count") == payload["summary"].get("transition_check_count")
            )
        return payload
    finally:
        close_bundle = getattr(provider_bundle, "close", None)
        if callable(close_bundle):
            close_bundle()


def main() -> int:
    """Run the CLI entrypoint and emit one JSON acceptance artifact."""

    parser = _build_argument_parser()
    args = parser.parse_args()
    payload = run_live_acceptance(
        env_file=args.env_file,
        suite=args.suite,
        timeout_s=args.timeout_s,
        remote_check_mode=args.remote_check_mode,
        run_id=args.run_id,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
