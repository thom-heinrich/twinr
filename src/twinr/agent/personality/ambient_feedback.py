"""Translate reserve-lane reactions into generic engagement learning signals.

The right-hand HDMI reserve lane should not only display calm conversation
openers. Twinr also needs to learn how the user reacts to those visible
prompts. This module keeps that learning path separate from both the display
publisher and the core signal extractor:

- `display` records what was shown
- `ambient_feedback.py` correlates later structured turn evidence with those
  exposures
- `learning.py` merges the resulting interaction signals into the normal
  background evolution path

The matching stays conservative and structured:

- no transcript regexes
- no topic hardcoding
- only bounded exact/containment checks over the shown anchors and later
  structured targets
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from twinr.agent.personality.models import InteractionSignal
from twinr.agent.personality.signals import (
    INTERACTION_SIGNAL_TOPIC_AFFINITY,
    INTERACTION_SIGNAL_TOPIC_AVERSION,
    INTERACTION_SIGNAL_TOPIC_COOLING,
    INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
    PersonalitySignalBatch,
)
from twinr.display.ambient_impulse_history import (
    DisplayAmbientImpulseExposure,
    DisplayAmbientImpulseHistoryStore,
)
from twinr.display.reserve_bus_feedback import DisplayReserveBusFeedbackStore
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
)
from twinr.text_utils import slugify_identifier, truncate_text

_POSITIVE_SIGNAL_KINDS = frozenset(
    {
        INTERACTION_SIGNAL_TOPIC_AFFINITY,
        INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
        "continuity",
        "place",
        "world",
    }
)
_NEGATIVE_SIGNAL_KINDS = frozenset(
    {
        INTERACTION_SIGNAL_TOPIC_COOLING,
        INTERACTION_SIGNAL_TOPIC_AVERSION,
    }
)
_PENDING_MAX_AGE_HOURS = 12.0
_RECENT_EXPOSURE_WINDOW = timedelta(hours=2)
_ACTIVE_PICKUP_GRACE = timedelta(minutes=2)


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse arbitrary text into one bounded single-line string."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _match_tokens(value: object | None) -> tuple[str, ...]:
    """Return conservative normalized matching tokens for one structured label."""

    compact = _compact_text(value, max_len=160).casefold()
    if not compact:
        return ()
    slug = slugify_identifier(compact, fallback="")
    if slug and slug != compact:
        return (compact, slug)
    return (compact,)


def _tokens_match(left: Sequence[str], right: Sequence[str]) -> bool:
    """Return whether two bounded token sets refer to the same shown topic."""

    if not left or not right:
        return False
    for left_token in left:
        if not left_token:
            continue
        for right_token in right:
            if not right_token:
                continue
            if left_token == right_token:
                return True
            if len(left_token) >= 14 and left_token in right_token:
                return True
            if len(right_token) >= 14 and right_token in left_token:
                return True
    return False


@dataclass(frozen=True, slots=True)
class _BatchTarget:
    """Describe one structured target surfaced by the current turn."""

    signal_kind: str
    target: str
    summary: str
    confidence: float
    evidence_count: int
    source_event_ids: tuple[str, ...]

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return bounded normalized matching tokens for the target."""

        return _match_tokens(self.target)


@dataclass(slots=True)
class AmbientImpulseFeedbackExtractor:
    """Correlate recent reserve-card exposures with later structured turn evidence."""

    history_store: DisplayAmbientImpulseHistoryStore
    reserve_bus_feedback_store: DisplayReserveBusFeedbackStore | None = None
    pending_max_age_hours: float = _PENDING_MAX_AGE_HOURS

    @classmethod
    def from_config(cls, config) -> "AmbientImpulseFeedbackExtractor":
        """Build one extractor and its display-history store from configuration."""

        return cls(
            history_store=DisplayAmbientImpulseHistoryStore.from_config(config),
            reserve_bus_feedback_store=DisplayReserveBusFeedbackStore.from_config(config),
        )

    def extract_from_consolidation(
        self,
        *,
        turn: LongTermConversationTurn,
        consolidation: LongTermConsolidationResultV1,
        extracted_batch: PersonalitySignalBatch,
    ) -> PersonalitySignalBatch:
        """Return additional display-reaction interaction signals for one turn."""

        occurred_at = consolidation.occurred_at.astimezone(timezone.utc)
        pending = self.history_store.load_pending(
            now=occurred_at,
            max_age_hours=self.pending_max_age_hours,
        )
        if not pending:
            return PersonalitySignalBatch()

        batch_targets = self._collect_batch_targets(batch=extracted_batch)
        positive_targets = tuple(target for target in batch_targets if target.signal_kind in _POSITIVE_SIGNAL_KINDS)
        negative_targets = tuple(target for target in batch_targets if target.signal_kind in _NEGATIVE_SIGNAL_KINDS)
        strong_other_topic_present = bool(batch_targets)

        for exposure in pending:
            negative_match = self._first_matching_target(exposure, negative_targets)
            if negative_match is not None:
                signal = self._build_negative_signal(
                    exposure=exposure,
                    matched_target=negative_match,
                    turn_id=consolidation.turn_id,
                    occurred_at=occurred_at,
                )
                self._resolve_exposure(
                    exposure=exposure,
                    status="avoided" if negative_match.signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION else "cooled",
                    sentiment="negative",
                    occurred_at=occurred_at,
                    turn_id=consolidation.turn_id,
                    target=negative_match.target,
                    summary=negative_match.summary or signal.summary,
                )
                return PersonalitySignalBatch(interaction_signals=(signal,))

            positive_match = self._first_matching_target(exposure, positive_targets)
            if positive_match is not None:
                signal = self._build_positive_signal(
                    exposure=exposure,
                    matched_target=positive_match,
                    turn_id=consolidation.turn_id,
                    occurred_at=occurred_at,
                )
                self._resolve_exposure(
                    exposure=exposure,
                    status="engaged",
                    sentiment="positive",
                    occurred_at=occurred_at,
                    turn_id=consolidation.turn_id,
                    target=positive_match.target,
                    summary=positive_match.summary or signal.summary,
                )
                return PersonalitySignalBatch(interaction_signals=(signal,))

            if self._should_count_as_ignored(
                exposure=exposure,
                occurred_at=occurred_at,
                strong_other_topic_present=strong_other_topic_present,
            ):
                signal = self._build_ignored_signal(
                    exposure=exposure,
                    turn_id=consolidation.turn_id,
                    occurred_at=occurred_at,
                )
                self._resolve_exposure(
                    exposure=exposure,
                    status="ignored",
                    sentiment="neutral",
                    occurred_at=occurred_at,
                    turn_id=consolidation.turn_id,
                    target=signal.target,
                    summary=signal.summary,
                )
                return PersonalitySignalBatch(interaction_signals=(signal,))

        return PersonalitySignalBatch()

    def _collect_batch_targets(self, *, batch: PersonalitySignalBatch) -> tuple[_BatchTarget, ...]:
        """Flatten structured batch content into generic match targets."""

        targets: list[_BatchTarget] = []
        for interaction_signal in batch.interaction_signals:
            targets.append(
                _BatchTarget(
                    signal_kind=interaction_signal.signal_kind,
                    target=interaction_signal.target,
                    summary=interaction_signal.summary,
                    confidence=float(interaction_signal.confidence),
                    evidence_count=int(interaction_signal.evidence_count),
                    source_event_ids=tuple(interaction_signal.source_event_ids),
                )
            )
        for thread in batch.continuity_threads:
            targets.append(
                _BatchTarget(
                    signal_kind="continuity",
                    target=thread.title,
                    summary=thread.summary,
                    confidence=float(thread.salience),
                    evidence_count=1,
                    source_event_ids=(),
                )
            )
        for place_signal in batch.place_signals:
            targets.append(
                _BatchTarget(
                    signal_kind="place",
                    target=place_signal.place_name,
                    summary=place_signal.summary,
                    confidence=float(place_signal.confidence),
                    evidence_count=int(place_signal.evidence_count),
                    source_event_ids=tuple(place_signal.source_event_ids),
                )
            )
        for world_signal in batch.world_signals:
            targets.append(
                _BatchTarget(
                    signal_kind="world",
                    target=world_signal.topic,
                    summary=world_signal.summary,
                    confidence=float(world_signal.salience),
                    evidence_count=int(world_signal.evidence_count),
                    source_event_ids=tuple(world_signal.source_event_ids),
                )
            )
        return tuple(targets)

    def _first_matching_target(
        self,
        exposure: DisplayAmbientImpulseExposure,
        targets: Sequence[_BatchTarget],
    ) -> _BatchTarget | None:
        """Return the first structured target that matches the shown card anchors."""

        exposure_tokens: list[str] = []
        for anchor in exposure.anchors():
            exposure_tokens.extend(_match_tokens(anchor))
        unique_exposure_tokens = tuple(dict.fromkeys(exposure_tokens))
        for target in targets:
            if _tokens_match(unique_exposure_tokens, target.tokens):
                return target
        return None

    def _build_positive_signal(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        matched_target: _BatchTarget,
        turn_id: str,
        occurred_at: datetime,
    ) -> InteractionSignal:
        """Build one engagement booster signal from a positively picked-up card."""

        exposure_count = max(
            1,
            self.history_store.topic_exposure_count(topic_key=exposure.semantic_key(), now=occurred_at),
        )
        response_mode, response_latency_seconds = self._response_profile(
            exposure=exposure,
            occurred_at=occurred_at,
            status="engaged",
        )
        immediate_bonus = 0.14 if response_mode == "voice_immediate_pickup" else 0.0
        active_bonus = 0.08 if occurred_at <= exposure.expires_at_datetime() else 0.0
        signal_confidence = min(0.98, max(0.58, matched_target.confidence + 0.08 + active_bonus + immediate_bonus))
        evidence_count = max(2, matched_target.evidence_count, exposure_count + (1 if immediate_bonus > 0.0 else 0))
        signal_target = matched_target.target or exposure.title or exposure.semantic_key()
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        target_slug = slugify_identifier(signal_target, fallback="topic")
        return InteractionSignal(
            signal_id=f"signal:interaction:{turn_slug}:{target_slug}:display_pickup",
            signal_kind=INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
            target=signal_target,
            summary=truncate_text(
                f"The user picked up a displayed reserve-card thread about {signal_target}.",
                limit=180,
            ),
            confidence=signal_confidence,
            impact=signal_confidence * (0.52 if immediate_bonus > 0.0 else 0.42),
            evidence_count=evidence_count,
            source_event_ids=matched_target.source_event_ids or (turn_id,),
            explicit_user_requested=False,
            metadata={
                "signal_source": "display_reserve_card",
                "exposure_id": exposure.exposure_id,
                "cue_action": exposure.action,
                "cue_attention_state": exposure.attention_state,
                "engagement_kind": "display_pickup",
                "engagement_direction": "positive",
                "exposure_count": exposure_count,
                "response_mode": response_mode,
                "response_latency_seconds": response_latency_seconds,
            },
        )

    def _build_negative_signal(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        matched_target: _BatchTarget,
        turn_id: str,
        occurred_at: datetime,
    ) -> InteractionSignal:
        """Build one stronger cooling/aversion signal from negative card reaction."""

        exposure_count = max(
            1,
            self.history_store.topic_exposure_count(topic_key=exposure.semantic_key(), now=occurred_at),
        )
        response_mode, response_latency_seconds = self._response_profile(
            exposure=exposure,
            occurred_at=occurred_at,
            status="avoided" if matched_target.signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION else "cooled",
        )
        immediate_penalty = 0.08 if response_mode == "voice_immediate_pushback" else 0.0
        signal_target = matched_target.target or exposure.title or exposure.semantic_key()
        signal_kind = (
            INTERACTION_SIGNAL_TOPIC_AVERSION
            if matched_target.signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION
            else INTERACTION_SIGNAL_TOPIC_COOLING
        )
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        target_slug = slugify_identifier(signal_target, fallback="topic")
        impact = -max(
            0.18,
            matched_target.confidence * (
                (0.40 if signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION else 0.30) + immediate_penalty
            ),
        )
        metadata = {
            "signal_source": "display_reserve_card",
            "exposure_id": exposure.exposure_id,
            "cue_action": exposure.action,
            "cue_attention_state": exposure.attention_state,
            "engagement_direction": "negative",
            "engagement_kind": "display_rejection",
            "exposure_count": exposure_count,
            "non_reengagement_count": 1,
            "deflection_count": 1 if signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION else 0,
            "response_mode": response_mode,
            "response_latency_seconds": response_latency_seconds,
        }
        return InteractionSignal(
            signal_id=f"signal:interaction:{turn_slug}:{target_slug}:display_reaction",
            signal_kind=signal_kind,
            target=signal_target,
            summary=truncate_text(
                f"The user pushed back on a displayed reserve-card thread about {signal_target}.",
                limit=180,
            ),
            confidence=min(0.98, max(0.58, matched_target.confidence + 0.06 + immediate_penalty)),
            impact=impact,
            evidence_count=max(2, matched_target.evidence_count, exposure_count + (1 if immediate_penalty > 0.0 else 0)),
            source_event_ids=matched_target.source_event_ids or (turn_id,),
            explicit_user_requested=False,
            metadata=metadata,
        )

    def _build_ignored_signal(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        turn_id: str,
        occurred_at: datetime,
    ) -> InteractionSignal:
        """Build one mild cooling signal after repeated non-pickup of the same card topic."""

        exposure_count = max(
            2,
            self.history_store.topic_exposure_count(topic_key=exposure.semantic_key(), now=occurred_at),
        )
        signal_target = exposure.title or exposure.semantic_key()
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        target_slug = slugify_identifier(signal_target, fallback="topic")
        return InteractionSignal(
            signal_id=f"signal:interaction:{turn_slug}:{target_slug}:display_non_reengagement",
            signal_kind=INTERACTION_SIGNAL_TOPIC_COOLING,
            target=signal_target,
            summary=truncate_text(
                f"The displayed reserve-card topic about {signal_target} did not pull the user back in again.",
                limit=180,
            ),
            confidence=0.54,
            impact=-0.16,
            evidence_count=exposure_count,
            source_event_ids=(turn_id,),
            explicit_user_requested=False,
            metadata={
                "signal_source": "display_reserve_card",
                "exposure_id": exposure.exposure_id,
                "cue_action": exposure.action,
                "cue_attention_state": exposure.attention_state,
                "engagement_direction": "negative",
                "engagement_kind": "display_non_reengagement",
                "exposure_count": exposure_count,
                "non_reengagement_count": 1,
                "deflection_count": 0,
                "exposure_aware": True,
                "response_mode": "no_voice_pickup",
                "response_latency_seconds": max(
                    0.0,
                    round((occurred_at - exposure.shown_at_datetime()).total_seconds(), 3),
                ),
            },
        )

    def _should_count_as_ignored(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        occurred_at: datetime,
        strong_other_topic_present: bool,
    ) -> bool:
        """Return whether the shown topic should count as mild repeated non-pickup."""

        if not strong_other_topic_present:
            return False
        exposure_count = self.history_store.topic_exposure_count(
            topic_key=exposure.semantic_key(),
            now=occurred_at,
        )
        if exposure_count < 2:
            return False
        return occurred_at >= (exposure.shown_at_datetime() + _RECENT_EXPOSURE_WINDOW) or occurred_at > exposure.expires_at_datetime()

    def _resolve_exposure(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        status: str,
        sentiment: str,
        occurred_at: datetime,
        turn_id: str,
        target: str | None,
        summary: str,
    ) -> None:
        """Persist the resolved reaction outcome for one exposure."""

        response_mode, response_latency_seconds = self._response_profile(
            exposure=exposure,
            occurred_at=occurred_at,
            status=status,
        )
        self.history_store.resolve_feedback(
            exposure_id=exposure.exposure_id,
            response_status=status,
            response_sentiment=sentiment,
            response_at=occurred_at,
            response_mode=response_mode,
            response_latency_seconds=response_latency_seconds,
            response_turn_id=turn_id,
            response_target=target,
            response_summary=summary,
        )
        self._record_reserve_bus_feedback(
            exposure=exposure,
            status=status,
            occurred_at=occurred_at,
            summary=summary,
            response_mode=response_mode,
        )

    def _response_profile(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        occurred_at: datetime,
        status: str,
    ) -> tuple[str, float]:
        """Describe the response mode and pickup latency for one resolved exposure."""

        latency_seconds = max(0.0, round((occurred_at - exposure.shown_at_datetime()).total_seconds(), 3))
        active_until = exposure.expires_at_datetime() + _ACTIVE_PICKUP_GRACE
        if status == "engaged":
            if occurred_at <= active_until:
                return ("voice_immediate_pickup", latency_seconds)
            return ("voice_delayed_pickup", latency_seconds)
        if status == "avoided":
            if occurred_at <= active_until:
                return ("voice_immediate_pushback", latency_seconds)
            return ("voice_delayed_pushback", latency_seconds)
        if status == "cooled":
            if occurred_at <= active_until:
                return ("voice_immediate_cooling", latency_seconds)
            return ("voice_delayed_cooling", latency_seconds)
        if status == "ignored":
            return ("no_voice_pickup", latency_seconds)
        return ("neutral", latency_seconds)

    def _record_reserve_bus_feedback(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        status: str,
        occurred_at: datetime,
        summary: str,
        response_mode: str,
    ) -> None:
        """Persist one short-lived reserve-bus feedback hint for faster replanning."""

        if self.reserve_bus_feedback_store is None:
            return
        reaction = {
            "engaged": "immediate_engagement" if response_mode == "voice_immediate_pickup" else "engaged",
            "cooled": "cooled",
            "avoided": "avoided",
            "ignored": "ignored",
        }.get(status)
        if reaction is None:
            return
        intensity = {
            "immediate_engagement": 1.0,
            "engaged": 0.72,
            "cooled": 0.56,
            "avoided": 0.84,
            "ignored": 0.38,
        }[reaction]
        self.reserve_bus_feedback_store.record_reaction(
            topic_key=exposure.semantic_key(),
            reaction=reaction,
            intensity=intensity,
            reason=summary,
            now=occurred_at,
            source="display_reserve_card",
        )


__all__ = ["AmbientImpulseFeedbackExtractor"]
