# CHANGELOG: 2026-03-27
# BUG-1: Fixed single-exposure early-return behavior that left other pending exposures unresolved,
#        causing duplicate future learning, stale backlog growth, and nondeterministic attribution.
# BUG-2: Fixed first-match attribution bias by switching to bounded best-match scoring
#        (exact/containment only) instead of target-order-dependent matching.
# BUG-3: Fixed false "ignored" cooling by requiring a strong competing structured topic
#        instead of treating any batch target as evidence of non-pickup.
# SEC-1: Added hard caps for pending exposure groups, anchors, and match tokens to prevent
#        trivial local DoS on Raspberry Pi deployments via malformed or flooded display history.
# IMP-1: Added exposure-group consolidation, so repeated displays of the same semantic topic
#        are resolved together while emitting at most one learning signal per topic per turn.
# IMP-2: Added delay-aware and exposure-aware match metadata for downstream bandit/OPE layers
#        (match score, active-state, group size, matched signal kind, competing-topic strength).
# IMP-3: Made optional reserve-bus persistence and store lookups fail-open to keep the
#        main consolidation path alive under transient storage/config faults.

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

import logging
import math
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

LOGGER = logging.getLogger(__name__)

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
_FUTURE_SKEW_GRACE = timedelta(seconds=5)

# Edge hardening for Raspberry Pi class deployments.
_MAX_PENDING_GROUPS = 32
_MAX_SIGNALS_PER_TURN = 4
_MAX_ANCHORS_PER_EXPOSURE = 8
_MAX_MATCH_TOKENS = 12
_MIN_CONTAINMENT_MATCH_LEN = 14
_IGNORED_COMPETING_TARGET_MIN_CONFIDENCE = 0.55


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


def _to_utc(value: datetime) -> datetime:
    """Normalize one datetime to aware UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _clamp(value: float, *, lower: float, upper: float) -> float:
    """Clamp one numeric value into the closed interval [lower, upper]."""

    return max(lower, min(upper, value))


def _tokens_match(left: Sequence[str], right: Sequence[str]) -> bool:
    """Return whether two bounded token sets refer to the same shown topic."""

    if not left or not right:
        return False
    for left_token in left[:_MAX_MATCH_TOKENS]:
        if not left_token:
            continue
        for right_token in right[:_MAX_MATCH_TOKENS]:
            if not right_token:
                continue
            if left_token == right_token:
                return True
            if len(left_token) >= _MIN_CONTAINMENT_MATCH_LEN and left_token in right_token:
                return True
            if len(right_token) >= _MIN_CONTAINMENT_MATCH_LEN and right_token in left_token:
                return True
    return False


def _token_match_quality(left_token: str, right_token: str) -> tuple[float, str]:
    """Return conservative token match strength and quality label."""

    if not left_token or not right_token:
        return (0.0, "")
    if left_token == right_token:
        return (1.0, "exact")
    shorter_len = min(len(left_token), len(right_token))
    longer_len = max(len(left_token), len(right_token))
    if shorter_len >= _MIN_CONTAINMENT_MATCH_LEN and (left_token in right_token or right_token in left_token):
        ratio = shorter_len / longer_len
        return (_clamp(0.78 + (0.14 * ratio), lower=0.78, upper=0.92), "containment")
    return (0.0, "")


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

    @property
    def target_key(self) -> str:
        """Return a bounded normalized identity key for reuse suppression."""

        return _compact_text(self.target, max_len=160).casefold()


@dataclass(frozen=True, slots=True)
class _TargetMatch:
    """Describe one scored match between an exposure and one structured target."""

    target: _BatchTarget
    match_strength: float
    decision_score: float
    quality: str


@dataclass(frozen=True, slots=True)
class _ExposureGroup:
    """Group repeated pending exposures of the same semantic topic."""

    group_key: str
    primary: DisplayAmbientImpulseExposure
    related: tuple[DisplayAmbientImpulseExposure, ...]

    @property
    def group_size(self) -> int:
        """Return the number of pending exposures represented by this group."""

        return len(self.related)


@dataclass(slots=True)
class AmbientImpulseFeedbackExtractor:
    """Correlate recent reserve-card exposures with later structured turn evidence."""

    history_store: DisplayAmbientImpulseHistoryStore
    reserve_bus_feedback_store: DisplayReserveBusFeedbackStore | None = None
    pending_max_age_hours: float = _PENDING_MAX_AGE_HOURS

    @classmethod
    def from_config(cls, config) -> "AmbientImpulseFeedbackExtractor":
        """Build one extractor and its display-history store from configuration."""

        reserve_bus_feedback_store: DisplayReserveBusFeedbackStore | None
        try:
            reserve_bus_feedback_store = DisplayReserveBusFeedbackStore.from_config(config)
        except Exception:
            reserve_bus_feedback_store = None
            LOGGER.exception("Ambient reserve-bus feedback store disabled due to configuration failure.")
        return cls(
            history_store=DisplayAmbientImpulseHistoryStore.from_config(config),
            reserve_bus_feedback_store=reserve_bus_feedback_store,
        )

    def extract_from_consolidation(
        self,
        *,
        turn: LongTermConversationTurn,
        consolidation: LongTermConsolidationResultV1,
        extracted_batch: PersonalitySignalBatch,
    ) -> PersonalitySignalBatch:
        """Return additional display-reaction interaction signals for one turn."""

        del turn  # reserved for future contextual scoring hooks
        occurred_at = _to_utc(consolidation.occurred_at)
        pending = self._load_pending(occurred_at=occurred_at)
        if not pending:
            return PersonalitySignalBatch()

        pending_groups = self._prepare_pending_groups(pending=pending, occurred_at=occurred_at)
        if not pending_groups:
            return PersonalitySignalBatch()

        batch_targets = self._collect_batch_targets(batch=extracted_batch)
        positive_targets = tuple(target for target in batch_targets if target.signal_kind in _POSITIVE_SIGNAL_KINDS)
        negative_targets = tuple(target for target in batch_targets if target.signal_kind in _NEGATIVE_SIGNAL_KINDS)

        interaction_signals: list[InteractionSignal] = []
        used_target_keys: set[tuple[str, str]] = set()

        # BREAKING: one turn can now emit multiple interaction signals when several
        # independently shown reserve topics are resolved by the same consolidated turn.
        for group in pending_groups:
            if len(interaction_signals) >= _MAX_SIGNALS_PER_TURN:
                break

            negative_match = self._best_matching_target(
                exposure=group.primary,
                targets=negative_targets,
                used_target_keys=used_target_keys,
            )
            positive_match = self._best_matching_target(
                exposure=group.primary,
                targets=positive_targets,
                used_target_keys=used_target_keys,
            )
            competing_topic_strength = self._competing_topic_strength(
                exposure=group.primary,
                targets=batch_targets,
            )

            if negative_match is not None and (
                positive_match is None or negative_match.decision_score >= (positive_match.decision_score + 0.03)
            ):
                signal = self._build_negative_signal(
                    exposure=group.primary,
                    exposure_group_size=group.group_size,
                    matched_target=negative_match,
                    turn_id=consolidation.turn_id,
                    occurred_at=occurred_at,
                )
                status = (
                    "avoided"
                    if negative_match.target.signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION
                    else "cooled"
                )
                self._resolve_exposure_group(
                    group=group,
                    status=status,
                    sentiment="negative",
                    occurred_at=occurred_at,
                    turn_id=consolidation.turn_id,
                    target=negative_match.target.target,
                    summary=negative_match.target.summary or signal.summary,
                )
                interaction_signals.append(signal)
                used_target_keys.add((negative_match.target.signal_kind, negative_match.target.target_key))
                continue

            if positive_match is not None:
                signal = self._build_positive_signal(
                    exposure=group.primary,
                    exposure_group_size=group.group_size,
                    matched_target=positive_match,
                    turn_id=consolidation.turn_id,
                    occurred_at=occurred_at,
                )
                self._resolve_exposure_group(
                    group=group,
                    status="engaged",
                    sentiment="positive",
                    occurred_at=occurred_at,
                    turn_id=consolidation.turn_id,
                    target=positive_match.target.target,
                    summary=positive_match.target.summary or signal.summary,
                )
                interaction_signals.append(signal)
                used_target_keys.add((positive_match.target.signal_kind, positive_match.target.target_key))
                continue

            if self._should_count_as_ignored(
                exposure=group.primary,
                occurred_at=occurred_at,
                competing_topic_strength=competing_topic_strength,
            ):
                signal = self._build_ignored_signal(
                    exposure=group.primary,
                    exposure_group_size=group.group_size,
                    turn_id=consolidation.turn_id,
                    occurred_at=occurred_at,
                    competing_topic_strength=competing_topic_strength,
                )
                self._resolve_exposure_group(
                    group=group,
                    status="ignored",
                    sentiment="neutral",
                    occurred_at=occurred_at,
                    turn_id=consolidation.turn_id,
                    target=signal.target,
                    summary=signal.summary,
                )
                interaction_signals.append(signal)

        if not interaction_signals:
            return PersonalitySignalBatch()
        return PersonalitySignalBatch(interaction_signals=tuple(interaction_signals))

    def _load_pending(self, *, occurred_at: datetime) -> tuple[DisplayAmbientImpulseExposure, ...]:
        """Load pending exposures safely without breaking the main consolidation path."""

        max_age_hours = self.pending_max_age_hours
        if not isinstance(max_age_hours, (int, float)) or not math.isfinite(float(max_age_hours)) or max_age_hours <= 0.0:
            max_age_hours = _PENDING_MAX_AGE_HOURS
        try:
            pending = self.history_store.load_pending(
                now=occurred_at,
                max_age_hours=float(max_age_hours),
            )
        except Exception:
            LOGGER.exception("Ambient exposure feedback skipped because pending history could not be loaded.")
            return ()
        return tuple(pending)

    def _prepare_pending_groups(
        self,
        *,
        pending: Sequence[DisplayAmbientImpulseExposure],
        occurred_at: datetime,
    ) -> tuple[_ExposureGroup, ...]:
        """Deduplicate and prioritize pending exposures by semantic topic key."""

        grouped: dict[str, list[DisplayAmbientImpulseExposure]] = {}
        for exposure in pending:
            shown_at = _to_utc(exposure.shown_at_datetime())
            if shown_at > (occurred_at + _FUTURE_SKEW_GRACE):
                continue
            group_key = self._exposure_group_key(exposure)
            grouped.setdefault(group_key, []).append(exposure)

        groups: list[_ExposureGroup] = []
        for group_key, exposures in grouped.items():
            ordered = sorted(exposures, key=lambda item: _to_utc(item.shown_at_datetime()), reverse=True)
            primary = ordered[0]
            groups.append(
                _ExposureGroup(
                    group_key=group_key,
                    primary=primary,
                    related=tuple(ordered),
                )
            )

        groups.sort(key=lambda group: _to_utc(group.primary.shown_at_datetime()), reverse=True)
        return tuple(groups[:_MAX_PENDING_GROUPS])

    def _collect_batch_targets(self, *, batch: PersonalitySignalBatch) -> tuple[_BatchTarget, ...]:
        """Flatten structured batch content into generic match targets."""

        targets: list[_BatchTarget] = []
        for interaction_signal in batch.interaction_signals:
            targets.append(
                _BatchTarget(
                    signal_kind=interaction_signal.signal_kind,
                    target=interaction_signal.target,
                    summary=interaction_signal.summary,
                    confidence=_clamp(float(interaction_signal.confidence), lower=0.0, upper=1.0),
                    evidence_count=max(0, int(interaction_signal.evidence_count)),
                    source_event_ids=tuple(interaction_signal.source_event_ids),
                )
            )
        for thread in batch.continuity_threads:
            targets.append(
                _BatchTarget(
                    signal_kind="continuity",
                    target=thread.title,
                    summary=thread.summary,
                    confidence=_clamp(float(thread.salience), lower=0.0, upper=1.0),
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
                    confidence=_clamp(float(place_signal.confidence), lower=0.0, upper=1.0),
                    evidence_count=max(0, int(place_signal.evidence_count)),
                    source_event_ids=tuple(place_signal.source_event_ids),
                )
            )
        for world_signal in batch.world_signals:
            targets.append(
                _BatchTarget(
                    signal_kind="world",
                    target=world_signal.topic,
                    summary=world_signal.summary,
                    confidence=_clamp(float(world_signal.salience), lower=0.0, upper=1.0),
                    evidence_count=max(0, int(world_signal.evidence_count)),
                    source_event_ids=tuple(world_signal.source_event_ids),
                )
            )
        return tuple(targets)

    def _best_matching_target(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        targets: Sequence[_BatchTarget],
        used_target_keys: set[tuple[str, str]],
    ) -> _TargetMatch | None:
        """Return the highest-scoring structured target that matches the shown anchors."""

        exposure_tokens = self._exposure_tokens(exposure)
        if not exposure_tokens or not targets:
            return None

        best: _TargetMatch | None = None
        for target in targets:
            target_key = (target.signal_kind, target.target_key)
            if target_key in used_target_keys:
                continue
            match_strength, quality = self._score_target_match(exposure_tokens, target)
            if match_strength <= 0.0:
                continue
            decision_score = (
                match_strength
                + (0.08 * target.confidence)
                + min(0.06, 0.015 * math.log1p(max(0, target.evidence_count)))
            )
            candidate = _TargetMatch(
                target=target,
                match_strength=match_strength,
                decision_score=decision_score,
                quality=quality,
            )
            if best is None or candidate.decision_score > best.decision_score:
                best = candidate
        return best

    def _score_target_match(
        self,
        exposure_tokens: Sequence[str],
        target: _BatchTarget,
    ) -> tuple[float, str]:
        """Score one conservative target match using exact/containment checks only."""

        best_strength = 0.0
        best_quality = ""
        target_tokens = target.tokens
        if not target_tokens:
            return (0.0, "")
        for exposure_token in exposure_tokens[:_MAX_MATCH_TOKENS]:
            for target_token in target_tokens[:_MAX_MATCH_TOKENS]:
                strength, quality = _token_match_quality(exposure_token, target_token)
                if strength > best_strength:
                    best_strength = strength
                    best_quality = quality
        return (best_strength, best_quality)

    def _build_positive_signal(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        exposure_group_size: int,
        matched_target: _TargetMatch,
        turn_id: str,
        occurred_at: datetime,
    ) -> InteractionSignal:
        """Build one engagement booster signal from a positively picked-up card."""

        exposure_count = max(
            1,
            self._topic_exposure_count(topic_key=self._topic_key(exposure), now=occurred_at, default=1),
        )
        response_mode, response_latency_seconds = self._response_profile(
            exposure=exposure,
            occurred_at=occurred_at,
            status="engaged",
        )
        immediate_bonus = 0.14 if response_mode == "voice_immediate_pickup" else 0.0
        active_bonus = 0.08 if occurred_at <= exposure.expires_at_datetime() else 0.0
        repeated_exposure_penalty = min(0.08, 0.02 * max(0, exposure_count - 1))
        signal_confidence = _clamp(
            matched_target.target.confidence
            + (0.10 * matched_target.match_strength)
            + active_bonus
            + immediate_bonus
            - repeated_exposure_penalty,
            lower=0.58,
            upper=0.98,
        )
        evidence_count = max(
            2,
            matched_target.target.evidence_count,
            exposure_count + (1 if immediate_bonus > 0.0 else 0),
        )
        signal_target = matched_target.target.target or exposure.title or self._topic_key(exposure)
        signal_id = self._signal_id(
            turn_id=turn_id,
            signal_target=signal_target,
            exposure=exposure,
            suffix="display_pickup",
        )
        return InteractionSignal(
            signal_id=signal_id,
            signal_kind=INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
            target=signal_target,
            summary=truncate_text(
                f"The user picked up a displayed reserve-card thread about {signal_target}.",
                limit=180,
            ),
            confidence=signal_confidence,
            impact=signal_confidence * (0.40 + (0.10 if immediate_bonus > 0.0 else 0.0)) * (0.75 + (0.25 * matched_target.match_strength)),
            evidence_count=evidence_count,
            source_event_ids=matched_target.target.source_event_ids or (turn_id,),
            explicit_user_requested=False,
            metadata={
                "signal_source": "display_reserve_card",
                "exposure_id": exposure.exposure_id,
                "topic_key": self._topic_key(exposure),
                "cue_action": exposure.action,
                "cue_attention_state": exposure.attention_state,
                "engagement_kind": "display_pickup",
                "engagement_direction": "positive",
                "exposure_count": exposure_count,
                "exposure_group_size": exposure_group_size,
                "response_mode": response_mode,
                "response_latency_seconds": response_latency_seconds,
                "match_score": round(matched_target.match_strength, 4),
                "match_quality": matched_target.quality,
                "matched_signal_kind": matched_target.target.signal_kind,
                "matched_target_confidence": round(matched_target.target.confidence, 4),
                "matched_target_evidence_count": matched_target.target.evidence_count,
                "active_when_resolved": occurred_at <= exposure.expires_at_datetime(),
            },
        )

    def _build_negative_signal(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        exposure_group_size: int,
        matched_target: _TargetMatch,
        turn_id: str,
        occurred_at: datetime,
    ) -> InteractionSignal:
        """Build one stronger cooling/aversion signal from negative card reaction."""

        exposure_count = max(
            1,
            self._topic_exposure_count(topic_key=self._topic_key(exposure), now=occurred_at, default=1),
        )
        status = (
            "avoided"
            if matched_target.target.signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION
            else "cooled"
        )
        response_mode, response_latency_seconds = self._response_profile(
            exposure=exposure,
            occurred_at=occurred_at,
            status=status,
        )
        immediate_penalty = 0.08 if response_mode == "voice_immediate_pushback" else 0.0
        repeated_exposure_bonus = min(0.08, 0.02 * max(0, exposure_count - 1))
        signal_target = matched_target.target.target or exposure.title or self._topic_key(exposure)
        signal_kind = (
            INTERACTION_SIGNAL_TOPIC_AVERSION
            if matched_target.target.signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION
            else INTERACTION_SIGNAL_TOPIC_COOLING
        )
        signal_confidence = _clamp(
            matched_target.target.confidence
            + (0.08 * matched_target.match_strength)
            + immediate_penalty
            + repeated_exposure_bonus,
            lower=0.58,
            upper=0.98,
        )
        base_impact = 0.40 if signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION else 0.30
        impact = -max(
            0.18,
            signal_confidence
            * (base_impact + immediate_penalty + (0.03 * max(0, exposure_count - 1))),
        ) * (0.75 + (0.25 * matched_target.match_strength))
        signal_id = self._signal_id(
            turn_id=turn_id,
            signal_target=signal_target,
            exposure=exposure,
            suffix="display_reaction",
        )
        metadata = {
            "signal_source": "display_reserve_card",
            "exposure_id": exposure.exposure_id,
            "topic_key": self._topic_key(exposure),
            "cue_action": exposure.action,
            "cue_attention_state": exposure.attention_state,
            "engagement_direction": "negative",
            "engagement_kind": "display_rejection",
            "exposure_count": exposure_count,
            "exposure_group_size": exposure_group_size,
            "non_reengagement_count": 1,
            "deflection_count": 1 if signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION else 0,
            "response_mode": response_mode,
            "response_latency_seconds": response_latency_seconds,
            "match_score": round(matched_target.match_strength, 4),
            "match_quality": matched_target.quality,
            "matched_signal_kind": matched_target.target.signal_kind,
            "matched_target_confidence": round(matched_target.target.confidence, 4),
            "matched_target_evidence_count": matched_target.target.evidence_count,
            "active_when_resolved": occurred_at <= exposure.expires_at_datetime(),
        }
        return InteractionSignal(
            signal_id=signal_id,
            signal_kind=signal_kind,
            target=signal_target,
            summary=truncate_text(
                f"The user pushed back on a displayed reserve-card thread about {signal_target}.",
                limit=180,
            ),
            confidence=signal_confidence,
            impact=impact,
            evidence_count=max(
                2,
                matched_target.target.evidence_count,
                exposure_count + (1 if immediate_penalty > 0.0 else 0),
            ),
            source_event_ids=matched_target.target.source_event_ids or (turn_id,),
            explicit_user_requested=False,
            metadata=metadata,
        )

    def _build_ignored_signal(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        exposure_group_size: int,
        turn_id: str,
        occurred_at: datetime,
        competing_topic_strength: float,
    ) -> InteractionSignal:
        """Build one mild cooling signal after repeated non-pickup of the same card topic."""

        exposure_count = max(
            2,
            self._topic_exposure_count(topic_key=self._topic_key(exposure), now=occurred_at, default=2),
        )
        signal_target = exposure.title or self._topic_key(exposure)
        signal_id = self._signal_id(
            turn_id=turn_id,
            signal_target=signal_target,
            exposure=exposure,
            suffix="display_non_reengagement",
        )
        confidence = _clamp(
            0.52
            + min(0.10, 0.02 * max(0, exposure_count - 2))
            + (0.06 * competing_topic_strength),
            lower=0.52,
            upper=0.82,
        )
        impact = -min(
            0.28,
            0.14
            + min(0.08, 0.02 * max(0, exposure_count - 2))
            + (0.04 * competing_topic_strength),
        )
        return InteractionSignal(
            signal_id=signal_id,
            signal_kind=INTERACTION_SIGNAL_TOPIC_COOLING,
            target=signal_target,
            summary=truncate_text(
                f"The displayed reserve-card topic about {signal_target} did not pull the user back in again.",
                limit=180,
            ),
            confidence=confidence,
            impact=impact,
            evidence_count=exposure_count,
            source_event_ids=(turn_id,),
            explicit_user_requested=False,
            metadata={
                "signal_source": "display_reserve_card",
                "exposure_id": exposure.exposure_id,
                "topic_key": self._topic_key(exposure),
                "cue_action": exposure.action,
                "cue_attention_state": exposure.attention_state,
                "engagement_direction": "negative",
                "engagement_kind": "display_non_reengagement",
                "exposure_count": exposure_count,
                "exposure_group_size": exposure_group_size,
                "non_reengagement_count": 1,
                "deflection_count": 0,
                "exposure_aware": True,
                "competing_topic_strength": round(competing_topic_strength, 4),
                "response_mode": "no_voice_pickup",
                "response_latency_seconds": max(
                    0.0,
                    round((occurred_at - _to_utc(exposure.shown_at_datetime())).total_seconds(), 3),
                ),
            },
        )

    def _should_count_as_ignored(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        occurred_at: datetime,
        competing_topic_strength: float,
    ) -> bool:
        """Return whether the shown topic should count as mild repeated non-pickup."""

        if competing_topic_strength < _IGNORED_COMPETING_TARGET_MIN_CONFIDENCE:
            return False
        exposure_count = self._topic_exposure_count(
            topic_key=self._topic_key(exposure),
            now=occurred_at,
            default=0,
        )
        if exposure_count < 2:
            return False
        shown_at = _to_utc(exposure.shown_at_datetime())
        expires_at = _to_utc(exposure.expires_at_datetime())
        return occurred_at >= (shown_at + _RECENT_EXPOSURE_WINDOW) or occurred_at > expires_at

    def _resolve_exposure_group(
        self,
        *,
        group: _ExposureGroup,
        status: str,
        sentiment: str,
        occurred_at: datetime,
        turn_id: str,
        target: str | None,
        summary: str,
    ) -> None:
        """Persist the resolved reaction outcome for one grouped topic exposure."""

        for exposure in group.related:
            self._resolve_single_exposure(
                exposure=exposure,
                status=status,
                sentiment=sentiment,
                occurred_at=occurred_at,
                turn_id=turn_id,
                target=target,
                summary=summary,
                record_reserve_bus_feedback=(exposure is group.primary),
            )

    def _resolve_single_exposure(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        status: str,
        sentiment: str,
        occurred_at: datetime,
        turn_id: str,
        target: str | None,
        summary: str,
        record_reserve_bus_feedback: bool,
    ) -> None:
        """Persist the resolved reaction outcome for one exposure."""

        response_mode, response_latency_seconds = self._response_profile(
            exposure=exposure,
            occurred_at=occurred_at,
            status=status,
        )
        try:
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
        except Exception:
            LOGGER.exception(
                "Ambient exposure feedback resolution failed for exposure_id=%s.",
                getattr(exposure, "exposure_id", "<unknown>"),
            )
        if record_reserve_bus_feedback:
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

        shown_at = _to_utc(exposure.shown_at_datetime())
        expires_at = _to_utc(exposure.expires_at_datetime())
        latency_seconds = max(0.0, round((occurred_at - shown_at).total_seconds(), 3))
        active_until = expires_at + _ACTIVE_PICKUP_GRACE
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
        try:
            self.reserve_bus_feedback_store.record_reaction(
                topic_key=self._topic_key(exposure),
                reaction=reaction,
                intensity=intensity,
                reason=summary,
                now=occurred_at,
                source="display_reserve_card",
            )
        except Exception:
            LOGGER.exception(
                "Reserve-bus feedback write failed for exposure_id=%s.",
                getattr(exposure, "exposure_id", "<unknown>"),
            )

    def _competing_topic_strength(
        self,
        *,
        exposure: DisplayAmbientImpulseExposure,
        targets: Sequence[_BatchTarget],
    ) -> float:
        """Return the strongest unrelated structured target in the current batch."""

        exposure_tokens = self._exposure_tokens(exposure)
        if not exposure_tokens:
            return 0.0
        best = 0.0
        for target in targets:
            if not target.tokens:
                continue
            if _tokens_match(exposure_tokens, target.tokens):
                continue
            if target.confidence < _IGNORED_COMPETING_TARGET_MIN_CONFIDENCE and target.evidence_count < 2:
                continue
            strength = _clamp(
                (0.75 * target.confidence) + min(0.25, 0.05 * max(0, target.evidence_count - 1)),
                lower=0.0,
                upper=1.0,
            )
            if strength > best:
                best = strength
        return best

    def _topic_key(self, exposure: DisplayAmbientImpulseExposure) -> str:
        """Return the safest topic key for counting, grouping, and reserve-bus feedback."""

        semantic_key = _compact_text(exposure.semantic_key(), max_len=160)
        if semantic_key:
            return semantic_key
        return self._exposure_group_key(exposure)

    def _exposure_group_key(self, exposure: DisplayAmbientImpulseExposure) -> str:
        """Return a stable grouping key for repeated displays of the same topic."""

        semantic_key = _compact_text(exposure.semantic_key(), max_len=160)
        if semantic_key:
            return semantic_key
        title = _compact_text(getattr(exposure, "title", ""), max_len=160)
        if title:
            return title.casefold()
        exposure_id = _compact_text(getattr(exposure, "exposure_id", ""), max_len=160)
        return exposure_id or "unknown_exposure"

    def _exposure_tokens(self, exposure: DisplayAmbientImpulseExposure) -> tuple[str, ...]:
        """Return bounded normalized exposure tokens for conservative structured matching."""

        raw_labels: list[str] = []
        title = _compact_text(getattr(exposure, "title", ""), max_len=160)
        if title:
            raw_labels.append(title)

        anchor_count = 0
        for anchor in exposure.anchors():
            if anchor_count >= _MAX_ANCHORS_PER_EXPOSURE:
                break
            compact_anchor = _compact_text(anchor, max_len=160)
            if compact_anchor:
                raw_labels.append(compact_anchor)
                anchor_count += 1

        semantic_key = _compact_text(exposure.semantic_key(), max_len=160)
        if semantic_key:
            raw_labels.append(semantic_key)

        tokens: list[str] = []
        for label in raw_labels:
            for token in _match_tokens(label):
                if token:
                    tokens.append(token)
                    if len(tokens) >= _MAX_MATCH_TOKENS:
                        return tuple(dict.fromkeys(tokens))
        return tuple(dict.fromkeys(tokens))

    def _signal_id(
        self,
        *,
        turn_id: str,
        signal_target: str,
        exposure: DisplayAmbientImpulseExposure,
        suffix: str,
    ) -> str:
        """Return an idempotent signal id that stays unique under multi-signal turns."""

        turn_slug = slugify_identifier(turn_id, fallback="turn")
        target_slug = slugify_identifier(signal_target, fallback="topic")
        exposure_slug = slugify_identifier(self._exposure_group_key(exposure), fallback="cue")
        # BREAKING: signal ids now include the exposure topic key so retries stay idempotent
        # even when one turn resolves multiple displayed reserve-card topics.
        return f"signal:interaction:{turn_slug}:{target_slug}:{exposure_slug}:{suffix}"

    def _topic_exposure_count(self, *, topic_key: str, now: datetime, default: int) -> int:
        """Return exposure count safely without breaking interaction extraction."""

        try:
            return int(self.history_store.topic_exposure_count(topic_key=topic_key, now=now))
        except Exception:
            LOGGER.exception("Ambient topic exposure count lookup failed for topic_key=%s.", topic_key)
            return default


__all__ = ["AmbientImpulseFeedbackExtractor"]