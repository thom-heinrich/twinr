# CHANGELOG: 2026-03-29
# BUG-1: Normalize naive / malformed / future timestamps so one bad history row cannot crash the
# BUG-1: builder or silently dominate learning after Raspberry Pi clock skew / late NTP sync.
# BUG-2: Separate "shown" weight from "resolved feedback" weight so pending / unknown statuses no
# BUG-2: longer dilute topic scores; normalize response_mode tokens consistently for pickup logic.
# SEC-1: Bound history rows, tracked key cardinality, inverse-propensity amplification, and string
# SEC-1: compaction cost so corrupted / poisoned history cannot cause practical memory or CPU blowups.
# IMP-1: Upgrade scoring to confidence-aware Bayesian long-horizon learning with recent-trend signals.
# IMP-2: Add clipped propensity-aware weighting when logged action probabilities are available.
# IMP-3: Expose richer JSON-safe context for downstream ranking / copy generation without breaking
# IMP-3: the existing public API.

"""Learn which reserve-lane openers actually pull the user in over days.

The right HDMI reserve lane should not only react to the latest short-lived
pickup signal. Twinr also needs a slower memory of what kinds of silent
conversation openers have been welcome across multiple days.

This module derives that long-horizon profile from the durable reserve-lane
exposure history:

- topic-level engagement pressure for recurring themes
- family-level fit such as memory follow-ups vs. world updates
- action-level fit such as question-first vs. brief updates
- recent repetition pressure so stale topics cool down naturally

The result is a bounded, display-specific learning profile that can bias the
next day plan without hardcoding any topic names or parsing raw transcripts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from heapq import heappush, heappushpop
import math

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.display.ambient_impulse_history import (
    DisplayAmbientImpulseExposure,
    DisplayAmbientImpulseHistoryStore,
)

_DEFAULT_WINDOW_DAYS = 21.0
_DEFAULT_HALF_LIFE_DAYS = 7.0
_DEFAULT_RECENT_WINDOW_DAYS = 2.0
_DEFAULT_FUTURE_TOLERANCE_MINUTES = 15.0
_DEFAULT_PRIOR_STRENGTH = 1.75
_DEFAULT_MAX_HISTORY_ROWS = 4096
_DEFAULT_MAX_TOPIC_KEYS = 2048
_DEFAULT_MAX_FAMILY_KEYS = 256
_DEFAULT_MAX_ACTION_KEYS = 64
_DEFAULT_MAX_INVERSE_PROPENSITY_WEIGHT = 4.0
_DEFAULT_TREND_MIN_EVIDENCE_WEIGHT = 0.75

_PROBABILITY_METADATA_KEYS = (
    "selection_probability",
    "display_probability",
    "logged_probability",
    "propensity",
    "probability",
)


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _finite_float(value: object) -> float | None:
    """Return one finite float or ``None`` for invalid numeric-like values."""

    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _coerce_float(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Coerce one config-like float into a bounded finite float."""

    number = _finite_float(value)
    if number is None:
        return default
    return max(minimum, min(maximum, number))


def _coerce_int(
    value: object,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Coerce one config-like integer into a bounded finite int."""

    number = _finite_float(value)
    if number is None:
        return default
    try:
        coerced = int(number)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, coerced))


def _coerce_days(value: object, *, default: float, minimum: float, maximum: float) -> float:
    """Coerce one config-like day value into a bounded finite float."""

    return _coerce_float(
        value,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse one value into bounded single-line text.

    Non-scalar values are reduced to their type name so corrupted history rows
    cannot explode CPU / memory by forcing giant ``str(...)`` calls.
    """

    if value is None:
        return ""
    if isinstance(value, str):
        raw = value[: max(512, max_len * 4)]
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value[: max(512, max_len * 4)]).decode("utf-8", errors="replace")
    elif isinstance(value, (int, float, bool)):
        raw = str(value)
    else:
        raw = type(value).__name__
    compact = " ".join(raw.split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _normalized_token(value: object | None, *, max_len: int, default: str = "") -> str:
    """Return one compact, case-folded token."""

    token = _compact_text(value, max_len=max_len).casefold()
    return token or default


def _coerce_utc_datetime(
    value: object | None,
    *,
    now: datetime | None = None,
    future_tolerance: timedelta | None = None,
) -> datetime | None:
    """Normalize one timestamp-like value to an aware UTC datetime.

    Naive datetimes are treated as UTC because the durable history store is
    expected to persist machine timestamps, and treating them as local time
    would create host-dependent learning drift.
    """

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    if now is not None and future_tolerance is not None:
        if dt > now + future_tolerance:
            return None
        if dt > now:
            return now
    return dt


def _history_family(exposure: DisplayAmbientImpulseExposure) -> str:
    """Return the generic candidate family recorded for one exposure."""

    metadata = exposure.metadata if isinstance(exposure.metadata, Mapping) else {}
    family = _normalized_token(metadata.get("candidate_family"), max_len=40)
    if family:
        return family
    return _normalized_token(exposure.source, max_len=40, default="general")


def _history_topic(exposure: DisplayAmbientImpulseExposure) -> str:
    """Return the grouped semantic topic key recorded for one exposure."""

    topic = _normalized_token(exposure.semantic_topic_key, max_len=96)
    if topic:
        return topic
    return _normalized_token(exposure.topic_key, max_len=96)


def _coerce_float_bucket(value: object, *, default: float = 0.0) -> float:
    """Return one finite float from a mutable accumulator bucket."""

    number = _finite_float(value)
    return number if number is not None else default


def _coerce_int_bucket(value: object, *, default: int = 0) -> int:
    """Return one bounded int from a mutable accumulator bucket."""

    number = _finite_float(value)
    if number is None:
        return default
    try:
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return default


def _feedback_summary(
    exposure: DisplayAmbientImpulseExposure,
) -> tuple[bool, float, float, bool, bool, bool]:
    """Return normalized feedback summary for one exposure.

    The tuple is:
    ``(has_evidence, signed_outcome, reward_01, is_positive, is_negative,
    immediate_pickup)``.
    """

    status = _normalized_token(exposure.response_status, max_len=24)
    response_mode = _normalized_token(exposure.response_mode, max_len=48)
    immediate_pickup = response_mode == "voice_immediate_pickup"
    immediate_pushback = response_mode == "voice_immediate_pushback"

    if status == "engaged":
        return (
            True,
            1.15 if immediate_pickup else 0.72,
            1.0 if immediate_pickup else 0.84,
            True,
            False,
            immediate_pickup,
        )
    if status == "avoided":
        return (
            True,
            -1.05 if immediate_pushback else -0.88,
            0.0 if immediate_pushback else 0.08,
            False,
            True,
            False,
        )
    if status == "cooled":
        return True, -0.62, 0.22, False, True, False
    if status == "ignored":
        return True, -0.34, 0.38, False, True, False
    return False, 0.0, 0.5, False, False, immediate_pickup


def _propensity_weight(
    exposure: DisplayAmbientImpulseExposure,
    *,
    max_inverse_weight: float,
) -> float:
    """Return one clipped inverse-propensity weight if the log provides it."""

    metadata = exposure.metadata if isinstance(exposure.metadata, Mapping) else {}
    for key in _PROBABILITY_METADATA_KEYS:
        probability = _finite_float(metadata.get(key))
        if probability is None:
            continue
        if 0.0 < probability <= 1.0:
            return min(max_inverse_weight, 1.0 / max(probability, 1e-6))
        break
    return 1.0


@dataclass(frozen=True, slots=True)
class DisplayReserveLearningSignal:
    """Summarize long-horizon reserve-lane outcome learning for one key."""

    key: str
    exposure_weight: float
    outcome_score: float
    immediate_pickup_weight: float
    positive_weight: float
    negative_weight: float
    recent_exposure_count: int
    recent_positive_count: int
    last_shown_at: str | None = None
    last_positive_at: str | None = None
    last_negative_at: str | None = None
    evidence_weight: float = 0.0
    posterior_reward_mean: float = 0.5
    confidence: float = 0.0
    trend_score: float = 0.0
    evidence_ratio: float = 0.0
    recent_exposure_weight: float = 0.0
    recent_positive_weight: float = 0.0
    recent_negative_weight: float = 0.0

    @property
    def normalized_score(self) -> float:
        """Return the bounded long-horizon outcome score in the inclusive -1..1 range."""

        score = (self.posterior_reward_mean * 2.0) - 1.0
        if self.immediate_pickup_weight > 0.0:
            score += min(0.18, 0.16 * self.immediate_pickup_weight)
        if self.negative_weight > self.positive_weight:
            score -= min(0.20, 0.08 * (self.negative_weight - self.positive_weight))
        return max(-1.0, min(1.0, score))

    @property
    def repetition_pressure(self) -> float:
        """Return how stale / repetitive this key has become recently."""

        exposure_pressure = (
            self.recent_exposure_weight
            if self.recent_exposure_weight > 0.0
            else float(self.recent_exposure_count)
        )
        positive_relief = (
            self.recent_positive_weight
            if self.recent_positive_weight > 0.0
            else float(self.recent_positive_count)
        )
        negative_push = max(0.0, self.recent_negative_weight * 0.35)
        raw_pressure = max(0.0, exposure_pressure - positive_relief - 0.75) + negative_push
        scale = 2.5 + (0.75 * max(0.0, self.confidence))
        return max(0.0, min(1.0, raw_pressure / scale))

    @property
    def topic_state(self) -> str:
        """Return a coarse topical state for display-copy conditioning."""

        score = self.normalized_score
        if score >= 0.45 or self.immediate_pickup_weight >= 1.0:
            return "pulling"
        if self.confidence >= 0.35 and score >= 0.25 and self.trend_score >= 0.2:
            return "pulling"
        if score <= -0.45 or self.negative_weight >= self.positive_weight + 0.6:
            return "cooling"
        if self.confidence >= 0.35 and score <= -0.2 and self.trend_score <= -0.2:
            return "cooling"
        return "neutral"


@dataclass(frozen=True, slots=True)
class DisplayReserveLearningProfile:
    """Expose long-horizon reserve-lane learning to candidate ranking."""

    topics: Mapping[str, DisplayReserveLearningSignal]
    families: Mapping[str, DisplayReserveLearningSignal]
    actions: Mapping[str, DisplayReserveLearningSignal]
    window_days: float
    half_life_days: float

    def topic_signal(self, topic_key: str) -> DisplayReserveLearningSignal | None:
        """Return the stored topic signal for one normalized topic key."""

        key = _normalized_token(topic_key, max_len=96)
        if not key:
            return None
        return self.topics.get(key)

    def family_signal(self, family: str) -> DisplayReserveLearningSignal | None:
        """Return the stored family signal for one normalized candidate family."""

        key = _normalized_token(family, max_len=40)
        if not key:
            return None
        return self.families.get(key)

    def action_signal(self, action: str) -> DisplayReserveLearningSignal | None:
        """Return the stored action signal for one normalized action token."""

        key = _normalized_token(action, max_len=24)
        if not key:
            return None
        return self.actions.get(key)

    def candidate_adjustment(self, candidate: AmbientDisplayImpulseCandidate) -> float:
        """Return one bounded salience adjustment for a reserve candidate."""

        topic = self.topic_signal(candidate.semantic_key())
        family = self.family_signal(
            _compact_text(getattr(candidate, "candidate_family", None), max_len=40)
            or candidate.source
        )
        action = self.action_signal(candidate.action)

        adjustment = 0.0
        if topic is not None:
            adjustment += topic.normalized_score * (0.12 + (0.10 * topic.confidence))
            adjustment += topic.trend_score * 0.08
            adjustment += min(0.22, 0.22 * max(0.0, topic.immediate_pickup_weight))
            adjustment -= topic.repetition_pressure * (0.08 + (0.05 * topic.confidence))
            if topic.evidence_ratio < 0.35 and topic.recent_exposure_count > 0:
                adjustment -= 0.03
        if family is not None:
            adjustment += family.normalized_score * (0.05 + (0.07 * family.confidence))
            adjustment += family.trend_score * 0.04
        if action is not None:
            adjustment += action.normalized_score * (0.03 + (0.05 * action.confidence))
            adjustment += action.trend_score * 0.03
        if topic is not None and family is not None:
            if topic.normalized_score > 0.0 and family.normalized_score > 0.0:
                adjustment += 0.02 * min(topic.confidence, family.confidence)
            if topic.normalized_score < -0.45 and family.normalized_score < 0.0:
                adjustment -= 0.02 * min(topic.confidence, family.confidence)

        return max(-0.4, min(0.4, adjustment))

    def context_for_candidate(self, candidate: AmbientDisplayImpulseCandidate) -> dict[str, object]:
        """Return one JSON-safe learning context block for LLM copy generation."""

        topic = self.topic_signal(candidate.semantic_key())
        family_name = (
            _compact_text(getattr(candidate, "candidate_family", None), max_len=40)
            or candidate.source
        )
        family = self.family_signal(family_name)
        action = self.action_signal(candidate.action)
        return {
            "topic_state": topic.topic_state if topic is not None else "unknown",
            "topic_score": round(topic.normalized_score, 3) if topic is not None else 0.0,
            "topic_repetition_pressure": round(topic.repetition_pressure, 3) if topic is not None else 0.0,
            "topic_confidence": round(topic.confidence, 3) if topic is not None else 0.0,
            "topic_trend_score": round(topic.trend_score, 3) if topic is not None else 0.0,
            "family_state": family.topic_state if family is not None else "unknown",
            "family_score": round(family.normalized_score, 3) if family is not None else 0.0,
            "family_confidence": round(family.confidence, 3) if family is not None else 0.0,
            "family_trend_score": round(family.trend_score, 3) if family is not None else 0.0,
            "action_score": round(action.normalized_score, 3) if action is not None else 0.0,
            "action_confidence": round(action.confidence, 3) if action is not None else 0.0,
            "action_trend_score": round(action.trend_score, 3) if action is not None else 0.0,
        }


@dataclass(slots=True)
class DisplayReserveLearningProfileBuilder:
    """Build one long-horizon reserve-lane learning profile from exposure history."""

    history_store: DisplayAmbientImpulseHistoryStore
    window_days: float = _DEFAULT_WINDOW_DAYS
    half_life_days: float = _DEFAULT_HALF_LIFE_DAYS
    recent_window_days: float = _DEFAULT_RECENT_WINDOW_DAYS
    future_tolerance_minutes: float = _DEFAULT_FUTURE_TOLERANCE_MINUTES
    prior_strength: float = _DEFAULT_PRIOR_STRENGTH
    max_history_rows: int = _DEFAULT_MAX_HISTORY_ROWS
    max_topic_keys: int = _DEFAULT_MAX_TOPIC_KEYS
    max_family_keys: int = _DEFAULT_MAX_FAMILY_KEYS
    max_action_keys: int = _DEFAULT_MAX_ACTION_KEYS
    max_inverse_propensity_weight: float = _DEFAULT_MAX_INVERSE_PROPENSITY_WEIGHT
    trend_min_evidence_weight: float = _DEFAULT_TREND_MIN_EVIDENCE_WEIGHT

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveLearningProfileBuilder":
        """Resolve one history-backed learning-profile builder from configuration."""

        return cls(
            history_store=DisplayAmbientImpulseHistoryStore.from_config(config),
            window_days=_coerce_days(
                getattr(config, "display_reserve_bus_learning_window_days", _DEFAULT_WINDOW_DAYS),
                default=_DEFAULT_WINDOW_DAYS,
                minimum=3.0,
                maximum=90.0,
            ),
            half_life_days=_coerce_days(
                getattr(config, "display_reserve_bus_learning_half_life_days", _DEFAULT_HALF_LIFE_DAYS),
                default=_DEFAULT_HALF_LIFE_DAYS,
                minimum=1.0,
                maximum=30.0,
            ),
            recent_window_days=_coerce_days(
                getattr(config, "display_reserve_bus_learning_recent_window_days", _DEFAULT_RECENT_WINDOW_DAYS),
                default=_DEFAULT_RECENT_WINDOW_DAYS,
                minimum=0.25,
                maximum=14.0,
            ),
            future_tolerance_minutes=_coerce_float(
                getattr(
                    config,
                    "display_reserve_bus_learning_future_tolerance_minutes",
                    _DEFAULT_FUTURE_TOLERANCE_MINUTES,
                ),
                default=_DEFAULT_FUTURE_TOLERANCE_MINUTES,
                minimum=0.0,
                maximum=240.0,
            ),
            prior_strength=_coerce_float(
                getattr(config, "display_reserve_bus_learning_prior_strength", _DEFAULT_PRIOR_STRENGTH),
                default=_DEFAULT_PRIOR_STRENGTH,
                minimum=0.1,
                maximum=20.0,
            ),
            max_history_rows=_coerce_int(
                getattr(config, "display_reserve_bus_learning_max_history_rows", _DEFAULT_MAX_HISTORY_ROWS),
                default=_DEFAULT_MAX_HISTORY_ROWS,
                minimum=128,
                maximum=50_000,
            ),
            max_topic_keys=_coerce_int(
                getattr(config, "display_reserve_bus_learning_max_topic_keys", _DEFAULT_MAX_TOPIC_KEYS),
                default=_DEFAULT_MAX_TOPIC_KEYS,
                minimum=64,
                maximum=10_000,
            ),
            max_family_keys=_coerce_int(
                getattr(config, "display_reserve_bus_learning_max_family_keys", _DEFAULT_MAX_FAMILY_KEYS),
                default=_DEFAULT_MAX_FAMILY_KEYS,
                minimum=16,
                maximum=1_000,
            ),
            max_action_keys=_coerce_int(
                getattr(config, "display_reserve_bus_learning_max_action_keys", _DEFAULT_MAX_ACTION_KEYS),
                default=_DEFAULT_MAX_ACTION_KEYS,
                minimum=8,
                maximum=512,
            ),
            max_inverse_propensity_weight=_coerce_float(
                getattr(
                    config,
                    "display_reserve_bus_learning_max_inverse_propensity_weight",
                    _DEFAULT_MAX_INVERSE_PROPENSITY_WEIGHT,
                ),
                default=_DEFAULT_MAX_INVERSE_PROPENSITY_WEIGHT,
                minimum=1.0,
                maximum=100.0,
            ),
            trend_min_evidence_weight=_coerce_float(
                getattr(
                    config,
                    "display_reserve_bus_learning_trend_min_evidence_weight",
                    _DEFAULT_TREND_MIN_EVIDENCE_WEIGHT,
                ),
                default=_DEFAULT_TREND_MIN_EVIDENCE_WEIGHT,
                minimum=0.1,
                maximum=10.0,
            ),
        )

    def build(self, *, now: datetime | None = None) -> DisplayReserveLearningProfile:
        """Summarize reserve-lane history into topic / family / action outcome signals."""

        effective_now = _coerce_utc_datetime(now or _utc_now()) or _utc_now()
        minimum_shown_at = effective_now - timedelta(days=self.window_days)
        recent_cutoff = effective_now - timedelta(days=self.recent_window_days)
        topics: dict[str, dict[str, object]] = {}
        families: dict[str, dict[str, object]] = {}
        actions: dict[str, dict[str, object]] = {}

        retained_exposures = self._retained_history(
            self.history_store.load(),
            now=effective_now,
            minimum_shown_at=minimum_shown_at,
        )

        for exposure, shown_at in retained_exposures:
            base_weight = self._time_weight(now=effective_now, shown_at=shown_at)
            combined_weight = base_weight * _propensity_weight(
                exposure,
                max_inverse_weight=self.max_inverse_propensity_weight,
            )
            family = _history_family(exposure)
            action = _normalized_token(exposure.action, max_len=24, default="hint")
            has_evidence, outcome, reward_01, is_positive, is_negative, immediate_pickup = _feedback_summary(
                exposure
            )
            evidence_weight = combined_weight if has_evidence else 0.0

            self._accumulate(
                topics,
                key=_history_topic(exposure),
                weight=combined_weight,
                evidence_weight=evidence_weight,
                reward_01=reward_01,
                outcome=outcome,
                shown_at=shown_at,
                is_positive=is_positive,
                is_negative=is_negative,
                immediate_pickup=immediate_pickup,
                recent_cutoff=recent_cutoff,
            )
            self._accumulate(
                families,
                key=family,
                weight=combined_weight,
                evidence_weight=evidence_weight,
                reward_01=reward_01,
                outcome=outcome,
                shown_at=shown_at,
                is_positive=is_positive,
                is_negative=is_negative,
                immediate_pickup=immediate_pickup,
                recent_cutoff=recent_cutoff,
            )
            self._accumulate(
                actions,
                key=action,
                weight=combined_weight,
                evidence_weight=evidence_weight,
                reward_01=reward_01,
                outcome=outcome,
                shown_at=shown_at,
                is_positive=is_positive,
                is_negative=is_negative,
                immediate_pickup=immediate_pickup,
                recent_cutoff=recent_cutoff,
            )

        return DisplayReserveLearningProfile(
            topics=self._freeze_signal_map(topics, max_keys=self.max_topic_keys),
            families=self._freeze_signal_map(families, max_keys=self.max_family_keys),
            actions=self._freeze_signal_map(actions, max_keys=self.max_action_keys),
            window_days=self.window_days,
            half_life_days=self.half_life_days,
        )

    def _retained_history(
        self,
        exposures: Iterable[DisplayAmbientImpulseExposure],
        *,
        now: datetime,
        minimum_shown_at: datetime,
    ) -> list[tuple[DisplayAmbientImpulseExposure, datetime]]:
        """Return the most recent, valid history rows inside the learning window."""

        future_tolerance = timedelta(minutes=self.future_tolerance_minutes)
        retained: list[tuple[float, int, DisplayAmbientImpulseExposure, datetime]] = []

        for index, exposure in enumerate(exposures):
            try:
                shown_at_raw = exposure.shown_at_datetime()
            except Exception:
                continue
            shown_at = _coerce_utc_datetime(
                shown_at_raw,
                now=now,
                future_tolerance=future_tolerance,
            )
            if shown_at is None or shown_at < minimum_shown_at:
                continue

            item = (shown_at.timestamp(), index, exposure, shown_at)
            if len(retained) < self.max_history_rows:
                heappush(retained, item)
                continue
            if item[0] > retained[0][0]:
                heappushpop(retained, item)

        retained.sort(key=lambda row: (row[0], row[1]))
        return [(exposure, shown_at) for _, _, exposure, shown_at in retained]

    def _time_weight(self, *, now: datetime, shown_at: datetime) -> float:
        """Return one decay weight for an exposure inside the learning window."""

        age_days = max(0.0, (now - shown_at).total_seconds() / 86_400.0)
        return max(0.05, 0.5 ** (age_days / max(0.5, self.half_life_days)))

    def _accumulate(
        self,
        bucket: dict[str, dict[str, object]],
        *,
        key: str,
        weight: float,
        evidence_weight: float,
        reward_01: float,
        outcome: float,
        shown_at: datetime,
        is_positive: bool,
        is_negative: bool,
        immediate_pickup: bool,
        recent_cutoff: datetime,
    ) -> None:
        """Accumulate one exposure into a generic keyed outcome bucket."""

        normalized_key = _normalized_token(key, max_len=96)
        if not normalized_key:
            return

        state = bucket.setdefault(
            normalized_key,
            {
                "exposure_weight": 0.0,
                "evidence_weight": 0.0,
                "reward_sum_01": 0.0,
                "recent_evidence_weight": 0.0,
                "recent_reward_sum_01": 0.0,
                "outcome_score": 0.0,
                "immediate_pickup_weight": 0.0,
                "positive_weight": 0.0,
                "negative_weight": 0.0,
                "recent_exposure_count": 0,
                "recent_positive_count": 0,
                "recent_exposure_weight": 0.0,
                "recent_positive_weight": 0.0,
                "recent_negative_weight": 0.0,
                "last_shown_at": None,
                "last_positive_at": None,
                "last_negative_at": None,
            },
        )

        state["exposure_weight"] = _coerce_float_bucket(state.get("exposure_weight")) + weight
        if evidence_weight > 0.0:
            state["evidence_weight"] = _coerce_float_bucket(state.get("evidence_weight")) + evidence_weight
            state["reward_sum_01"] = _coerce_float_bucket(state.get("reward_sum_01")) + (
                reward_01 * evidence_weight
            )
            state["outcome_score"] = _coerce_float_bucket(state.get("outcome_score")) + (
                outcome * evidence_weight
            )

        if immediate_pickup and evidence_weight > 0.0:
            state["immediate_pickup_weight"] = _coerce_float_bucket(
                state.get("immediate_pickup_weight")
            ) + evidence_weight
        if is_positive and evidence_weight > 0.0:
            state["positive_weight"] = _coerce_float_bucket(state.get("positive_weight")) + evidence_weight
            state["last_positive_at"] = shown_at
        if is_negative and evidence_weight > 0.0:
            state["negative_weight"] = _coerce_float_bucket(state.get("negative_weight")) + evidence_weight
            state["last_negative_at"] = shown_at

        if shown_at >= recent_cutoff:
            state["recent_exposure_count"] = _coerce_int_bucket(state.get("recent_exposure_count")) + 1
            state["recent_exposure_weight"] = _coerce_float_bucket(
                state.get("recent_exposure_weight")
            ) + weight
            if evidence_weight > 0.0:
                state["recent_evidence_weight"] = _coerce_float_bucket(
                    state.get("recent_evidence_weight")
                ) + evidence_weight
                state["recent_reward_sum_01"] = _coerce_float_bucket(
                    state.get("recent_reward_sum_01")
                ) + (reward_01 * evidence_weight)
            if is_positive and evidence_weight > 0.0:
                state["recent_positive_count"] = _coerce_int_bucket(
                    state.get("recent_positive_count")
                ) + 1
                state["recent_positive_weight"] = _coerce_float_bucket(
                    state.get("recent_positive_weight")
                ) + evidence_weight
            if is_negative and evidence_weight > 0.0:
                state["recent_negative_weight"] = _coerce_float_bucket(
                    state.get("recent_negative_weight")
                ) + evidence_weight

        last_shown_at = state.get("last_shown_at")
        if not isinstance(last_shown_at, datetime) or shown_at >= last_shown_at:
            state["last_shown_at"] = shown_at

    def _freeze_signal_map(
        self,
        raw: Mapping[str, Mapping[str, object]],
        *,
        max_keys: int,
    ) -> dict[str, DisplayReserveLearningSignal]:
        """Convert mutable accumulator buckets into frozen learning signals."""

        frozen: dict[str, DisplayReserveLearningSignal] = {}
        for key, payload in self._trim_to_top_keys(raw, max_keys=max_keys).items():
            exposure_weight = _coerce_float_bucket(payload.get("exposure_weight"))
            evidence_weight = _coerce_float_bucket(payload.get("evidence_weight"))
            reward_sum_01 = _coerce_float_bucket(payload.get("reward_sum_01"))
            recent_evidence_weight = _coerce_float_bucket(payload.get("recent_evidence_weight"))
            recent_reward_sum_01 = _coerce_float_bucket(payload.get("recent_reward_sum_01"))

            posterior_reward_mean = self._posterior_reward_mean(
                reward_sum_01=reward_sum_01,
                evidence_weight=evidence_weight,
            )
            confidence = self._confidence(evidence_weight=evidence_weight)
            evidence_ratio = (
                max(0.0, min(1.0, evidence_weight / exposure_weight)) if exposure_weight > 0.0 else 0.0
            )

            recent_mean = (
                recent_reward_sum_01 / recent_evidence_weight
                if recent_evidence_weight > 0.0
                else posterior_reward_mean
            )
            older_evidence_weight = max(0.0, evidence_weight - recent_evidence_weight)
            older_reward_sum_01 = max(0.0, reward_sum_01 - recent_reward_sum_01)
            older_mean = (
                older_reward_sum_01 / older_evidence_weight
                if older_evidence_weight > 0.0
                else posterior_reward_mean
            )
            trend_score = self._trend_score(
                posterior_reward_mean=posterior_reward_mean,
                recent_mean=recent_mean,
                recent_evidence_weight=recent_evidence_weight,
                older_mean=older_mean,
                older_evidence_weight=older_evidence_weight,
            )

            frozen[key] = DisplayReserveLearningSignal(
                key=key,
                exposure_weight=exposure_weight,
                outcome_score=_coerce_float_bucket(payload.get("outcome_score")),
                immediate_pickup_weight=_coerce_float_bucket(payload.get("immediate_pickup_weight")),
                positive_weight=_coerce_float_bucket(payload.get("positive_weight")),
                negative_weight=_coerce_float_bucket(payload.get("negative_weight")),
                recent_exposure_count=_coerce_int_bucket(payload.get("recent_exposure_count")),
                recent_positive_count=_coerce_int_bucket(payload.get("recent_positive_count")),
                last_shown_at=self._iso(payload.get("last_shown_at")),
                last_positive_at=self._iso(payload.get("last_positive_at")),
                last_negative_at=self._iso(payload.get("last_negative_at")),
                evidence_weight=evidence_weight,
                posterior_reward_mean=posterior_reward_mean,
                confidence=confidence,
                trend_score=trend_score,
                evidence_ratio=evidence_ratio,
                recent_exposure_weight=_coerce_float_bucket(payload.get("recent_exposure_weight")),
                recent_positive_weight=_coerce_float_bucket(payload.get("recent_positive_weight")),
                recent_negative_weight=_coerce_float_bucket(payload.get("recent_negative_weight")),
            )
        return frozen

    def _trim_to_top_keys(
        self,
        raw: Mapping[str, Mapping[str, object]],
        *,
        max_keys: int,
    ) -> dict[str, Mapping[str, object]]:
        """Keep the heaviest keys only so one corrupted history file cannot fan out unboundedly."""

        if len(raw) <= max_keys:
            return dict(raw)
        ranked = sorted(
            raw.items(),
            key=lambda item: (
                -_coerce_float_bucket(item[1].get("exposure_weight")),
                -_coerce_float_bucket(item[1].get("evidence_weight")),
                item[0],
            ),
        )
        return dict(ranked[:max_keys])

    def _posterior_reward_mean(self, *, reward_sum_01: float, evidence_weight: float) -> float:
        """Return one Bayesian-smoothed reward mean in the inclusive 0..1 range."""

        neutral_mass = self.prior_strength * 0.5
        mean = (neutral_mass + reward_sum_01) / max(self.prior_strength + evidence_weight, 1e-6)
        return max(0.0, min(1.0, mean))

    def _confidence(self, *, evidence_weight: float) -> float:
        """Return one smooth 0..1 confidence score for a signal."""

        return max(0.0, min(1.0, evidence_weight / max(self.prior_strength + evidence_weight, 1e-6)))

    def _trend_score(
        self,
        *,
        posterior_reward_mean: float,
        recent_mean: float,
        recent_evidence_weight: float,
        older_mean: float,
        older_evidence_weight: float,
    ) -> float:
        """Return one bounded recent-trend score in the inclusive -1..1 range."""

        if recent_evidence_weight <= 0.0:
            return 0.0

        if older_evidence_weight >= self.trend_min_evidence_weight:
            baseline = older_mean
            evidence_scale = min(1.0, recent_evidence_weight / self.trend_min_evidence_weight)
        else:
            baseline = posterior_reward_mean
            evidence_scale = min(0.6, recent_evidence_weight / self.trend_min_evidence_weight)

        delta = recent_mean - baseline
        return max(-1.0, min(1.0, delta * 2.0 * evidence_scale))

    def _iso(self, value: object | None) -> str | None:
        """Render one optional datetime as UTC ISO-8601 text."""

        dt = _coerce_utc_datetime(value)
        if dt is None:
            return None
        return dt.astimezone(timezone.utc).isoformat()


__all__ = [
    "DisplayReserveLearningProfile",
    "DisplayReserveLearningProfileBuilder",
    "DisplayReserveLearningSignal",
]
