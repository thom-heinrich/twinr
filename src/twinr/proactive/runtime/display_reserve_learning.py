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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse one value into bounded single-line text."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _coerce_days(value: object, *, default: float, minimum: float, maximum: float) -> float:
    """Coerce one config-like day value into a bounded finite float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(minimum, min(maximum, number))


def _history_family(exposure: DisplayAmbientImpulseExposure) -> str:
    """Return the generic candidate family recorded for one exposure."""

    metadata = exposure.metadata if isinstance(exposure.metadata, Mapping) else {}
    family = _compact_text(metadata.get("candidate_family"), max_len=40).casefold()
    if family:
        return family
    return _compact_text(exposure.source, max_len=40).casefold() or "general"


def _outcome_weight(exposure: DisplayAmbientImpulseExposure) -> float:
    """Return one signed outcome weight for a resolved exposure."""

    status = _compact_text(exposure.response_status, max_len=24).casefold()
    response_mode = _compact_text(exposure.response_mode, max_len=48).casefold()
    if status == "engaged":
        return 1.15 if response_mode == "voice_immediate_pickup" else 0.72
    if status == "avoided":
        return -1.05 if response_mode == "voice_immediate_pushback" else -0.88
    if status == "cooled":
        return -0.62
    if status == "ignored":
        return -0.34
    return 0.0


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

    @property
    def normalized_score(self) -> float:
        """Return the signed outcome average in the inclusive -1..1 range."""

        if self.exposure_weight <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, self.outcome_score / self.exposure_weight))

    @property
    def repetition_pressure(self) -> float:
        """Return how stale/repetitive this key has become recently."""

        raw_pressure = max(0, self.recent_exposure_count - self.recent_positive_count - 1)
        return max(0.0, min(1.0, raw_pressure / 3.0))

    @property
    def topic_state(self) -> str:
        """Return a coarse topical state for display-copy conditioning."""

        score = self.normalized_score
        if score >= 0.4 or self.immediate_pickup_weight >= 1.0:
            return "pulling"
        if score <= -0.45 or self.negative_weight >= self.positive_weight + 0.6:
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

        key = _compact_text(topic_key, max_len=96).casefold()
        if not key:
            return None
        return self.topics.get(key)

    def family_signal(self, family: str) -> DisplayReserveLearningSignal | None:
        """Return the stored family signal for one normalized candidate family."""

        key = _compact_text(family, max_len=40).casefold()
        if not key:
            return None
        return self.families.get(key)

    def action_signal(self, action: str) -> DisplayReserveLearningSignal | None:
        """Return the stored action signal for one normalized action token."""

        key = _compact_text(action, max_len=24).casefold()
        if not key:
            return None
        return self.actions.get(key)

    def candidate_adjustment(self, candidate: AmbientDisplayImpulseCandidate) -> float:
        """Return one bounded salience adjustment for a reserve candidate."""

        topic = self.topic_signal(candidate.topic_key)
        family = self.family_signal(
            _compact_text(getattr(candidate, "candidate_family", None), max_len=40)
            or candidate.source
        )
        action = self.action_signal(candidate.action)
        adjustment = 0.0
        if topic is not None:
            adjustment += topic.normalized_score * 0.22
            adjustment -= topic.repetition_pressure * 0.12
        if family is not None:
            adjustment += family.normalized_score * 0.12
        if action is not None:
            adjustment += action.normalized_score * 0.08
        return max(-0.35, min(0.35, adjustment))

    def context_for_candidate(self, candidate: AmbientDisplayImpulseCandidate) -> dict[str, object]:
        """Return one JSON-safe learning context block for LLM copy generation."""

        topic = self.topic_signal(candidate.topic_key)
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
            "family_state": family.topic_state if family is not None else "unknown",
            "family_score": round(family.normalized_score, 3) if family is not None else 0.0,
            "action_score": round(action.normalized_score, 3) if action is not None else 0.0,
        }


@dataclass(slots=True)
class DisplayReserveLearningProfileBuilder:
    """Build one long-horizon reserve-lane learning profile from exposure history."""

    history_store: DisplayAmbientImpulseHistoryStore
    window_days: float = _DEFAULT_WINDOW_DAYS
    half_life_days: float = _DEFAULT_HALF_LIFE_DAYS
    recent_window_days: float = _DEFAULT_RECENT_WINDOW_DAYS

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
            recent_window_days=_DEFAULT_RECENT_WINDOW_DAYS,
        )

    def build(self, *, now: datetime | None = None) -> DisplayReserveLearningProfile:
        """Summarize reserve-lane history into topic/family/action outcome signals."""

        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        minimum_shown_at = effective_now - timedelta(days=self.window_days)
        recent_cutoff = effective_now - timedelta(days=self.recent_window_days)
        topics: dict[str, dict[str, object]] = {}
        families: dict[str, dict[str, object]] = {}
        actions: dict[str, dict[str, object]] = {}

        for exposure in self.history_store.load():
            shown_at = exposure.shown_at_datetime()
            if shown_at < minimum_shown_at:
                continue
            weight = self._time_weight(now=effective_now, shown_at=shown_at)
            family = _history_family(exposure)
            action = _compact_text(exposure.action, max_len=24).casefold() or "hint"
            outcome = _outcome_weight(exposure)
            is_positive = outcome > 0.0
            is_negative = outcome < 0.0
            immediate_pickup = exposure.response_mode == "voice_immediate_pickup"
            self._accumulate(
                topics,
                key=exposure.topic_key,
                weight=weight,
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
                weight=weight,
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
                weight=weight,
                outcome=outcome,
                shown_at=shown_at,
                is_positive=is_positive,
                is_negative=is_negative,
                immediate_pickup=immediate_pickup,
                recent_cutoff=recent_cutoff,
            )

        return DisplayReserveLearningProfile(
            topics=self._freeze_signal_map(topics),
            families=self._freeze_signal_map(families),
            actions=self._freeze_signal_map(actions),
            window_days=self.window_days,
            half_life_days=self.half_life_days,
        )

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
        outcome: float,
        shown_at: datetime,
        is_positive: bool,
        is_negative: bool,
        immediate_pickup: bool,
        recent_cutoff: datetime,
    ) -> None:
        """Accumulate one exposure into a generic keyed outcome bucket."""

        normalized_key = _compact_text(key, max_len=96).casefold()
        if not normalized_key:
            return
        state = bucket.setdefault(
            normalized_key,
            {
                "exposure_weight": 0.0,
                "outcome_score": 0.0,
                "immediate_pickup_weight": 0.0,
                "positive_weight": 0.0,
                "negative_weight": 0.0,
                "recent_exposure_count": 0,
                "recent_positive_count": 0,
                "last_shown_at": None,
                "last_positive_at": None,
                "last_negative_at": None,
            },
        )
        state["exposure_weight"] = float(state["exposure_weight"]) + weight
        state["outcome_score"] = float(state["outcome_score"]) + outcome * weight
        if immediate_pickup:
            state["immediate_pickup_weight"] = float(state["immediate_pickup_weight"]) + weight
        if is_positive:
            state["positive_weight"] = float(state["positive_weight"]) + weight
            state["last_positive_at"] = shown_at
        if is_negative:
            state["negative_weight"] = float(state["negative_weight"]) + weight
            state["last_negative_at"] = shown_at
        if shown_at >= recent_cutoff:
            state["recent_exposure_count"] = int(state["recent_exposure_count"]) + 1
            if is_positive:
                state["recent_positive_count"] = int(state["recent_positive_count"]) + 1
        last_shown_at = state.get("last_shown_at")
        if not isinstance(last_shown_at, datetime) or shown_at >= last_shown_at:
            state["last_shown_at"] = shown_at

    def _freeze_signal_map(
        self,
        raw: Mapping[str, Mapping[str, object]],
    ) -> dict[str, DisplayReserveLearningSignal]:
        """Convert mutable accumulator buckets into frozen learning signals."""

        frozen: dict[str, DisplayReserveLearningSignal] = {}
        for key, payload in raw.items():
            frozen[key] = DisplayReserveLearningSignal(
                key=key,
                exposure_weight=float(payload.get("exposure_weight", 0.0) or 0.0),
                outcome_score=float(payload.get("outcome_score", 0.0) or 0.0),
                immediate_pickup_weight=float(payload.get("immediate_pickup_weight", 0.0) or 0.0),
                positive_weight=float(payload.get("positive_weight", 0.0) or 0.0),
                negative_weight=float(payload.get("negative_weight", 0.0) or 0.0),
                recent_exposure_count=int(payload.get("recent_exposure_count", 0) or 0),
                recent_positive_count=int(payload.get("recent_positive_count", 0) or 0),
                last_shown_at=self._iso(payload.get("last_shown_at")),
                last_positive_at=self._iso(payload.get("last_positive_at")),
                last_negative_at=self._iso(payload.get("last_negative_at")),
            )
        return frozen

    def _iso(self, value: object | None) -> str | None:
        """Render one optional datetime as UTC ISO-8601 text."""

        if not isinstance(value, datetime):
            return None
        return value.astimezone(timezone.utc).isoformat()


__all__ = [
    "DisplayReserveLearningProfile",
    "DisplayReserveLearningProfileBuilder",
    "DisplayReserveLearningSignal",
]
