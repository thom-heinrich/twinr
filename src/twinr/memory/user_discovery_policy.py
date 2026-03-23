"""Score Twinr's guided user-discovery topics from long-horizon engagement.

This module keeps discovery planning separate from the main discovery state
machine. It reads the existing reserve-lane feedback artifacts and turns them
into generic topic-level signals that can bias:

- which discovery topic is worth asking about next
- how softly or directly Twinr should phrase the next discovery question
- when a short profile review is due because corrections accumulated
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.ambient_impulse_history import (
    DisplayAmbientImpulseExposure,
    DisplayAmbientImpulseHistoryStore,
)
from twinr.display.reserve_bus_feedback import DisplayReserveBusFeedbackStore

_DEFAULT_WINDOW_DAYS = 30.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _compact_text(value: object | None, *, max_len: int) -> str:
    compact = " ".join(str(value or "").split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "..."


def _coerce_days(value: object, *, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(minimum, min(maximum, number))


def _coerce_timestamp(value: str | None) -> datetime | None:
    text = _compact_text(value, max_len=64)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _topic_id_from_exposure(exposure: DisplayAmbientImpulseExposure) -> str | None:
    metadata = exposure.metadata if isinstance(exposure.metadata, Mapping) else {}
    direct = _compact_text(metadata.get("topic_id"), max_len=48).lower().replace("-", "_").replace(" ", "_")
    if direct:
        return direct
    topic_key = _compact_text(exposure.topic_key, max_len=96).lower()
    if not topic_key.startswith("user_discovery:"):
        return None
    _, _phase, topic_id = (topic_key.split(":", 2) + ["", "", ""])[:3]
    normalized = _compact_text(topic_id, max_len=48).lower().replace("-", "_").replace(" ", "_")
    return normalized or None


@dataclass(frozen=True, slots=True)
class UserDiscoveryEngagementSignals:
    """Summarize recent reserve-lane discovery feedback for one topic."""

    topic_id: str
    engaged_count: int = 0
    ignored_count: int = 0
    avoided_count: int = 0
    cooled_count: int = 0
    positive_weight: float = 0.0
    negative_weight: float = 0.0
    bus_reaction: str | None = None
    bus_intensity: float = 0.0

    @property
    def net_score(self) -> float:
        return self.positive_weight - self.negative_weight

    @property
    def engagement_state(self) -> str:
        if self.net_score >= 0.55 or self.engaged_count >= max(2, self.ignored_count + self.avoided_count):
            return "pulling"
        if self.avoided_count > 0 or self.net_score <= -0.45:
            return "cooling"
        if self.ignored_count > self.engaged_count:
            return "hesitant"
        return "neutral"


@dataclass(frozen=True, slots=True)
class UserDiscoveryTopicPolicy:
    """Describe how discovery should approach one topic next."""

    topic_id: str
    score: float
    engagement_state: str
    question_style: str
    invite_style: str
    invite_salience_adjustment: float


@dataclass(slots=True)
class UserDiscoveryPolicyEngine:
    """Build generic discovery-planning signals from reserve-lane feedback."""

    history_store: DisplayAmbientImpulseHistoryStore
    reserve_bus_feedback_store: DisplayReserveBusFeedbackStore
    window_days: float = _DEFAULT_WINDOW_DAYS

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "UserDiscoveryPolicyEngine":
        return cls(
            history_store=DisplayAmbientImpulseHistoryStore.from_config(config),
            reserve_bus_feedback_store=DisplayReserveBusFeedbackStore.from_config(config),
            window_days=_coerce_days(
                getattr(config, "display_reserve_learning_window_days", _DEFAULT_WINDOW_DAYS),
                default=_DEFAULT_WINDOW_DAYS,
                minimum=7.0,
                maximum=60.0,
            ),
        )

    def load_signals(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, UserDiscoveryEngagementSignals]:
        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        minimum_shown_at = effective_now - timedelta(days=self.window_days)
        aggregated: dict[str, dict[str, object]] = {}
        for exposure in self.history_store.load():
            topic_id = _topic_id_from_exposure(exposure)
            if topic_id is None:
                continue
            shown_at = exposure.shown_at_datetime()
            if shown_at < minimum_shown_at:
                continue
            slot = aggregated.setdefault(
                topic_id,
                {
                    "engaged_count": 0,
                    "ignored_count": 0,
                    "avoided_count": 0,
                    "cooled_count": 0,
                    "positive_weight": 0.0,
                    "negative_weight": 0.0,
                    "bus_reaction": None,
                    "bus_intensity": 0.0,
                },
            )
            status = _compact_text(exposure.response_status, max_len=24).lower()
            if status == "engaged":
                slot["engaged_count"] = int(slot["engaged_count"]) + 1
                slot["positive_weight"] = float(slot["positive_weight"]) + 0.72
            elif status == "ignored":
                slot["ignored_count"] = int(slot["ignored_count"]) + 1
                slot["negative_weight"] = float(slot["negative_weight"]) + 0.24
            elif status == "avoided":
                slot["avoided_count"] = int(slot["avoided_count"]) + 1
                slot["negative_weight"] = float(slot["negative_weight"]) + 0.52
            elif status == "cooled":
                slot["cooled_count"] = int(slot["cooled_count"]) + 1
                slot["negative_weight"] = float(slot["negative_weight"]) + 0.34

        bus_signal = self.reserve_bus_feedback_store.load_active(now=effective_now)
        if bus_signal is not None:
            topic_key = _compact_text(bus_signal.topic_key, max_len=96).lower()
            if topic_key.startswith("user_discovery:"):
                topic_id = _compact_text(topic_key.split(":")[-1], max_len=48).lower().replace("-", "_").replace(" ", "_")
                if topic_id:
                    slot = aggregated.setdefault(
                        topic_id,
                        {
                            "engaged_count": 0,
                            "ignored_count": 0,
                            "avoided_count": 0,
                            "cooled_count": 0,
                            "positive_weight": 0.0,
                            "negative_weight": 0.0,
                            "bus_reaction": None,
                            "bus_intensity": 0.0,
                        },
                    )
                    slot["bus_reaction"] = _compact_text(bus_signal.reaction, max_len=32).lower()
                    slot["bus_intensity"] = float(bus_signal.intensity)

        return {
            topic_id: UserDiscoveryEngagementSignals(
                topic_id=topic_id,
                engaged_count=int(values["engaged_count"]),
                ignored_count=int(values["ignored_count"]),
                avoided_count=int(values["avoided_count"]),
                cooled_count=int(values["cooled_count"]),
                positive_weight=float(values["positive_weight"]),
                negative_weight=float(values["negative_weight"]),
                bus_reaction=str(values["bus_reaction"]) if values["bus_reaction"] else None,
                bus_intensity=float(values["bus_intensity"]),
            )
            for topic_id, values in aggregated.items()
        }

    def topic_policy(
        self,
        *,
        topic_id: str,
        completed_once: bool,
        fact_count: int,
        skip_count: int,
        correction_count: int,
        last_completed_at: str | None,
        initial_order: int,
        now: datetime | None = None,
        signals: Mapping[str, UserDiscoveryEngagementSignals] | None = None,
    ) -> UserDiscoveryTopicPolicy:
        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        topic_signals = (signals or self.load_signals(now=effective_now)).get(
            topic_id,
            UserDiscoveryEngagementSignals(topic_id=topic_id),
        )
        last_completed = _coerce_timestamp(last_completed_at)
        staleness_days = 365.0 if last_completed is None else max(
            0.0,
            (effective_now - last_completed).total_seconds() / 86400.0,
        )
        freshness_bonus = min(1.25, staleness_days / 14.0) * 0.28
        fact_gap_bonus = max(0, 3 - max(0, int(fact_count))) * 0.22
        completion_bonus = 0.88 if not completed_once else 0.0
        correction_bonus = min(max(0, int(correction_count)) * 0.24, 0.72)
        skip_penalty = min(max(0, int(skip_count)) * 0.24, 0.84)
        engagement_bonus = topic_signals.net_score * 0.55
        if topic_signals.bus_reaction == "immediate_engagement":
            engagement_bonus += 0.18 * max(0.3, topic_signals.bus_intensity)
        elif topic_signals.bus_reaction in {"avoided", "ignored", "cooled"}:
            engagement_bonus -= 0.16 * max(0.3, topic_signals.bus_intensity)
        score = completion_bonus + fact_gap_bonus + freshness_bonus + correction_bonus + engagement_bonus - skip_penalty
        score -= max(0, int(initial_order)) * 0.015
        engagement_state = topic_signals.engagement_state
        if correction_count > 0 or engagement_state in {"hesitant", "cooling"}:
            question_style = "gentle_optional"
            invite_style = "soft"
        elif topic_signals.net_score >= 0.5 and fact_count > 0:
            question_style = "deeper_follow_up"
            invite_style = "confident"
        else:
            question_style = "standard"
            invite_style = "warm"
        return UserDiscoveryTopicPolicy(
            topic_id=topic_id,
            score=score,
            engagement_state=engagement_state,
            question_style=question_style,
            invite_style=invite_style,
            invite_salience_adjustment=max(-0.16, min(0.16, engagement_bonus * 0.18)),
        )

    def review_due(
        self,
        *,
        active_fact_count: int,
        correction_total: int,
        review_count: int,
        last_reviewed_at: str | None,
        now: datetime | None = None,
    ) -> bool:
        if active_fact_count < 3:
            return False
        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        last_reviewed = _coerce_timestamp(last_reviewed_at)
        if correction_total > 0 and (last_reviewed is None or (effective_now - last_reviewed) >= timedelta(days=2)):
            return True
        if review_count <= 0 and active_fact_count >= 4:
            return True
        if last_reviewed is None:
            return False
        return (effective_now - last_reviewed) >= timedelta(days=12)


__all__ = [
    "UserDiscoveryEngagementSignals",
    "UserDiscoveryPolicyEngine",
    "UserDiscoveryTopicPolicy",
]
