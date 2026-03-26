"""Resolve visible discovery-invite reactions directly against reserve artifacts.

Generic ambient feedback matching is intentionally conservative and works well
for open-ended world, place, or continuity prompts. Guided user-discovery is
different: when the user explicitly starts, answers, skips, pauses, or reviews
the visible discovery invite, Twinr already knows exactly which reserve topic
that reaction belongs to.

This helper uses that explicit link to update both:

- the pending reserve exposure history entry, when the invite was shown, and
- the short-lived reserve-bus feedback signal so the same-day plan can retire
  or de-prioritize the reacted-to discovery card immediately.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseExposure, DisplayAmbientImpulseHistoryStore
from twinr.display.reserve_bus_feedback import DisplayReserveBusFeedbackStore
from twinr.memory.user_discovery import UserDiscoveryInvite

_PENDING_MAX_AGE_HOURS = 12.0
_ACTIVE_PICKUP_GRACE = timedelta(minutes=2)


def _topic_key(invite: UserDiscoveryInvite) -> str:
    return f"user_discovery:{invite.phase}:{invite.topic_id}".strip().casefold()


def _response_profile(
    exposure: DisplayAmbientImpulseExposure,
    *,
    occurred_at: datetime,
    status: str,
) -> tuple[str, float]:
    """Describe the response mode and latency for one direct discovery reaction."""

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
    return ("neutral", latency_seconds)


def _response_sentiment(status: str) -> str:
    if status == "engaged":
        return "positive"
    if status in {"avoided", "cooled"}:
        return "negative"
    return "unknown"


def _bus_reaction(status: str, *, response_mode: str | None) -> tuple[str, float] | None:
    if status == "engaged":
        if response_mode == "voice_immediate_pickup":
            return ("immediate_engagement", 1.0)
        return ("engaged", 0.72)
    if status == "cooled":
        return ("cooled", 0.56)
    if status == "avoided":
        return ("avoided", 0.84)
    return None


def record_user_discovery_invite_feedback(
    config: TwinrConfig,
    *,
    invite: UserDiscoveryInvite,
    status: str,
    occurred_at: datetime,
    summary: str,
    response_turn_id: str | None = None,
) -> bool:
    """Resolve the newest pending exposure for one explicit discovery invite reaction."""

    topic_key = _topic_key(invite)
    if not topic_key:
        return False
    effective_now = occurred_at.astimezone(timezone.utc)
    history_store = DisplayAmbientImpulseHistoryStore.from_config(config)
    feedback_store = DisplayReserveBusFeedbackStore.from_config(config)
    resolved = False
    response_mode: str | None = None

    for exposure in history_store.load_pending(
        now=effective_now,
        max_age_hours=_PENDING_MAX_AGE_HOURS,
        limit=12,
    ):
        if exposure.semantic_key() != topic_key:
            continue
        response_mode, latency_seconds = _response_profile(
            exposure,
            occurred_at=effective_now,
            status=status,
        )
        history_store.resolve_feedback(
            exposure_id=exposure.exposure_id,
            response_status=status,
            response_sentiment=_response_sentiment(status),
            response_at=effective_now,
            response_mode=response_mode,
            response_latency_seconds=latency_seconds,
            response_turn_id=response_turn_id,
            response_target=invite.display_topic_label or invite.topic_label or invite.topic_id,
            response_summary=summary,
        )
        resolved = True
        break

    reaction = _bus_reaction(status, response_mode=response_mode)
    if reaction is not None:
        feedback_store.record_reaction(
            topic_key=topic_key,
            reaction=reaction[0],
            intensity=reaction[1],
            reason=summary,
            now=effective_now,
            source="user_discovery",
        )
    return resolved


__all__ = ["record_user_discovery_invite_feedback"]
