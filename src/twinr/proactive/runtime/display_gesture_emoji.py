"""Mirror recognized camera gestures into bounded HDMI emoji acknowledgements.

This module keeps gesture-to-emoji acknowledgement policy separate from the
main proactive monitor orchestration. It listens only to stabilized rising-edge
camera gesture events and publishes a short-lived right-hand HDMI emoji cue
without touching the face channel.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.emoji_cues import DisplayEmojiController, DisplayEmojiCueStore, DisplayEmojiSymbol

from ..social.camera_surface import ProactiveCameraSnapshot, ProactiveCameraSurfaceUpdate
from ..social.engine import SocialFineHandGesture, SocialGestureEvent


_SOURCE = "proactive_gesture_ack"
_DEFAULT_HOLD_SECONDS = 2.8
_COARSE_GESTURE_EVENT_NAMES = frozenset({"camera.gesture_detected", "camera.coarse_arm_gesture_detected"})
_FINE_GESTURE_EVENT_NAME = "camera.fine_hand_gesture_detected"
_MOTION_COARSE_GESTURES = frozenset(
    {
        SocialGestureEvent.WAVE,
        SocialGestureEvent.DISMISS,
        SocialGestureEvent.TWO_HAND_DISMISS,
    }
)
_MOTION_GESTURE_DOMINANCE_WINDOW_S = 1.0
_MIN_REFRESH_INTERVAL_S = 0.1
_CUSTOM_ONLY_FINE_GESTURES = frozenset(
    {
        SocialFineHandGesture.OK_SIGN,
        SocialFineHandGesture.MIDDLE_FINGER,
    }
)

_FINE_HAND_GESTURE_MAP: dict[SocialFineHandGesture, tuple[DisplayEmojiSymbol, str]] = {
    SocialFineHandGesture.THUMBS_UP: (DisplayEmojiSymbol.THUMBS_UP, "success"),
    SocialFineHandGesture.THUMBS_DOWN: (DisplayEmojiSymbol.THUMBS_DOWN, "warm"),
    SocialFineHandGesture.POINTING: (DisplayEmojiSymbol.POINTING_HAND, "info"),
    SocialFineHandGesture.PEACE_SIGN: (DisplayEmojiSymbol.VICTORY_HAND, "warm"),
    SocialFineHandGesture.OPEN_PALM: (DisplayEmojiSymbol.RAISED_HAND, "info"),
    SocialFineHandGesture.OK_SIGN: (DisplayEmojiSymbol.OK_HAND, "success"),
    SocialFineHandGesture.MIDDLE_FINGER: (DisplayEmojiSymbol.WARNING, "alert"),
}

_COARSE_GESTURE_MAP: dict[SocialGestureEvent, tuple[DisplayEmojiSymbol, str]] = {
    SocialGestureEvent.WAVE: (DisplayEmojiSymbol.WAVING_HAND, "warm"),
    SocialGestureEvent.STOP: (DisplayEmojiSymbol.RAISED_HAND, "info"),
    SocialGestureEvent.DISMISS: (DisplayEmojiSymbol.WAVING_HAND, "neutral"),
    SocialGestureEvent.TWO_HAND_DISMISS: (DisplayEmojiSymbol.WAVING_HAND, "neutral"),
    SocialGestureEvent.CONFIRM: (DisplayEmojiSymbol.CHECK, "success"),
}


def resolve_display_gesture_refresh_interval(config: TwinrConfig) -> float | None:
    """Return the bounded local refresh cadence for HDMI gesture acknowledgement."""

    raw_interval = getattr(
        config,
        "display_gesture_refresh_interval_s",
        getattr(config, "display_attention_refresh_interval_s", 0.2),
    )
    try:
        interval_s = float(raw_interval or 0.0)
    except (TypeError, ValueError):
        return 0.2
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        return None
    return max(_MIN_REFRESH_INTERVAL_S, interval_s)


def display_gesture_refresh_supported(
    *,
    config: TwinrConfig,
    vision_observer: object | None,
) -> bool:
    """Return whether a dedicated HDMI gesture-refresh path is safe to run."""

    if resolve_display_gesture_refresh_interval(config) is None:
        return False
    display_driver = str(getattr(config, "display_driver", "") or "").strip().lower()
    if not display_driver.startswith("hdmi"):
        return False
    supports_gesture_refresh = getattr(vision_observer, "supports_gesture_refresh", None)
    if supports_gesture_refresh is True:
        return True
    if supports_gesture_refresh is False:
        return False
    return callable(getattr(vision_observer, "observe_gesture", None))


def decision_for_fine_hand_gesture(
    gesture: SocialFineHandGesture,
) -> DisplayGestureEmojiDecision:
    """Return the user-facing emoji decision for one fine-hand gesture."""

    mapped = _FINE_HAND_GESTURE_MAP.get(gesture)
    if mapped is None:
        return DisplayGestureEmojiDecision(reason="unsupported_fine_hand_gesture")
    symbol, accent = mapped
    return DisplayGestureEmojiDecision(
        active=True,
        reason=f"fine_hand_gesture:{gesture.value}",
        symbol=symbol,
        accent=accent,
    )


def decision_for_coarse_gesture(
    gesture: SocialGestureEvent,
    *,
    motion_priority: bool = False,
) -> DisplayGestureEmojiDecision:
    """Return the user-facing emoji decision for one coarse-arm gesture."""

    mapped = _COARSE_GESTURE_MAP.get(gesture)
    if mapped is None:
        return DisplayGestureEmojiDecision(reason="unsupported_coarse_gesture")
    symbol, accent = mapped
    reason_prefix = "motion_coarse_gesture" if motion_priority else "coarse_gesture"
    return DisplayGestureEmojiDecision(
        active=True,
        reason=f"{reason_prefix}:{gesture.value}",
        symbol=symbol,
        accent=accent,
    )


@dataclass(frozen=True, slots=True)
class DisplayGestureEmojiDecision:
    """Describe one optional HDMI emoji acknowledgement."""

    active: bool = False
    reason: str = "inactive"
    source: str = _SOURCE
    symbol: DisplayEmojiSymbol = DisplayEmojiSymbol.SPARKLES
    accent: str = "neutral"
    hold_seconds: float = _DEFAULT_HOLD_SECONDS


@dataclass(frozen=True, slots=True)
class DisplayGestureEmojiPublishResult:
    """Summarize one bounded publish attempt."""

    action: str
    decision: DisplayGestureEmojiDecision
    owner: str | None = None


def derive_display_gesture_emoji(
    *,
    snapshot: ProactiveCameraSnapshot,
    event_names: Iterable[str],
) -> DisplayGestureEmojiDecision:
    """Translate one stabilized camera gesture update into one emoji ack."""

    names = {str(name or "").strip() for name in event_names}
    if (
        snapshot.gesture_event in _MOTION_COARSE_GESTURES
        and not snapshot.gesture_event_unknown
        and (
            _COARSE_GESTURE_EVENT_NAMES.intersection(names)
            or (
                _FINE_GESTURE_EVENT_NAME in names
                and snapshot.fine_hand_gesture == SocialFineHandGesture.OPEN_PALM
                and not snapshot.fine_hand_gesture_unknown
            )
        )
    ):
        decision = decision_for_coarse_gesture(
            snapshot.gesture_event,
            motion_priority=True,
        )
        if decision.active:
            return decision

    if _FINE_GESTURE_EVENT_NAME in names:
        if not snapshot.fine_hand_gesture_unknown:
            decision = decision_for_fine_hand_gesture(snapshot.fine_hand_gesture)
            if decision.active:
                return decision
        return DisplayGestureEmojiDecision(reason="unsupported_fine_hand_gesture")

    if _COARSE_GESTURE_EVENT_NAMES.intersection(names):
        if not snapshot.gesture_event_unknown:
            decision = decision_for_coarse_gesture(snapshot.gesture_event)
            if decision.active:
                return decision
        return DisplayGestureEmojiDecision(reason="unsupported_coarse_gesture")

    return DisplayGestureEmojiDecision(reason="no_gesture_event")


@dataclass(slots=True)
class DisplayGestureEmojiPublisher:
    """Persist short-lived HDMI emoji acknowledgements for user gestures."""

    controller: DisplayEmojiController
    source: str = _SOURCE
    _recent_motion_priority_until: datetime | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayGestureEmojiPublisher":
        """Build one publisher from the configured emoji cue store."""

        return cls(
            controller=DisplayEmojiController.from_config(
                config,
                default_source=_SOURCE,
            )
        )

    @property
    def store(self) -> DisplayEmojiCueStore:
        """Expose the underlying emoji cue store for tests."""

        return self.controller.store

    def publish_update(
        self,
        update: ProactiveCameraSurfaceUpdate,
        *,
        now: datetime | None = None,
    ) -> DisplayGestureEmojiPublishResult:
        """Derive and publish one emoji acknowledgement from a camera update."""

        effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        decision = derive_display_gesture_emoji(
            snapshot=update.snapshot,
            event_names=update.event_names,
        )
        decision = self._apply_recent_motion_priority(
            update=update,
            decision=decision,
            now=effective_now,
        )
        return self.publish(decision, now=effective_now)

    def publish(
        self,
        decision: DisplayGestureEmojiDecision,
        *,
        now: datetime | None = None,
    ) -> DisplayGestureEmojiPublishResult:
        """Persist one gesture acknowledgement unless another producer owns the surface."""

        if not decision.active:
            return DisplayGestureEmojiPublishResult(action="inactive", decision=decision)
        effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        active_cue = self.store.load_active(now=effective_now)
        active_owner = None if active_cue is None else str(active_cue.source or "").strip() or None
        if active_owner is not None and active_owner != self.source:
            return DisplayGestureEmojiPublishResult(
                action="blocked_foreign_cue",
                decision=decision,
                owner=active_owner,
            )
        self.controller.show_symbol(
            decision.symbol,
            accent=decision.accent,
            source=self.source,
            hold_seconds=decision.hold_seconds,
            now=effective_now,
        )
        return DisplayGestureEmojiPublishResult(
            action="updated" if active_owner == self.source else "created",
            decision=decision,
            owner=active_owner,
        )

    def _apply_recent_motion_priority(
        self,
        *,
        update: ProactiveCameraSurfaceUpdate,
        decision: DisplayGestureEmojiDecision,
        now: datetime,
    ) -> DisplayGestureEmojiDecision:
        """Prefer recent motion gestures briefly over conflicting custom-only fine-hand blips."""

        if (
            decision.active
            and decision.reason.startswith("motion_coarse_gesture:")
            and update.snapshot.gesture_event in _MOTION_COARSE_GESTURES
        ):
            self._recent_motion_priority_until = now + timedelta(
                seconds=min(decision.hold_seconds, _MOTION_GESTURE_DOMINANCE_WINDOW_S)
            )
            return decision

        if not decision.active or not decision.reason.startswith("fine_hand_gesture:"):
            return decision

        if update.snapshot.fine_hand_gesture not in _CUSTOM_ONLY_FINE_GESTURES:
            return decision

        recent_motion_until = self._recent_motion_priority_until
        if recent_motion_until is None or now >= recent_motion_until:
            return decision

        return DisplayGestureEmojiDecision(reason="suppressed_by_recent_motion_gesture")


__all__ = [
    "DisplayGestureEmojiDecision",
    "DisplayGestureEmojiPublishResult",
    "DisplayGestureEmojiPublisher",
    "decision_for_coarse_gesture",
    "decision_for_fine_hand_gesture",
    "display_gesture_refresh_supported",
    "derive_display_gesture_emoji",
    "resolve_display_gesture_refresh_interval",
]
