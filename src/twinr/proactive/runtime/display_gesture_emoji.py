"""Mirror recognized camera gestures into bounded HDMI emoji acknowledgements.

This module keeps gesture-to-emoji acknowledgement policy separate from the
main proactive monitor orchestration. It listens only to stabilized rising-edge
camera gesture events and publishes a short-lived right-hand HDMI emoji cue
without touching the face channel.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

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

_FINE_HAND_GESTURE_MAP: dict[SocialFineHandGesture, tuple[DisplayEmojiSymbol, str]] = {
    SocialFineHandGesture.THUMBS_UP: (DisplayEmojiSymbol.THUMBS_UP, "success"),
    SocialFineHandGesture.THUMBS_DOWN: (DisplayEmojiSymbol.THUMBS_DOWN, "warm"),
    SocialFineHandGesture.POINTING: (DisplayEmojiSymbol.POINTING_HAND, "info"),
    SocialFineHandGesture.OPEN_PALM: (DisplayEmojiSymbol.RAISED_HAND, "info"),
    SocialFineHandGesture.OK_SIGN: (DisplayEmojiSymbol.OK_HAND, "success"),
}

_COARSE_GESTURE_MAP: dict[SocialGestureEvent, tuple[DisplayEmojiSymbol, str]] = {
    SocialGestureEvent.WAVE: (DisplayEmojiSymbol.WAVING_HAND, "warm"),
    SocialGestureEvent.STOP: (DisplayEmojiSymbol.RAISED_HAND, "info"),
    SocialGestureEvent.DISMISS: (DisplayEmojiSymbol.WAVING_HAND, "neutral"),
    SocialGestureEvent.TWO_HAND_DISMISS: (DisplayEmojiSymbol.WAVING_HAND, "neutral"),
    SocialGestureEvent.CONFIRM: (DisplayEmojiSymbol.CHECK, "success"),
}


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
        mapped = _COARSE_GESTURE_MAP.get(snapshot.gesture_event)
        if mapped is not None:
            symbol, accent = mapped
            return DisplayGestureEmojiDecision(
                active=True,
                reason=f"motion_coarse_gesture:{snapshot.gesture_event.value}",
                symbol=symbol,
                accent=accent,
            )

    if _FINE_GESTURE_EVENT_NAME in names:
        mapped = _FINE_HAND_GESTURE_MAP.get(snapshot.fine_hand_gesture)
        if mapped is not None and not snapshot.fine_hand_gesture_unknown:
            symbol, accent = mapped
            return DisplayGestureEmojiDecision(
                active=True,
                reason=f"fine_hand_gesture:{snapshot.fine_hand_gesture.value}",
                symbol=symbol,
                accent=accent,
            )
        return DisplayGestureEmojiDecision(reason="unsupported_fine_hand_gesture")

    if _COARSE_GESTURE_EVENT_NAMES.intersection(names):
        mapped = _COARSE_GESTURE_MAP.get(snapshot.gesture_event)
        if mapped is not None and not snapshot.gesture_event_unknown:
            symbol, accent = mapped
            return DisplayGestureEmojiDecision(
                active=True,
                reason=f"coarse_gesture:{snapshot.gesture_event.value}",
                symbol=symbol,
                accent=accent,
            )
        return DisplayGestureEmojiDecision(reason="unsupported_coarse_gesture")

    return DisplayGestureEmojiDecision(reason="no_gesture_event")


@dataclass(slots=True)
class DisplayGestureEmojiPublisher:
    """Persist short-lived HDMI emoji acknowledgements for user gestures."""

    controller: DisplayEmojiController
    source: str = _SOURCE

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

        decision = derive_display_gesture_emoji(
            snapshot=update.snapshot,
            event_names=update.event_names,
        )
        return self.publish(decision, now=now)

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


__all__ = [
    "DisplayGestureEmojiDecision",
    "DisplayGestureEmojiPublishResult",
    "DisplayGestureEmojiPublisher",
    "derive_display_gesture_emoji",
]
