"""Throttle expensive ReSpeaker host-control reads behind a bounded schedule.

This wrapper keeps the XVF3800 signal path cheap enough for Pi runtime loops by
refreshing heavy host-control snapshots aggressively only while interaction is
active and otherwise serving a recent cached snapshot. The fallback ambient
audio path still runs every proactive tick, so Twinr keeps continuous cheap
audio sensing while direction and host-control polling stay out of the hot
path when the room is idle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.respeaker.models import ReSpeakerSignalSnapshot


class _SignalSnapshotProvider(Protocol):
    """Describe the small surface required from an XVF3800 signal provider."""

    def observe(self) -> ReSpeakerSignalSnapshot:
        """Return one current ReSpeaker signal snapshot."""

    def close(self) -> None:
        """Release nested resources when supported."""


@dataclass(frozen=True, slots=True)
class ReSpeakerSchedulingContext:
    """Describe the current runtime activity that should influence polling."""

    observed_at: float
    motion_active: bool = False
    inspect_requested: bool = False
    presence_session_armed: bool = False
    assistant_output_active: bool = False


class ScheduledReSpeakerSignalProvider:
    """Cache XVF3800 host-control reads behind bounded activity-aware refresh."""

    def __init__(
        self,
        *,
        provider: _SignalSnapshotProvider,
        active_refresh_interval_s: float,
        degraded_refresh_interval_s: float,
        idle_refresh_interval_s: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize one scheduled wrapper around a real signal provider."""

        self.provider = provider
        self.active_refresh_interval_s = _coerce_seconds(
            active_refresh_interval_s,
            default=0.5,
            minimum=0.1,
            maximum=10.0,
        )
        self.degraded_refresh_interval_s = _coerce_seconds(
            degraded_refresh_interval_s,
            default=min(self.active_refresh_interval_s, 1.0),
            minimum=0.1,
            maximum=10.0,
        )
        self.idle_refresh_interval_s = _coerce_seconds(
            idle_refresh_interval_s,
            default=max(self.active_refresh_interval_s, 6.0),
            minimum=self.active_refresh_interval_s,
            maximum=30.0,
        )
        self._clock = clock if clock is not None else time.monotonic
        self._latest_snapshot: ReSpeakerSignalSnapshot | None = None
        self._latest_observed_at: float | None = None
        self._context = ReSpeakerSchedulingContext(observed_at=0.0)

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        provider: _SignalSnapshotProvider,
        clock: Callable[[], float] | None = None,
    ) -> "ScheduledReSpeakerSignalProvider":
        """Build one scheduler from existing proactive cadence config."""

        active_refresh = _coerce_seconds(
            getattr(config, "proactive_poll_interval_s", 4.0),
            default=4.0,
            minimum=0.1,
            maximum=8.0,
        )
        idle_refresh = max(
            active_refresh,
            _coerce_seconds(
                getattr(config, "proactive_capture_interval_s", 6.0),
                default=6.0,
                minimum=0.5,
                maximum=30.0,
            ),
        )
        degraded_refresh = min(active_refresh, 1.0)
        return cls(
            provider=provider,
            active_refresh_interval_s=active_refresh,
            degraded_refresh_interval_s=degraded_refresh,
            idle_refresh_interval_s=idle_refresh,
            clock=clock,
        )

    def note_runtime_context(
        self,
        *,
        observed_at: float,
        motion_active: bool,
        inspect_requested: bool,
        presence_session_armed: bool,
        assistant_output_active: bool,
    ) -> None:
        """Update the current runtime context used for refresh decisions."""

        self._context = ReSpeakerSchedulingContext(
            observed_at=_coerce_seconds(observed_at, default=0.0, minimum=0.0, maximum=10.0**9),
            motion_active=motion_active is True,
            inspect_requested=inspect_requested is True,
            presence_session_armed=presence_session_armed is True,
            assistant_output_active=assistant_output_active is True,
        )

    def observe(self) -> ReSpeakerSignalSnapshot:
        """Return a cached or freshly refreshed ReSpeaker signal snapshot."""

        now = self._safe_clock()
        if self._should_refresh(now):
            snapshot = self.provider.observe()
            self._latest_snapshot = snapshot
            self._latest_observed_at = now
        if self._latest_snapshot is None:
            snapshot = self.provider.observe()
            self._latest_snapshot = snapshot
            self._latest_observed_at = now
        return self._latest_snapshot

    def close(self) -> None:
        """Close the wrapped provider when it supports shutdown."""

        self.provider.close()

    def _should_refresh(self, now: float) -> bool:
        """Return whether the underlying provider should refresh now."""

        snapshot = self._latest_snapshot
        if snapshot is None:
            return True
        last_observed_at = self._latest_observed_at
        if last_observed_at is None:
            return True
        age_s = max(0.0, now - last_observed_at)
        interval_s = self._current_refresh_interval(snapshot=snapshot)
        return age_s >= interval_s

    def _current_refresh_interval(self, *, snapshot: ReSpeakerSignalSnapshot) -> float:
        """Choose the current refresh interval from cached state and context."""

        if snapshot.device_runtime_mode != "audio_ready" or snapshot.host_control_ready is not True:
            return self.degraded_refresh_interval_s
        if self._active_context(snapshot=snapshot):
            return self.active_refresh_interval_s
        return self.idle_refresh_interval_s

    def _active_context(self, *, snapshot: ReSpeakerSignalSnapshot) -> bool:
        """Return whether the runtime currently needs faster ReSpeaker refresh."""

        if self._context.assistant_output_active:
            return True
        if self._context.presence_session_armed:
            return True
        if self._context.motion_active:
            return True
        if self._context.inspect_requested:
            return True
        if snapshot.speech_detected is True:
            return True
        if snapshot.recent_speech_age_s is not None and snapshot.recent_speech_age_s <= 5.0:
            return True
        return False

    def _safe_clock(self) -> float:
        """Read the local monotonic clock conservatively."""

        try:
            value = float(self._clock())
        except (TypeError, ValueError):
            return 0.0
        if value != value or value < 0.0:
            return 0.0
        return value


def _coerce_seconds(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Return one finite bounded duration in seconds."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if number != number:
        number = default
    return max(minimum, min(maximum, number))


__all__ = [
    "ReSpeakerSchedulingContext",
    "ScheduledReSpeakerSignalProvider",
]
