"""Expose runtime-facing XVF3800 signal observations for Twinr."""

from __future__ import annotations

from threading import Lock
from typing import Callable
import time

from twinr.hardware.respeaker.models import ReSpeakerPrimitiveSnapshot, ReSpeakerSignalSnapshot
from twinr.hardware.respeaker.snapshot_service import capture_respeaker_primitive_snapshot
from twinr.hardware.respeaker.transport import ReSpeakerLibusbTransport


_DEFAULT_LOCK_TIMEOUT_S = 0.25


class ReSpeakerSignalProvider:
    """Read conservative runtime-facing signals from the XVF3800.

    The provider wraps one bounded primitive snapshot read and adds tiny
    runtime-only state such as ``recent_speech_age_s``. It deliberately emits
    ``None`` for unknown values instead of guessing across host-control or
    permission failures.
    """

    def __init__(
        self,
        *,
        sensor_window_ms: int = 0,
        lock_timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
        transport: ReSpeakerLibusbTransport | None = None,
        snapshot_factory: Callable[..., ReSpeakerPrimitiveSnapshot] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize one provider with bounded lock and snapshot settings."""

        self.sensor_window_ms = max(0, int(sensor_window_ms))
        self.lock_timeout_s = max(0.01, float(lock_timeout_s))
        self.transport = transport or ReSpeakerLibusbTransport()
        self._snapshot_factory = snapshot_factory or capture_respeaker_primitive_snapshot
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._lock = Lock()
        self._last_speech_monotonic: float | None = None

    def observe(self) -> ReSpeakerSignalSnapshot:
        """Capture one conservative runtime-facing XVF3800 signal snapshot."""

        if not self._lock.acquire(timeout=self.lock_timeout_s):
            return ReSpeakerSignalSnapshot(
                captured_at=time.time(),
                source="respeaker_xvf3800",
                source_type="observed",
                sensor_window_ms=self.sensor_window_ms,
                device_runtime_mode="provider_lock_timeout",
                host_control_ready=False,
                transport_reason="provider_lock_timeout",
            )
        try:
            primitive = self._snapshot_factory(transport=self.transport)
            now_monotonic = self._monotonic_clock()
            speech_detected = primitive.direction.speech_detected
            if speech_detected is True:
                self._last_speech_monotonic = now_monotonic
                recent_speech_age_s = 0.0
            elif primitive.host_control_ready and self._last_speech_monotonic is not None:
                recent_speech_age_s = max(0.0, now_monotonic - self._last_speech_monotonic)
            else:
                recent_speech_age_s = None
            return ReSpeakerSignalSnapshot(
                captured_at=primitive.captured_at,
                source="respeaker_xvf3800",
                source_type="observed",
                sensor_window_ms=self.sensor_window_ms,
                device_runtime_mode=primitive.device_runtime_mode,
                host_control_ready=primitive.host_control_ready,
                transport_reason=primitive.transport.reason,
                requires_elevated_permissions=primitive.transport.requires_elevated_permissions,
                firmware_version=primitive.firmware_version,
                speech_detected=speech_detected,
                room_quiet=primitive.direction.room_quiet,
                recent_speech_age_s=recent_speech_age_s,
                azimuth_deg=primitive.direction.doa_degrees,
                beam_activity=primitive.direction.beam_speech_energies,
                mute_active=primitive.mute.mute_active,
                gpo_logic_levels=primitive.mute.gpo_logic_levels,
            )
        except Exception as exc:
            return ReSpeakerSignalSnapshot(
                captured_at=time.time(),
                source="respeaker_xvf3800",
                source_type="observed",
                sensor_window_ms=self.sensor_window_ms,
                device_runtime_mode="signal_provider_error",
                host_control_ready=False,
                transport_reason=f"signal_provider_error:{exc.__class__.__name__}",
            )
        finally:
            self._lock.release()

    def close(self) -> None:
        """Close the provider.

        The current libusb snapshot path is per-call and does not hold open
        device resources between observations, so close is intentionally a
        no-op.
        """

        return None
