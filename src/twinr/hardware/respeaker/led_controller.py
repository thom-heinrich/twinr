"""Bounded XVF3800 LED writes for Twinr runtime feedback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from twinr.hardware.respeaker.led_profiles import ReSpeakerLedProfile
from twinr.hardware.respeaker.transport import ReSpeakerLibusbTransport
from twinr.hardware.respeaker.write_specs import LED_COLOR_PARAMETER, LED_EFFECT_PARAMETER


_LED_EFFECT_OFF = 0
_LED_EFFECT_SINGLE_COLOR = 3
_EFFECT_REFRESH_INTERVAL_S = 1.0


def _default_emit(line: str) -> None:
    """Print one bounded telemetry line."""

    print(line, flush=True)


@dataclass(slots=True)
class ReSpeakerLedController:
    """Drive the XVF3800 ring with explicit Twinr-owned RGB pulses."""

    transport: ReSpeakerLibusbTransport = field(default_factory=ReSpeakerLibusbTransport)
    emit: Callable[[str], None] = _default_emit
    effect_refresh_interval_s: float = _EFFECT_REFRESH_INTERVAL_S
    _last_effect: int | None = field(default=None, init=False, repr=False)
    _last_effect_write_s: float | None = field(default=None, init=False, repr=False)
    _last_color_value: int | None = field(default=None, init=False, repr=False)
    _last_error_reason: str | None = field(default=None, init=False, repr=False)

    def render(self, profile: ReSpeakerLedProfile, *, at_monotonic_s: float) -> bool:
        """Render one color sample for the current LED profile."""

        rgb = profile.scaled_rgb(timestamp_s=at_monotonic_s)
        return self._set_single_color(rgb, at_monotonic_s=at_monotonic_s)

    def off(self) -> bool:
        """Switch the XVF3800 ring off."""

        availability = self.transport.write_parameter(LED_EFFECT_PARAMETER, (_LED_EFFECT_OFF,))
        if not availability.available:
            self._emit_error(f"off_failed:{availability.reason or 'unknown'}")
            return False
        self._clear_error()
        self._last_effect = _LED_EFFECT_OFF
        self._last_effect_write_s = None
        self._last_color_value = None
        return True

    def _set_single_color(self, rgb: tuple[int, int, int], *, at_monotonic_s: float) -> bool:
        if self._should_refresh_single_color_effect(at_monotonic_s=at_monotonic_s):
            effect_result = self.transport.write_parameter(
                LED_EFFECT_PARAMETER,
                (_LED_EFFECT_SINGLE_COLOR,),
            )
            if not effect_result.available:
                self._emit_error(f"effect_failed:{effect_result.reason or 'unknown'}")
                return False
            self._clear_error()
            self._last_effect = _LED_EFFECT_SINGLE_COLOR
            self._last_effect_write_s = max(0.0, float(at_monotonic_s))
            self._last_color_value = None

        color_value = _rgb_to_uint32(rgb)
        if color_value == self._last_color_value:
            return False

        color_result = self.transport.write_parameter(LED_COLOR_PARAMETER, (color_value,))
        if not color_result.available:
            self._emit_error(f"color_failed:{color_result.reason or 'unknown'}")
            return False
        self._clear_error()
        self._last_color_value = color_value
        return True

    def _should_refresh_single_color_effect(self, *, at_monotonic_s: float) -> bool:
        if self._last_effect != _LED_EFFECT_SINGLE_COLOR:
            return True
        if self._last_effect_write_s is None:
            return True
        return (max(0.0, float(at_monotonic_s)) - self._last_effect_write_s) >= self.effect_refresh_interval_s

    def _emit_error(self, reason: str) -> None:
        normalized = " ".join(str(reason).split()).strip() or "unknown"
        if normalized == self._last_error_reason:
            return
        self._last_error_reason = normalized
        self.emit(f"respeaker_led_write_failed={normalized}")

    def _clear_error(self) -> None:
        self._last_error_reason = None


def _rgb_to_uint32(rgb: tuple[int, int, int]) -> int:
    red, green, blue = (_normalize_channel(value) for value in rgb)
    return (red << 16) | (green << 8) | blue


def _normalize_channel(value: int) -> int:
    return max(0, min(255, int(value)))


__all__ = [
    "ReSpeakerLedController",
]
