"""Resolve calm XVF3800 LED profiles from Twinr runtime state."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ReSpeakerLedProfile:
    """Describe one calm RGB pulse profile for the XVF3800 ring."""

    state: str
    color_rgb: tuple[int, int, int]
    pulse_hz: float
    min_scale: float
    max_scale: float
    gamma: float = 1.6

    def scaled_rgb(self, *, timestamp_s: float) -> tuple[int, int, int]:
        """Return one time-varying RGB color for the current pulse phase."""

        pulse_hz = max(0.0, float(self.pulse_hz))
        if pulse_hz <= 0.0:
            scale = self.max_scale
        else:
            wave = 0.5 - (0.5 * math.cos((2.0 * math.pi * pulse_hz) * max(0.0, float(timestamp_s))))
            scale = self.min_scale + ((self.max_scale - self.min_scale) * (wave**self.gamma))
        red, green, blue = self.color_rgb
        return (
            _scale_channel(red, scale),
            _scale_channel(green, scale),
            _scale_channel(blue, scale),
        )


def _scale_channel(channel: int, scale: float) -> int:
    bounded_channel = max(0, min(255, int(channel)))
    bounded_scale = max(0.0, min(1.0, float(scale)))
    return max(0, min(255, int(round(bounded_channel * bounded_scale))))


# The requested LED rates only make sense as glanceable HCI when interpreted as
# pulses per second; minute-scale breathing would look almost static on device.
# On-device feedback showed the initial rates felt about 50% too fast, so the
# production profiles run at two-thirds of the first tuning pass.
_FIELD_TUNED_PULSE_SCALE = 2.0 / 3.0

WAITING_LED_PROFILE = ReSpeakerLedProfile(
    state="waiting",
    color_rgb=(255, 176, 84),
    pulse_hz=0.8 * _FIELD_TUNED_PULSE_SCALE,
    min_scale=0.18,
    max_scale=0.74,
    gamma=1.8,
)
LISTENING_LED_PROFILE = ReSpeakerLedProfile(
    state="listening",
    color_rgb=(255, 244, 188),
    pulse_hz=1.6 * _FIELD_TUNED_PULSE_SCALE,
    min_scale=0.28,
    max_scale=1.0,
    gamma=1.55,
)
PROCESSING_LED_PROFILE = ReSpeakerLedProfile(
    state="processing",
    color_rgb=(88, 224, 232),
    pulse_hz=1.0 * _FIELD_TUNED_PULSE_SCALE,
    min_scale=0.22,
    max_scale=0.95,
    gamma=1.6,
)
ANSWERING_LED_PROFILE = ReSpeakerLedProfile(
    state="answering",
    color_rgb=(72, 132, 255),
    pulse_hz=1.6 * _FIELD_TUNED_PULSE_SCALE,
    min_scale=0.28,
    max_scale=1.0,
    gamma=1.45,
)
ERROR_LED_PROFILE = ReSpeakerLedProfile(
    state="error",
    color_rgb=(255, 68, 68),
    pulse_hz=0.5 * _FIELD_TUNED_PULSE_SCALE,
    min_scale=0.20,
    max_scale=1.0,
    gamma=1.35,
)


def resolve_respeaker_led_profile(
    *,
    runtime_status: str | None,
    error_message: str | None = None,
) -> ReSpeakerLedProfile:
    """Return the calm LED profile for one Twinr runtime status."""

    normalized_status = str(runtime_status or "").strip().lower()
    normalized_error = str(error_message or "").strip()
    if normalized_status == "error" or normalized_error:
        return ERROR_LED_PROFILE
    if normalized_status == "listening":
        return LISTENING_LED_PROFILE
    if normalized_status in {"processing", "printing"}:
        return PROCESSING_LED_PROFILE
    if normalized_status == "answering":
        return ANSWERING_LED_PROFILE
    return WAITING_LED_PROFILE


__all__ = [
    "ANSWERING_LED_PROFILE",
    "ERROR_LED_PROFILE",
    "LISTENING_LED_PROFILE",
    "PROCESSING_LED_PROFILE",
    "ReSpeakerLedProfile",
    "WAITING_LED_PROFILE",
    "resolve_respeaker_led_profile",
]
