"""Official XVF3800 write specs used by Twinr LED control."""

from __future__ import annotations

from typing import Final

from twinr.hardware.respeaker.models import ReSpeakerParameterSpec, ReSpeakerParameterType


LED_EFFECT_PARAMETER: Final[ReSpeakerParameterSpec] = ReSpeakerParameterSpec(
    name="LED_EFFECT",
    resid=20,
    cmdid=12,
    value_count=1,
    access_mode="rw",
    value_type=ReSpeakerParameterType.UINT8,
    description="Set the LED effect mode, 0=off, 3=single color.",
)

LED_COLOR_PARAMETER: Final[ReSpeakerParameterSpec] = ReSpeakerParameterSpec(
    name="LED_COLOR",
    resid=20,
    cmdid=16,
    value_count=1,
    access_mode="rw",
    value_type=ReSpeakerParameterType.UINT32,
    description="Set the LED color for breath or single-color mode as 0xRRGGBB.",
)


__all__ = [
    "LED_COLOR_PARAMETER",
    "LED_EFFECT_PARAMETER",
]
