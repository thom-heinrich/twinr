"""Define the stable config constants shared by Twinr runtime loading.

Purpose and boundaries:
- Own only immutable constant values and the voice-activation defaults import.
- Stay free of parsing, normalization, and env-loading logic.
- Provide a single source of truth for cross-module config defaults.
"""

from __future__ import annotations

from twinr.orchestrator.voice_activation import DEFAULT_VOICE_ACTIVATION_PHRASES

DEFAULT_BUTTON_PROBE_LINES = (
    4,
    5,
    6,
    12,
    13,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
)
SUPPORTED_DISPLAY_DRIVERS = (
    "hdmi_wayland",
    "hdmi_fbdev",
    "waveshare_4in2_v2",
)
GPIO_DISPLAY_DRIVERS = frozenset({"waveshare_4in2_v2"})
SUPPORTED_DISPLAY_LAYOUTS = (
    "default",
    "debug_log",
    "debug_face",
)
DEFAULT_OPENAI_MAIN_MODEL = "gpt-5.4-mini"

__all__ = [
    "DEFAULT_BUTTON_PROBE_LINES",
    "SUPPORTED_DISPLAY_DRIVERS",
    "GPIO_DISPLAY_DRIVERS",
    "SUPPORTED_DISPLAY_LAYOUTS",
    "DEFAULT_OPENAI_MAIN_MODEL",
    "DEFAULT_VOICE_ACTIVATION_PHRASES",
]
