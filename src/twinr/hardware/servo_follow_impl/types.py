"""Define small value objects used by the servo-follow controller."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class AttentionServoDecision:
    """Describe one servo-follow update for tests and bounded telemetry."""

    observed_at: float | None = None
    active: bool = False
    reason: str = "disabled"
    confidence: float = 0.0
    target_center_x: float | None = None
    applied_center_x: float | None = None
    target_pulse_width_us: int | None = None
    commanded_pulse_width_us: int | None = None
