"""Open-loop heading planner for continuous-rotation attention servos.

Continuous-rotation servos do not accept absolute angle targets; pulse widths
control direction and speed around a neutral stop pulse. This helper keeps a
bounded virtual heading estimate so higher attention logic can still reason in
desired headings while the hardware path stays explicitly open-loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True, slots=True)
class ContinuousRotationServoConfig:
    """Tune the open-loop heading estimator for one continuous servo."""

    center_pulse_width_us: int = 1500
    min_pulse_width_us: int = 1050
    max_pulse_width_us: int = 1950
    max_heading_degrees: float = 90.0
    max_speed_degrees_per_s: float = 120.0
    slow_zone_degrees: float = 45.0
    stop_tolerance_degrees: float = 4.0
    min_speed_pulse_delta_us: int = 70
    max_speed_pulse_delta_us: int = 160

    def __post_init__(self) -> None:
        min_pulse = int(self.min_pulse_width_us)
        max_pulse = int(self.max_pulse_width_us)
        center_pulse = int(self.center_pulse_width_us)
        if max_pulse < min_pulse:
            min_pulse, max_pulse = max_pulse, min_pulse
        center_pulse = max(min_pulse, min(max_pulse, center_pulse))
        available_span = max(1, min(center_pulse - min_pulse, max_pulse - center_pulse))
        min_speed_delta = max(0, min(int(self.min_speed_pulse_delta_us), available_span))
        max_speed_delta = max(min_speed_delta, min(int(self.max_speed_pulse_delta_us), available_span))
        max_heading = max(1.0, float(self.max_heading_degrees))
        max_speed = max(1.0, float(self.max_speed_degrees_per_s))
        slow_zone = max(0.5, min(float(self.slow_zone_degrees), max_heading))
        stop_tolerance = max(0.0, min(float(self.stop_tolerance_degrees), slow_zone))
        object.__setattr__(self, "min_pulse_width_us", min_pulse)
        object.__setattr__(self, "max_pulse_width_us", max_pulse)
        object.__setattr__(self, "center_pulse_width_us", center_pulse)
        object.__setattr__(self, "max_heading_degrees", max_heading)
        object.__setattr__(self, "max_speed_degrees_per_s", max_speed)
        object.__setattr__(self, "slow_zone_degrees", slow_zone)
        object.__setattr__(self, "stop_tolerance_degrees", stop_tolerance)
        object.__setattr__(self, "min_speed_pulse_delta_us", min_speed_delta)
        object.__setattr__(self, "max_speed_pulse_delta_us", max_speed_delta)


class ContinuousRotationServoPlanner:
    """Estimate heading from commanded speed pulses and derive the next command."""

    def __init__(self, config: ContinuousRotationServoConfig) -> None:
        self.config = config
        self._estimated_heading_degrees = 0.0
        self._desired_heading_degrees = 0.0
        self._last_observed_at: float | None = None
        self._last_commanded_pulse_width_us = config.center_pulse_width_us
        self._last_command_speed_scale = 1.0

    @property
    def estimated_heading_degrees(self) -> float:
        return self._estimated_heading_degrees

    @property
    def desired_heading_degrees(self) -> float:
        return self._desired_heading_degrees

    @property
    def last_commanded_pulse_width_us(self) -> int:
        return self._last_commanded_pulse_width_us

    def target_pulse_width_for_heading(self, desired_heading_degrees: float, *, observed_at: float | None) -> int:
        """Return one bounded speed command that moves the estimate toward the desired heading."""

        self._advance(observed_at=observed_at)
        bounded_desired_heading = _clamp(
            float(desired_heading_degrees),
            minimum=-self.config.max_heading_degrees,
            maximum=self.config.max_heading_degrees,
        )
        self._desired_heading_degrees = bounded_desired_heading
        error_degrees = bounded_desired_heading - self._estimated_heading_degrees
        if abs(error_degrees) <= self.config.stop_tolerance_degrees:
            return self.config.center_pulse_width_us
        error_beyond_stop = max(0.0, abs(error_degrees) - self.config.stop_tolerance_degrees)
        usable_zone = max(
            0.5,
            self.config.slow_zone_degrees - self.config.stop_tolerance_degrees,
        )
        speed_ratio = min(1.0, error_beyond_stop / usable_zone)
        pulse_delta_us = self.config.min_speed_pulse_delta_us
        if self.config.max_speed_pulse_delta_us > self.config.min_speed_pulse_delta_us:
            pulse_delta_us += int(
                round(
                    (self.config.max_speed_pulse_delta_us - self.config.min_speed_pulse_delta_us) * speed_ratio
                )
            )
        direction = 1 if error_degrees > 0.0 else -1
        bounded_pulse = self.config.center_pulse_width_us + (direction * pulse_delta_us)
        return max(
            self.config.min_pulse_width_us,
            min(self.config.max_pulse_width_us, bounded_pulse),
        )

    def note_commanded_pulse_width(
        self,
        pulse_width_us: int,
        *,
        observed_at: float | None,
        speed_scale: float = 1.0,
    ) -> None:
        """Advance the estimate to `observed_at` and remember the actual applied command."""

        self._advance(observed_at=observed_at)
        self._last_commanded_pulse_width_us = max(
            self.config.min_pulse_width_us,
            min(self.config.max_pulse_width_us, int(pulse_width_us)),
        )
        checked_speed_scale = float(speed_scale)
        if not math.isfinite(checked_speed_scale):
            checked_speed_scale = 1.0
        self._last_command_speed_scale = max(0.0, checked_speed_scale)

    def note_stopped(self, *, observed_at: float | None) -> None:
        """Advance once more and freeze subsequent motion around the neutral pulse."""

        self._advance(observed_at=observed_at)
        self._last_commanded_pulse_width_us = self.config.center_pulse_width_us
        self._last_command_speed_scale = 1.0

    def reset(self, *, heading_degrees: float = 0.0, observed_at: float | None = None) -> None:
        """Reset the virtual heading and neutralize the remembered command."""

        self._estimated_heading_degrees = _clamp(
            float(heading_degrees),
            minimum=-self.config.max_heading_degrees,
            maximum=self.config.max_heading_degrees,
        )
        self._desired_heading_degrees = self._estimated_heading_degrees
        self._last_commanded_pulse_width_us = self.config.center_pulse_width_us
        self._last_command_speed_scale = 1.0
        self._last_observed_at = None if observed_at is None else float(observed_at)

    def _advance(self, *, observed_at: float | None) -> None:
        checked_at = None if observed_at is None else float(observed_at)
        previous_at = self._last_observed_at
        if checked_at is None:
            return
        if previous_at is None:
            self._last_observed_at = checked_at
            return
        dt_s = checked_at - previous_at
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            self._last_observed_at = checked_at
            return
        speed_degrees_per_s = (
            self._speed_degrees_per_s_for_pulse(self._last_commanded_pulse_width_us)
            * self._last_command_speed_scale
        )
        if speed_degrees_per_s != 0.0:
            self._estimated_heading_degrees = _clamp(
                self._estimated_heading_degrees + (speed_degrees_per_s * dt_s),
                minimum=-self.config.max_heading_degrees,
                maximum=self.config.max_heading_degrees,
            )
        self._last_observed_at = checked_at

    def _speed_degrees_per_s_for_pulse(self, pulse_width_us: int) -> float:
        pulse_delta_us = int(pulse_width_us) - self.config.center_pulse_width_us
        if abs(pulse_delta_us) < self.config.min_speed_pulse_delta_us:
            return 0.0
        usable_delta_us = max(
            1,
            self.config.max_speed_pulse_delta_us - self.config.min_speed_pulse_delta_us,
        )
        extra_delta_us = min(
            usable_delta_us,
            max(0, abs(pulse_delta_us) - self.config.min_speed_pulse_delta_us),
        )
        speed_ratio = extra_delta_us / usable_delta_us
        speed_degrees_per_s = self.config.max_speed_degrees_per_s * (0.35 + (0.65 * speed_ratio))
        return speed_degrees_per_s if pulse_delta_us > 0 else -speed_degrees_per_s
