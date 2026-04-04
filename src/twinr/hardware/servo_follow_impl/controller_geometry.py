"""Keep pure servo geometry and target-to-pulse conversions separate."""

from __future__ import annotations

from .common import _clamp, _clamp_ratio
from .controller_attrs import ControllerAttrsMixin

class ControllerGeometryMixin(ControllerAttrsMixin):
    def _clamped_follow_center_x(self, center_x: float) -> float:
        """Clamp exit-only follow targets to the configured off-center angle."""

        normalized_center_x = _clamp_ratio(center_x)
        if not self.config.follow_exit_only:
            return normalized_center_x
        normalized_offset = (normalized_center_x - 0.5) * 2.0
        limited_offset = _clamp(
            normalized_offset,
            minimum=-self.config.exit_follow_offset_limit,
            maximum=self.config.exit_follow_offset_limit,
        )
        return _clamp_ratio(0.5 + (limited_offset * 0.5))

    def _maximum_target_heading_degrees(self) -> float:
        if self.config.follow_exit_only:
            return max(1.0, float(self.config.exit_follow_max_degrees))
        return self.config.resolved_position_follow_max_degrees

    def _pulse_width_for_heading_degrees(self, heading_degrees: float) -> int:
        """Convert one bounded heading request into the calibrated pulse envelope."""

        bounded_heading_degrees = _clamp(
            float(heading_degrees),
            minimum=-self._maximum_target_heading_degrees(),
            maximum=self._maximum_target_heading_degrees(),
        )
        mechanical_half_range_degrees = max(1.0, float(self.config.mechanical_range_degrees) * 0.5)
        normalized_heading = _clamp(
            bounded_heading_degrees / mechanical_half_range_degrees,
            minimum=-1.0,
            maximum=1.0,
        )
        if normalized_heading >= 0.0:
            span = self.config.safe_max_pulse_width_us - self.config.center_pulse_width_us
        else:
            span = self.config.center_pulse_width_us - self.config.safe_min_pulse_width_us
        pulse_width = int(round(self.config.center_pulse_width_us + (normalized_heading * span)))
        return max(self.config.safe_min_pulse_width_us, min(self.config.safe_max_pulse_width_us, pulse_width))

    def _desired_heading_degrees_for_center_x(self, center_x: float) -> float:
        normalized_center_x = self._clamped_follow_center_x(center_x)
        normalized_offset = (normalized_center_x - 0.5) * 2.0
        if self.config.invert_direction:
            normalized_offset *= -1.0
        if abs(normalized_offset) <= self.config.deadband:
            normalized_offset = 0.0
        return normalized_offset * self._maximum_target_heading_degrees()

    def _pulse_width_for_center_x(self, center_x: float) -> int:
        if self.config.follow_exit_only:
            normalized_center_x = self._clamped_follow_center_x(center_x)
            normalized_offset = (normalized_center_x - 0.5) * 2.0
            if self.config.invert_direction:
                normalized_offset *= -1.0
            if abs(normalized_offset) <= self.config.deadband:
                normalized_offset = 0.0
            if normalized_offset >= 0.0:
                span = self.config.safe_max_pulse_width_us - self.config.center_pulse_width_us
            else:
                span = self.config.center_pulse_width_us - self.config.safe_min_pulse_width_us
            pulse_width = int(round(self.config.center_pulse_width_us + (normalized_offset * span)))
            return max(self.config.safe_min_pulse_width_us, min(self.config.safe_max_pulse_width_us, pulse_width))
        return self._pulse_width_for_heading_degrees(
            self._desired_heading_degrees_for_center_x(center_x)
        )

    def _continuous_tracking_pulse_width_for_center_x(self, center_x: float) -> int:
        """Map live image error to one bounded continuous-servo speed command."""

        normalized_center_x = self._clamped_follow_center_x(center_x)
        normalized_offset = (normalized_center_x - 0.5) * 2.0
        if self.config.invert_direction:
            normalized_offset *= -1.0
        absolute_offset = abs(normalized_offset)
        if absolute_offset <= self.config.deadband:
            return self.config.center_pulse_width_us
        offset_ratio = (absolute_offset - self.config.deadband) / max(1e-6, 1.0 - self.config.deadband)
        bounded_offset_ratio = max(0.0, min(1.0, offset_ratio))
        pulse_delta_us = self.config.continuous_min_speed_pulse_delta_us
        if self.config.continuous_max_speed_pulse_delta_us > self.config.continuous_min_speed_pulse_delta_us:
            pulse_delta_us += int(
                round(
                    (
                        self.config.continuous_max_speed_pulse_delta_us
                        - self.config.continuous_min_speed_pulse_delta_us
                    )
                    * bounded_offset_ratio
                )
            )
        direction_sign = 1 if normalized_offset > 0.0 else -1
        pulse_width_us = self.config.center_pulse_width_us + (direction_sign * pulse_delta_us)
        return max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, pulse_width_us),
        )
