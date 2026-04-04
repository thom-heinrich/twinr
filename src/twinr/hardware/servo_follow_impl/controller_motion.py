"""Keep generic pulse-planning and release heuristics focused."""

from __future__ import annotations

import math

from .constants import (
    _MAX_RELEASE_TOLERANCE_US,
    _MIN_RELEASE_TOLERANCE_US,
)

from .common import _clamp, _clamp_ratio
from .controller_state import ControllerStateMixin
from .types import AttentionServoDecision

class ControllerMotionMixin(ControllerStateMixin):
    def _write_target_pulse_width(
        self,
        *,
        observed_at: float | None,
        target_pulse_width_us: int,
        motion_profile: str,
    ) -> int:
        planned_pulse_width_us = self._advance_planned_pulse_width(
            target_pulse_width_us,
            observed_at=observed_at,
            motion_profile=motion_profile,
        )
        commanded_pulse_width_us = self._command_pulse_width_for_plan(
            planned_pulse_width_us,
            target_pulse_width_us=target_pulse_width_us,
            motion_profile=motion_profile,
        )
        try:
            if self.config.gpio is None:
                raise RuntimeError("Attention servo output is enabled without a configured output channel")
            if (
                self._last_commanded_pulse_width_us is None
                or commanded_pulse_width_us != self._last_commanded_pulse_width_us
            ):
                self._close_active_movement_journal_segment(
                    observed_at=observed_at,
                    record=not self._return_to_zero_requested,
                )
                self._pulse_writer.write(
                    gpio_chip=self.config.gpio_chip,
                    gpio=self.config.gpio,
                    pulse_width_us=commanded_pulse_width_us,
                )
                if self._continuous_planner is not None:
                    self._continuous_planner.note_commanded_pulse_width(
                        commanded_pulse_width_us,
                        observed_at=observed_at,
                    )
                self._last_commanded_pulse_width_us = commanded_pulse_width_us
                self._last_physical_pulse_width_us = commanded_pulse_width_us
                self._persist_runtime_state(observed_at=observed_at)
            if (
                str(motion_profile or "").strip().lower() == "rest"
                and commanded_pulse_width_us == self.config.center_pulse_width_us
            ):
                self._reset_motion_state(commanded_pulse_width_us)
            self._last_update_at = observed_at
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        return commanded_pulse_width_us

    def _write_immediate_target_pulse_width(
        self,
        *,
        observed_at: float | None,
        target_pulse_width_us: int,
        planner_speed_scale: float = 1.0,
    ) -> int:
        """Write one already-bounded continuous-servo target without extra smoothing.

        The continuous zero-return path is already slowed by explicit move
        windows around neutral. A second motion-profile ramp can shrink the
        effective command below the servo's real movement threshold and trap
        the controller in endless no-progress nudges.
        """

        commanded_pulse_width_us = max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, int(target_pulse_width_us)),
        )
        try:
            if self.config.gpio is None:
                raise RuntimeError("Attention servo output is enabled without a configured output channel")
            if (
                self._last_commanded_pulse_width_us is None
                or commanded_pulse_width_us != self._last_commanded_pulse_width_us
            ):
                self._pulse_writer.write(
                    gpio_chip=self.config.gpio_chip,
                    gpio=self.config.gpio,
                    pulse_width_us=commanded_pulse_width_us,
                )
                if self._continuous_planner is not None:
                    self._continuous_planner.note_commanded_pulse_width(
                        commanded_pulse_width_us,
                        observed_at=observed_at,
                        speed_scale=planner_speed_scale,
                    )
                self._last_commanded_pulse_width_us = commanded_pulse_width_us
                self._last_physical_pulse_width_us = commanded_pulse_width_us
                if not self._return_to_zero_requested:
                    self._start_active_movement_journal_segment(
                        pulse_width_us=commanded_pulse_width_us,
                        observed_at=observed_at,
                    )
                self._persist_runtime_state(observed_at=observed_at)
            self._last_update_at = observed_at
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        return commanded_pulse_width_us

    def _effective_dt_s(self, *, observed_at: float | None) -> float:
        previous = self._last_update_at
        if observed_at is None or previous is None or observed_at <= previous:
            return self.config.reference_interval_s
        return max(0.001, min(observed_at - previous, self.config.reference_interval_s))

    def _direct_hardware_tracking_target(self, *, target_pulse_width_us: int) -> int:
        """Return one direct target for hardware-ramped visible tracking.

        The Maestro can shape motion internally with speed and acceleration
        limits. During active visible tracking, sending only tiny software
        increments fights that hardware ramp and produces barely visible jitter.
        """

        checked_target_us = max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, int(target_pulse_width_us)),
        )
        if self._last_physical_pulse_width_us is not None:
            self._reset_motion_state(self._last_physical_pulse_width_us)
        previous_commanded_pulse_width_us = self._last_commanded_pulse_width_us
        if previous_commanded_pulse_width_us is not None and abs(
            checked_target_us - previous_commanded_pulse_width_us
        ) < self.config.min_command_delta_us:
            return previous_commanded_pulse_width_us
        return checked_target_us

    def _stabilize_visible_target_pulse_width(self, *, target_pulse_width_us: int, reason: str) -> int:
        """Latch visible targets so servo follow ignores millimeter re-justification."""

        checked_target_us = max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, int(target_pulse_width_us)),
        )
        if reason != "following_target":
            self._visible_target_pulse_width_us = None
            return checked_target_us
        tolerance_us = max(0, self.config.visible_retarget_tolerance_us)
        latched_target_us = self._visible_target_pulse_width_us
        if tolerance_us <= 0 or latched_target_us is None:
            self._visible_target_pulse_width_us = checked_target_us
            return checked_target_us
        if abs(checked_target_us - latched_target_us) <= tolerance_us:
            return latched_target_us
        self._visible_target_pulse_width_us = checked_target_us
        return checked_target_us

    def _smoothed_target_center_x(
        self,
        *,
        observed_at: float | None,
        target_center_x: float,
        active_tracking: bool = False,
    ) -> float:
        normalized_target = _clamp_ratio(target_center_x)
        previous = 0.5 if self._smoothed_center_x is None else self._smoothed_center_x
        if (
            active_tracking
            and self._continuous_planner is not None
            and ((previous - 0.5) * (normalized_target - 0.5)) < 0.0
        ):
            # Active visible follow should be able to cross image center
            # promptly. Reusing the old-side smoothed center here creates a
            # practical center barrier where the servo keeps drifting on the
            # stale side long after the live target has crossed over.
            smoothed = normalized_target
            self._visible_target_pulse_width_us = None
        elif self.config.target_smoothing_s <= 0.0:
            smoothed = normalized_target
        else:
            dt = self._effective_dt_s(observed_at=observed_at)
            alpha = max(0.0, min(1.0, dt / max(dt, self.config.target_smoothing_s)))
            smoothed = previous + ((normalized_target - previous) * alpha)
        bounded_smoothed = _clamp_ratio(smoothed)
        self._smoothed_center_x = bounded_smoothed
        return bounded_smoothed

    def _release_tolerance_us(self) -> int:
        return max(
            _MIN_RELEASE_TOLERANCE_US,
            min(_MAX_RELEASE_TOLERANCE_US, self.config.max_step_us * 2),
        )

    def _seeded_planned_pulse_width_us(self) -> float:
        if self._planned_pulse_width_us is not None:
            return self._planned_pulse_width_us
        if self._last_commanded_pulse_width_us is not None:
            return float(self._last_commanded_pulse_width_us)
        if self._last_physical_pulse_width_us is not None:
            return float(self._last_physical_pulse_width_us)
        if self._released_pulse_width_us is not None:
            return float(self._released_pulse_width_us)
        return float(self.config.center_pulse_width_us)

    def _advance_planned_pulse_width(
        self,
        target_pulse_width_us: int,
        *,
        observed_at: float | None,
        motion_profile: str,
    ) -> float:
        current_pulse_width_us = self._seeded_planned_pulse_width_us()
        bounded_target_us = float(
            max(
                self.config.safe_min_pulse_width_us,
                min(self.config.safe_max_pulse_width_us, int(target_pulse_width_us)),
            )
        )
        error_us = bounded_target_us - current_pulse_width_us
        if abs(error_us) <= 0.5:
            self._planned_pulse_width_us = bounded_target_us
            self._planned_velocity_us_per_s = 0.0
            self._planned_acceleration_us_per_s2 = 0.0
            return bounded_target_us

        dt = self._effective_dt_s(observed_at=observed_at)
        (
            max_velocity_us_per_s,
            max_acceleration_us_per_s2,
            max_jerk_us_per_s3,
        ) = self._motion_limits_for_profile(motion_profile)
        stopping_velocity_us_per_s = math.sqrt(max(0.0, 2.0 * max_acceleration_us_per_s2 * abs(error_us)))
        desired_velocity_us_per_s = math.copysign(
            min(max_velocity_us_per_s, stopping_velocity_us_per_s),
            error_us,
        )
        desired_acceleration_us_per_s2 = (desired_velocity_us_per_s - self._planned_velocity_us_per_s) / dt
        desired_acceleration_us_per_s2 = _clamp(
            desired_acceleration_us_per_s2,
            minimum=-max_acceleration_us_per_s2,
            maximum=max_acceleration_us_per_s2,
        )
        max_acceleration_delta_us_per_s2 = max_jerk_us_per_s3 * dt
        next_acceleration_us_per_s2 = self._planned_acceleration_us_per_s2 + _clamp(
            desired_acceleration_us_per_s2 - self._planned_acceleration_us_per_s2,
            minimum=-max_acceleration_delta_us_per_s2,
            maximum=max_acceleration_delta_us_per_s2,
        )
        next_acceleration_us_per_s2 = _clamp(
            next_acceleration_us_per_s2,
            minimum=-max_acceleration_us_per_s2,
            maximum=max_acceleration_us_per_s2,
        )
        next_velocity_us_per_s = self._planned_velocity_us_per_s + (next_acceleration_us_per_s2 * dt)
        next_velocity_us_per_s = _clamp(
            next_velocity_us_per_s,
            minimum=-max_velocity_us_per_s,
            maximum=max_velocity_us_per_s,
        )
        next_pulse_width_us = current_pulse_width_us + (
            (self._planned_velocity_us_per_s + next_velocity_us_per_s) * 0.5 * dt
        )
        if (error_us > 0.0 and next_pulse_width_us >= bounded_target_us) or (
            error_us < 0.0 and next_pulse_width_us <= bounded_target_us
        ):
            next_pulse_width_us = bounded_target_us
            next_velocity_us_per_s = 0.0
            next_acceleration_us_per_s2 = 0.0
        next_pulse_width_us = _clamp(
            next_pulse_width_us,
            minimum=float(self.config.safe_min_pulse_width_us),
            maximum=float(self.config.safe_max_pulse_width_us),
        )
        self._planned_pulse_width_us = next_pulse_width_us
        self._planned_velocity_us_per_s = next_velocity_us_per_s
        self._planned_acceleration_us_per_s2 = next_acceleration_us_per_s2
        return next_pulse_width_us

    def _motion_limits_for_profile(self, motion_profile: str) -> tuple[float, float, float]:
        normalized_profile = str(motion_profile or "tracking").strip().lower() or "tracking"
        if normalized_profile == "rest":
            return (
                max(1.0, self.config.rest_max_velocity_us_per_s),
                max(1.0, self.config.rest_max_acceleration_us_per_s2),
                max(1.0, self.config.rest_max_jerk_us_per_s3),
            )
        return (
            max(1.0, self.config.max_velocity_us_per_s),
            max(1.0, self.config.max_acceleration_us_per_s2),
            max(1.0, self.config.max_jerk_us_per_s3),
        )

    def _command_pulse_width_for_plan(
        self,
        planned_pulse_width_us: float,
        *,
        target_pulse_width_us: int,
        motion_profile: str,
    ) -> int:
        checked_target_us = max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, int(target_pulse_width_us)),
        )
        if (
            str(motion_profile or "").strip().lower() == "tracking"
            and self._hardware_position_ramp_enabled()
        ):
            return self._direct_hardware_tracking_target(
                target_pulse_width_us=checked_target_us,
            )
        candidate_pulse_width_us = int(round(planned_pulse_width_us))
        previous_commanded_pulse_width_us = (
            self._last_commanded_pulse_width_us
            if self._last_commanded_pulse_width_us is not None
            else (
                self._last_physical_pulse_width_us
                if self._last_physical_pulse_width_us is not None
                else self.config.center_pulse_width_us
            )
        )
        if (
            str(motion_profile or "").strip().lower() == "rest"
            and checked_target_us == self.config.center_pulse_width_us
            and abs(checked_target_us - previous_commanded_pulse_width_us) <= self._release_tolerance_us()
        ):
            return checked_target_us

        command_delta_us = candidate_pulse_width_us - previous_commanded_pulse_width_us
        if (
            self._last_commanded_pulse_width_us is not None
            and abs(command_delta_us) < self.config.min_command_delta_us
            and abs(checked_target_us - previous_commanded_pulse_width_us) >= self.config.min_command_delta_us
        ):
            return previous_commanded_pulse_width_us
        if abs(command_delta_us) > self.config.max_step_us:
            candidate_pulse_width_us = previous_commanded_pulse_width_us + (
                self.config.max_step_us if command_delta_us > 0 else -self.config.max_step_us
            )
        if (
            abs(command_delta_us) < self.config.min_command_delta_us
            and self._last_commanded_pulse_width_us is not None
        ):
            return previous_commanded_pulse_width_us
        return max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, candidate_pulse_width_us),
        )

    def _maybe_release_idle(
        self,
        *,
        observed_at: float | None,
        active: bool,
        target_pulse_width_us: int,
    ) -> bool:
        """Release the output after calm recentering so idle servos do not buzz."""

        if active or self.config.idle_release_s <= 0.0:
            self._centered_since = None
            return False
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - self.config.center_pulse_width_us) > tolerance_us:
            self._centered_since = None
            return False
        current_pulse_width_us = int(round(self._seeded_planned_pulse_width_us()))
        if current_pulse_width_us != self.config.center_pulse_width_us:
            self._centered_since = None
            return False
        if observed_at is None:
            return False
        if self._centered_since is None or observed_at < self._centered_since:
            self._centered_since = observed_at
            return False
        if (observed_at - self._centered_since) < self.config.idle_release_s:
            return False
        if self._last_commanded_pulse_width_us is None:
            return True
        try:
            self._pulse_writer.disable(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio if self.config.gpio is not None else 0,
            )
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        if self._continuous_planner is not None:
            self._continuous_planner.note_stopped(observed_at=observed_at)
        self._last_commanded_pulse_width_us = None
        self._last_physical_pulse_width_us = self.config.center_pulse_width_us
        self._reset_motion_state(self.config.center_pulse_width_us)
        self._persist_runtime_state(observed_at=observed_at, force=True)
        return True

    def _maybe_hold_released_target(
        self,
        *,
        observed_at: float | None,
        active: bool,
        visible_target_present: bool,
        reason: str,
        confidence: float,
        target_center_x: float | None,
        applied_center_x: float,
        target_pulse_width_us: int,
    ) -> AttentionServoDecision | None:
        released_pulse_width_us = self._released_pulse_width_us
        if released_pulse_width_us is None:
            return None
        if active and visible_target_present:
            # A live visible target must re-engage the physical servo instead of
            # staying parked on one previously released off-center pulse.
            self._released_pulse_width_us = None
            self._settled_since = None
            return None
        if target_pulse_width_us == self.config.center_pulse_width_us:
            if released_pulse_width_us != self.config.center_pulse_width_us:
                self._reset_motion_state(released_pulse_width_us)
                self._released_pulse_width_us = None
                self._settled_since = None
                return None
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - released_pulse_width_us) > tolerance_us:
            self._reset_motion_state(released_pulse_width_us)
            self._released_pulse_width_us = None
            self._settled_since = None
            return None
        self._settled_since = observed_at
        self._last_update_at = observed_at
        return AttentionServoDecision(
            observed_at=observed_at,
            active=active,
            reason="settled_released",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=None,
        )

    def _maybe_release_exit_hold(
        self,
        *,
        observed_at: float | None,
        reason: str,
        confidence: float,
        target_center_x: float | None,
        applied_center_x: float,
        target_pulse_width_us: int,
        commanded_pulse_width_us: int,
    ) -> AttentionServoDecision | None:
        """Release the servo after it reaches the projected exit endpoint."""

        if (
            self._continuous_planner is not None
            or not self.config.follow_exit_only
            or reason not in {"holding_projected_trajectory", "holding_exit_position"}
            or self._last_commanded_pulse_width_us is None
        ):
            return None
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - commanded_pulse_width_us) > tolerance_us:
            return None
        try:
            self._pulse_writer.disable(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio if self.config.gpio is not None else 0,
            )
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        if self._continuous_planner is not None:
            self._continuous_planner.note_stopped(observed_at=observed_at)
        self._released_pulse_width_us = commanded_pulse_width_us
        self._last_commanded_pulse_width_us = None
        self._last_physical_pulse_width_us = commanded_pulse_width_us
        self._reset_motion_state(commanded_pulse_width_us)
        self._settled_since = None
        self._persist_runtime_state(observed_at=observed_at, force=True)
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="exit_hold_released",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=None,
        )

    def _maybe_release_settled_target(
        self,
        *,
        observed_at: float | None,
        active: bool,
        visible_target_present: bool,
        target_pulse_width_us: int,
        commanded_pulse_width_us: int,
    ) -> bool:
        """Release a stable off-center target so loaded servos can relax quietly."""

        if (
            self._continuous_planner is not None
            or observed_at is None
            or (active and visible_target_present)
            or self.config.settled_release_s <= 0.0
            or self._last_commanded_pulse_width_us is None
        ):
            self._settled_since = None
            return False
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - self.config.center_pulse_width_us) <= tolerance_us:
            self._settled_since = None
            return False
        if (
            abs(target_pulse_width_us - commanded_pulse_width_us) > tolerance_us
            or commanded_pulse_width_us != self._last_commanded_pulse_width_us
        ):
            self._settled_since = None
            return False
        if self._settled_since is None or observed_at < self._settled_since:
            self._settled_since = observed_at
            return False
        if (observed_at - self._settled_since) < self.config.settled_release_s:
            return False
        try:
            self._pulse_writer.disable(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio if self.config.gpio is not None else 0,
            )
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        if self._continuous_planner is not None:
            self._continuous_planner.note_stopped(observed_at=observed_at)
        self._released_pulse_width_us = commanded_pulse_width_us
        self._last_commanded_pulse_width_us = None
        self._last_physical_pulse_width_us = commanded_pulse_width_us
        self._reset_motion_state(commanded_pulse_width_us)
        self._settled_since = None
        self._persist_runtime_state(observed_at=observed_at, force=True)
        return True
