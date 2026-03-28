"""Handle continuous-rotation zero-return behavior separately from tracking."""

from __future__ import annotations

from .constants import (
    _MOVEMENT_JOURNAL_MIN_DURATION_S,
)

from .controller_motion import ControllerMotionMixin
from .types import AttentionServoDecision

class ControllerContinuousMixin(ControllerMotionMixin):
    def _reject_estimated_zero_return(
        self,
        *,
        observed_at: float | None,
        checked_confidence: float,
        target_center_x: float | None,
        reason: str,
    ) -> AttentionServoDecision:
        self._reset_zero_return_cycle_state()
        rejection_state = self._runtime_state_snapshot(
            observed_at=observed_at,
            hold_until_armed=True,
            return_to_zero_requested=False,
        )
        if rejection_state is not None:
            self._adopt_runtime_state(rejection_state, observed_at=observed_at)
        self._apply_manual_hold(observed_at=observed_at)
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason=reason,
            confidence=checked_confidence,
            target_center_x=target_center_x,
            applied_center_x=0.5,
            target_pulse_width_us=self.config.center_pulse_width_us,
            commanded_pulse_width_us=None,
        )

    def _update_return_to_estimated_zero(
        self,
        *,
        observed_at: float | None,
        checked_confidence: float,
        target_center_x: float | None,
    ) -> AttentionServoDecision:
        """Drive a bounded open-loop return toward the persisted virtual zero."""

        if self._continuous_planner is None:
            return self._reject_estimated_zero_return(
                observed_at=observed_at,
                checked_confidence=checked_confidence,
                target_center_x=target_center_x,
                reason="estimated_zero_rejected_unavailable",
            )
        if not self._zero_reference_confirmed:
            return self._reject_estimated_zero_return(
                observed_at=observed_at,
                checked_confidence=checked_confidence,
                target_center_x=target_center_x,
                reason="estimated_zero_rejected_unconfirmed",
            )
        if (
            self._heading_uncertainty_degrees
            > self.config.estimated_zero_max_uncertainty_degrees
        ):
            return self._reject_estimated_zero_return(
                observed_at=observed_at,
                checked_confidence=checked_confidence,
                target_center_x=target_center_x,
                reason="estimated_zero_rejected_uncertainty",
            )
        if self._movement_journal or self._movement_journal_active_pulse_width_us is not None:
            return self._update_return_to_estimated_zero_from_movement_journal(
                observed_at=observed_at,
                checked_confidence=checked_confidence,
                target_center_x=target_center_x,
            )
        target_pulse_width_us, settled_at_zero = self._prepare_monotonic_zero_return_target(
            observed_at=observed_at,
        )
        if settled_at_zero:
            completed_state = self._runtime_state_snapshot(
                observed_at=observed_at,
                heading_degrees=0.0,
                hold_until_armed=True,
                return_to_zero_requested=False,
            )
            if completed_state is not None:
                self._adopt_runtime_state(completed_state, observed_at=observed_at)
            self._apply_manual_hold(observed_at=observed_at)
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="estimated_zero_hold",
                confidence=checked_confidence,
                target_center_x=target_center_x,
                applied_center_x=0.5,
                target_pulse_width_us=self.config.center_pulse_width_us,
                commanded_pulse_width_us=None,
            )
        if target_pulse_width_us is None:
            raise RuntimeError("estimated-zero target was unexpectedly unavailable")
        target_pulse_width_us = self._slow_estimated_zero_target_pulse_width(
            target_pulse_width_us=target_pulse_width_us,
            observed_at=observed_at,
        )
        commanded_pulse_width_us = self._write_continuous_zero_return_target_pulse_width(
            observed_at=observed_at,
            target_pulse_width_us=target_pulse_width_us,
        )
        return AttentionServoDecision(
            observed_at=observed_at,
            active=commanded_pulse_width_us is not None
            and commanded_pulse_width_us != self.config.center_pulse_width_us,
            reason="returning_to_estimated_zero",
            confidence=checked_confidence,
            target_center_x=target_center_x,
            applied_center_x=0.5,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        )

    def _update_return_to_estimated_zero_from_movement_journal(
        self,
        *,
        observed_at: float | None,
        checked_confidence: float,
        target_center_x: float | None,
    ) -> AttentionServoDecision:
        """Replay the logged outbound motion in reverse with exact segment deadlines."""

        self._close_active_movement_journal_segment(
            observed_at=observed_at,
            record=True,
        )
        replay_completion = self._consume_movement_journal_replay_completion(
            observed_at=observed_at,
        )
        if replay_completion is not None and self._movement_journal:
            self._movement_journal.pop()
            self._persist_runtime_state(observed_at=observed_at, force=True)
        active_replay_segment = self._active_movement_journal_replay_segment()
        if active_replay_segment is not None:
            return AttentionServoDecision(
                observed_at=observed_at,
                active=active_replay_segment.pulse_width_us != self.config.center_pulse_width_us,
                reason="returning_to_estimated_zero",
                confidence=checked_confidence,
                target_center_x=target_center_x,
                applied_center_x=0.5,
                target_pulse_width_us=active_replay_segment.pulse_width_us,
                commanded_pulse_width_us=active_replay_segment.pulse_width_us,
            )
        if self._movement_journal:
            current_segment = self._movement_journal[-1]
            target_pulse_width_us = self._inverse_movement_journal_pulse_width_us(
                current_segment.pulse_width_us
            )
            started_segment = self._start_movement_journal_replay_segment(
                observed_at=observed_at,
                target_pulse_width_us=target_pulse_width_us,
                duration_s=max(_MOVEMENT_JOURNAL_MIN_DURATION_S, current_segment.duration_s),
            )
            return AttentionServoDecision(
                observed_at=observed_at,
                active=started_segment.pulse_width_us != self.config.center_pulse_width_us,
                reason="returning_to_estimated_zero",
                confidence=checked_confidence,
                target_center_x=target_center_x,
                applied_center_x=0.5,
                target_pulse_width_us=started_segment.pulse_width_us,
                commanded_pulse_width_us=started_segment.pulse_width_us,
            )
        completed_state = self._runtime_state_snapshot(
            observed_at=observed_at,
            heading_degrees=0.0,
            movement_journal=(),
            hold_until_armed=True,
            return_to_zero_requested=False,
        )
        if completed_state is not None:
            self._adopt_runtime_state(completed_state, observed_at=observed_at)
        self._apply_manual_hold(observed_at=observed_at)
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="estimated_zero_hold",
            confidence=checked_confidence,
            target_center_x=target_center_x,
            applied_center_x=0.5,
            target_pulse_width_us=self.config.center_pulse_width_us,
            commanded_pulse_width_us=None,
        )

    def _maybe_hold_continuous_absence_before_return_to_zero(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        visible_target_present: bool,
        active: bool,
        reason: str,
    ) -> AttentionServoDecision | None:
        """Release the continuous servo before a delayed return-to-zero phase.

        Continuous servos cannot hold a position electrically. When the person
        leaves frame, Twinr should stop driving, keep the last estimated
        heading, and only start the slow return to virtual zero after the
        configured no-user delay has elapsed.
        """

        if (
            self._continuous_planner is None
            or visible_target_present
            or active
            or reason != "recentering"
        ):
            return None
        absence_hold_s = max(0.0, self.config.continuous_return_to_zero_after_s)
        if absence_hold_s <= 0.0 or observed_at is None:
            return None
        last_visible_target_at = self._last_visible_target_at
        if last_visible_target_at is None:
            return None
        if observed_at < last_visible_target_at:
            self._last_visible_target_at = observed_at
            return None
        if (observed_at - last_visible_target_at) >= absence_hold_s:
            return None
        self._reset_zero_return_cycle_state()
        self._release_active_output(observed_at=observed_at)
        self._last_update_at = observed_at
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="absence_hold_released",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=0.5,
            target_pulse_width_us=self.config.center_pulse_width_us,
            commanded_pulse_width_us=None,
        )

    def _write_continuous_zero_return_target_pulse_width(
        self,
        *,
        observed_at: float | None,
        target_pulse_width_us: int,
    ) -> int | None:
        """Emit one slow zero-return command, releasing output during pause windows.

        Continuous-servo zero-return already uses explicit short move windows.
        Driving the configured stop pulse during the off-window can still cause
        reverse creep on a real servo whose neutral trim is imperfect. Treat
        that off-window as an electrical release instead of a center-pulse
        command so the shaft pauses without counter-steering.
        """

        if int(target_pulse_width_us) == self.config.center_pulse_width_us:
            self._release_active_output(observed_at=observed_at)
            self._last_update_at = observed_at
            return None
        return self._write_immediate_target_pulse_width(
            observed_at=observed_at,
            target_pulse_width_us=target_pulse_width_us,
            planner_speed_scale=self.config.estimated_zero_speed_scale,
        )

    def _prepare_monotonic_zero_return_target(
        self,
        *,
        observed_at: float | None,
    ) -> tuple[int | None, bool]:
        """Return one zero-return target that will not intentionally reverse direction.

        Continuous-servo homing is open-loop. Once the virtual heading estimate
        crosses zero, issuing the opposite pulse creates visible left/right
        chatter. Treat that first sign flip as "close enough", snap the virtual
        heading back to zero, and stop instead of counter-steering.
        """

        if self._continuous_planner is None:
            return None, False
        center_pulse_width_us = self.config.center_pulse_width_us
        target_pulse_width_us = self._target_pulse_width_for_center_x(
            center_x=0.5,
            observed_at=observed_at,
        )
        estimated_heading_degrees = self._continuous_planner.estimated_heading_degrees
        stop_tolerance_degrees = self.config.estimated_zero_settle_tolerance_degrees
        if (
            abs(estimated_heading_degrees) <= stop_tolerance_degrees
            and target_pulse_width_us == center_pulse_width_us
        ):
            self._reset_zero_return_cycle_state()
            self._continuous_planner.reset(heading_degrees=0.0, observed_at=observed_at)
            return None, True
        return_direction_sign = 0
        if estimated_heading_degrees < (-stop_tolerance_degrees):
            return_direction_sign = 1
        elif estimated_heading_degrees > stop_tolerance_degrees:
            return_direction_sign = -1
        if return_direction_sign == 0:
            self._reset_zero_return_cycle_state()
            self._continuous_planner.reset(heading_degrees=0.0, observed_at=observed_at)
            return None, True
        if self._zero_return_direction_sign is None:
            self._zero_return_direction_sign = return_direction_sign
        elif return_direction_sign != self._zero_return_direction_sign:
            self._reset_zero_return_cycle_state()
            self._continuous_planner.reset(heading_degrees=0.0, observed_at=observed_at)
            return None, True
        pulse_delta_us = abs(target_pulse_width_us - center_pulse_width_us)
        if pulse_delta_us <= 0:
            pulse_delta_us = self.config.continuous_min_speed_pulse_delta_us
        monotonic_target_pulse_width_us = center_pulse_width_us + (return_direction_sign * pulse_delta_us)
        monotonic_target_pulse_width_us = max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, monotonic_target_pulse_width_us),
        )
        return monotonic_target_pulse_width_us, False

    def _slow_estimated_zero_target_pulse_width(
        self,
        *,
        target_pulse_width_us: int,
        observed_at: float | None,
    ) -> int:
        """Return one extra-slow open-loop target for estimated-zero returns."""

        checked_target_us = max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, int(target_pulse_width_us)),
        )
        center_pulse_width_us = self.config.center_pulse_width_us
        pulse_delta_us = checked_target_us - center_pulse_width_us
        if pulse_delta_us == 0:
            return center_pulse_width_us
        max_move_delta_us = max(0, int(self.config.estimated_zero_move_pulse_delta_us))
        if max_move_delta_us == 0:
            return center_pulse_width_us
        if self.config.estimated_zero_move_duty_cycle >= 1.0:
            self._zero_return_phase_anchor_at = observed_at
            self._zero_return_move_phase_active = True
        elif observed_at is not None:
            checked_at = max(0.0, float(observed_at))
            active_window_s = (
                self.config.estimated_zero_move_period_s * self.config.estimated_zero_move_duty_cycle
            )
            release_window_s = max(0.0, self.config.estimated_zero_move_period_s - active_window_s)
            phase_anchor_at = self._zero_return_phase_anchor_at
            move_phase_active = self._zero_return_move_phase_active
            if phase_anchor_at is None or move_phase_active is None or checked_at < phase_anchor_at:
                phase_anchor_at = checked_at
                move_phase_active = True
                self._zero_return_phase_anchor_at = checked_at
                self._zero_return_move_phase_active = True
            elapsed_phase_s = max(0.0, checked_at - phase_anchor_at)
            if move_phase_active:
                if elapsed_phase_s >= active_window_s:
                    self._zero_return_phase_anchor_at = checked_at
                    self._zero_return_move_phase_active = False
                    return center_pulse_width_us
            elif elapsed_phase_s >= release_window_s:
                self._zero_return_phase_anchor_at = checked_at
                self._zero_return_move_phase_active = True
            else:
                return center_pulse_width_us
        bounded_delta_us = min(abs(pulse_delta_us), max_move_delta_us)
        if bounded_delta_us <= 0:
            return center_pulse_width_us
        if pulse_delta_us > 0:
            return min(self.config.safe_max_pulse_width_us, center_pulse_width_us + bounded_delta_us)
        return max(self.config.safe_min_pulse_width_us, center_pulse_width_us - bounded_delta_us)
