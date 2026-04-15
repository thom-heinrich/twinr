"""Own controller state, persistence, replay, and fault recovery."""

from __future__ import annotations

import math
import time

from twinr.hardware.servo_continuous import (
    ContinuousRotationServoConfig,
    ContinuousRotationServoPlanner,
)
from twinr.hardware.servo_segment_player import (
    BoundedServoPulseSegmentPlayer,
    ServoPulseSegmentCompletion,
    ServoPulseSegmentPlayback,
)
from twinr.hardware.servo_state import (
    AttentionServoMovementSegment,
    AttentionServoRuntimeState,
    AttentionServoStateStore,
)

from .constants import (
    _DEFAULT_SERVO_DRIVER,
    _MOVEMENT_JOURNAL_MAX_SEGMENTS,
    _MOVEMENT_JOURNAL_MIN_DURATION_S,
    _RECOVERABLE_SERVO_FAULT_RETRY_S,
)

from .controller_geometry import ControllerGeometryMixin
class ControllerStateMixin(ControllerGeometryMixin):
    def _writer_reports_live_position(self) -> bool:
        """Return whether the active writer exposes the live emitted pulse width."""

        return bool(getattr(self._pulse_writer, "reports_live_position", False))

    def _hardware_position_ramp_enabled(self) -> bool:
        """Return whether visible positional tracking should delegate smoothing to hardware."""

        return (
            self._continuous_planner is None
            and bool(getattr(self._pulse_writer, "hardware_motion_profile_enabled", False))
        )

    def _refresh_last_physical_pulse_width_from_writer(self) -> None:
        """Refresh the live physical pulse when the writer can report it accurately."""

        if not self._writer_reports_live_position() or self.config.gpio is None:
            return
        current_pulse_reader = getattr(self._pulse_writer, "current_pulse_width_us", None)
        if not callable(current_pulse_reader):
            return
        try:
            pulse_width_us = current_pulse_reader(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio,
            )
        except Exception:
            return
        if pulse_width_us is None:
            if self._last_commanded_pulse_width_us is None:
                self._last_physical_pulse_width_us = self.config.center_pulse_width_us
                self._released_pulse_width_us = self.config.center_pulse_width_us
                self._reset_motion_state(self.config.center_pulse_width_us)
            return
        checked_pulse_width_us = max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, int(pulse_width_us)),
        )
        self._last_physical_pulse_width_us = checked_pulse_width_us
        if self._last_commanded_pulse_width_us is None:
            self._released_pulse_width_us = checked_pulse_width_us
        if self._hardware_position_ramp_enabled():
            self._reset_motion_state(checked_pulse_width_us)

    def _build_continuous_planner(self) -> ContinuousRotationServoPlanner | None:
        if not self.config.uses_continuous_rotation:
            return None
        return ContinuousRotationServoPlanner(
            ContinuousRotationServoConfig(
                center_pulse_width_us=self.config.center_pulse_width_us,
                min_pulse_width_us=self.config.safe_min_pulse_width_us,
                max_pulse_width_us=self.config.safe_max_pulse_width_us,
                max_heading_degrees=self._maximum_target_heading_degrees(),
                max_speed_degrees_per_s=self.config.continuous_max_speed_degrees_per_s,
                slow_zone_degrees=self.config.continuous_slow_zone_degrees,
                stop_tolerance_degrees=self.config.continuous_stop_tolerance_degrees,
                min_speed_pulse_delta_us=self.config.continuous_min_speed_pulse_delta_us,
                max_speed_pulse_delta_us=self.config.continuous_max_speed_pulse_delta_us,
            )
        )

    def _build_state_store(self) -> AttentionServoStateStore | None:
        if not self.config.uses_continuous_rotation:
            return None
        return AttentionServoStateStore(self.config.state_path)

    def _build_movement_journal_replay_player(self) -> BoundedServoPulseSegmentPlayer | None:
        if not self.config.uses_continuous_rotation or self.config.gpio is None:
            return None
        return BoundedServoPulseSegmentPlayer(
            pulse_writer=self._pulse_writer,
            gpio_chip=self.config.gpio_chip,
            gpio=self.config.gpio,
        )

    def _load_runtime_state(self) -> AttentionServoRuntimeState | None:
        if self._state_store is None:
            return None
        return self._state_store.load()

    def _runtime_state_snapshot(
        self,
        *,
        observed_at: float | None,
        heading_degrees: float | None = None,
        heading_uncertainty_degrees: float | None = None,
        movement_journal: tuple[AttentionServoMovementSegment, ...] | None = None,
        hold_until_armed: bool | None = None,
        return_to_zero_requested: bool | None = None,
        zero_reference_confirmed: bool | None = None,
    ) -> AttentionServoRuntimeState | None:
        if self._continuous_planner is None:
            return None
        resolved_heading_degrees = (
            self._continuous_planner.estimated_heading_degrees
            if heading_degrees is None
            else float(heading_degrees)
        )
        resolved_heading_uncertainty_degrees = (
            self._heading_uncertainty_degrees
            if heading_uncertainty_degrees is None
            else float(heading_uncertainty_degrees)
        )
        return AttentionServoRuntimeState(
            heading_degrees=round(resolved_heading_degrees, 3),
            heading_uncertainty_degrees=round(resolved_heading_uncertainty_degrees, 3),
            movement_journal=(
                self._movement_journal_snapshot(observed_at=observed_at)
                if movement_journal is None
                else movement_journal
            ),
            hold_until_armed=(
                self._startup_hold_until_armed
                if hold_until_armed is None
                else bool(hold_until_armed)
            ),
            return_to_zero_requested=(
                self._return_to_zero_requested
                if return_to_zero_requested is None
                else bool(return_to_zero_requested)
            ),
            zero_reference_confirmed=(
                self._zero_reference_confirmed
                if zero_reference_confirmed is None
                else bool(zero_reference_confirmed)
            ),
            updated_at=observed_at,
        )

    def _current_runtime_state(self, *, observed_at: float | None) -> AttentionServoRuntimeState | None:
        return self._runtime_state_snapshot(observed_at=observed_at)

    def _persist_runtime_state(self, *, observed_at: float | None, force: bool = False) -> None:
        current_state = self._current_runtime_state(observed_at=observed_at)
        if current_state is None or self._state_store is None:
            return
        previous_state = self._last_saved_runtime_state
        state_changed = (
            previous_state is None
            or previous_state.heading_degrees != current_state.heading_degrees
            or previous_state.heading_uncertainty_degrees != current_state.heading_uncertainty_degrees
            or previous_state.movement_journal != current_state.movement_journal
            or previous_state.hold_until_armed != current_state.hold_until_armed
            or previous_state.return_to_zero_requested != current_state.return_to_zero_requested
            or previous_state.zero_reference_confirmed != current_state.zero_reference_confirmed
        )
        if not force and not state_changed:
            return
        self._state_store.save(current_state)
        self._last_saved_runtime_state = current_state
        self._last_runtime_state_mtime_ns = self._state_store.mtime_ns()

    def _movement_journal_snapshot(
        self,
        *,
        observed_at: float | None,
    ) -> tuple[AttentionServoMovementSegment, ...]:
        snapshot = list(self._movement_journal)
        active_segment = self._active_movement_journal_segment(observed_at=observed_at)
        if active_segment is not None:
            snapshot.append(active_segment)
        return tuple(snapshot)

    def _active_movement_journal_segment(
        self,
        *,
        observed_at: float | None,
    ) -> AttentionServoMovementSegment | None:
        if (
            self._movement_journal_active_pulse_width_us is None
            or self._movement_journal_active_started_at is None
            or observed_at is None
        ):
            return None
        duration_s = max(0.0, float(observed_at) - self._movement_journal_active_started_at)
        if duration_s < _MOVEMENT_JOURNAL_MIN_DURATION_S:
            return None
        return AttentionServoMovementSegment(
            pulse_width_us=self._movement_journal_active_pulse_width_us,
            duration_s=duration_s,
        )

    def _append_movement_journal_segment(self, segment: AttentionServoMovementSegment) -> None:
        if (
            segment.duration_s < _MOVEMENT_JOURNAL_MIN_DURATION_S
            or segment.pulse_width_us == self.config.center_pulse_width_us
        ):
            return
        if self._movement_journal and self._movement_journal[-1].pulse_width_us == segment.pulse_width_us:
            merged_duration_s = self._movement_journal[-1].duration_s + segment.duration_s
            self._movement_journal[-1] = AttentionServoMovementSegment(
                pulse_width_us=segment.pulse_width_us,
                duration_s=merged_duration_s,
            )
        else:
            self._movement_journal.append(segment)
        if len(self._movement_journal) > _MOVEMENT_JOURNAL_MAX_SEGMENTS:
            self._movement_journal = self._movement_journal[-_MOVEMENT_JOURNAL_MAX_SEGMENTS :]

    def _close_active_movement_journal_segment(
        self,
        *,
        observed_at: float | None,
        record: bool,
    ) -> None:
        active_segment = self._active_movement_journal_segment(observed_at=observed_at)
        self._movement_journal_active_pulse_width_us = None
        self._movement_journal_active_started_at = None
        if record and active_segment is not None:
            self._append_movement_journal_segment(active_segment)

    def _start_active_movement_journal_segment(
        self,
        *,
        pulse_width_us: int,
        observed_at: float | None,
    ) -> None:
        if observed_at is None or pulse_width_us == self.config.center_pulse_width_us:
            self._movement_journal_active_pulse_width_us = None
            self._movement_journal_active_started_at = None
            return
        self._movement_journal_active_pulse_width_us = int(pulse_width_us)
        self._movement_journal_active_started_at = float(observed_at)

    def _clear_movement_journal(self) -> None:
        self._movement_journal.clear()
        self._movement_journal_active_pulse_width_us = None
        self._movement_journal_active_started_at = None
        if self._movement_journal_replay_player is not None:
            self._movement_journal_replay_player.cancel()

    def _inverse_movement_journal_pulse_width_us(self, pulse_width_us: int) -> int:
        center_pulse_width_us = self.config.center_pulse_width_us
        inverse_pulse_width_us = center_pulse_width_us - (int(pulse_width_us) - center_pulse_width_us)
        return max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, inverse_pulse_width_us),
        )

    def _clear_tracking_state_for_external_override(self) -> None:
        """Drop transient follow latches before a live operator state override."""

        if self._movement_journal_replay_player is not None:
            self._movement_journal_replay_player.cancel()
        self._last_target_center_x = None
        self._last_target_at = None
        self._last_target_velocity_x_per_s = 0.0
        self._smoothed_center_x = None
        self._centered_since = None
        self._settled_since = None
        self._visible_target_pulse_width_us = None
        self._clear_visible_retarget_cooldown()
        self._exit_cooldown_until_at = None
        self._reset_zero_return_cycle_state()
        self._clear_exit_pursuit(clear_recent_visible_targets=True)

    def _reset_zero_return_cycle_state(self) -> None:
        """Forget the current continuous-servo zero-return direction and phase."""

        self._zero_return_direction_sign = None
        self._zero_return_phase_anchor_at = None
        self._zero_return_move_phase_active = None

    def _adopt_runtime_state(
        self,
        state: AttentionServoRuntimeState,
        *,
        observed_at: float | None,
    ) -> None:
        """Adopt one runtime-state snapshot into the live controller."""

        previous_hold = self._startup_hold_until_armed
        previous_return_to_zero_requested = self._return_to_zero_requested
        previous_zero_reference = self._zero_reference_confirmed
        previous_heading_degrees = self._continuous_startup_heading_degrees
        previous_movement_journal = tuple(self._movement_journal)
        refreshed_heading_degrees = (
            0.0 if not state.zero_reference_confirmed else state.heading_degrees
        )
        refreshed_movement_journal = (
            tuple(state.movement_journal)
            if state.zero_reference_confirmed
            else ()
        )
        self._heading_uncertainty_degrees = state.heading_uncertainty_degrees
        self._startup_hold_until_armed = bool(state.hold_until_armed)
        self._return_to_zero_requested = bool(state.return_to_zero_requested)
        self._zero_reference_confirmed = bool(state.zero_reference_confirmed)
        self._continuous_startup_heading_degrees = refreshed_heading_degrees
        self._movement_journal = list(refreshed_movement_journal)
        self._movement_journal_active_pulse_width_us = None
        self._movement_journal_active_started_at = None
        entering_manual_hold = not previous_hold and self._startup_hold_until_armed
        leaving_manual_hold = previous_hold and not self._startup_hold_until_armed
        return_request_changed = previous_return_to_zero_requested != self._return_to_zero_requested
        heading_changed = not math.isclose(
            previous_heading_degrees,
            refreshed_heading_degrees,
            abs_tol=0.001,
        )
        movement_journal_changed = previous_movement_journal != refreshed_movement_journal
        zero_reference_changed = previous_zero_reference != self._zero_reference_confirmed
        if (
            entering_manual_hold
            or return_request_changed
            or heading_changed
            or movement_journal_changed
            or zero_reference_changed
        ):
            self._clear_tracking_state_for_external_override()
        if leaving_manual_hold and observed_at is not None:
            self._last_visible_target_at = float(observed_at)
        if self._continuous_planner is None:
            return
        if not heading_changed and not zero_reference_changed:
            return
        self._continuous_planner.reset(
            heading_degrees=refreshed_heading_degrees,
            observed_at=observed_at,
        )
        self._last_physical_pulse_width_us = self.config.center_pulse_width_us
        self._released_pulse_width_us = (
            self.config.center_pulse_width_us
            if self._last_commanded_pulse_width_us is None
            else None
        )
        self._reset_motion_state(self.config.center_pulse_width_us)

    def _apply_loaded_runtime_state(
        self,
        state: AttentionServoRuntimeState,
        *,
        observed_at: float | None,
    ) -> None:
        """Adopt one externally updated state snapshot into the live controller."""

        self._adopt_runtime_state(state, observed_at=observed_at)
        self._last_saved_runtime_state = state

    def _refresh_runtime_state_from_store(self, *, observed_at: float | None) -> None:
        """Reload the persisted runtime state when an operator changes it live."""

        if self._state_store is None:
            return
        refreshed_state = self._state_store.load()
        self._last_runtime_state_mtime_ns = self._state_store.mtime_ns()
        if refreshed_state is None or refreshed_state == self._last_saved_runtime_state:
            return
        self._apply_loaded_runtime_state(refreshed_state, observed_at=observed_at)

    def _apply_manual_hold(self, *, observed_at: float | None) -> None:
        """Keep a continuous servo electrically released while startup hold is active.

        A 360-degree servo has no absolute position hold. Driving the configured
        center pulse during startup hold therefore still commands real motion
        when the individual servo's neutral trim is even slightly off. Safe
        startup hold for this path must mean "release output" instead of
        "continuously drive center".
        """

        current_pulse_width_us = None
        if self._movement_journal_replay_player is not None:
            self._movement_journal_replay_player.cancel()
        current_pulse_reader = getattr(self._pulse_writer, "current_pulse_width_us", None)
        if callable(current_pulse_reader) and self.config.gpio is not None:
            try:
                current_pulse_width_us = current_pulse_reader(
                    gpio_chip=self.config.gpio_chip,
                    gpio=self.config.gpio,
                )
            except Exception:
                current_pulse_width_us = None
        try:
            if self._last_commanded_pulse_width_us is not None or current_pulse_width_us is not None:
                self._pulse_writer.disable(
                    gpio_chip=self.config.gpio_chip,
                    gpio=self.config.gpio if self.config.gpio is not None else 0,
                )
            self._close_active_movement_journal_segment(
                observed_at=observed_at,
                record=not self._return_to_zero_requested,
            )
            if self._continuous_planner is not None:
                self._continuous_planner.note_stopped(observed_at=observed_at)
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        self._last_commanded_pulse_width_us = None
        self._last_physical_pulse_width_us = self.config.center_pulse_width_us
        self._released_pulse_width_us = self.config.center_pulse_width_us
        self._reset_zero_return_cycle_state()
        self._reset_motion_state(self.config.center_pulse_width_us)
        self._persist_runtime_state(observed_at=observed_at, force=True)

    def _active_movement_journal_replay_segment(self) -> ServoPulseSegmentPlayback | None:
        if self._movement_journal_replay_player is None:
            return None
        return self._movement_journal_replay_player.active_segment()

    def _movement_journal_replay_due_in_seconds(
        self,
        *,
        observed_at: float | None,
    ) -> float | None:
        active_segment = self._active_movement_journal_replay_segment()
        if active_segment is None or observed_at is None:
            return None
        return round(max(0.0, active_segment.due_at - float(observed_at)), 3)

    def _consume_movement_journal_replay_completion(
        self,
        *,
        observed_at: float | None,
    ) -> ServoPulseSegmentCompletion | None:
        if self._movement_journal_replay_player is None:
            return None
        completion = self._movement_journal_replay_player.consume_completion()
        if completion is None:
            return None
        if completion.error:
            self._fault_reason = completion.error
            raise RuntimeError(completion.error)
        if self._continuous_planner is not None:
            self._continuous_planner.note_stopped(observed_at=observed_at)
        self._released_pulse_width_us = completion.playback.pulse_width_us
        self._last_physical_pulse_width_us = completion.playback.pulse_width_us
        self._last_commanded_pulse_width_us = None
        self._reset_motion_state(self._released_pulse_width_us)
        self._last_update_at = observed_at
        return completion

    def _start_movement_journal_replay_segment(
        self,
        *,
        observed_at: float | None,
        target_pulse_width_us: int,
        duration_s: float,
    ) -> ServoPulseSegmentPlayback:
        if self._movement_journal_replay_player is None:
            raise RuntimeError("movement-journal replay player is unavailable")
        started_segment = self._movement_journal_replay_player.start_segment(
            pulse_width_us=target_pulse_width_us,
            duration_s=duration_s,
        )
        if self._continuous_planner is not None:
            self._continuous_planner.note_commanded_pulse_width(
                started_segment.pulse_width_us,
                observed_at=observed_at,
            )
        self._last_commanded_pulse_width_us = started_segment.pulse_width_us
        self._last_physical_pulse_width_us = started_segment.pulse_width_us
        self._released_pulse_width_us = None
        self._last_update_at = observed_at
        return started_segment

    def _prime_last_physical_pulse_width_from_writer(self) -> None:
        """Seed the motion planner from the kernel's remembered pulse width on controller startup.

        The Twinr kernel writer persists the last pulse width even after the GPIO
        line is released. Reusing that value avoids a fresh controller assuming
        the servo is already centered and yanking loaded hardware back to rest.
        """

        if self._continuous_planner is not None:
            self._continuous_planner.reset(heading_degrees=self._continuous_startup_heading_degrees)
            self._last_physical_pulse_width_us = self.config.center_pulse_width_us
            self._released_pulse_width_us = self.config.center_pulse_width_us
            self._reset_motion_state(self.config.center_pulse_width_us)
            return
        if self.config.gpio is None:
            return
        if self._writer_reports_live_position():
            self._refresh_last_physical_pulse_width_from_writer()
            if self._last_physical_pulse_width_us is not None:
                if self._last_physical_pulse_width_us != self.config.center_pulse_width_us:
                    self._startup_rest_alignment_pending = True
                return
        current_pulse_reader = getattr(self._pulse_writer, "current_pulse_width_us", None)
        if not callable(current_pulse_reader):
            return
        try:
            pulse_width_us = current_pulse_reader(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio,
            )
        except Exception:
            return
        if pulse_width_us is None:
            return
        checked_pulse_width_us = max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, int(pulse_width_us)),
        )
        self._last_physical_pulse_width_us = checked_pulse_width_us
        self._released_pulse_width_us = checked_pulse_width_us
        self._reset_motion_state(checked_pulse_width_us)
        if checked_pulse_width_us != self.config.center_pulse_width_us:
            self._startup_rest_alignment_pending = True

    def _release_active_output(self, *, observed_at: float | None = None) -> None:
        if self._movement_journal_replay_player is not None:
            self._movement_journal_replay_player.cancel()
        if self._last_commanded_pulse_width_us is None:
            if self._continuous_planner is not None:
                self._close_active_movement_journal_segment(
                    observed_at=observed_at,
                    record=not self._return_to_zero_requested,
                )
                self._continuous_planner.note_stopped(observed_at=observed_at)
                self._persist_runtime_state(observed_at=observed_at, force=True)
            return
        try:
            self._pulse_writer.disable(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio if self.config.gpio is not None else 0,
            )
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        self._close_active_movement_journal_segment(
            observed_at=observed_at,
            record=not self._return_to_zero_requested,
        )
        if self._continuous_planner is not None:
            self._continuous_planner.note_stopped(observed_at=observed_at)
        self._released_pulse_width_us = self._last_commanded_pulse_width_us
        self._last_physical_pulse_width_us = self._last_commanded_pulse_width_us
        self._last_commanded_pulse_width_us = None
        self._reset_motion_state(self._released_pulse_width_us)
        self._persist_runtime_state(observed_at=observed_at, force=True)

    def _release_stale_output_if_disabled(self) -> None:
        """Best-effort release for outputs left active by an earlier process when this controller starts disabled."""

        if self.config.enabled or self.config.gpio is None:
            return
        try:
            self._pulse_writer.disable(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio,
            )
        except Exception:
            return

    def _driver_supports_fault_recovery(self) -> bool:
        return str(self.config.driver or _DEFAULT_SERVO_DRIVER).strip().lower() in {
            "pololu_maestro",
            "peer_pololu_maestro",
        }

    def _maybe_recover_from_fault(self, *, observed_at: float | None) -> bool:
        if self._fault_reason is None:
            return True
        if not self._driver_supports_fault_recovery():
            return False
        if self._movement_journal_replay_player is not None:
            self._movement_journal_replay_player.cancel()
        retry_anchor_at = time.monotonic() if observed_at is None else observed_at
        if self._fault_retry_after_at is not None and retry_anchor_at < self._fault_retry_after_at:
            return False
        close_writer = getattr(self._pulse_writer, "close", None)
        if callable(close_writer):
            try:
                close_writer()
            except Exception:
                pass
        try:
            if self.config.gpio is not None:
                self._pulse_writer.probe(self.config.gpio)
            self._fault_reason = None
            self._disabled_due_to_fault = False
            self._fault_retry_after_at = None
            self._last_commanded_pulse_width_us = None
            self._last_target_center_x = None
            self._last_target_at = None
            self._last_target_velocity_x_per_s = 0.0
            self._smoothed_center_x = None
            if self._continuous_planner is not None:
                self._heading_uncertainty_degrees = max(
                    self._heading_uncertainty_degrees,
                    self.config.estimated_zero_max_uncertainty_degrees + self.config.continuous_stop_tolerance_degrees,
                )
                self._return_to_zero_requested = False
                self._continuous_planner.reset(
                    heading_degrees=self._continuous_planner.estimated_heading_degrees,
                    observed_at=observed_at,
                )
                self._last_physical_pulse_width_us = self.config.center_pulse_width_us
                self._released_pulse_width_us = self.config.center_pulse_width_us
                self._reset_motion_state(self.config.center_pulse_width_us)
            else:
                self._prime_last_physical_pulse_width_from_writer()
            return True
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            self._fault_retry_after_at = retry_anchor_at + _RECOVERABLE_SERVO_FAULT_RETRY_S
            return False

    def _reset_motion_state(self, pulse_width_us: int | None = None) -> None:
        seeded_pulse_width = (
            None
            if pulse_width_us is None
            else float(
                max(
                    self.config.safe_min_pulse_width_us,
                    min(self.config.safe_max_pulse_width_us, int(pulse_width_us)),
                )
            )
        )
        self._planned_pulse_width_us = seeded_pulse_width
        self._planned_velocity_us_per_s = 0.0
        self._planned_acceleration_us_per_s2 = 0.0
