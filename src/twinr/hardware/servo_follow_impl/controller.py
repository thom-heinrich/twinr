"""Expose the public attention-servo controller over decomposed mixins."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import sys
from typing import Callable, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.servo_segment_player import ServoPulseSegmentPlayer
from twinr.hardware.servo_state import AttentionServoMovementSegment

from .common import _bounded_float, _clamp_ratio
from .config import AttentionServoConfig
from .controller_exit_only import ControllerExitOnlyMixin
from .types import AttentionServoDecision
from .writers import (
    _NoopServoPulseWriter,
    ServoPulseWriter,
    _default_pulse_writer_for_config,
)


def _resolve_default_pulse_writer_factory() -> Callable[[AttentionServoConfig], ServoPulseWriter]:
    """Resolve the public wrapper factory so legacy monkeypatches still apply."""

    public_module = sys.modules.get("twinr.hardware.servo_follow")
    if public_module is None:
        return _default_pulse_writer_for_config
    return cast(
        Callable[[AttentionServoConfig], ServoPulseWriter],
        getattr(public_module, "_default_pulse_writer_for_config", _default_pulse_writer_for_config),
    )


class AttentionServoController(ControllerExitOnlyMixin):
    def __init__(
        self,
        *,
        config: AttentionServoConfig,
        pulse_writer: ServoPulseWriter | None = None,
        replay_segment_player: ServoPulseSegmentPlayer | None = None,
    ) -> None:
        self.config = config
        default_factory = _resolve_default_pulse_writer_factory()
        self._pulse_writer = pulse_writer or default_factory(config)
        self._continuous_planner = self._build_continuous_planner()
        self._state_store = self._build_state_store()
        startup_state = self._load_runtime_state()
        self._continuous_startup_heading_degrees = (
            0.0
            if startup_state is None or not startup_state.zero_reference_confirmed
            else startup_state.heading_degrees
        )
        self._heading_uncertainty_degrees = (
            0.0 if startup_state is None else startup_state.heading_uncertainty_degrees
        )
        self._movement_journal: list[AttentionServoMovementSegment] = list(
            () if startup_state is None else startup_state.movement_journal
        )
        self._startup_hold_until_armed = bool(startup_state and startup_state.hold_until_armed)
        self._return_to_zero_requested = bool(startup_state and startup_state.return_to_zero_requested)
        self._zero_reference_confirmed = bool(startup_state and startup_state.zero_reference_confirmed)
        self._last_saved_runtime_state = startup_state
        self._last_runtime_state_mtime_ns = (
            None if self._state_store is None else self._state_store.mtime_ns()
        )
        self._last_target_center_x: float | None = None
        self._last_target_at: float | None = None
        self._last_visible_target_at: float | None = None
        self._movement_journal_active_pulse_width_us: int | None = None
        self._movement_journal_active_started_at: float | None = None
        self._movement_journal_replay_player = (
            replay_segment_player or self._build_movement_journal_replay_player()
        )
        self._zero_return_direction_sign: int | None = None
        self._zero_return_phase_anchor_at: float | None = None
        self._zero_return_move_phase_active: bool | None = None
        self._last_target_velocity_x_per_s = 0.0
        self._last_commanded_pulse_width_us: int | None = None
        self._last_physical_pulse_width_us: int | None = None
        self._planned_pulse_width_us: float | None = None
        self._planned_velocity_us_per_s = 0.0
        self._planned_acceleration_us_per_s2 = 0.0
        self._smoothed_center_x: float | None = None
        self._last_update_at: float | None = None
        self._centered_since: float | None = None
        self._settled_since: float | None = None
        self._released_pulse_width_us: int | None = None
        self._visible_target_pulse_width_us: int | None = None
        self._last_exit_hold_center_x: float | None = None
        self._recent_visible_targets: deque[tuple[float, float]] = deque()
        self._recent_visible_target_boxes: deque[tuple[float, float, float]] = deque()
        self._exit_pursuit_target_pulse_width_us: int | None = None
        self._exit_pursuit_center_x: float | None = None
        self._exit_pursuit_settled_at: float | None = None
        self._exit_cooldown_until_at: float | None = None
        self._visible_edge_departure_since_at: float | None = None
        self._visible_recenter_since_at: float | None = None
        self._visible_recenter_target_pulse_width_us: int | None = None
        self._visible_recenter_center_x: float | None = None
        self._visible_recenter_settled_at: float | None = None
        self._startup_rest_alignment_pending = False
        self._fault_reason: str | None = None
        self._disabled_due_to_fault = False
        self._fault_retry_after_at: float | None = None
        self._prime_last_physical_pulse_width_from_writer()
        self._release_stale_output_if_disabled()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AttentionServoController":
        """Build one servo controller from the global Twinr config."""

        servo_config = AttentionServoConfig.from_config(config)
        try:
            return cls(config=servo_config)
        except Exception as exc:
            disabled_config = replace(servo_config, enabled=False)
            controller = cls(config=disabled_config, pulse_writer=_NoopServoPulseWriter())
            controller._fault_reason = f"startup_preflight_failed: {exc.__class__.__name__}: {exc}"
            controller._disabled_due_to_fault = True
            return controller

    def update(
        self,
        *,
        observed_at: float | None,
        active: bool,
        target_center_x: float | None,
        confidence: float | None,
        visible_target_present: bool | None = None,
        visible_target_box_left: float | None = None,
        visible_target_box_right: float | None = None,
    ) -> AttentionServoDecision:
        """Apply one bounded servo update from a pre-derived attention target."""

        checked_at = None if observed_at is None else float(observed_at)
        checked_confidence = _clamp_ratio(
            _bounded_float(confidence, default=0.0, minimum=0.0, maximum=1.0)
        )
        checked_visible_target_present = (
            self._visible_target_present(
                active=active,
                target_center_x=target_center_x,
            )
            if visible_target_present is None
            else (
                bool(visible_target_present)
                and target_center_x is not None
            )
        )
        effective_active = (
            active
            if visible_target_present is None
            else (active and bool(visible_target_present))
        )
        if not self.config.enabled and not self._disabled_due_to_fault:
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="disabled",
                confidence=checked_confidence,
                target_center_x=target_center_x,
            )
        if self.config.gpio is None:
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="unconfigured_gpio",
                confidence=checked_confidence,
                target_center_x=target_center_x,
            )
        if self._fault_reason is not None and not self._maybe_recover_from_fault(observed_at=checked_at):
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="faulted",
                confidence=checked_confidence,
                target_center_x=target_center_x,
            )
        self._refresh_last_physical_pulse_width_from_writer()
        self._refresh_runtime_state_from_store(observed_at=checked_at)
        if self._startup_hold_until_armed and self._continuous_planner is not None:
            self._apply_manual_hold(observed_at=checked_at)
            self._last_update_at = checked_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="manual_hold",
                confidence=checked_confidence,
                target_center_x=target_center_x,
                applied_center_x=0.5,
                target_pulse_width_us=self.config.center_pulse_width_us,
                commanded_pulse_width_us=None,
            )
        if self._return_to_zero_requested and self._continuous_planner is not None:
            decision = self._update_return_to_estimated_zero(
                observed_at=checked_at,
                checked_confidence=checked_confidence,
                target_center_x=target_center_x,
            )
            self._last_update_at = checked_at
            return decision
        if self.config.follow_exit_only:
            return self._update_follow_exit_only(
                observed_at=checked_at,
                target_center_x=target_center_x,
                confidence=checked_confidence,
                visible_target_present=checked_visible_target_present,
                visible_target_box_left=visible_target_box_left,
                visible_target_box_right=visible_target_box_right,
            )
        if self.config.follow_exit_only and checked_visible_target_present:
            if target_center_x is None:
                raise RuntimeError("visible exit-only target requires a concrete target_center_x")
            return self._update_visible_target_for_exit_only(
                observed_at=checked_at,
                target_center_x=float(target_center_x),
                confidence=checked_confidence,
            )

        applied_center_x, effective_active, reason = self._resolve_target(
            observed_at=checked_at,
            active=effective_active,
            target_center_x=target_center_x,
            confidence=checked_confidence,
            visible_target_present=checked_visible_target_present,
        )
        if reason == "awaiting_exit_confirmation":
            self._reset_zero_return_cycle_state()
            return self._await_exit_confirmation(
                observed_at=checked_at,
                confidence=checked_confidence,
                target_center_x=target_center_x,
                applied_center_x=applied_center_x,
            )
        if not self._return_to_zero_requested and reason != "recentering":
            self._reset_zero_return_cycle_state()
        absence_hold_decision = self._maybe_hold_continuous_absence_before_return_to_zero(
            observed_at=checked_at,
            confidence=checked_confidence,
            target_center_x=target_center_x,
            visible_target_present=checked_visible_target_present,
            active=effective_active,
            reason=reason,
        )
        if absence_hold_decision is not None:
            return absence_hold_decision
        if (
            self._continuous_planner is not None
            and not effective_active
            and reason == "recentering"
        ):
            rest_decision = self._drive_rest_position(
                observed_at=checked_at,
                confidence=checked_confidence,
                target_center_x=target_center_x,
            )
            self._last_update_at = checked_at
            return rest_decision
        applied_center_x = self._clamped_follow_center_x(applied_center_x)
        applied_center_x = self._smoothed_target_center_x(
            observed_at=checked_at,
            target_center_x=applied_center_x,
            active_tracking=effective_active,
        )
        target_pulse_width_us = self._target_pulse_width_for_center_x(
            center_x=applied_center_x,
            observed_at=checked_at,
            active_tracking=effective_active,
        )
        target_pulse_width_us = self._stabilize_visible_target_pulse_width(
            target_pulse_width_us=target_pulse_width_us,
            reason=reason,
        )
        if self._maybe_release_idle(
            observed_at=checked_at,
            active=effective_active,
            target_pulse_width_us=target_pulse_width_us,
        ):
            self._last_update_at = checked_at
            return AttentionServoDecision(
                observed_at=checked_at,
                active=False,
                reason="idle_released",
                confidence=checked_confidence,
                target_center_x=target_center_x,
                applied_center_x=applied_center_x,
                target_pulse_width_us=target_pulse_width_us,
                commanded_pulse_width_us=None,
            )
        released_decision = self._maybe_hold_released_target(
            observed_at=checked_at,
            active=effective_active,
            visible_target_present=checked_visible_target_present,
            reason=reason,
            confidence=checked_confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
        )
        if released_decision is not None:
            return released_decision
        planned_pulse_width_us = self._advance_planned_pulse_width(
            target_pulse_width_us,
            observed_at=checked_at,
            motion_profile="tracking",
        )
        commanded_pulse_width_us = self._command_pulse_width_for_plan(
            planned_pulse_width_us,
            target_pulse_width_us=target_pulse_width_us,
            motion_profile="tracking",
        )
        released_exit_decision = self._maybe_release_exit_hold(
            observed_at=checked_at,
            reason=reason,
            confidence=checked_confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        )
        if released_exit_decision is not None:
            self._last_update_at = checked_at
            return released_exit_decision
        if self._maybe_release_settled_target(
            observed_at=checked_at,
            active=effective_active,
            visible_target_present=checked_visible_target_present,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        ):
            self._last_update_at = checked_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=effective_active,
                reason="settled_released",
                confidence=checked_confidence,
                target_center_x=target_center_x,
                applied_center_x=applied_center_x,
                target_pulse_width_us=target_pulse_width_us,
                commanded_pulse_width_us=None,
            )

        try:
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
                        observed_at=checked_at,
                    )
                self._last_commanded_pulse_width_us = commanded_pulse_width_us
                if not self._writer_reports_live_position():
                    self._last_physical_pulse_width_us = commanded_pulse_width_us
                self._persist_runtime_state(observed_at=checked_at)
            self._last_update_at = checked_at
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise

        return AttentionServoDecision(
            observed_at=observed_at,
            active=effective_active,
            reason=reason,
            confidence=checked_confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        )

    def close(self) -> None:
        """Stop the current pulse train and release any underlying resources."""

        gpio = self.config.gpio
        try:
            if self._movement_journal_replay_player is not None:
                self._movement_journal_replay_player.close()
            if self.config.enabled and gpio is not None and self._last_commanded_pulse_width_us is not None:
                self._pulse_writer.disable(
                    gpio_chip=self.config.gpio_chip,
                    gpio=gpio,
                )
        finally:
            self._last_commanded_pulse_width_us = None
            self._reset_zero_return_cycle_state()
            self._reset_motion_state()
            if self._continuous_planner is not None:
                self._continuous_planner.reset()
            self._last_target_velocity_x_per_s = 0.0
            self._recent_visible_targets.clear()
            self._recent_visible_target_boxes.clear()
            self._exit_pursuit_target_pulse_width_us = None
            self._exit_pursuit_center_x = None
            self._exit_pursuit_settled_at = None
            self._exit_cooldown_until_at = None
            self._visible_edge_departure_since_at = None
            self._clear_visible_recenter(reset_timer=True)
            self._smoothed_center_x = None
            self._last_update_at = None
            self._centered_since = None
            self._settled_since = None
            self._released_pulse_width_us = None
            self._visible_target_pulse_width_us = None
            self._last_physical_pulse_width_us = None
            self._last_exit_hold_center_x = None
            self._pulse_writer.close()

    def debug_snapshot(self, *, observed_at: float | None) -> dict[str, object]:
        """Return one bounded debug snapshot of the controller's internal state."""

        checked_at = None if observed_at is None else float(observed_at)
        recent_targets: list[dict[str, object]] = []
        for sample_at, sample_center_x in list(self._recent_visible_targets)[-6:]:
            age_s = None if checked_at is None else round(max(0.0, checked_at - sample_at), 3)
            recent_targets.append(
                {
                    "age_s": age_s,
                    "center_x": round(sample_center_x, 4),
                }
            )
        last_target_age_s = None
        if checked_at is not None and self._last_target_at is not None:
            last_target_age_s = round(max(0.0, checked_at - self._last_target_at), 3)
        visible_departure_age_s = None
        if checked_at is not None and self._visible_edge_departure_since_at is not None:
            visible_departure_age_s = round(max(0.0, checked_at - self._visible_edge_departure_since_at), 3)
        visible_recenter_age_s = None
        if checked_at is not None and self._visible_recenter_since_at is not None:
            visible_recenter_age_s = round(max(0.0, checked_at - self._visible_recenter_since_at), 3)
        visible_recenter_settled_age_s = None
        if checked_at is not None and self._visible_recenter_settled_at is not None:
            visible_recenter_settled_age_s = round(max(0.0, checked_at - self._visible_recenter_settled_at), 3)
        exit_cooldown_remaining_s = None
        if checked_at is not None and self._exit_cooldown_until_at is not None:
            exit_cooldown_remaining_s = round(max(0.0, self._exit_cooldown_until_at - checked_at), 3)
        recent_exit_anchor_center_x = (
            None if checked_at is None else self._recent_exit_anchor_center_x(observed_at=checked_at)
        )
        recent_exit_box_edge = (
            None if checked_at is None else self._recent_exit_anchor_box_edge(observed_at=checked_at)
        )
        return {
            "last_target_center_x": None if self._last_target_center_x is None else round(self._last_target_center_x, 4),
            "last_target_age_s": last_target_age_s,
            "last_target_velocity_x_per_s": round(self._last_target_velocity_x_per_s, 4),
            "recent_visible_targets": recent_targets,
            "recent_exit_anchor_center_x": (
                None if recent_exit_anchor_center_x is None else round(recent_exit_anchor_center_x, 4)
            ),
            "recent_exit_box_edge": None if recent_exit_box_edge is None else round(recent_exit_box_edge, 4),
            "visible_edge_departure_age_s": visible_departure_age_s,
            "visible_recenter_age_s": visible_recenter_age_s,
            "visible_recenter_center_x": (
                None if self._visible_recenter_center_x is None else round(self._visible_recenter_center_x, 4)
            ),
            "visible_recenter_target_pulse_width_us": self._visible_recenter_target_pulse_width_us,
            "visible_recenter_settled_age_s": visible_recenter_settled_age_s,
            "exit_pursuit_center_x": None if self._exit_pursuit_center_x is None else round(self._exit_pursuit_center_x, 4),
            "exit_pursuit_target_pulse_width_us": self._exit_pursuit_target_pulse_width_us,
            "exit_pursuit_settled_at": self._exit_pursuit_settled_at,
            "exit_cooldown_remaining_s": exit_cooldown_remaining_s,
            "last_commanded_pulse_width_us": self._last_commanded_pulse_width_us,
            "last_physical_pulse_width_us": self._last_physical_pulse_width_us,
            "released_pulse_width_us": self._released_pulse_width_us,
            "visible_target_pulse_width_us": self._visible_target_pulse_width_us,
            "startup_rest_alignment_pending": self._startup_rest_alignment_pending,
            "startup_hold_until_armed": self._startup_hold_until_armed,
            "return_to_zero_requested": self._return_to_zero_requested,
            "zero_reference_confirmed": self._zero_reference_confirmed,
            "movement_journal_segments": len(self._movement_journal_snapshot(observed_at=checked_at)),
            "movement_journal_total_duration_s": round(
                sum(segment.duration_s for segment in self._movement_journal_snapshot(observed_at=checked_at)),
                3,
            ),
            "movement_journal_replay_active": self._active_movement_journal_replay_segment() is not None,
            "movement_journal_replay_due_in_s": self._movement_journal_replay_due_in_seconds(
                observed_at=checked_at,
            ),
            "continuous_estimated_heading_degrees": (
                None
                if self._continuous_planner is None
                else round(self._continuous_planner.estimated_heading_degrees, 3)
            ),
            "continuous_heading_uncertainty_degrees": round(self._heading_uncertainty_degrees, 3),
            "continuous_desired_heading_degrees": (
                None
                if self._continuous_planner is None
                else round(self._continuous_planner.desired_heading_degrees, 3)
            ),
            "fault_reason": self._fault_reason,
            "disabled_due_to_fault": self._disabled_due_to_fault,
            "config": {
                "control_mode": self.config.control_mode,
                "follow_exit_only": self.config.follow_exit_only,
                "visible_recenter_interval_s": round(self.config.visible_recenter_interval_s, 3),
                "visible_recenter_center_tolerance": round(
                    self.config.visible_recenter_center_tolerance,
                    4,
                ),
                "exit_visible_edge_threshold": round(self.config.exit_visible_edge_threshold, 4),
                "exit_visible_box_edge_threshold": round(self.config.exit_visible_box_edge_threshold, 4),
                "exit_activation_delay_s": round(self.config.exit_activation_delay_s, 3),
                "exit_reacquire_center_tolerance": round(self.config.exit_reacquire_center_tolerance, 4),
                "exit_cooldown_s": round(self.config.exit_cooldown_s, 3),
                "exit_follow_offset_limit": round(self.config.exit_follow_offset_limit, 4),
                "min_confidence": round(self.config.min_confidence, 4),
                "hold_min_confidence": round(self.config.hold_min_confidence, 4),
                "deadband": round(self.config.deadband, 4),
                "state_path": self.config.state_path,
                "estimated_zero_max_uncertainty_degrees": round(
                    self.config.estimated_zero_max_uncertainty_degrees,
                    3,
                ),
                "estimated_zero_settle_tolerance_degrees": round(
                    self.config.estimated_zero_settle_tolerance_degrees,
                    3,
                ),
                "estimated_zero_speed_scale": round(self.config.estimated_zero_speed_scale, 3),
                "estimated_zero_move_pulse_delta_us": self.config.estimated_zero_move_pulse_delta_us,
                "estimated_zero_move_period_s": round(self.config.estimated_zero_move_period_s, 3),
                "estimated_zero_move_duty_cycle": round(self.config.estimated_zero_move_duty_cycle, 3),
            },
        }
