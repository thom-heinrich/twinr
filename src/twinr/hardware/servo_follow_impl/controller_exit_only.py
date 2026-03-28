"""Isolate exit-only follow, cooldown, and visible-recenter orchestration."""

from __future__ import annotations

import time

from .common import _clamp_ratio
from .controller_targets import ControllerTargetMixin
from .types import AttentionServoDecision

class ControllerExitOnlyMixin(ControllerTargetMixin):
    def _update_follow_exit_only(
        self,
        *,
        observed_at: float | None,
        target_center_x: float | None,
        confidence: float,
        visible_target_present: bool,
        visible_target_box_left: float | None,
        visible_target_box_right: float | None,
    ) -> AttentionServoDecision:
        normalized_target_center_x = None if target_center_x is None else _clamp_ratio(float(target_center_x))
        if (
            self._exit_cooldown_until_at is not None
            and observed_at is not None
            and observed_at >= self._exit_cooldown_until_at
        ):
            self._exit_cooldown_until_at = None
        if self._exit_cooldown_until_at is not None:
            return self._hold_exit_cooldown(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
                normalized_target_center_x=normalized_target_center_x,
                visible_target_present=visible_target_present,
            )
        if self._startup_rest_alignment_pending and self._exit_pursuit_target_pulse_width_us is None:
            if visible_target_present and normalized_target_center_x is not None:
                return self._handle_visible_target_during_startup_alignment(
                    observed_at=observed_at,
                    confidence=confidence,
                    target_center_x=target_center_x,
                    normalized_target_center_x=normalized_target_center_x,
                )
            rest_decision = self._drive_rest_position(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
            )
            if rest_decision.reason == "idle_released":
                self._startup_rest_alignment_pending = False
            return rest_decision
        if visible_target_present and normalized_target_center_x is not None:
            return self._handle_visible_target_for_exit_only(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
                normalized_target_center_x=normalized_target_center_x,
                visible_target_box_left=visible_target_box_left,
                visible_target_box_right=visible_target_box_right,
            )
        self._clear_visible_recenter(reset_timer=True)
        if self._exit_pursuit_target_pulse_width_us is not None:
            if self._should_return_to_rest_from_exit_pursuit(
                observed_at=observed_at,
                visible_target_present=visible_target_present,
            ):
                self._clear_exit_pursuit(clear_recent_visible_targets=False)
                self._last_target_center_x = None
                self._last_target_at = None
                self._last_target_velocity_x_per_s = 0.0
                self._smoothed_center_x = None
                self._visible_target_pulse_width_us = None
                return self._drive_rest_position(
                    observed_at=observed_at,
                    confidence=confidence,
                    target_center_x=target_center_x,
                )
            return self._drive_exit_pursuit_target(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
                reason="pursuing_exit_direction",
            )
        if (
            self._last_target_center_x is not None
            and self._last_target_at is not None
            and observed_at is not None
            and (observed_at - self._last_target_at) <= self.config.target_hold_s
        ):
            loss_elapsed_s = max(0.0, observed_at - self._last_target_at)
            if loss_elapsed_s < self.config.exit_activation_delay_s:
                self._last_exit_hold_center_x = None
                self._release_active_output(observed_at=observed_at)
                self._last_update_at = observed_at
                return AttentionServoDecision(
                    observed_at=observed_at,
                    active=False,
                    reason="awaiting_exit_confirmation",
                    confidence=confidence,
                    target_center_x=target_center_x,
                    applied_center_x=self._last_target_center_x,
                    target_pulse_width_us=None,
                    commanded_pulse_width_us=None,
                )
            if self._begin_exit_pursuit(observed_at=observed_at) is not None:
                return self._drive_exit_pursuit_target(
                    observed_at=observed_at,
                    confidence=confidence,
                    target_center_x=target_center_x,
                    reason="pursuing_exit_direction",
                )
        self._clear_exit_pursuit(clear_recent_visible_targets=False)
        return self._drive_rest_position(
            observed_at=observed_at,
            confidence=confidence,
            target_center_x=target_center_x,
        )

    def _handle_visible_target_for_exit_only(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        normalized_target_center_x: float,
        visible_target_box_left: float | None,
        visible_target_box_right: float | None,
    ) -> AttentionServoDecision:
        self._remember_visible_target_box(
            observed_at=observed_at,
            box_left=visible_target_box_left,
            box_right=visible_target_box_right,
        )
        self._remember_visible_target(
            observed_at=observed_at,
            target_center_x=normalized_target_center_x,
        )
        self._update_target_velocity(
            observed_at=observed_at,
            target_center_x=normalized_target_center_x,
        )
        self._last_target_center_x = normalized_target_center_x
        self._last_target_at = observed_at
        self._last_exit_hold_center_x = None
        self._smoothed_center_x = normalized_target_center_x
        edge_departure_confirmed = self._visible_edge_departure_confirmed(
            observed_at=observed_at,
            target_center_x=normalized_target_center_x,
        )
        if self._exit_pursuit_target_pulse_width_us is None:
            if edge_departure_confirmed and self._begin_exit_pursuit(observed_at=observed_at) is not None:
                return self._drive_exit_pursuit_target(
                    observed_at=observed_at,
                    confidence=confidence,
                    target_center_x=target_center_x,
                    reason="pursuing_edge_departure",
                )
            if self._visible_recenter_target_pulse_width_us is not None:
                return self._drive_visible_recenter_target(
                    observed_at=observed_at,
                    confidence=confidence,
                    target_center_x=target_center_x,
                    normalized_target_center_x=normalized_target_center_x,
                )
            if self._target_is_visible_recenter_centered(normalized_target_center_x):
                self._clear_visible_recenter(reset_timer=True)
                self._exit_pursuit_settled_at = None
                self._release_active_output(observed_at=observed_at)
                self._last_update_at = observed_at
                return AttentionServoDecision(
                    observed_at=observed_at,
                    active=False,
                    reason="waiting_for_exit",
                    confidence=confidence,
                    target_center_x=target_center_x,
                    applied_center_x=normalized_target_center_x,
                    target_pulse_width_us=None,
                    commanded_pulse_width_us=None,
                )
            if self._should_begin_visible_recenter(observed_at=observed_at):
                if self._begin_visible_recenter(normalized_target_center_x=normalized_target_center_x):
                    return self._drive_visible_recenter_target(
                        observed_at=observed_at,
                        confidence=confidence,
                        target_center_x=target_center_x,
                        normalized_target_center_x=normalized_target_center_x,
                    )
                self._clear_visible_recenter(reset_timer=True)
            self._exit_pursuit_settled_at = None
            self._release_active_output(observed_at=observed_at)
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="waiting_for_exit",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=normalized_target_center_x,
                target_pulse_width_us=None,
                commanded_pulse_width_us=None,
            )
        if not self._target_is_exit_reacquired_centered(normalized_target_center_x):
            return self._drive_exit_pursuit_target(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
                reason="pursuing_edge_departure",
            )
        target_pulse_width_us = self._exit_pursuit_target_pulse_width_us
        self._start_exit_cooldown(observed_at=observed_at)
        self._release_active_output(observed_at=observed_at)
        self._last_update_at = observed_at
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="reacquired_visible_cooldown",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=normalized_target_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=None,
        )

    def _hold_exit_cooldown(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        normalized_target_center_x: float | None,
        visible_target_present: bool,
    ) -> AttentionServoDecision:
        if visible_target_present and normalized_target_center_x is not None:
            self._remember_visible_target(
                observed_at=observed_at,
                target_center_x=normalized_target_center_x,
            )
            self._update_target_velocity(
                observed_at=observed_at,
                target_center_x=normalized_target_center_x,
            )
            self._last_target_center_x = normalized_target_center_x
            self._last_target_at = observed_at
            self._smoothed_center_x = normalized_target_center_x
        self._release_active_output(observed_at=observed_at)
        self._last_update_at = observed_at
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="exit_cooldown",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=normalized_target_center_x,
            target_pulse_width_us=None,
            commanded_pulse_width_us=None,
        )

    def _drive_exit_pursuit_target(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        reason: str,
    ) -> AttentionServoDecision:
        applied_center_x = self._exit_pursuit_center_x
        if applied_center_x is None:
            return self._drive_rest_position(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
            )
        target_pulse_width_us = self._target_pulse_width_for_center_x(
            center_x=applied_center_x,
            observed_at=observed_at,
        )
        self._exit_pursuit_target_pulse_width_us = target_pulse_width_us
        if (
            self._exit_pursuit_settled_at is not None
            and self._last_commanded_pulse_width_us is None
        ):
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="holding_exit_limit",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=applied_center_x,
                target_pulse_width_us=target_pulse_width_us,
                commanded_pulse_width_us=None,
            )
        commanded_pulse_width_us = self._write_target_pulse_width(
            observed_at=observed_at,
            target_pulse_width_us=target_pulse_width_us,
            motion_profile="exit",
        )
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - commanded_pulse_width_us) <= tolerance_us:
            if (
                observed_at is not None
                and self._exit_pursuit_settled_at is not None
                and observed_at >= self._exit_pursuit_settled_at
                and (observed_at - self._exit_pursuit_settled_at) >= self.config.exit_settle_hold_s
            ):
                self._release_active_output(observed_at=observed_at)
                self._last_update_at = observed_at
                return AttentionServoDecision(
                    observed_at=observed_at,
                    active=False,
                    reason="holding_exit_limit",
                    confidence=confidence,
                    target_center_x=target_center_x,
                    applied_center_x=applied_center_x,
                    target_pulse_width_us=target_pulse_width_us,
                    commanded_pulse_width_us=None,
                )
            if observed_at is not None and (
                self._exit_pursuit_settled_at is None or observed_at < self._exit_pursuit_settled_at
            ):
                self._exit_pursuit_settled_at = observed_at
        else:
            self._exit_pursuit_settled_at = None
        return AttentionServoDecision(
            observed_at=observed_at,
            active=True,
            reason=reason,
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        )

    def _drive_visible_recenter_target(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        normalized_target_center_x: float,
    ) -> AttentionServoDecision:
        applied_center_x = self._visible_recenter_center_x
        if applied_center_x is None:
            self._clear_visible_recenter(reset_timer=True)
            self._release_active_output(observed_at=observed_at)
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="waiting_for_exit",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=normalized_target_center_x,
                target_pulse_width_us=None,
                commanded_pulse_width_us=None,
            )
        target_pulse_width_us = self._target_pulse_width_for_center_x(
            center_x=applied_center_x,
            observed_at=observed_at,
        )
        self._visible_recenter_target_pulse_width_us = target_pulse_width_us
        if self._target_is_visible_recenter_centered(normalized_target_center_x):
            self._clear_visible_recenter(reset_timer=True)
            self._release_active_output(observed_at=observed_at)
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="waiting_for_exit",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=normalized_target_center_x,
                target_pulse_width_us=target_pulse_width_us,
                commanded_pulse_width_us=None,
            )
        target_side = self._exit_side_for_center_x(normalized_target_center_x)
        latched_side = self._exit_side_for_center_x(applied_center_x)
        if target_side != 0 and latched_side != 0 and target_side != latched_side:
            self._clear_visible_recenter(reset_timer=True)
            self._release_active_output(observed_at=observed_at)
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="waiting_for_exit",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=normalized_target_center_x,
                target_pulse_width_us=None,
                commanded_pulse_width_us=None,
            )
        commanded_pulse_width_us = self._write_target_pulse_width(
            observed_at=observed_at,
            target_pulse_width_us=target_pulse_width_us,
            motion_profile="tracking",
        )
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - commanded_pulse_width_us) <= tolerance_us:
            if (
                observed_at is not None
                and self._visible_recenter_settled_at is not None
                and observed_at >= self._visible_recenter_settled_at
                and (observed_at - self._visible_recenter_settled_at) >= self.config.settled_release_s
            ):
                self._clear_visible_recenter(reset_timer=True)
                self._release_active_output(observed_at=observed_at)
                self._last_update_at = observed_at
                return AttentionServoDecision(
                    observed_at=observed_at,
                    active=False,
                    reason="waiting_for_exit",
                    confidence=confidence,
                    target_center_x=target_center_x,
                    applied_center_x=normalized_target_center_x,
                    target_pulse_width_us=target_pulse_width_us,
                    commanded_pulse_width_us=None,
                )
            if observed_at is not None and (
                self._visible_recenter_settled_at is None
                or observed_at < self._visible_recenter_settled_at
            ):
                self._visible_recenter_settled_at = observed_at
        else:
            self._visible_recenter_settled_at = None
        return AttentionServoDecision(
            observed_at=observed_at,
            active=True,
            reason="pursuing_visible_recenter",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        )

    def _drive_rest_position(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
    ) -> AttentionServoDecision:
        if self._continuous_planner is not None:
            target_pulse_width_us, settled_at_zero = self._prepare_monotonic_zero_return_target(
                observed_at=observed_at,
            )
            if settled_at_zero:
                self._release_active_output(observed_at=observed_at)
                self._last_update_at = observed_at
                return AttentionServoDecision(
                    observed_at=observed_at,
                    active=False,
                    reason="idle_released",
                    confidence=confidence,
                    target_center_x=target_center_x,
                    applied_center_x=0.5,
                    target_pulse_width_us=target_pulse_width_us,
                    commanded_pulse_width_us=None,
                )
            if target_pulse_width_us is None:
                raise RuntimeError("continuous zero-return target was unexpectedly unavailable")
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
                reason="recentering",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=0.5,
                target_pulse_width_us=target_pulse_width_us,
                commanded_pulse_width_us=commanded_pulse_width_us,
            )
        target_pulse_width_us = self.config.center_pulse_width_us
        if self._maybe_release_idle(
            observed_at=observed_at,
            active=False,
            target_pulse_width_us=target_pulse_width_us,
        ):
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="idle_released",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=0.5,
                target_pulse_width_us=target_pulse_width_us,
                commanded_pulse_width_us=None,
            )
        commanded_pulse_width_us = self._write_target_pulse_width(
            observed_at=observed_at,
            target_pulse_width_us=target_pulse_width_us,
            motion_profile="rest",
        )
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="recentering",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=0.5,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        )

    def _begin_exit_pursuit(self, *, observed_at: float | None) -> float | None:
        anchor_center_x = self._recent_exit_anchor_center_x(observed_at=observed_at)
        if anchor_center_x is None:
            return None
        anchor_offset = anchor_center_x - 0.5
        if abs(anchor_offset) <= self.config.deadband:
            return None
        exit_sign = 1.0 if anchor_offset > 0.0 else -1.0
        exit_center_x = _clamp_ratio(0.5 + (0.5 * self.config.exit_follow_offset_limit * exit_sign))
        self._clear_visible_recenter(reset_timer=True)
        self._exit_pursuit_center_x = exit_center_x
        self._exit_pursuit_target_pulse_width_us = self._target_pulse_width_for_center_x(
            center_x=exit_center_x,
            observed_at=observed_at,
        )
        self._exit_pursuit_settled_at = None
        self._last_exit_hold_center_x = exit_center_x
        return exit_center_x

    def _start_exit_cooldown(self, *, observed_at: float | None) -> None:
        cooldown_anchor_at = time.monotonic() if observed_at is None else observed_at
        self._exit_cooldown_until_at = cooldown_anchor_at + self.config.exit_cooldown_s
        self._clear_exit_pursuit(clear_recent_visible_targets=True)
        self._last_target_center_x = None
        self._last_target_at = None
        self._last_target_velocity_x_per_s = 0.0
        self._smoothed_center_x = None
        self._visible_target_pulse_width_us = None

    def _clear_exit_pursuit(self, *, clear_recent_visible_targets: bool) -> None:
        self._exit_pursuit_target_pulse_width_us = None
        self._exit_pursuit_center_x = None
        self._exit_pursuit_settled_at = None
        self._last_exit_hold_center_x = None
        self._visible_edge_departure_since_at = None
        self._clear_visible_recenter(reset_timer=True)
        if clear_recent_visible_targets:
            self._recent_visible_targets.clear()
            self._recent_visible_target_boxes.clear()

    def _should_return_to_rest_from_exit_pursuit(
        self,
        *,
        observed_at: float | None,
        visible_target_present: bool,
    ) -> bool:
        if visible_target_present or self._exit_pursuit_settled_at is None:
            return False
        if observed_at is None:
            return False
        return_anchor_at = self._last_target_at
        if return_anchor_at is None:
            return_anchor_at = self._exit_pursuit_settled_at
        return (observed_at - return_anchor_at) >= self.config.exit_cooldown_s

    def _update_visible_target_for_exit_only(
        self,
        *,
        observed_at: float | None,
        target_center_x: float,
        confidence: float,
    ) -> AttentionServoDecision:
        """Remember the visible target trajectory without physically following it."""

        normalized_center_x = _clamp_ratio(target_center_x)
        self._remember_visible_target(
            observed_at=observed_at,
            target_center_x=normalized_center_x,
        )
        self._update_target_velocity(
            observed_at=observed_at,
            target_center_x=normalized_center_x,
        )
        self._last_target_center_x = normalized_center_x
        self._last_target_at = observed_at
        self._last_exit_hold_center_x = None
        self._smoothed_center_x = normalized_center_x
        self._visible_target_pulse_width_us = None
        self._centered_since = None
        self._settled_since = None
        self._release_active_output(observed_at=observed_at)
        self._last_update_at = observed_at
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="waiting_for_exit",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=normalized_center_x,
            target_pulse_width_us=None,
            commanded_pulse_width_us=None,
        )

    def _handle_visible_target_during_startup_alignment(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        normalized_target_center_x: float,
    ) -> AttentionServoDecision:
        """Return stale startup alignment to neutral before enabling exit-only waiting semantics.

        The kernel writer remembers the last pulse width after release. If the
        runtime restarts while the servo is still off-center from an older exit
        move, entering `waiting_for_exit` immediately would treat that stale
        physical pose as neutral and later produce only a tiny nudge on loss.
        During this bounded one-time alignment phase we still remember the
        visible target history, but physically return to center first.
        """

        self._remember_visible_target(
            observed_at=observed_at,
            target_center_x=normalized_target_center_x,
        )
        self._update_target_velocity(
            observed_at=observed_at,
            target_center_x=normalized_target_center_x,
        )
        self._last_target_center_x = normalized_target_center_x
        self._last_target_at = observed_at
        self._smoothed_center_x = normalized_target_center_x
        rest_decision = self._drive_rest_position(
            observed_at=observed_at,
            confidence=confidence,
            target_center_x=target_center_x,
        )
        if rest_decision.reason == "idle_released":
            self._startup_rest_alignment_pending = False
        if rest_decision.reason == "recentering":
            return AttentionServoDecision(
                observed_at=rest_decision.observed_at,
                active=False,
                reason="startup_recentering",
                confidence=rest_decision.confidence,
                target_center_x=rest_decision.target_center_x,
                applied_center_x=rest_decision.applied_center_x,
                target_pulse_width_us=rest_decision.target_pulse_width_us,
                commanded_pulse_width_us=rest_decision.commanded_pulse_width_us,
            )
        return rest_decision
