"""Resolve visible targets, projection, and exit-anchor state."""

from __future__ import annotations

import time

from .common import _clamp, _clamp_ratio
from .controller_continuous import ControllerContinuousMixin
from .types import AttentionServoDecision

class ControllerTargetMixin(ControllerContinuousMixin):
    def _target_available(
        self,
        *,
        active: bool,
        target_center_x: float | None,
        confidence: float,
        visible_target_present: bool,
    ) -> bool:
        confidence_floor = self.config.min_confidence
        if (
            visible_target_present
            and active
            and target_center_x is not None
            and self._last_target_center_x is not None
            and self._last_target_at is not None
        ):
            # Acquisition can stay strict while an already-visible track gets a
            # lower release threshold to avoid confidence-edge thrashing.
            confidence_floor = min(confidence_floor, self.config.hold_min_confidence)
        return (
            active
            and target_center_x is not None
            and confidence >= confidence_floor
        )

    def _visible_target_present(
        self,
        *,
        active: bool,
        target_center_x: float | None,
    ) -> bool:
        return active and target_center_x is not None

    def _remember_visible_target(
        self,
        *,
        observed_at: float | None,
        target_center_x: float,
    ) -> None:
        """Keep a short visible history so exit-follow prefers the outward edge over brief inward tracker wobble."""

        if observed_at is None:
            return
        checked_at = float(observed_at)
        self._last_visible_target_at = checked_at
        history_window_s = max(
            self.config.target_hold_s,
            self.config.loss_extrapolation_s,
            self.config.exit_activation_delay_s,
        )
        cutoff_at = checked_at - max(0.0, history_window_s)
        while self._recent_visible_targets and self._recent_visible_targets[0][0] < cutoff_at:
            self._recent_visible_targets.popleft()
        self._recent_visible_targets.append((checked_at, _clamp_ratio(target_center_x)))

    def _remember_visible_target_box(
        self,
        *,
        observed_at: float | None,
        box_left: float | None,
        box_right: float | None,
    ) -> None:
        """Keep a short visible-box history so edge pursuit keys off real frame boundaries."""

        if observed_at is None or box_left is None or box_right is None:
            return
        checked_at = float(observed_at)
        normalized_left = _clamp_ratio(float(box_left))
        normalized_right = _clamp_ratio(float(box_right))
        if normalized_right < normalized_left:
            normalized_left, normalized_right = normalized_right, normalized_left
        history_window_s = max(
            self.config.target_hold_s,
            self.config.loss_extrapolation_s,
            self.config.exit_activation_delay_s,
        )
        cutoff_at = checked_at - max(0.0, history_window_s)
        while self._recent_visible_target_boxes and self._recent_visible_target_boxes[0][0] < cutoff_at:
            self._recent_visible_target_boxes.popleft()
        self._recent_visible_target_boxes.append((checked_at, normalized_left, normalized_right))

    def _visible_edge_departure_confirmed(
        self,
        *,
        observed_at: float | None,
        target_center_x: float,
    ) -> bool:
        """Return whether one visible side-departure should trigger monotone pursuit.

        The HDMI fast path can keep a user weakly visible as a small edge anchor
        while they are already leaving the frame, and the attention target can
        briefly snap inward again before the exit delay elapses. In exit-only
        mode that should still unlock the same one-direction pursuit after a
        short confirmation delay instead of blocking forever on a broad
        `person_visible=true` or a single centered wobble.
        """

        if observed_at is None:
            self._visible_edge_departure_since_at = None
            return False
        checked_at = float(observed_at)
        candidate_center_x = self._recent_exit_anchor_center_x(observed_at=checked_at)
        if candidate_center_x is None:
            candidate_center_x = float(target_center_x)
        exit_side = self._exit_side_for_center_x(candidate_center_x)
        edge_threshold = self.config.exit_visible_edge_threshold
        if edge_threshold <= 0.5:
            self._visible_edge_departure_since_at = None
            return False
        distance_from_center = abs(candidate_center_x - 0.5)
        required_distance = max(0.0, edge_threshold - 0.5)
        if distance_from_center < required_distance:
            self._visible_edge_departure_since_at = None
            return False
        edge_position = self._recent_exit_anchor_box_edge(observed_at=checked_at)
        if edge_position is not None and exit_side != 0:
            box_edge_threshold = self.config.exit_visible_box_edge_threshold
            if exit_side > 0 and edge_position < box_edge_threshold:
                self._visible_edge_departure_since_at = None
                return False
            if exit_side < 0 and edge_position > (1.0 - box_edge_threshold):
                self._visible_edge_departure_since_at = None
                return False
        if self._visible_edge_departure_since_at is None:
            self._visible_edge_departure_since_at = checked_at
            return False
        return (checked_at - self._visible_edge_departure_since_at) >= self.config.exit_activation_delay_s

    def _target_is_exit_reacquired_centered(self, target_center_x: float) -> bool:
        """Return whether one visible target is centered enough to end exit pursuit."""

        return abs(float(target_center_x) - 0.5) <= self.config.exit_reacquire_center_tolerance

    def _target_is_visible_recenter_centered(self, target_center_x: float) -> bool:
        """Return whether one visible target is close enough to center to skip periodic recentering."""

        return abs(float(target_center_x) - 0.5) <= self.config.visible_recenter_center_tolerance

    def _clear_visible_recenter(self, *, reset_timer: bool) -> None:
        self._visible_recenter_target_pulse_width_us = None
        self._visible_recenter_center_x = None
        self._visible_recenter_settled_at = None
        if reset_timer:
            self._visible_recenter_since_at = None

    def _should_begin_visible_recenter(self, *, observed_at: float | None) -> bool:
        if self._visible_recenter_target_pulse_width_us is not None:
            return True
        checked_at = time.monotonic() if observed_at is None else float(observed_at)
        interval_s = max(0.0, self.config.visible_recenter_interval_s)
        if interval_s <= 0.0:
            self._visible_recenter_since_at = checked_at
            return True
        since_at = self._visible_recenter_since_at
        if since_at is None or checked_at < since_at:
            self._visible_recenter_since_at = checked_at
            return False
        return (checked_at - since_at) >= interval_s

    def _begin_visible_recenter(self, *, normalized_target_center_x: float) -> bool:
        if self._target_is_visible_recenter_centered(normalized_target_center_x):
            return False
        self._visible_recenter_center_x = normalized_target_center_x
        self._visible_recenter_target_pulse_width_us = self._target_pulse_width_for_center_x(
            center_x=normalized_target_center_x,
            observed_at=self._last_update_at,
        )
        self._visible_recenter_settled_at = None
        return True

    def _recent_exit_anchor_center_x(self, *, observed_at: float | None) -> float | None:
        """Return the furthest recent visible point on the current exit side."""

        last_target_center_x = self._last_target_center_x
        if last_target_center_x is None:
            return None
        checked_at = None if observed_at is None else float(observed_at)
        history_window_s = max(
            self.config.target_hold_s,
            self.config.loss_extrapolation_s,
            self.config.exit_activation_delay_s,
        )
        if checked_at is not None:
            cutoff_at = checked_at - max(0.0, history_window_s)
            while self._recent_visible_targets and self._recent_visible_targets[0][0] < cutoff_at:
                self._recent_visible_targets.popleft()
        exit_side = self._current_exit_side(last_target_center_x)
        if exit_side == 0:
            return last_target_center_x
        anchor_center_x = last_target_center_x
        anchor_offset = abs(anchor_center_x - 0.5)
        for _, sample_center_x in self._recent_visible_targets:
            sample_offset = sample_center_x - 0.5
            if exit_side > 0 and sample_offset <= 0.0:
                continue
            if exit_side < 0 and sample_offset >= 0.0:
                continue
            sample_distance = abs(sample_offset)
            if sample_distance > anchor_offset:
                anchor_center_x = sample_center_x
                anchor_offset = sample_distance
        return anchor_center_x

    def _recent_exit_anchor_box_edge(self, *, observed_at: float | None) -> float | None:
        """Return the furthest recent visible box edge on the current exit side."""

        last_target_center_x = self._last_target_center_x
        if last_target_center_x is None:
            return None
        checked_at = None if observed_at is None else float(observed_at)
        history_window_s = max(
            self.config.target_hold_s,
            self.config.loss_extrapolation_s,
            self.config.exit_activation_delay_s,
        )
        if checked_at is not None:
            cutoff_at = checked_at - max(0.0, history_window_s)
            while self._recent_visible_target_boxes and self._recent_visible_target_boxes[0][0] < cutoff_at:
                self._recent_visible_target_boxes.popleft()
        exit_side = self._current_exit_side(last_target_center_x)
        if exit_side == 0:
            return None
        anchor_edge = None
        for _, sample_left, sample_right in self._recent_visible_target_boxes:
            sample_center_x = (sample_left + sample_right) * 0.5
            if exit_side > 0:
                if sample_center_x <= 0.5:
                    continue
                candidate_edge = sample_right
                if anchor_edge is None or candidate_edge > anchor_edge:
                    anchor_edge = candidate_edge
            else:
                if sample_center_x >= 0.5:
                    continue
                candidate_edge = sample_left
                if anchor_edge is None or candidate_edge < anchor_edge:
                    anchor_edge = candidate_edge
        return anchor_edge

    def _current_exit_side(self, last_target_center_x: float) -> int:
        """Return the most plausible exit side from recent target position and velocity."""

        last_side = self._exit_side_for_center_x(last_target_center_x)
        velocity_side = 0
        if self._last_target_velocity_x_per_s > 1e-4:
            velocity_side = 1
        elif self._last_target_velocity_x_per_s < -1e-4:
            velocity_side = -1
        exit_side = velocity_side or last_side
        if last_side != 0 and velocity_side != 0 and velocity_side != last_side:
            exit_side = last_side
        return exit_side

    def _exit_side_for_center_x(self, center_x: float | None) -> int:
        """Return the horizontal side token for one normalized target center."""

        if center_x is None:
            return 0
        offset = float(center_x) - 0.5
        if offset > 0.0:
            return 1
        if offset < 0.0:
            return -1
        return 0

    def _await_exit_confirmation(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        applied_center_x: float,
    ) -> AttentionServoDecision:
        """Keep the servo still until visibility loss persists long enough."""

        self._visible_target_pulse_width_us = None
        self._centered_since = None
        self._settled_since = None
        self._release_active_output(observed_at=observed_at)
        self._last_update_at = observed_at
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="awaiting_exit_confirmation",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=None,
            commanded_pulse_width_us=None,
        )

    def _resolve_target(
        self,
        *,
        observed_at: float | None,
        active: bool,
        target_center_x: float | None,
        confidence: float,
        visible_target_present: bool,
    ) -> tuple[float, bool, str]:
        checked_at = None if observed_at is None else float(observed_at)
        target_available = self._target_available(
            active=active,
            target_center_x=target_center_x,
            confidence=confidence,
            visible_target_present=visible_target_present,
        )
        if target_available:
            if target_center_x is None:
                raise RuntimeError("Target availability requires a concrete normalized target center")
            normalized_center_x = _clamp_ratio(float(target_center_x))
            self._last_visible_target_at = checked_at
            self._reset_zero_return_cycle_state()
            self._update_target_velocity(
                observed_at=checked_at,
                target_center_x=normalized_center_x,
            )
            self._last_target_center_x = normalized_center_x
            self._last_target_at = checked_at
            self._last_exit_hold_center_x = None
            return normalized_center_x, True, "following_target"

        if (
            self._last_target_center_x is not None
            and self._last_target_at is not None
            and checked_at is not None
            and (checked_at - self._last_target_at) <= self.config.target_hold_s
        ):
            if self.config.follow_exit_only:
                loss_elapsed_s = max(0.0, checked_at - self._last_target_at)
                if loss_elapsed_s < self.config.exit_activation_delay_s:
                    self._last_exit_hold_center_x = None
                    return self._last_target_center_x, False, "awaiting_exit_confirmation"
            projected_center_x, projected_reason = self._projected_target_center_x(
                observed_at=checked_at,
                loss_activation_delay_s=(
                    self.config.exit_activation_delay_s
                    if self.config.follow_exit_only
                    else 0.0
                ),
            )
            if self.config.follow_exit_only:
                self._last_exit_hold_center_x = projected_center_x
            return projected_center_x, True, projected_reason

        if self.config.follow_exit_only and self._last_exit_hold_center_x is not None:
            return self._last_exit_hold_center_x, False, "holding_exit_position"

        self._last_target_center_x = None
        self._last_target_at = None
        self._last_target_velocity_x_per_s = 0.0
        self._last_exit_hold_center_x = None
        return 0.5, False, "recentering"

    def _update_target_velocity(self, *, observed_at: float | None, target_center_x: float) -> None:
        previous_center_x = self._last_target_center_x
        previous_at = self._last_target_at
        if (
            observed_at is None
            or previous_center_x is None
            or previous_at is None
            or observed_at <= previous_at
        ):
            self._last_target_velocity_x_per_s = 0.0
            return
        dt = max(0.001, observed_at - previous_at)
        observed_velocity_x_per_s = (target_center_x - previous_center_x) / dt
        observed_velocity_x_per_s = _clamp(
            observed_velocity_x_per_s,
            minimum=-2.0,
            maximum=2.0,
        )
        self._last_target_velocity_x_per_s = (
            (self._last_target_velocity_x_per_s * 0.6)
            + (observed_velocity_x_per_s * 0.4)
        )

    def _projected_target_center_x(
        self,
        *,
        observed_at: float,
        loss_activation_delay_s: float = 0.0,
    ) -> tuple[float, str]:
        seed_center_x = self._recent_exit_anchor_center_x(observed_at=observed_at)
        if seed_center_x is None or self._last_target_at is None:
            return 0.5, "recentering"
        raw_elapsed_s = max(0.0, observed_at - self._last_target_at)
        elapsed_s = max(0.0, raw_elapsed_s - max(0.0, loss_activation_delay_s))
        projection_window_s = min(
            max(0.0, self.config.target_hold_s - max(0.0, loss_activation_delay_s)),
            self.config.loss_extrapolation_s,
        )
        if projection_window_s <= 0.0:
            return seed_center_x, "holding_recent_target"
        effective_elapsed_s = min(elapsed_s, projection_window_s)
        decay_progress = effective_elapsed_s / projection_window_s
        seed_offset = seed_center_x - 0.5
        seed_side = 0
        if seed_offset > 0.0:
            seed_side = 1
        elif seed_offset < 0.0:
            seed_side = -1
        velocity_x_per_s = self._last_target_velocity_x_per_s
        if seed_side > 0 and velocity_x_per_s < 0.0:
            velocity_x_per_s = 0.0
        elif seed_side < 0 and velocity_x_per_s > 0.0:
            velocity_x_per_s = 0.0
        projected_offset = (
            velocity_x_per_s
            * self.config.loss_extrapolation_gain
            * effective_elapsed_s
            * max(0.0, 1.0 - (0.5 * decay_progress))
        )
        projected_center_x = _clamp_ratio(seed_center_x + projected_offset)
        if elapsed_s <= projection_window_s:
            return projected_center_x, "projecting_recent_trajectory"
        return projected_center_x, "holding_projected_trajectory"

    def _target_pulse_width_for_center_x(
        self,
        *,
        center_x: float,
        observed_at: float | None,
        active_tracking: bool = False,
    ) -> int:
        if self._continuous_planner is None:
            return self._pulse_width_for_center_x(center_x)
        if active_tracking:
            # Visible-target following already closes the loop through vision.
            # Reusing the virtual-heading planner here makes sparse Pi refresh
            # cadence overshoot the estimated heading and flip left/right on
            # the next frame even while the target stays on the same side.
            return self._continuous_tracking_pulse_width_for_center_x(center_x)
        return self._continuous_planner.target_pulse_width_for_heading(
            self._desired_heading_degrees_for_center_x(center_x),
            observed_at=observed_at,
        )
