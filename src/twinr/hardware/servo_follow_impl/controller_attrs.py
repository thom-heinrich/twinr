"""Declare shared controller attributes for mixin-based servo-follow logic."""

from __future__ import annotations

from typing import Any

class ControllerAttrsMixin:
    _clear_exit_pursuit: Any
    _centered_since: Any
    _continuous_planner: Any
    _continuous_startup_heading_degrees: Any
    _disabled_due_to_fault: Any
    _exit_cooldown_until_at: Any
    _exit_pursuit_center_x: Any
    _exit_pursuit_settled_at: Any
    _exit_pursuit_target_pulse_width_us: Any
    _fault_reason: Any
    _fault_retry_after_at: Any
    _heading_uncertainty_degrees: Any
    _last_commanded_pulse_width_us: Any
    _last_exit_hold_center_x: Any
    _last_physical_pulse_width_us: Any
    _last_runtime_state_mtime_ns: Any
    _last_saved_runtime_state: Any
    _last_target_at: Any
    _last_target_center_x: Any
    _last_target_velocity_x_per_s: Any
    _last_update_at: Any
    _last_visible_target_at: Any
    _movement_journal: Any
    _movement_journal_active_pulse_width_us: Any
    _movement_journal_active_started_at: Any
    _movement_journal_replay_player: Any
    _planned_acceleration_us_per_s2: Any
    _planned_pulse_width_us: Any
    _planned_velocity_us_per_s: Any
    _pulse_writer: Any
    _recent_visible_target_boxes: Any
    _recent_visible_targets: Any
    _released_pulse_width_us: Any
    _return_to_zero_requested: Any
    _settled_since: Any
    _smoothed_center_x: Any
    _startup_hold_until_armed: Any
    _startup_rest_alignment_pending: Any
    _state_store: Any
    _target_pulse_width_for_center_x: Any
    _visible_edge_departure_since_at: Any
    _visible_recenter_center_x: Any
    _visible_recenter_settled_at: Any
    _visible_recenter_since_at: Any
    _visible_recenter_target_pulse_width_us: Any
    _visible_retarget_cooldown_anchor_pulse_width_us: Any
    _visible_retarget_cooldown_until_at: Any
    _visible_target_pulse_width_us: Any
    _zero_reference_confirmed: Any
    _zero_return_direction_sign: Any
    _zero_return_move_phase_active: Any
    _zero_return_phase_anchor_at: Any
    config: Any
