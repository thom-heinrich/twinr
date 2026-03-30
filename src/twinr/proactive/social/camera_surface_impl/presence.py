# CHANGELOG: 2026-03-29
# BUG-1: Rejected out-of-order / clock-rollback observations so stale frames can no longer rewind
# BUG-2: Fixed stringly-typed boolean coercion ("false", "0", "no") that previously evaluated to True
# SEC-1: Bounded visible_persons ingestion to reduce practical RAM/CPU exhaustion risk on Raspberry Pi 4
# IMP-1: Added switch-aware center smoothing, per-signal hold windows, and monotonic-safe age handling
# IMP-2: Added lazy re-entrant state locking and deterministic visible-person ordering for threaded pipelines

"""Presence, health, and secondary-field resolution for the camera surface."""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable
from itertools import islice
from typing import Any

from ..engine import (
    SocialBodyPose,
    SocialMotionState,
    SocialPersonZone,
    SocialSpatialBox,
    SocialVisiblePerson,
)
from .coercion import (
    coerce_body_pose,
    coerce_motion_state,
    coerce_optional_ratio,
    coerce_optional_text,
    coerce_optional_timestamp,
    coerce_person_zone,
    coerce_spatial_box,
    coerce_timestamp,
    coerce_visible_persons,
)

_PRESENCE_LOCK_INIT_GUARD = threading.Lock()
_TRUTHY_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSY_STRINGS = frozenset({"0", "false", "f", "no", "n", "off", "", "none", "null"})


class ProactiveCameraPresenceMixin:
    """Resolve presence, health, motion, and held secondary camera fields."""

    def _presence_state_guard(self: Any) -> threading.RLock:
        """Return a lazily-created re-entrant lock for shared presence state."""
        lock = getattr(self, "_presence_state_lock_obj", None)
        if lock is None:
            with _PRESENCE_LOCK_INIT_GUARD:
                lock = getattr(self, "_presence_state_lock_obj", None)
                if lock is None:
                    lock = threading.RLock()
                    setattr(self, "_presence_state_lock_obj", lock)
        return lock

    def _state(self: Any, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)

    def _config_float(self: Any, name: str, default: float) -> float:
        config = getattr(self, "config", None)
        if config is None:
            return default
        raw_value = getattr(config, name, default)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(value):
            return default
        return value

    def _config_int(self: Any, name: str, default: int) -> int:
        config = getattr(self, "config", None)
        if config is None:
            return default
        raw_value = getattr(config, name, default)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return default
        return value

    def _config_ratio(self: Any, name: str, default: float) -> float:
        value = self._config_float(name, default)
        return max(0.0, min(1.0, value))

    def _secondary_hold_window_s(self: Any, field_name: str) -> float:
        specific = self._config_float(f"{field_name}_hold_s", float("nan"))
        if math.isfinite(specific):
            return max(0.0, specific)
        return max(0.0, self._config_float("secondary_unknown_hold_s", 0.0))

    def _coerce_timestamp_safe(
        self: Any,
        raw_value: object,
        *,
        fallback: float | None = None,
    ) -> float | None:
        try:
            value = coerce_timestamp(raw_value)
        except Exception:
            return fallback
        if value is None or not math.isfinite(value):
            return fallback
        return float(value)

    def _coerce_optional_timestamp_safe(
        self: Any,
        raw_value: object,
        *,
        fallback: float | None = None,
    ) -> float | None:
        try:
            value = coerce_optional_timestamp(raw_value)
        except Exception:
            return fallback
        if value is None:
            return None
        if not math.isfinite(value):
            return fallback
        return float(value)

    def _observed_ts(self: Any, observed_at: object) -> float:
        fallback = self._state("_presence_latest_observed_at")
        value = self._coerce_timestamp_safe(observed_at, fallback=fallback)
        if value is None:
            return 0.0
        return value

    def _age_seconds(
        self: Any,
        *,
        observed_at: float,
        last_seen_at: float | None,
    ) -> float | None:
        if last_seen_at is None:
            return None
        current = self._coerce_timestamp_safe(observed_at)
        previous = self._coerce_timestamp_safe(last_seen_at, fallback=last_seen_at)
        if current is None or previous is None:
            return None
        return current - previous

    def _is_out_of_order(
        self: Any,
        *,
        observed_at: float,
        last_seen_at: float | None,
    ) -> bool:
        age_s = self._age_seconds(observed_at=observed_at, last_seen_at=last_seen_at)
        if age_s is None:
            return False
        tolerance_s = max(0.0, self._config_float("presence_out_of_order_tolerance_s", 0.0))
        return age_s < (-tolerance_s)

    def _accept_global_update(self: Any, *, observed_at: float, cache_attr: str) -> bool:
        last_seen_at = self._state(cache_attr)
        if self._is_out_of_order(observed_at=observed_at, last_seen_at=last_seen_at):
            return False
        latest_global = self._state("_presence_latest_observed_at")
        if latest_global is None or observed_at >= latest_global:
            setattr(self, "_presence_latest_observed_at", observed_at)
        if last_seen_at is None or observed_at >= last_seen_at:
            setattr(self, cache_attr, observed_at)
        return True

    def _coerce_bool_like(self: Any, raw_value: object) -> bool:
        if isinstance(raw_value, bool):
            return raw_value
        if raw_value is None:
            return False
        if isinstance(raw_value, (bytes, bytearray)):
            try:
                raw_value = raw_value.decode("utf-8", errors="ignore")
            except Exception:
                return bool(raw_value)
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in _TRUTHY_STRINGS:
                return True
            if normalized in _FALSY_STRINGS:
                return False
        if isinstance(raw_value, (int, float)):
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                return bool(raw_value)
            if not math.isfinite(numeric):
                return False
            return numeric != 0.0
        return bool(raw_value)

    def _coerce_optional_ratio_safe(self: Any, raw_value: object) -> float | None:
        try:
            value = coerce_optional_ratio(raw_value)
        except Exception:
            return None
        if value is None:
            return None
        if not math.isfinite(value):
            return None
        return max(0.0, min(1.0, float(value)))

    def _coerce_person_zone_safe(self: Any, raw_value: object) -> SocialPersonZone:
        try:
            return coerce_person_zone(raw_value)
        except Exception:
            return SocialPersonZone.UNKNOWN

    def _coerce_spatial_box_safe(self: Any, raw_value: object) -> SocialSpatialBox | None:
        try:
            return coerce_spatial_box(raw_value)
        except Exception:
            return None

    def _coerce_body_pose_safe(self: Any, raw_value: object) -> SocialBodyPose:
        try:
            return coerce_body_pose(raw_value)
        except Exception:
            return SocialBodyPose.UNKNOWN

    def _coerce_motion_state_safe(self: Any, raw_value: object) -> SocialMotionState:
        try:
            return coerce_motion_state(raw_value)
        except Exception:
            return SocialMotionState.UNKNOWN

    def _bounded_raw_visible_persons(self: Any, raw_value: object) -> object:
        # BREAKING: visible_persons is capped by config.max_visible_persons (default 16) to bound RAM/CPU on Pi 4.
        # Set config.max_visible_persons <= 0 to disable the cap.
        max_visible_persons = self._config_int("max_visible_persons", 16)
        if max_visible_persons <= 0 or raw_value is None:
            return raw_value
        if isinstance(raw_value, tuple):
            return raw_value[:max_visible_persons]
        if isinstance(raw_value, list):
            return raw_value[:max_visible_persons]
        if isinstance(raw_value, Iterable) and not isinstance(raw_value, (str, bytes, bytearray, dict)):
            return list(islice(raw_value, max_visible_persons))
        return raw_value

    def _coerce_visible_persons_safe(self: Any, raw_value: object) -> tuple[SocialVisiblePerson, ...]:
        try:
            people = coerce_visible_persons(self._bounded_raw_visible_persons(raw_value))
        except Exception:
            return ()
        max_visible_persons = self._config_int("max_visible_persons", 16)
        if max_visible_persons > 0 and len(people) > max_visible_persons:
            return people[:max_visible_persons]
        return people

    def _canonicalize_visible_persons(
        self: Any,
        *,
        people: tuple[SocialVisiblePerson, ...],
        primary_person_box: SocialSpatialBox | None,
    ) -> tuple[SocialVisiblePerson, ...]:
        # BREAKING: visible_persons ordering is canonicalized (primary match first, then confidence)
        # so downstream consumers see a stable order even when detector ordering jitters frame to frame.
        if len(people) < 2:
            return people

        indexed_people = list(enumerate(people))

        def sort_key(item: tuple[int, SocialVisiblePerson]) -> tuple[int, float, int]:
            index, person = item
            confidence = getattr(person, "confidence", None)
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = -1.0
            if not math.isfinite(confidence_value):
                confidence_value = -1.0
            primary_match = 1 if primary_person_box is not None and getattr(person, "box", None) == primary_person_box else 0
            return primary_match, confidence_value, -index

        indexed_people.sort(key=sort_key, reverse=True)
        return tuple(person for _, person in indexed_people)

    def _resolve_person_returned(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
        person_visible: bool,
        person_visible_rising: bool,
    ) -> bool:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            person_returned = False
            if inspected:
                if not self._accept_global_update(observed_at=now, cache_attr="_last_person_presence_observed_at"):
                    return False

                has_seen_person = bool(self._state("_has_seen_person", False))
                absence_started_at = self._state("_absence_started_at")
                last_authoritative_visible = bool(self._state("_last_authoritative_person_visible", False))

                if person_visible:
                    if (
                        person_visible_rising
                        and has_seen_person
                        and absence_started_at is not None
                        and not self._is_out_of_order(observed_at=now, last_seen_at=absence_started_at)
                        and (now - absence_started_at) >= self._config_float("person_returned_absence_s", 0.0)
                    ):
                        person_returned = True
                    self._has_seen_person = True
                    self._last_authoritative_person_visible = True
                    self._last_person_seen_at = now
                    self._absence_started_at = None
                    if person_visible_rising:
                        self._person_appeared_at = now
                        self._person_appeared_seen_at = now
                else:
                    if last_authoritative_visible or absence_started_at is None:
                        self._absence_started_at = now
                    if last_authoritative_visible:
                        self._person_disappeared_at = now
                        self._person_disappeared_seen_at = now
                    self._last_authoritative_person_visible = False
            return person_returned

    def _resolve_person_recently_visible(
        self: Any,
        *,
        observed_at: float,
        person_visible: bool,
        person_visible_unknown: bool,
    ) -> tuple[bool, bool]:
        with self._presence_state_guard():
            if person_visible:
                return True, person_visible_unknown
            now = self._observed_ts(observed_at)
            age_s = self._age_seconds(observed_at=now, last_seen_at=self._state("_last_person_seen_at"))
            if age_s is not None and (
                age_s < 0
                or age_s <= self._config_float("person_recently_visible_window_s", 0.0)
            ):
                return True, person_visible_unknown
            return False, person_visible_unknown

    def _resolve_person_count(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        person_count: object,
    ) -> tuple[int, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("person_count")
            last_seen_at = self._state("_last_person_count_at")
            current_value = self._state("_last_person_count", 0)

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=0,
                        observed_at=now,
                        hold_s=hold_s,
                    )
                if person_visible:
                    count = max(1, self._coerce_non_negative_int(person_count, default=1))
                else:
                    count = 0
                self._last_person_count = count
                self._last_person_count_at = now
                return count, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=0,
                observed_at=now,
                hold_s=hold_s,
            )

    def _resolve_primary_person_zone(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        primary_person_zone: object,
    ) -> tuple[SocialPersonZone, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("primary_person_zone")
            last_seen_at = self._state("_last_primary_person_zone_at")
            current_value = self._state("_last_primary_person_zone", SocialPersonZone.UNKNOWN)

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=SocialPersonZone.UNKNOWN,
                        observed_at=now,
                        hold_s=hold_s,
                    )
                zone = self._coerce_person_zone_safe(primary_person_zone) if person_visible else SocialPersonZone.UNKNOWN
                self._last_primary_person_zone = zone
                self._last_primary_person_zone_at = now
                return zone, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=SocialPersonZone.UNKNOWN,
                observed_at=now,
                hold_s=hold_s,
            )

    def _resolve_primary_person_box(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        primary_person_box: object,
    ) -> tuple[SocialSpatialBox | None, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("primary_person_box")
            last_seen_at = self._state("_last_primary_person_box_at")
            current_value = self._state("_last_primary_person_box")

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=None,
                        observed_at=now,
                        hold_s=hold_s,
                    )
                box = self._coerce_spatial_box_safe(primary_person_box) if person_visible else None
                self._last_primary_person_box = box
                self._last_primary_person_box_at = now
                return box, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=None,
                observed_at=now,
                hold_s=hold_s,
            )

    def _resolve_visible_persons(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        visible_persons: object,
        primary_person_box: SocialSpatialBox | None,
        primary_person_zone: SocialPersonZone,
    ) -> tuple[tuple[SocialVisiblePerson, ...], bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("visible_persons")
            last_seen_at = self._state("_last_visible_persons_at")
            current_value = self._state("_last_visible_persons", ())

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=(),
                        observed_at=now,
                        hold_s=hold_s,
                    )

                if not person_visible:
                    people: tuple[SocialVisiblePerson, ...] = ()
                else:
                    people = self._coerce_visible_persons_safe(visible_persons)
                    if not people and primary_person_box is not None:
                        people = (
                            SocialVisiblePerson(
                                box=primary_person_box,
                                zone=primary_person_zone,
                                confidence=1.0,
                            ),
                        )
                    people = self._canonicalize_visible_persons(
                        people=people,
                        primary_person_box=primary_person_box,
                    )

                self._last_visible_persons = people
                self._last_visible_persons_at = now
                return people, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=(),
                observed_at=now,
                hold_s=hold_s,
            )

    def _resolve_primary_person_center(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_center: object,
        box: SocialSpatialBox | None,
        axis: str,
    ) -> tuple[float | None, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            if axis == "x":
                cache_attr = "_last_primary_person_center_x"
                cache_seen_attr = "_last_primary_person_center_x_at"
            else:
                cache_attr = "_last_primary_person_center_y"
                cache_seen_attr = "_last_primary_person_center_y_at"

            hold_s = self._secondary_hold_window_s("primary_person_center")
            last_seen_at = self._state(cache_seen_attr)
            current_value = self._state(cache_attr)

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=None,
                        observed_at=now,
                        hold_s=hold_s,
                    )

                if not person_visible:
                    value = None
                elif box is not None:
                    value = box.center_x if axis == "x" else box.center_y
                else:
                    value = self._coerce_optional_ratio_safe(raw_center)

                value = self._smooth_primary_person_center(
                    axis=axis,
                    observed_at=now,
                    value=value,
                )
                setattr(self, cache_attr, value)
                setattr(self, cache_seen_attr, now)
                return value, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=None,
                observed_at=now,
                hold_s=hold_s,
            )

    def _smooth_primary_person_center(
        self: Any,
        *,
        axis: str,
        observed_at: float,
        value: float | None,
    ) -> float | None:
        """Dampen primary-person center jitter without making target switches sluggish."""
        if value is None:
            return None

        if axis == "x":
            previous_value = self._state("_last_primary_person_center_x")
            previous_at = self._state("_last_primary_person_center_x_at")
        else:
            previous_value = self._state("_last_primary_person_center_y")
            previous_at = self._state("_last_primary_person_center_y_at")

        if previous_value is None or previous_at is None:
            return value

        age_s = self._age_seconds(
            observed_at=self._observed_ts(observed_at),
            last_seen_at=previous_at,
        )
        if age_s is None or age_s < 0:
            return value

        if age_s > self._config_float("primary_person_center_smoothing_window_s", 0.0):
            return value

        delta = value - previous_value
        jump_reset_delta = self._config_float("primary_person_center_jump_reset_delta", 0.35)
        if abs(delta) >= max(0.0, jump_reset_delta):
            return value

        deadband = max(0.0, self._config_float("primary_person_center_deadband", 0.0))
        if abs(delta) <= deadband:
            return previous_value

        alpha = self._config_ratio("primary_person_center_smoothing_alpha", 1.0)
        smoothed = previous_value + (delta * alpha)
        return max(0.0, min(1.0, smoothed))

    def _resolve_visual_attention_score(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_score: object,
    ) -> tuple[float | None, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("visual_attention_score")
            last_seen_at = self._state("_last_visual_attention_score_at")
            current_value = self._state("_last_visual_attention_score")

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=None,
                        observed_at=now,
                        hold_s=hold_s,
                    )
                value = self._coerce_optional_ratio_safe(raw_score) if person_visible else None
                self._last_visual_attention_score = value
                self._last_visual_attention_score_at = now
                return value, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=None,
                observed_at=now,
                hold_s=hold_s,
            )

    def _resolve_body_pose(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_pose: object,
    ) -> tuple[SocialBodyPose, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("body_pose")
            last_seen_at = self._state("_last_body_pose_at")
            current_value = self._state("_last_body_pose", SocialBodyPose.UNKNOWN)

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=SocialBodyPose.UNKNOWN,
                        observed_at=now,
                        hold_s=hold_s,
                    )

                pose = self._coerce_body_pose_safe(raw_pose) if person_visible else SocialBodyPose.UNKNOWN
                if pose != current_value:
                    self._body_state_changed_at = now
                    self._body_state_changed_seen_at = now
                self._last_body_pose = pose
                self._last_body_pose_at = now
                return pose, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=SocialBodyPose.UNKNOWN,
                observed_at=now,
                hold_s=hold_s,
            )

    def _resolve_pose_confidence(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_confidence: object,
    ) -> tuple[float | None, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("pose_confidence")
            last_seen_at = self._state("_last_pose_confidence_at")
            current_value = self._state("_last_pose_confidence")

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=None,
                        observed_at=now,
                        hold_s=hold_s,
                    )
                value = self._coerce_optional_ratio_safe(raw_confidence) if person_visible else None
                self._last_pose_confidence = value
                self._last_pose_confidence_at = now
                return value, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=None,
                observed_at=now,
                hold_s=hold_s,
            )

    def _resolve_body_state_changed_at(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
    ) -> tuple[float | None, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            value = self._state("_body_state_changed_at")
            last_seen_at = self._state("_body_state_changed_seen_at")
            if inspected:
                return value, False
            return self._hold_secondary(
                value=value,
                last_seen_at=last_seen_at,
                fallback=None,
                observed_at=now,
                hold_s=self._secondary_hold_window_s("transition_timestamp"),
            )

    def _resolve_motion_state(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_state: object,
    ) -> tuple[SocialMotionState, bool, bool]:
        """Resolve one held coarse-motion state and whether it changed materially."""
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("motion_state")
            last_seen_at = self._state("_last_motion_state_at")
            current_value = self._state("_last_motion_state", SocialMotionState.UNKNOWN)

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    state, unknown = self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=SocialMotionState.UNKNOWN,
                        observed_at=now,
                        hold_s=hold_s,
                    )
                    return state, unknown, False

                state = self._coerce_motion_state_safe(raw_state) if person_visible else SocialMotionState.UNKNOWN
                rising = False
                if state != current_value:
                    self._motion_state_changed_at = now
                    self._motion_state_changed_seen_at = now
                    if state not in {SocialMotionState.UNKNOWN, SocialMotionState.STILL}:
                        last_motion_emitted_at = self._state("_last_motion_emitted_at")
                        age_since_emit_s = self._age_seconds(observed_at=now, last_seen_at=last_motion_emitted_at)
                        if age_since_emit_s is None or age_since_emit_s < 0:
                            age_since_emit_s = None
                        if (
                            age_since_emit_s is None
                            or age_since_emit_s >= self._config_float("motion_event_cooldown_s", 0.0)
                        ):
                            self._last_motion_emitted_at = now
                            rising = True
                self._last_motion_state = state
                self._last_motion_state_at = now
                return state, False, rising

            state, unknown = self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=SocialMotionState.UNKNOWN,
                observed_at=now,
                hold_s=hold_s,
            )
            return state, unknown, False

    def _resolve_motion_confidence(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_confidence: object,
    ) -> tuple[float | None, bool]:
        """Resolve one held motion confidence score."""
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("motion_confidence")
            last_seen_at = self._state("_last_motion_confidence_at")
            current_value = self._state("_last_motion_confidence")

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=None,
                        observed_at=now,
                        hold_s=hold_s,
                    )
                value = self._coerce_optional_ratio_safe(raw_confidence) if person_visible else None
                self._last_motion_confidence = value
                self._last_motion_confidence_at = now
                return value, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=None,
                observed_at=now,
                hold_s=hold_s,
            )

    def _resolve_motion_state_changed_at(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
    ) -> tuple[float | None, bool]:
        """Resolve the last timestamp where the coarse motion state changed."""
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            value = self._state("_motion_state_changed_at")
            last_seen_at = self._state("_motion_state_changed_seen_at")
            if inspected:
                return value, False
            return self._hold_secondary(
                value=value,
                last_seen_at=last_seen_at,
                fallback=None,
                observed_at=now,
                hold_s=self._secondary_hold_window_s("transition_timestamp"),
            )

    def _resolve_smiling(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        smiling: bool,
    ) -> tuple[bool, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("smiling")
            last_seen_at = self._state("_last_smiling_at")
            current_value = self._state("_last_smiling", False)

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=False,
                        observed_at=now,
                        hold_s=hold_s,
                    )
                self._last_smiling = bool(person_visible and self._coerce_bool_like(smiling))
                self._last_smiling_at = now
                return self._last_smiling, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=False,
                observed_at=now,
                hold_s=hold_s,
            )

    def _resolve_showing_started_at(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
        showing_rising: bool,
    ) -> None:
        with self._presence_state_guard():
            if not inspected or not showing_rising:
                return
            now = self._observed_ts(observed_at)
            if not self._accept_global_update(observed_at=now, cache_attr="_last_showing_observed_at"):
                return
            self._showing_intent_started_at = now
            self._showing_intent_started_seen_at = now

    def _resolve_transition_timestamp(
        self: Any,
        *,
        observed_at: float,
        value: float | None,
        last_seen_at: float | None,
    ) -> tuple[float | None, bool]:
        with self._presence_state_guard():
            return self._hold_secondary(
                value=value,
                last_seen_at=last_seen_at,
                fallback=None,
                observed_at=self._observed_ts(observed_at),
                hold_s=self._secondary_hold_window_s("transition_timestamp"),
            )

    def _resolve_secondary_bool(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
        raw_value: object,
        cache_attr: str,
        cache_seen_attr: str,
    ) -> tuple[bool, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("secondary_bool")
            last_seen_at = self._state(cache_seen_attr)
            current_value = self._state(cache_attr, False)

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=False,
                        observed_at=now,
                        hold_s=hold_s,
                    )
                value = self._coerce_bool_like(raw_value)
                setattr(self, cache_attr, value)
                setattr(self, cache_seen_attr, now)
                return value, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=False,
                observed_at=now,
                hold_s=hold_s,
            )

    def _resolve_secondary_text(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
        raw_value: object,
        cache_attr: str,
        cache_seen_attr: str,
    ) -> tuple[str | None, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("secondary_text")
            last_seen_at = self._state(cache_seen_attr)
            current_value = self._state(cache_attr)

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=None,
                        observed_at=now,
                        hold_s=hold_s,
                    )
                value = coerce_optional_text(raw_value)
                setattr(self, cache_attr, value)
                setattr(self, cache_seen_attr, now)
                return value, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=None,
                observed_at=now,
                hold_s=hold_s,
            )

    def _resolve_secondary_timestamp(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
        raw_value: object,
        cache_attr: str,
        cache_seen_attr: str,
    ) -> tuple[float | None, bool]:
        with self._presence_state_guard():
            now = self._observed_ts(observed_at)
            hold_s = self._secondary_hold_window_s("secondary_timestamp")
            last_seen_at = self._state(cache_seen_attr)
            current_value = self._state(cache_attr)

            if inspected:
                if self._is_out_of_order(observed_at=now, last_seen_at=last_seen_at):
                    return self._hold_secondary(
                        value=current_value,
                        last_seen_at=last_seen_at,
                        fallback=None,
                        observed_at=now,
                        hold_s=hold_s,
                    )
                value = self._coerce_optional_timestamp_safe(raw_value)
                setattr(self, cache_attr, value)
                setattr(self, cache_seen_attr, now)
                return value, False

            return self._hold_secondary(
                value=current_value,
                last_seen_at=last_seen_at,
                fallback=None,
                observed_at=now,
                hold_s=hold_s,
            )

    def _hold_secondary(
        self: Any,
        *,
        value: Any,
        last_seen_at: float | None,
        fallback: Any,
        observed_at: float,
        hold_s: float | None = None,
    ) -> tuple[Any, bool]:
        ttl_s = self._secondary_hold_window_s("secondary") if hold_s is None else max(0.0, hold_s)
        age_s = self._age_seconds(observed_at=self._observed_ts(observed_at), last_seen_at=last_seen_at)
        if age_s is None:
            return fallback, True
        if age_s < 0:
            return value, True
        if age_s <= ttl_s:
            return value, True
        return fallback, True