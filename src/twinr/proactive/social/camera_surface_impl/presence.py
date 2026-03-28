"""Presence, health, and secondary-field resolution for the camera surface."""

from __future__ import annotations

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


class ProactiveCameraPresenceMixin:
    """Resolve presence, health, motion, and held secondary camera fields."""

    def _resolve_person_returned(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
        person_visible: bool,
        person_visible_rising: bool,
    ) -> bool:
        now = coerce_timestamp(observed_at)
        person_returned = False
        if inspected:
            if person_visible:
                if (
                    person_visible_rising
                    and self._has_seen_person
                    and self._absence_started_at is not None
                    and (now - self._absence_started_at) >= self.config.person_returned_absence_s
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
                if self._last_authoritative_person_visible or self._absence_started_at is None:
                    self._absence_started_at = now
                if self._last_authoritative_person_visible:
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
        if person_visible:
            return True, person_visible_unknown
        if (
            self._last_person_seen_at is not None
            and (observed_at - self._last_person_seen_at) <= self.config.person_recently_visible_window_s
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
        if inspected:
            if person_visible:
                count = max(1, self._coerce_non_negative_int(person_count, default=1))
            else:
                count = 0
            self._last_person_count = count
            self._last_person_count_at = coerce_timestamp(observed_at)
            return count, False
        return self._hold_secondary(
            value=self._last_person_count,
            last_seen_at=self._last_person_count_at,
            fallback=0,
            observed_at=observed_at,
        )

    def _resolve_primary_person_zone(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        primary_person_zone: object,
    ) -> tuple[SocialPersonZone, bool]:
        if inspected:
            zone = coerce_person_zone(primary_person_zone) if person_visible else SocialPersonZone.UNKNOWN
            self._last_primary_person_zone = zone
            self._last_primary_person_zone_at = coerce_timestamp(observed_at)
            return zone, False
        return self._hold_secondary(
            value=self._last_primary_person_zone,
            last_seen_at=self._last_primary_person_zone_at,
            fallback=SocialPersonZone.UNKNOWN,
            observed_at=observed_at,
        )

    def _resolve_primary_person_box(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        primary_person_box: object,
    ) -> tuple[SocialSpatialBox | None, bool]:
        if inspected:
            box = coerce_spatial_box(primary_person_box) if person_visible else None
            self._last_primary_person_box = box
            self._last_primary_person_box_at = coerce_timestamp(observed_at)
            return box, False
        return self._hold_secondary(
            value=self._last_primary_person_box,
            last_seen_at=self._last_primary_person_box_at,
            fallback=None,
            observed_at=observed_at,
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
        if inspected:
            if not person_visible:
                people: tuple[SocialVisiblePerson, ...] = ()
            else:
                people = coerce_visible_persons(visible_persons)
                if not people and primary_person_box is not None:
                    people = (
                        SocialVisiblePerson(
                            box=primary_person_box,
                            zone=primary_person_zone,
                            confidence=1.0,
                        ),
                    )
            self._last_visible_persons = people
            self._last_visible_persons_at = coerce_timestamp(observed_at)
            return people, False
        return self._hold_secondary(
            value=self._last_visible_persons,
            last_seen_at=self._last_visible_persons_at,
            fallback=(),
            observed_at=observed_at,
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
        if inspected:
            if not person_visible:
                value = None
            elif box is not None:
                value = box.center_x if axis == "x" else box.center_y
            else:
                value = coerce_optional_ratio(raw_center)
            value = self._smooth_primary_person_center(
                axis=axis,
                observed_at=observed_at,
                value=value,
            )
            if axis == "x":
                self._last_primary_person_center_x = value
                self._last_primary_person_center_x_at = coerce_timestamp(observed_at)
            else:
                self._last_primary_person_center_y = value
                self._last_primary_person_center_y_at = coerce_timestamp(observed_at)
            return value, False

        if axis == "x":
            return self._hold_secondary(
                value=self._last_primary_person_center_x,
                last_seen_at=self._last_primary_person_center_x_at,
                fallback=None,
                observed_at=observed_at,
            )
        return self._hold_secondary(
            value=self._last_primary_person_center_y,
            last_seen_at=self._last_primary_person_center_y_at,
            fallback=None,
            observed_at=observed_at,
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
        previous_value = (
            self._last_primary_person_center_x if axis == "x" else self._last_primary_person_center_y
        )
        previous_at = (
            self._last_primary_person_center_x_at if axis == "x" else self._last_primary_person_center_y_at
        )
        if previous_value is None or previous_at is None:
            return value
        previous_seen_at = coerce_timestamp(previous_at)
        current_seen_at = coerce_timestamp(observed_at)
        if previous_seen_at is None or current_seen_at is None:
            return value
        if (current_seen_at - previous_seen_at) > self.config.primary_person_center_smoothing_window_s:
            return value
        delta = value - previous_value
        if abs(delta) <= self.config.primary_person_center_deadband:
            return previous_value
        alpha = self.config.primary_person_center_smoothing_alpha
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
        if inspected:
            value = coerce_optional_ratio(raw_score) if person_visible else None
            self._last_visual_attention_score = value
            self._last_visual_attention_score_at = coerce_timestamp(observed_at)
            return value, False
        return self._hold_secondary(
            value=self._last_visual_attention_score,
            last_seen_at=self._last_visual_attention_score_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_body_pose(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_pose: object,
    ) -> tuple[SocialBodyPose, bool]:
        if inspected:
            pose = coerce_body_pose(raw_pose) if person_visible else SocialBodyPose.UNKNOWN
            if pose != self._last_body_pose:
                self._body_state_changed_at = coerce_timestamp(observed_at)
                self._body_state_changed_seen_at = coerce_timestamp(observed_at)
            self._last_body_pose = pose
            self._last_body_pose_at = coerce_timestamp(observed_at)
            return pose, False
        return self._hold_secondary(
            value=self._last_body_pose,
            last_seen_at=self._last_body_pose_at,
            fallback=SocialBodyPose.UNKNOWN,
            observed_at=observed_at,
        )

    def _resolve_pose_confidence(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_confidence: object,
    ) -> tuple[float | None, bool]:
        if inspected:
            value = coerce_optional_ratio(raw_confidence) if person_visible else None
            self._last_pose_confidence = value
            self._last_pose_confidence_at = coerce_timestamp(observed_at)
            return value, False
        return self._hold_secondary(
            value=self._last_pose_confidence,
            last_seen_at=self._last_pose_confidence_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_body_state_changed_at(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
    ) -> tuple[float | None, bool]:
        if inspected:
            return self._body_state_changed_at, False
        return self._hold_secondary(
            value=self._body_state_changed_at,
            last_seen_at=self._body_state_changed_seen_at,
            fallback=None,
            observed_at=observed_at,
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

        now = coerce_timestamp(observed_at)
        if inspected:
            state = coerce_motion_state(raw_state) if person_visible else SocialMotionState.UNKNOWN
            rising = False
            if state != self._last_motion_state:
                self._motion_state_changed_at = now
                self._motion_state_changed_seen_at = now
                if state not in {SocialMotionState.UNKNOWN, SocialMotionState.STILL}:
                    if (
                        self._last_motion_emitted_at is None
                        or (now - self._last_motion_emitted_at) >= self.config.motion_event_cooldown_s
                    ):
                        self._last_motion_emitted_at = now
                        rising = True
            self._last_motion_state = state
            self._last_motion_state_at = now
            return state, False, rising
        state, unknown = self._hold_secondary(
            value=self._last_motion_state,
            last_seen_at=self._last_motion_state_at,
            fallback=SocialMotionState.UNKNOWN,
            observed_at=observed_at,
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

        if inspected:
            value = coerce_optional_ratio(raw_confidence) if person_visible else None
            self._last_motion_confidence = value
            self._last_motion_confidence_at = coerce_timestamp(observed_at)
            return value, False
        return self._hold_secondary(
            value=self._last_motion_confidence,
            last_seen_at=self._last_motion_confidence_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_motion_state_changed_at(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
    ) -> tuple[float | None, bool]:
        """Resolve the last timestamp where the coarse motion state changed."""

        if inspected:
            return self._motion_state_changed_at, False
        return self._hold_secondary(
            value=self._motion_state_changed_at,
            last_seen_at=self._motion_state_changed_seen_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_smiling(
        self: Any,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        smiling: bool,
    ) -> tuple[bool, bool]:
        if inspected:
            self._last_smiling = bool(person_visible and smiling)
            self._last_smiling_at = coerce_timestamp(observed_at)
            return self._last_smiling, False
        return self._hold_secondary(
            value=self._last_smiling,
            last_seen_at=self._last_smiling_at,
            fallback=False,
            observed_at=observed_at,
        )

    def _resolve_showing_started_at(
        self: Any,
        *,
        inspected: bool,
        observed_at: float,
        showing_rising: bool,
    ) -> None:
        if inspected and showing_rising:
            self._showing_intent_started_at = coerce_timestamp(observed_at)
            self._showing_intent_started_seen_at = coerce_timestamp(observed_at)

    def _resolve_transition_timestamp(
        self: Any,
        *,
        observed_at: float,
        value: float | None,
        last_seen_at: float | None,
    ) -> tuple[float | None, bool]:
        return self._hold_secondary(
            value=value,
            last_seen_at=last_seen_at,
            fallback=None,
            observed_at=observed_at,
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
        if inspected:
            value = bool(raw_value)
            setattr(self, cache_attr, value)
            setattr(self, cache_seen_attr, coerce_timestamp(observed_at))
            return value, False
        return self._hold_secondary(
            value=getattr(self, cache_attr),
            last_seen_at=getattr(self, cache_seen_attr),
            fallback=False,
            observed_at=observed_at,
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
        if inspected:
            value = coerce_optional_text(raw_value)
            setattr(self, cache_attr, value)
            setattr(self, cache_seen_attr, coerce_timestamp(observed_at))
            return value, False
        return self._hold_secondary(
            value=getattr(self, cache_attr),
            last_seen_at=getattr(self, cache_seen_attr),
            fallback=None,
            observed_at=observed_at,
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
        if inspected:
            value = coerce_optional_timestamp(raw_value)
            setattr(self, cache_attr, value)
            setattr(self, cache_seen_attr, coerce_timestamp(observed_at))
            return value, False
        return self._hold_secondary(
            value=getattr(self, cache_attr),
            last_seen_at=getattr(self, cache_seen_attr),
            fallback=None,
            observed_at=observed_at,
        )

    def _hold_secondary(
        self: Any,
        *,
        value: Any,
        last_seen_at: float | None,
        fallback: Any,
        observed_at: float,
    ) -> tuple[Any, bool]:
        if last_seen_at is not None and (coerce_timestamp(observed_at) - last_seen_at) <= self.config.secondary_unknown_hold_s:
            return value, True
        return fallback, True
