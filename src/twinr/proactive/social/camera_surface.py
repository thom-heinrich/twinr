"""Stabilize camera observations into automation-friendly facts and events.

This module turns raw ``SocialVisionObservation`` ticks into a bounded camera
snapshot and a small rising-edge event surface. It owns debounce, cooldown, and
unknown-state handling for camera-derived automation signals so runtime
orchestrators can stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .engine import SocialBodyPose, SocialPersonZone, SocialVisionObservation

_PERSON_VISIBLE_EVENT = "camera.person_visible"
_HAND_NEAR_EVENT = "camera.hand_or_object_near_camera"
_PERSON_RETURNED_EVENT = "camera.person_returned"


@dataclass(frozen=True, slots=True)
class ProactiveCameraSurfaceConfig:
    """Store stabilization rules for camera-derived automation signals.

    The defaults intentionally follow the proactive capture cadence instead of
    assuming video-rate input. Camera positive transitions remain responsive,
    while negative transitions require confirmation so one skipped inspection
    does not immediately clear presence.
    """

    person_visible_on_samples: int = 1
    person_visible_off_samples: int = 2
    person_visible_unknown_hold_s: float = 9.0
    person_visible_event_cooldown_s: float = 9.0
    person_returned_absence_s: float = 20.0 * 60.0
    looking_toward_device_on_samples: int = 1
    looking_toward_device_off_samples: int = 2
    looking_toward_device_unknown_hold_s: float = 9.0
    hand_or_object_near_camera_on_samples: int = 1
    hand_or_object_near_camera_off_samples: int = 2
    hand_or_object_near_camera_unknown_hold_s: float = 9.0
    hand_or_object_near_camera_event_cooldown_s: float = 9.0
    secondary_unknown_hold_s: float = 9.0

    def __post_init__(self) -> None:
        """Reject malformed debounce and cooldown configuration eagerly."""

        _require_positive_int(self.person_visible_on_samples, field_name="person_visible_on_samples")
        _require_positive_int(self.person_visible_off_samples, field_name="person_visible_off_samples")
        _require_non_negative_float(
            self.person_visible_unknown_hold_s,
            field_name="person_visible_unknown_hold_s",
        )
        _require_non_negative_float(
            self.person_visible_event_cooldown_s,
            field_name="person_visible_event_cooldown_s",
        )
        _require_non_negative_float(self.person_returned_absence_s, field_name="person_returned_absence_s")
        _require_positive_int(
            self.looking_toward_device_on_samples,
            field_name="looking_toward_device_on_samples",
        )
        _require_positive_int(
            self.looking_toward_device_off_samples,
            field_name="looking_toward_device_off_samples",
        )
        _require_non_negative_float(
            self.looking_toward_device_unknown_hold_s,
            field_name="looking_toward_device_unknown_hold_s",
        )
        _require_positive_int(
            self.hand_or_object_near_camera_on_samples,
            field_name="hand_or_object_near_camera_on_samples",
        )
        _require_positive_int(
            self.hand_or_object_near_camera_off_samples,
            field_name="hand_or_object_near_camera_off_samples",
        )
        _require_non_negative_float(
            self.hand_or_object_near_camera_unknown_hold_s,
            field_name="hand_or_object_near_camera_unknown_hold_s",
        )
        _require_non_negative_float(
            self.hand_or_object_near_camera_event_cooldown_s,
            field_name="hand_or_object_near_camera_event_cooldown_s",
        )
        _require_non_negative_float(self.secondary_unknown_hold_s, field_name="secondary_unknown_hold_s")

    @classmethod
    def from_config(cls, config: object) -> ProactiveCameraSurfaceConfig:
        """Build one cadence-aware camera surface config from Twinr config."""

        interval_s = _coerce_positive_float(
            getattr(config, "proactive_capture_interval_s", 6.0),
            default=6.0,
        )
        unknown_hold_s = max(interval_s + 1.0, interval_s * 1.5)
        cooldown_s = max(interval_s, interval_s * 1.5)
        return cls(
            person_visible_unknown_hold_s=unknown_hold_s,
            person_visible_event_cooldown_s=cooldown_s,
            person_returned_absence_s=_coerce_positive_float(
                getattr(config, "proactive_person_returned_absence_s", 20.0 * 60.0),
                default=20.0 * 60.0,
            ),
            looking_toward_device_unknown_hold_s=unknown_hold_s,
            hand_or_object_near_camera_unknown_hold_s=unknown_hold_s,
            hand_or_object_near_camera_event_cooldown_s=cooldown_s,
            secondary_unknown_hold_s=unknown_hold_s,
        )


@dataclass(frozen=True, slots=True)
class ProactiveCameraSnapshot:
    """Capture one stabilized camera snapshot for synchronous consumers."""

    person_visible: bool
    person_visible_for_s: float
    person_visible_unknown: bool
    person_count: int
    person_count_unknown: bool
    person_returned_after_absence: bool
    primary_person_zone: SocialPersonZone
    primary_person_zone_unknown: bool
    looking_toward_device: bool
    looking_toward_device_unknown: bool
    body_pose: SocialBodyPose
    body_pose_unknown: bool
    smiling: bool
    smiling_unknown: bool
    hand_or_object_near_camera: bool
    hand_or_object_near_camera_for_s: float
    hand_or_object_near_camera_unknown: bool

    @property
    def unknown(self) -> bool:
        """Return whether any camera field is currently running on unknown data."""

        return any(
            (
                self.person_visible_unknown,
                self.person_count_unknown,
                self.primary_person_zone_unknown,
                self.looking_toward_device_unknown,
                self.body_pose_unknown,
                self.smiling_unknown,
                self.hand_or_object_near_camera_unknown,
            )
        )

    def to_automation_facts(self) -> dict[str, object]:
        """Render the snapshot into the runtime automation fact contract."""

        return {
            "person_visible": self.person_visible,
            "person_visible_for_s": round(self.person_visible_for_s, 3),
            "person_visible_unknown": self.person_visible_unknown,
            "person_count": self.person_count,
            "count_persons": self.person_count,
            "person_count_unknown": self.person_count_unknown,
            "person_returned_after_absence": self.person_returned_after_absence,
            "primary_person_zone": self.primary_person_zone.value,
            "primary_person_zone_unknown": self.primary_person_zone_unknown,
            "looking_toward_device": self.looking_toward_device,
            "looking_toward_device_unknown": self.looking_toward_device_unknown,
            "body_pose": self.body_pose.value,
            "body_pose_unknown": self.body_pose_unknown,
            "smiling": self.smiling,
            "smiling_unknown": self.smiling_unknown,
            "hand_or_object_near_camera": self.hand_or_object_near_camera,
            "hand_or_object_near_camera_for_s": round(self.hand_or_object_near_camera_for_s, 3),
            "hand_or_object_near_camera_unknown": self.hand_or_object_near_camera_unknown,
            "unknown": self.unknown,
        }


@dataclass(frozen=True, slots=True)
class ProactiveCameraSurfaceUpdate:
    """Combine one stabilized snapshot with any newly opened camera events."""

    snapshot: ProactiveCameraSnapshot
    event_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _BooleanSignalView:
    """Describe the latest debounced state of one boolean signal."""

    value: bool
    unknown: bool
    active_for_s: float
    rising_edge: bool


class _DebouncedBooleanSignal:
    """Track one cadence-aware boolean signal with unknown-state handling."""

    def __init__(
        self,
        *,
        on_samples: int,
        off_samples: int,
        unknown_hold_s: float,
        event_cooldown_s: float = 0.0,
    ) -> None:
        """Initialize one boolean stabilizer.

        Args:
            on_samples: Consecutive positive samples required to activate.
            off_samples: Consecutive negative samples required to clear.
            unknown_hold_s: How long to preserve the last stable value through
                missing/unknown observations before failing closed.
            event_cooldown_s: Minimum time between rising-edge events.
        """

        self.on_samples = _require_positive_int(on_samples, field_name="on_samples")
        self.off_samples = _require_positive_int(off_samples, field_name="off_samples")
        self.unknown_hold_s = _require_non_negative_float(unknown_hold_s, field_name="unknown_hold_s")
        self.event_cooldown_s = _require_non_negative_float(
            event_cooldown_s,
            field_name="event_cooldown_s",
        )
        self._stable_value = False
        self._active_since: float | None = None
        self._last_concrete_at: float | None = None
        self._last_observed_at: float | None = None
        self._pending_value: bool | None = None
        self._pending_count = 0
        self._pending_started_at: float | None = None
        self._last_rising_event_at: float | None = None

    def observe(self, value: bool | None, *, observed_at: float) -> _BooleanSignalView:
        """Observe one new value and return the debounced signal view."""

        now = _coerce_timestamp(observed_at, previous=self._last_observed_at)
        self._last_observed_at = now
        rising_edge = False

        if value is None:
            self._clear_pending()
            if (
                self._last_concrete_at is not None
                and self._stable_value
                and (now - self._last_concrete_at) > self.unknown_hold_s
            ):
                self._stable_value = False
                self._active_since = None
            return _BooleanSignalView(
                value=self._stable_value,
                unknown=True,
                active_for_s=_duration_since(self._active_since, now),
                rising_edge=False,
            )

        self._last_concrete_at = now
        current_value = bool(value)
        if current_value == self._stable_value:
            self._clear_pending()
        else:
            self._advance_pending(current_value, now=now)
            threshold = self.on_samples if current_value else self.off_samples
            if self._pending_count >= threshold:
                self._stable_value = current_value
                if current_value:
                    self._active_since = self._pending_started_at or now
                    rising_edge = self._allow_rising_edge(now)
                else:
                    self._active_since = None
                self._clear_pending()

        if self._stable_value and self._active_since is None:
            self._active_since = now
        return _BooleanSignalView(
            value=self._stable_value,
            unknown=False,
            active_for_s=_duration_since(self._active_since, now),
            rising_edge=rising_edge,
        )

    def _advance_pending(self, value: bool, *, now: float) -> None:
        """Advance the pending transition candidate for one new observation."""

        if self._pending_value is value:
            self._pending_count += 1
            return
        self._pending_value = value
        self._pending_count = 1
        self._pending_started_at = now

    def _clear_pending(self) -> None:
        """Clear any partially-confirmed transition candidate."""

        self._pending_value = None
        self._pending_count = 0
        self._pending_started_at = None

    def _allow_rising_edge(self, now: float) -> bool:
        """Return whether a rising-edge event may fire at ``now``."""

        if self._last_rising_event_at is not None and (now - self._last_rising_event_at) < self.event_cooldown_s:
            return False
        self._last_rising_event_at = now
        return True


class ProactiveCameraSurface:
    """Build a stabilized camera snapshot and bounded event surface."""

    @classmethod
    def from_config(cls, config: object) -> ProactiveCameraSurface:
        """Build one camera surface directly from Twinr config."""

        return cls(config=ProactiveCameraSurfaceConfig.from_config(config))

    def __init__(self, *, config: ProactiveCameraSurfaceConfig | None = None) -> None:
        """Initialize one camera surface from an optional config object."""

        self.config = config or ProactiveCameraSurfaceConfig()
        self._person_visible = _DebouncedBooleanSignal(
            on_samples=self.config.person_visible_on_samples,
            off_samples=self.config.person_visible_off_samples,
            unknown_hold_s=self.config.person_visible_unknown_hold_s,
            event_cooldown_s=self.config.person_visible_event_cooldown_s,
        )
        self._looking_toward_device = _DebouncedBooleanSignal(
            on_samples=self.config.looking_toward_device_on_samples,
            off_samples=self.config.looking_toward_device_off_samples,
            unknown_hold_s=self.config.looking_toward_device_unknown_hold_s,
        )
        self._hand_near_camera = _DebouncedBooleanSignal(
            on_samples=self.config.hand_or_object_near_camera_on_samples,
            off_samples=self.config.hand_or_object_near_camera_off_samples,
            unknown_hold_s=self.config.hand_or_object_near_camera_unknown_hold_s,
            event_cooldown_s=self.config.hand_or_object_near_camera_event_cooldown_s,
        )
        self._last_body_pose = SocialBodyPose.UNKNOWN
        self._last_body_pose_at: float | None = None
        self._last_person_count = 0
        self._last_person_count_at: float | None = None
        self._last_primary_person_zone = SocialPersonZone.UNKNOWN
        self._last_primary_person_zone_at: float | None = None
        self._last_smiling = False
        self._last_smiling_at: float | None = None
        self._has_seen_person = False
        self._last_authoritative_person_visible = False
        self._absence_started_at: float | None = None

    def observe(
        self,
        *,
        inspected: bool,
        observed_at: float,
        observation: SocialVisionObservation,
    ) -> ProactiveCameraSurfaceUpdate:
        """Consume one raw observation and return the stabilized camera update."""

        person_sample = self._person_visible.observe(
            observation.person_visible if inspected else None,
            observed_at=observed_at,
        )
        person_returned_after_absence = self._resolve_person_returned(
            inspected=inspected,
            observed_at=observed_at,
            person_visible=person_sample.value,
            person_visible_rising=person_sample.rising_edge,
        )
        person_count, person_count_unknown = self._resolve_person_count(
            inspected=inspected,
            person_visible=person_sample.value,
            observed_at=observed_at,
            person_count=getattr(observation, "person_count", 0),
        )
        primary_person_zone, primary_person_zone_unknown = self._resolve_primary_person_zone(
            inspected=inspected,
            person_visible=person_sample.value,
            observed_at=observed_at,
            primary_person_zone=getattr(observation, "primary_person_zone", SocialPersonZone.UNKNOWN),
        )
        looking_input = None
        hand_input = None
        if inspected:
            looking_input = bool(observation.person_visible and observation.looking_toward_device)
            hand_input = observation.hand_or_object_near_camera
        looking_sample = self._looking_toward_device.observe(looking_input, observed_at=observed_at)
        hand_sample = self._hand_near_camera.observe(hand_input, observed_at=observed_at)
        body_pose, body_pose_unknown = self._resolve_body_pose(
            inspected=inspected,
            person_visible=person_sample.value,
            observed_at=observed_at,
            raw_pose=observation.body_pose,
        )
        smiling, smiling_unknown = self._resolve_smiling(
            inspected=inspected,
            person_visible=person_sample.value,
            observed_at=observed_at,
            smiling=observation.smiling,
        )
        snapshot = ProactiveCameraSnapshot(
            person_visible=person_sample.value,
            person_visible_for_s=person_sample.active_for_s,
            person_visible_unknown=person_sample.unknown,
            person_count=person_count,
            person_count_unknown=person_count_unknown,
            person_returned_after_absence=person_returned_after_absence,
            primary_person_zone=primary_person_zone,
            primary_person_zone_unknown=primary_person_zone_unknown,
            looking_toward_device=looking_sample.value,
            looking_toward_device_unknown=looking_sample.unknown,
            body_pose=body_pose,
            body_pose_unknown=body_pose_unknown,
            smiling=smiling,
            smiling_unknown=smiling_unknown,
            hand_or_object_near_camera=hand_sample.value,
            hand_or_object_near_camera_for_s=hand_sample.active_for_s,
            hand_or_object_near_camera_unknown=hand_sample.unknown,
        )
        event_names: list[str] = []
        if person_sample.rising_edge:
            event_names.append(_PERSON_VISIBLE_EVENT)
        if person_returned_after_absence:
            event_names.append(_PERSON_RETURNED_EVENT)
        if hand_sample.rising_edge:
            event_names.append(_HAND_NEAR_EVENT)
        return ProactiveCameraSurfaceUpdate(snapshot=snapshot, event_names=tuple(event_names))

    def _resolve_body_pose(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_pose: object,
    ) -> tuple[SocialBodyPose, bool]:
        """Return the stabilized coarse body pose and unknown flag."""

        if inspected:
            pose = _coerce_body_pose(raw_pose) if person_visible else SocialBodyPose.UNKNOWN
            self._last_body_pose = pose
            self._last_body_pose_at = _coerce_timestamp(observed_at)
            return pose, False
        return self._hold_secondary(
            value=self._last_body_pose,
            last_seen_at=self._last_body_pose_at,
            fallback=SocialBodyPose.UNKNOWN,
            observed_at=observed_at,
        )

    def _resolve_person_returned(
        self,
        *,
        inspected: bool,
        observed_at: float,
        person_visible: bool,
        person_visible_rising: bool,
    ) -> bool:
        """Return whether the latest visible tick qualifies as a return event."""

        now = _coerce_timestamp(observed_at)
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
                self._absence_started_at = None
            else:
                if self._last_authoritative_person_visible or self._absence_started_at is None:
                    self._absence_started_at = now
                self._last_authoritative_person_visible = False
        return person_returned

    def _resolve_person_count(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        person_count: object,
    ) -> tuple[int, bool]:
        """Return the stabilized visible-person count and unknown flag."""

        if inspected:
            if person_visible:
                count = max(1, _coerce_non_negative_int(person_count, default=1))
            else:
                count = 0
            self._last_person_count = count
            self._last_person_count_at = _coerce_timestamp(observed_at)
            return count, False
        return self._hold_secondary(
            value=self._last_person_count,
            last_seen_at=self._last_person_count_at,
            fallback=0,
            observed_at=observed_at,
        )

    def _resolve_primary_person_zone(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        primary_person_zone: object,
    ) -> tuple[SocialPersonZone, bool]:
        """Return the stabilized primary-person zone and unknown flag."""

        if inspected:
            zone = _coerce_person_zone(primary_person_zone) if person_visible else SocialPersonZone.UNKNOWN
            self._last_primary_person_zone = zone
            self._last_primary_person_zone_at = _coerce_timestamp(observed_at)
            return zone, False
        return self._hold_secondary(
            value=self._last_primary_person_zone,
            last_seen_at=self._last_primary_person_zone_at,
            fallback=SocialPersonZone.UNKNOWN,
            observed_at=observed_at,
        )

    def _resolve_smiling(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        smiling: bool,
    ) -> tuple[bool, bool]:
        """Return the stabilized smiling flag and unknown state."""

        if inspected:
            self._last_smiling = bool(person_visible and smiling)
            self._last_smiling_at = _coerce_timestamp(observed_at)
            return self._last_smiling, False
        return self._hold_secondary(
            value=self._last_smiling,
            last_seen_at=self._last_smiling_at,
            fallback=False,
            observed_at=observed_at,
        )

    def _hold_secondary(
        self,
        *,
        value: Any,
        last_seen_at: float | None,
        fallback: Any,
        observed_at: float,
    ) -> tuple[Any, bool]:
        """Hold one secondary field through brief unknown windows only."""

        if last_seen_at is not None and (_coerce_timestamp(observed_at) - last_seen_at) <= self.config.secondary_unknown_hold_s:
            return value, True
        return fallback, True


def _coerce_timestamp(value: float, *, previous: float | None = None) -> float:
    """Normalize one timestamp into a monotonic finite float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        now = 0.0 if previous is None else previous
        return now
    number = float(value)
    if not math.isfinite(number):
        number = 0.0 if previous is None else previous
    if previous is not None and number < previous:
        return previous
    return number


def _duration_since(since: float | None, now: float) -> float:
    """Return the elapsed duration since ``since`` with a safe lower bound."""

    if since is None:
        return 0.0
    return max(0.0, now - since)


def _coerce_body_pose(value: object) -> SocialBodyPose:
    """Normalize a raw pose token into the coarse ``SocialBodyPose`` enum."""

    if isinstance(value, SocialBodyPose):
        return value
    if value is None:
        return SocialBodyPose.UNKNOWN
    token = str(value).strip().lower()
    for pose in SocialBodyPose:
        if pose.value == token:
            return pose
    return SocialBodyPose.UNKNOWN


def _coerce_person_zone(value: object) -> SocialPersonZone:
    """Normalize a raw zone token into the coarse ``SocialPersonZone`` enum."""

    if isinstance(value, SocialPersonZone):
        return value
    if value is None:
        return SocialPersonZone.UNKNOWN
    token = str(value).strip().lower()
    for zone in SocialPersonZone:
        if zone.value == token:
            return zone
    return SocialPersonZone.UNKNOWN


def _coerce_positive_float(value: object, *, default: float) -> float:
    """Return a finite positive float, falling back to ``default`` when needed."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return default
    return number


def _require_positive_int(value: int, *, field_name: str) -> int:
    """Validate one strictly positive integer configuration value."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return value


def _coerce_non_negative_int(value: object, *, default: int) -> int:
    """Return a non-negative integer, falling back to ``default`` when needed."""

    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if number < 0:
        return default
    return number


def _require_non_negative_float(value: float, *, field_name: str) -> float:
    """Validate one finite non-negative float configuration value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be a finite number >= 0.")
    return number
