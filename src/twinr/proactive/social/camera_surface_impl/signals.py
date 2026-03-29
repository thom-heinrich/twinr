"""Cadence-aware signal trackers for the camera surface."""

from __future__ import annotations

from dataclasses import dataclass

from ..engine import SocialDetectedObject
from .coercion import coerce_timestamp, duration_since
from .validation import require_non_negative_float, require_positive_int


@dataclass(frozen=True, slots=True)
class BooleanSignalView:
    """Describe the latest debounced state of one boolean signal."""

    value: bool
    unknown: bool
    active_for_s: float
    rising_edge: bool


class DebouncedBooleanSignal:
    """Track one cadence-aware boolean signal with unknown-state handling."""

    def __init__(
        self,
        *,
        on_samples: int,
        off_samples: int,
        unknown_hold_s: float,
        event_cooldown_s: float = 0.0,
    ) -> None:
        self.on_samples = require_positive_int(on_samples, field_name="on_samples")
        self.off_samples = require_positive_int(off_samples, field_name="off_samples")
        self.unknown_hold_s = require_non_negative_float(unknown_hold_s, field_name="unknown_hold_s")
        self.event_cooldown_s = require_non_negative_float(event_cooldown_s, field_name="event_cooldown_s")
        self._stable_value = False
        self._active_since: float | None = None
        self._last_concrete_at: float | None = None
        self._last_observed_at: float | None = None
        self._pending_value: bool | None = None
        self._pending_count = 0
        self._pending_started_at: float | None = None
        self._last_rising_event_at: float | None = None

    def observe(self, value: bool | None, *, observed_at: float) -> BooleanSignalView:
        """Observe one new value and return the debounced signal view."""

        now = coerce_timestamp(observed_at, previous=self._last_observed_at)
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
            return BooleanSignalView(
                value=self._stable_value,
                unknown=True,
                active_for_s=duration_since(self._active_since, now),
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
        return BooleanSignalView(
            value=self._stable_value,
            unknown=False,
            active_for_s=duration_since(self._active_since, now),
            rising_edge=rising_edge,
        )

    def observe_authoritative(
        self,
        value: bool | None,
        *,
        observed_at: float,
    ) -> BooleanSignalView:
        """Adopt one already-stabilized upstream value without local debounce."""

        now = coerce_timestamp(observed_at, previous=self._last_observed_at)
        self._last_observed_at = now

        if value is None:
            self._clear_pending()
            if (
                self._last_concrete_at is not None
                and self._stable_value
                and (now - self._last_concrete_at) > self.unknown_hold_s
            ):
                self._stable_value = False
                self._active_since = None
            return BooleanSignalView(
                value=self._stable_value,
                unknown=True,
                active_for_s=duration_since(self._active_since, now),
                rising_edge=False,
            )

        self._last_concrete_at = now
        self._clear_pending()
        current_value = bool(value)
        was_active = self._stable_value
        self._stable_value = current_value
        rising_edge = False
        if current_value:
            if not was_active:
                self._active_since = now
                rising_edge = self._allow_rising_edge(now)
            elif self._active_since is None:
                self._active_since = now
        else:
            self._active_since = None
        return BooleanSignalView(
            value=self._stable_value,
            unknown=False,
            active_for_s=duration_since(self._active_since, now),
            rising_edge=rising_edge,
        )

    def _advance_pending(self, value: bool, *, now: float) -> None:
        if self._pending_value is value:
            self._pending_count += 1
            return
        self._pending_value = value
        self._pending_count = 1
        self._pending_started_at = now

    def _clear_pending(self) -> None:
        self._pending_value = None
        self._pending_count = 0
        self._pending_started_at = None

    def _allow_rising_edge(self, now: float) -> bool:
        if self._last_rising_event_at is not None and (now - self._last_rising_event_at) < self.event_cooldown_s:
            return False
        self._last_rising_event_at = now
        return True


@dataclass(frozen=True, slots=True)
class ObjectTrackerView:
    """Describe the current stable-object surface."""

    objects: tuple[SocialDetectedObject, ...]
    unknown: bool
    rising_objects: tuple[SocialDetectedObject, ...]


class StableObjectTracker:
    """Track stable object detections across cadence-based observation ticks."""

    def __init__(
        self,
        *,
        on_samples: int,
        off_samples: int,
        unknown_hold_s: float,
    ) -> None:
        self.on_samples = require_positive_int(on_samples, field_name="on_samples")
        self.off_samples = require_positive_int(off_samples, field_name="off_samples")
        self.unknown_hold_s = require_non_negative_float(unknown_hold_s, field_name="unknown_hold_s")
        self._stable: dict[tuple[str, str], SocialDetectedObject] = {}
        self._seen_counts: dict[tuple[str, str], int] = {}
        self._missing_counts: dict[tuple[str, str], int] = {}
        self._last_concrete_at: float | None = None

    def observe(
        self,
        objects: tuple[SocialDetectedObject, ...] | None,
        *,
        observed_at: float,
    ) -> ObjectTrackerView:
        """Observe one new object list and return the stable surface."""

        now = coerce_timestamp(observed_at)
        if objects is None:
            if self._last_concrete_at is not None and (now - self._last_concrete_at) > self.unknown_hold_s:
                self._stable.clear()
                self._seen_counts.clear()
                self._missing_counts.clear()
            return ObjectTrackerView(objects=self._sorted_stable(), unknown=True, rising_objects=())

        self._last_concrete_at = now
        current: dict[tuple[str, str], SocialDetectedObject] = {}
        for item in objects:
            key = _object_key(item)
            previous = current.get(key)
            if previous is None or item.confidence >= previous.confidence:
                current[key] = item

        rising_objects: list[SocialDetectedObject] = []
        for key, item in current.items():
            self._missing_counts[key] = 0
            self._seen_counts[key] = self._seen_counts.get(key, 0) + 1
            if key not in self._stable and self._seen_counts[key] >= self.on_samples:
                stable_item = _stable_object(item)
                self._stable[key] = stable_item
                rising_objects.append(stable_item)
            elif key in self._stable:
                self._stable[key] = _stable_object(item)

        for key in tuple(self._stable):
            if key in current:
                continue
            self._missing_counts[key] = self._missing_counts.get(key, 0) + 1
            if self._missing_counts[key] >= self.off_samples:
                self._stable.pop(key, None)
                self._seen_counts.pop(key, None)
                self._missing_counts.pop(key, None)

        for key in tuple(self._seen_counts):
            if key not in current and key not in self._stable:
                self._seen_counts.pop(key, None)
                self._missing_counts.pop(key, None)

        return ObjectTrackerView(
            objects=self._sorted_stable(),
            unknown=False,
            rising_objects=tuple(sorted(rising_objects, key=lambda item: (item.label, item.zone.value))),
        )

    def _sorted_stable(self) -> tuple[SocialDetectedObject, ...]:
        return tuple(sorted(self._stable.values(), key=lambda item: (item.label, item.zone.value)))


def _object_key(item: SocialDetectedObject) -> tuple[str, str]:
    """Return one stable tracker key for an object detection."""

    return (item.label, item.zone.value)


def _stable_object(item: SocialDetectedObject) -> SocialDetectedObject:
    """Return one detection marked as stable for policy consumers."""

    return SocialDetectedObject(
        label=item.label,
        confidence=item.confidence,
        zone=item.zone,
        stable=True,
        box=item.box,
    )
