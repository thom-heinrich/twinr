"""Cadence-aware signal trackers for the camera surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..engine import SocialDetectedObject
from .coercion import coerce_timestamp, duration_since
from .validation import require_non_negative_float, require_positive_int

# CHANGELOG: 2026-03-29
# BUG-1: StableObjectTracker now normalizes observed_at with the previous observation too.
#        The old code used coerce_timestamp(observed_at) without previous=..., so out-of-order /
#        clock-adjusted timestamps could make staleness and unknown-hold logic run backward.
# BUG-2: The module claimed "cadence-aware" behavior but used pure sample counts. On a Pi 4,
#        variable camera cadence changed activation/deactivation latency in real deployments.
#        The tracker now auto-normalizes debounce latency to observed cadence while preserving
#        an opt-out path for legacy pure sample semantics.
# SEC-1: Added bounded per-tick fan-in and bounded internal track state with deterministic
#        pruning to mitigate input-driven memory / CPU exhaustion on edge deployments.
# IMP-1: StableObjectTracker now prefers upstream tracker IDs and otherwise performs lightweight
#        IoU-based identity association instead of collapsing everything to (label, zone).
# IMP-2: Added confidence EMA smoothing and instance-level outputs while preserving the legacy
#        surface-level objects API.
# BREAKING: Debounce semantics are now time-aware by default. To force legacy pure sample-count
#           behavior, pass on_min_s=0.0 and off_min_s=0.0 explicitly.


def _require_fraction(value: float, *, field_name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0 inclusive")
    return value


def _coerce_optional_non_negative_float(
    value: float | None,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return require_non_negative_float(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class BooleanSignalView:
    """Describe the latest debounced state of one boolean signal."""

    value: bool
    unknown: bool
    active_for_s: float
    rising_edge: bool


class DebouncedBooleanSignal:
    """Track one boolean signal with debounce, cooldown, and unknown-state handling."""

    def __init__(
        self,
        *,
        on_samples: int,
        off_samples: int,
        unknown_hold_s: float,
        event_cooldown_s: float = 0.0,
        on_min_s: float | None = None,
        off_min_s: float | None = None,
    ) -> None:
        self.on_samples = require_positive_int(on_samples, field_name="on_samples")
        self.off_samples = require_positive_int(off_samples, field_name="off_samples")
        self.unknown_hold_s = require_non_negative_float(unknown_hold_s, field_name="unknown_hold_s")
        self.event_cooldown_s = require_non_negative_float(event_cooldown_s, field_name="event_cooldown_s")
        self.on_min_s = _coerce_optional_non_negative_float(on_min_s, field_name="on_min_s")
        self.off_min_s = _coerce_optional_non_negative_float(off_min_s, field_name="off_min_s")

        self._stable_value = False
        self._active_since: float | None = None
        self._last_concrete_at: float | None = None
        self._last_observed_at: float | None = None
        self._cadence_ema_s: float | None = None

        self._pending_value: bool | None = None
        self._pending_count = 0
        self._pending_started_at: float | None = None

        self._last_rising_event_at: float | None = None

    def observe(self, value: bool | None, *, observed_at: float) -> BooleanSignalView:
        """Observe one new value and return the debounced signal view."""

        now = coerce_timestamp(observed_at, previous=self._last_observed_at)
        if value is not None:
            self._update_cadence(now, previous=self._last_concrete_at)
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
            configured_min_s = self.on_min_s if current_value else self.off_min_s
            minimum_s = self._resolve_transition_min_s(configured_min_s, threshold_samples=threshold)
            minimum_samples = self._resolve_transition_min_samples(configured_min_s, threshold_samples=threshold)
            if (
                self._pending_count >= minimum_samples
                and duration_since(self._pending_started_at, now) >= minimum_s
            ):
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
        if value is not None:
            self._update_cadence(now, previous=self._last_concrete_at)
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

    def _update_cadence(self, now: float, *, previous: float | None) -> None:
        if previous is None or now <= previous:
            return
        delta_s = now - previous
        if self._cadence_ema_s is None:
            self._cadence_ema_s = delta_s
            return
        self._cadence_ema_s = (0.25 * delta_s) + (0.75 * self._cadence_ema_s)

    def _resolve_transition_min_s(
        self,
        configured_min_s: float | None,
        *,
        threshold_samples: int,
    ) -> float:
        if configured_min_s is not None:
            return configured_min_s
        if threshold_samples <= 1 or self._cadence_ema_s is None:
            return 0.0
        return max(0.0, (threshold_samples - 1) * self._cadence_ema_s)

    def _resolve_transition_min_samples(
        self,
        configured_min_s: float | None,
        *,
        threshold_samples: int,
    ) -> int:
        if configured_min_s is not None:
            return threshold_samples
        if self._cadence_ema_s is None:
            return threshold_samples
        if threshold_samples <= 1:
            return 1
        return 2

    def _advance_pending(self, value: bool, *, now: float) -> None:
        if self._pending_value == value:
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
    """Describe the current stable-object surface and the stable tracked instances."""

    objects: tuple[SocialDetectedObject, ...]
    unknown: bool
    rising_objects: tuple[SocialDetectedObject, ...]
    stable_instances: tuple[SocialDetectedObject, ...] = ()
    rising_instances: tuple[SocialDetectedObject, ...] = ()


@dataclass(slots=True)
class _TrackState:
    """Internal per-instance tracking state."""

    identity: str
    item: SocialDetectedObject
    confidence_ema: float
    seen_streak: int = 0
    miss_streak: int = 0
    visible_run_started_at: float | None = None
    missing_run_started_at: float | None = None
    stable: bool = False
    stable_since: float | None = None
    last_seen_at: float | None = None
    last_updated_at: float | None = None


class StableObjectTracker:
    """Track stable object detections across cadence-normalized observation ticks."""

    def __init__(
        self,
        *,
        on_samples: int,
        off_samples: int,
        unknown_hold_s: float,
        on_min_s: float | None = None,
        off_min_s: float | None = None,
        iou_match_threshold: float = 0.3,
        confidence_alpha: float = 0.35,
        max_objects_per_observation: int = 128,
        max_tracks: int = 512,
    ) -> None:
        self.on_samples = require_positive_int(on_samples, field_name="on_samples")
        self.off_samples = require_positive_int(off_samples, field_name="off_samples")
        self.unknown_hold_s = require_non_negative_float(unknown_hold_s, field_name="unknown_hold_s")
        self.on_min_s = _coerce_optional_non_negative_float(on_min_s, field_name="on_min_s")
        self.off_min_s = _coerce_optional_non_negative_float(off_min_s, field_name="off_min_s")
        self.iou_match_threshold = _require_fraction(iou_match_threshold, field_name="iou_match_threshold")
        self.confidence_alpha = _require_fraction(confidence_alpha, field_name="confidence_alpha")
        self.max_objects_per_observation = require_positive_int(
            max_objects_per_observation,
            field_name="max_objects_per_observation",
        )
        self.max_tracks = require_positive_int(max_tracks, field_name="max_tracks")

        self._tracks: dict[str, _TrackState] = {}
        self._last_concrete_at: float | None = None
        self._last_observed_at: float | None = None
        self._cadence_ema_s: float | None = None
        self._next_track_serial = 0
        self._previous_surface_keys: set[tuple[str, str]] = set()

    def observe(
        self,
        objects: tuple[SocialDetectedObject, ...] | None,
        *,
        observed_at: float,
    ) -> ObjectTrackerView:
        """Observe one object list and return the stable surface plus stable instances."""

        return self._observe_impl(objects, observed_at=observed_at, authoritative=False)

    def observe_authoritative(
        self,
        objects: tuple[SocialDetectedObject, ...] | None,
        *,
        observed_at: float,
    ) -> ObjectTrackerView:
        """Adopt one already-stabilized upstream tracked list without local debounce."""

        return self._observe_impl(objects, observed_at=observed_at, authoritative=True)

    def _observe_impl(
        self,
        objects: tuple[SocialDetectedObject, ...] | None,
        *,
        observed_at: float,
        authoritative: bool,
    ) -> ObjectTrackerView:
        now = coerce_timestamp(observed_at, previous=self._last_observed_at)
        self._last_observed_at = now

        if objects is None:
            if self._last_concrete_at is not None and (now - self._last_concrete_at) > self.unknown_hold_s:
                self._clear_all()
            return self._build_view(unknown=True, rising_instance_ids=set())

        self._update_cadence(now, previous=self._last_concrete_at)
        self._last_concrete_at = now

        limited = _limit_objects(objects, max_objects=self.max_objects_per_observation)
        assignments = self._associate(limited)
        seen_ids = set(assignments)
        rising_instance_ids: set[str] = set()

        for identity, item in assignments.items():
            state = self._tracks.get(identity)
            if state is None:
                state = _TrackState(
                    identity=identity,
                    item=item,
                    confidence_ema=float(item.confidence),
                )
                self._tracks[identity] = state

            if state.miss_streak > 0:
                state.seen_streak = 1
                state.visible_run_started_at = now
            else:
                state.seen_streak += 1
                if state.visible_run_started_at is None:
                    state.visible_run_started_at = now

            state.item = item
            state.confidence_ema = self._blend_confidence(state.confidence_ema, item.confidence)
            state.miss_streak = 0
            state.missing_run_started_at = None
            state.last_seen_at = now
            state.last_updated_at = now

            if authoritative:
                if not state.stable:
                    state.stable = True
                    state.stable_since = now
                    rising_instance_ids.add(identity)
                continue

            if not state.stable and self._ready_to_activate(state, now):
                state.stable = True
                state.stable_since = state.visible_run_started_at or now
                rising_instance_ids.add(identity)

        for identity in tuple(self._tracks):
            if identity in seen_ids:
                continue

            state = self._tracks[identity]
            state.last_updated_at = now

            if authoritative:
                self._tracks.pop(identity, None)
                continue

            state.miss_streak += 1
            if state.missing_run_started_at is None:
                state.missing_run_started_at = now

            if state.stable and self._ready_to_deactivate(state, now):
                state.stable = False
                state.stable_since = None

            if (not state.stable) and self._ready_to_evict(state, now):
                self._tracks.pop(identity, None)

        self._prune_tracks()

        return self._build_view(unknown=False, rising_instance_ids=rising_instance_ids)

    def _update_cadence(self, now: float, *, previous: float | None) -> None:
        if previous is None or now <= previous:
            return
        delta_s = now - previous
        if self._cadence_ema_s is None:
            self._cadence_ema_s = delta_s
            return
        self._cadence_ema_s = (0.25 * delta_s) + (0.75 * self._cadence_ema_s)

    def _resolve_transition_min_s(
        self,
        configured_min_s: float | None,
        *,
        threshold_samples: int,
    ) -> float:
        if configured_min_s is not None:
            return configured_min_s
        if threshold_samples <= 1 or self._cadence_ema_s is None:
            return 0.0
        return max(0.0, (threshold_samples - 1) * self._cadence_ema_s)

    def _resolve_transition_min_samples(
        self,
        configured_min_s: float | None,
        *,
        threshold_samples: int,
    ) -> int:
        if configured_min_s is not None:
            return threshold_samples
        if self._cadence_ema_s is None:
            return threshold_samples
        if threshold_samples <= 1:
            return 1
        return 2

    def _associate(
        self,
        objects: tuple[SocialDetectedObject, ...],
    ) -> dict[str, SocialDetectedObject]:
        assignments: dict[str, SocialDetectedObject] = {}
        reserved: set[str] = set()
        deferred: list[SocialDetectedObject] = []

        # 1) Exact upstream tracking IDs are the highest-quality identity signal.
        for item in sorted(objects, key=lambda candidate: candidate.confidence, reverse=True):
            explicit = _explicit_identity(item)
            if explicit is None:
                deferred.append(item)
                continue
            if explicit in assignments:
                if item.confidence > assignments[explicit].confidence:
                    assignments[explicit] = item
                reserved.add(explicit)
                continue
            assignments[explicit] = item
            reserved.add(explicit)

        # 2) If no upstream ID is available, associate by lightweight IoU.
        for item in deferred:
            identity = self._match_existing_track(item, reserved=reserved)
            if identity is None:
                # 3) Without IDs or usable boxes, preserve legacy surface semantics.
                fallback_identity = self._surface_identity(item)
                if _coerce_box(item.box) is None:
                    identity = fallback_identity
                elif fallback_identity not in reserved and fallback_identity in self._tracks:
                    identity = fallback_identity
                else:
                    identity = self._new_anon_identity()

            current = assignments.get(identity)
            if current is None or item.confidence >= current.confidence:
                assignments[identity] = item
            reserved.add(identity)

        return assignments

    def _match_existing_track(
        self,
        item: SocialDetectedObject,
        *,
        reserved: set[str],
    ) -> str | None:
        box = _coerce_box(item.box)
        if box is None:
            return None

        best_identity: str | None = None
        best_score = self.iou_match_threshold
        for identity, state in self._tracks.items():
            if identity in reserved:
                continue
            if state.item.label != item.label:
                continue
            if state.last_seen_at is None:
                continue

            candidate_box = _coerce_box(state.item.box)
            if candidate_box is None:
                continue

            iou = _box_iou(box, candidate_box)
            if iou < self.iou_match_threshold:
                continue

            score = iou
            if state.item.zone.value == item.zone.value:
                score += 1e-6  # Keep same-zone matches stable when IoU ties.
            if score >= best_score:
                best_identity = identity
                best_score = score

        return best_identity

    def _ready_to_activate(self, state: _TrackState, now: float) -> bool:
        minimum_s = self._resolve_transition_min_s(self.on_min_s, threshold_samples=self.on_samples)
        minimum_samples = self._resolve_transition_min_samples(self.on_min_s, threshold_samples=self.on_samples)
        return (
            state.seen_streak >= minimum_samples
            and duration_since(state.visible_run_started_at, now) >= minimum_s
        )

    def _ready_to_deactivate(self, state: _TrackState, now: float) -> bool:
        minimum_s = self._resolve_transition_min_s(self.off_min_s, threshold_samples=self.off_samples)
        minimum_samples = self._resolve_transition_min_samples(self.off_min_s, threshold_samples=self.off_samples)
        return (
            state.miss_streak >= minimum_samples
            and duration_since(state.missing_run_started_at, now) >= minimum_s
        )

    def _ready_to_evict(self, state: _TrackState, now: float) -> bool:
        if state.last_seen_at is None:
            return True
        hold_s = max(
            self.unknown_hold_s,
            self._resolve_transition_min_s(self.off_min_s, threshold_samples=self.off_samples),
        )
        return duration_since(state.last_seen_at, now) >= hold_s

    def _build_view(
        self,
        *,
        unknown: bool,
        rising_instance_ids: set[str],
    ) -> ObjectTrackerView:
        stable_states = [state for state in self._tracks.values() if state.stable]

        stable_instance_map = {
            state.identity: _stable_object(state.item, confidence=state.confidence_ema)
            for state in stable_states
        }
        stable_instances = tuple(
            sorted(
                stable_instance_map.values(),
                key=lambda item: (item.label, item.zone.value, -float(item.confidence)),
            )
        )

        surface_best: dict[tuple[str, str], SocialDetectedObject] = {}
        for item in stable_instances:
            key = (item.label, item.zone.value)
            previous = surface_best.get(key)
            if previous is None or item.confidence >= previous.confidence:
                surface_best[key] = item

        current_surface_keys = set(surface_best)
        if unknown:
            rising_objects: tuple[SocialDetectedObject, ...] = ()
            rising_instances: tuple[SocialDetectedObject, ...] = ()
        else:
            rising_surface_keys = current_surface_keys - self._previous_surface_keys
            rising_objects = tuple(
                sorted(
                    (surface_best[key] for key in rising_surface_keys),
                    key=lambda item: (item.label, item.zone.value),
                )
            )
            rising_instances = tuple(
                sorted(
                    (
                        stable_instance_map[identity]
                        for identity in rising_instance_ids
                        if identity in stable_instance_map
                    ),
                    key=lambda item: (item.label, item.zone.value, -float(item.confidence)),
                )
            )
            self._previous_surface_keys = current_surface_keys

        return ObjectTrackerView(
            objects=tuple(sorted(surface_best.values(), key=lambda item: (item.label, item.zone.value))),
            unknown=unknown,
            rising_objects=rising_objects,
            stable_instances=stable_instances,
            rising_instances=rising_instances,
        )

    def _blend_confidence(self, previous: float, current: float) -> float:
        alpha = self.confidence_alpha
        return (alpha * float(current)) + ((1.0 - alpha) * float(previous))

    def _new_anon_identity(self) -> str:
        self._next_track_serial += 1
        return f"anon:{self._next_track_serial}"

    def _surface_identity(self, item: SocialDetectedObject) -> str:
        return f"surface:{item.label}:{item.zone.value}"

    def _prune_tracks(self) -> None:
        if len(self._tracks) <= self.max_tracks:
            return

        # Prefer pruning oldest non-stable tracks; only prune stable tracks if the cap is exceeded.
        prune_order = sorted(
            self._tracks.values(),
            key=lambda state: (
                state.stable,
                state.last_seen_at if state.last_seen_at is not None else float("-inf"),
                state.identity,
            ),
        )
        overflow = len(self._tracks) - self.max_tracks
        for state in prune_order[:overflow]:
            self._tracks.pop(state.identity, None)

    def _clear_all(self) -> None:
        self._tracks.clear()
        self._previous_surface_keys.clear()


def _limit_objects(
    objects: tuple[SocialDetectedObject, ...],
    *,
    max_objects: int,
) -> tuple[SocialDetectedObject, ...]:
    """Bound per-observation fan-in while preserving the most confident detections."""

    if len(objects) <= max_objects:
        return objects
    return tuple(sorted(objects, key=lambda item: item.confidence, reverse=True)[:max_objects])


def _explicit_identity(item: SocialDetectedObject) -> str | None:
    """Return an upstream track/object identity if the detection already carries one."""

    for attribute in ("tracker_id", "track_id", "object_id", "instance_id", "id"):
        value = getattr(item, attribute, None)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        return f"track:{attribute}:{text}"
    return None


def _coerce_box(box: Any) -> tuple[float, float, float, float] | None:
    """Normalize common box representations to xyxy floats."""

    if box is None:
        return None

    if isinstance(box, (tuple, list)):
        if len(box) == 4:
            try:
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
            except (TypeError, ValueError):
                return None
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)
        if len(box) == 1:
            return _coerce_box(box[0])

    for names in (
        ("x1", "y1", "x2", "y2"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ):
        if all(hasattr(box, name) for name in names):
            try:
                x1 = float(getattr(box, names[0]))
                y1 = float(getattr(box, names[1]))
                x2 = float(getattr(box, names[2]))
                y2 = float(getattr(box, names[3]))
            except (TypeError, ValueError):
                return None
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)

    xyxy = getattr(box, "xyxy", None)
    if xyxy is not None:
        return _coerce_box(xyxy)

    return None


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return the IoU of two xyxy boxes."""

    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _object_key(item: SocialDetectedObject) -> tuple[str, str]:
    """Return the legacy surface key for one detection."""

    return (item.label, item.zone.value)


def _stable_object(
    item: SocialDetectedObject,
    *,
    confidence: float | None = None,
) -> SocialDetectedObject:
    """Return one detection marked as stable for policy consumers."""

    return SocialDetectedObject(
        label=item.label,
        confidence=item.confidence if confidence is None else float(confidence),
        zone=item.zone,
        stable=True,
        box=item.box,
    )