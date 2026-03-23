"""Publish bounded HDMI header debug signals for camera and fusion inspection.

This module keeps short-lived operator debugging pills separate from the main
runtime snapshot. It mirrors current camera states such as ``POSE_UPRIGHT`` and
``MOTION_STILL`` plus brief trigger/event holds such as ``POSSIBLE_FALL`` into
the dedicated HDMI debug-signal store without bloating the monitor loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.debug_signals import (
    DisplayDebugSignal,
    DisplayDebugSignalSnapshot,
    DisplayDebugSignalStore,
)


_SOURCE = "proactive_display_debug_signals"
_DEFAULT_EVENT_HOLD_S = 8.0
_DEFAULT_SAFETY_HOLD_S = 10.0
_MATCHING_SNAPSHOT_REFRESH_LOOKAHEAD_S = 1.5
_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})

_CURRENT_SIGNAL_SPECS: tuple[tuple[str, str, str, int], ...] = (
    ("hand_or_object_near_camera", "HAND_NEAR", "info", 51),
    ("showing_intent_likely", "INTENT_LIKELY", "success", 52),
    ("person_recently_visible", "PERSON_RECENT", "neutral", 100),
)

_POSE_SIGNAL_MAP: dict[str, tuple[str, str, int]] = {
    "upright": ("POSE_UPRIGHT", "neutral", 80),
    "seated": ("POSE_SEATED", "neutral", 81),
    "slumped": ("POSE_SLUMPED", "warning", 25),
    "lying_low": ("POSE_LYING", "warning", 26),
    "floor": ("POSE_FLOOR", "alert", 15),
}

_MOTION_SIGNAL_MAP: dict[str, tuple[str, str, int]] = {
    "still": ("MOTION_STILL", "neutral", 90),
    "walking": ("MOTION_WALKING", "info", 91),
    "approaching": ("MOTION_APPROACHING", "success", 92),
    "leaving": ("MOTION_LEAVING", "warning", 93),
}

_EVENT_SIGNAL_MAP: dict[str, tuple[DisplayDebugSignal, float]] = {
    "camera.person_returned": (
        DisplayDebugSignal(
            key="person_returned",
            label="PERSON_RETURNED",
            accent="success",
            priority=40,
        ),
        _DEFAULT_EVENT_HOLD_S,
    ),
    "camera.attention_window_opened": (
        DisplayDebugSignal(
            key="attention_window",
            label="ATTENTION_WINDOW",
            accent="info",
            priority=30,
        ),
        _DEFAULT_EVENT_HOLD_S,
    ),
    "camera.showing_intent_started": (
        DisplayDebugSignal(
            key="showing_intent",
            label="SHOWING_INTENT",
            accent="success",
            priority=31,
        ),
        _DEFAULT_EVENT_HOLD_S,
    ),
}

_TRIGGER_SIGNAL_MAP: dict[str, tuple[DisplayDebugSignal, float]] = {
    "slumped_quiet": (
        DisplayDebugSignal(
            key="slumped_quiet",
            label="SLUMPED_QUIET",
            accent="warning",
            priority=20,
        ),
        _DEFAULT_SAFETY_HOLD_S,
    ),
    "possible_fall": (
        DisplayDebugSignal(
            key="possible_fall",
            label="POSSIBLE_FALL",
            accent="alert",
            priority=10,
        ),
        _DEFAULT_SAFETY_HOLD_S,
    ),
    "floor_stillness": (
        DisplayDebugSignal(
            key="floor_stillness",
            label="FLOOR_STILLNESS",
            accent="alert",
            priority=11,
        ),
        _DEFAULT_SAFETY_HOLD_S,
    ),
    "distress_possible": (
        DisplayDebugSignal(
            key="distress_possible",
            label="DISTRESS_POSSIBLE",
            accent="alert",
            priority=12,
        ),
        _DEFAULT_SAFETY_HOLD_S,
    ),
    "positive_contact": (
        DisplayDebugSignal(
            key="positive_contact",
            label="POSITIVE_CONTACT",
            accent="success",
            priority=32,
        ),
        _DEFAULT_EVENT_HOLD_S,
    ),
}


def _utc_now() -> datetime:
    """Return the current aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _coerce_mapping(value: object | None) -> Mapping[str, object]:
    """Return one mapping-like payload or an empty mapping."""

    return value if isinstance(value, Mapping) else {}


def _normalize_text(value: object | None) -> str:
    """Normalize one optional text value into a compact lowercase token."""

    return " ".join(str(value or "").split()).strip().lower()


def _coerce_optional_bool(value: object | None) -> bool | None:
    """Parse one optional bool-like runtime fact."""

    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = _normalize_text(value)
    if normalized in _TRUE_TOKENS:
        return True
    if normalized in _FALSE_TOKENS:
        return False
    return None


def _coerce_optional_int(value: object | None) -> int | None:
    """Parse one optional integer-like runtime fact."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class DisplayDebugSignalPublishResult:
    """Summarize one debug-signal publish step for tests and telemetry."""

    action: str
    signal_keys: tuple[str, ...]
    hold_seconds: float


@dataclass(slots=True)
class DisplayDebugSignalPublisher:
    """Persist short-lived HDMI header debug pills from proactive camera facts."""

    store: DisplayDebugSignalStore
    source: str = _SOURCE
    _held_signals: dict[str, tuple[DisplayDebugSignal, datetime]] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayDebugSignalPublisher":
        """Build one publisher from the configured HDMI debug-signal store."""

        return cls(store=DisplayDebugSignalStore.from_config(config))

    def publish_from_camera_facts(
        self,
        *,
        camera_facts: Mapping[str, object] | object,
        event_names: Iterable[str] = (),
        trigger_ids: Iterable[str] = (),
        now: datetime | None = None,
    ) -> DisplayDebugSignalPublishResult:
        """Persist the current active debug pills from camera facts and recent triggers."""

        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        camera = _coerce_mapping(camera_facts)
        self._refresh_holds(now=effective_now, event_names=event_names, trigger_ids=trigger_ids)
        signals = self._ordered_signals(
            (
                *self._current_camera_signals(camera),
                *self._active_held_signals(now=effective_now),
            )
        )
        hold_seconds = max(self.store.default_ttl_s, self._max_remaining_hold_seconds(now=effective_now))
        active_snapshot = self.store.load_active(now=effective_now)
        action = "updated"
        if active_snapshot is not None and active_snapshot.source == self.source and active_snapshot.signature() == tuple(
            signal.signature() for signal in signals
        ):
            if not self._active_snapshot_needs_refresh(
                snapshot=active_snapshot,
                hold_seconds=hold_seconds,
                now=effective_now,
            ):
                return DisplayDebugSignalPublishResult(
                    action="unchanged",
                    signal_keys=tuple(signal.key for signal in signals),
                    hold_seconds=hold_seconds,
                )
            action = "refreshed"
        self.store.save(
            DisplayDebugSignalSnapshot(
                source=self.source,
                signals=signals,
            ),
            now=effective_now,
            hold_seconds=hold_seconds,
        )
        return DisplayDebugSignalPublishResult(
            action=action,
            signal_keys=tuple(signal.key for signal in signals),
            hold_seconds=hold_seconds,
        )

    def _current_camera_signals(
        self,
        camera: Mapping[str, object],
    ) -> tuple[DisplayDebugSignal, ...]:
        """Return the currently active camera-derived debug pills."""

        signals: list[DisplayDebugSignal] = []
        person_visible = _coerce_optional_bool(camera.get("person_visible"))
        person_visible_unknown = _coerce_optional_bool(camera.get("person_visible_unknown"))
        if person_visible is True and person_visible_unknown is not True:
            person_count = None
            if _coerce_optional_bool(camera.get("person_count_unknown")) is not True:
                person_count = _coerce_optional_int(camera.get("person_count"))
            if person_count is not None and person_count > 0:
                person_label = "PERSON_3P" if person_count >= 3 else f"PERSON_{person_count}"
            else:
                person_label = "PERSON_VISIBLE"
            signals.append(
                DisplayDebugSignal(
                    key="person_visible",
                    label=person_label,
                    accent="info",
                    priority=70,
                )
            )

        if _coerce_optional_bool(camera.get("person_returned_after_absence")) is True:
            signals.append(
                DisplayDebugSignal(
                    key="person_returned",
                    label="PERSON_RETURNED",
                    accent="success",
                    priority=40,
                )
            )

        for fact_key, label, accent, priority in _CURRENT_SIGNAL_SPECS:
            if _coerce_optional_bool(camera.get(f"{fact_key}_unknown")) is True:
                continue
            if _coerce_optional_bool(camera.get(fact_key)) is not True:
                continue
            signals.append(
                DisplayDebugSignal(
                    key=fact_key,
                    label=label,
                    accent=accent,
                    priority=priority,
                )
            )

        if _coerce_optional_bool(camera.get("engaged_with_device_unknown")) is not True and _coerce_optional_bool(
            camera.get("engaged_with_device")
        ) is True:
            looking_confirmed = (
                _coerce_optional_bool(camera.get("looking_toward_device")) is True
                and _normalize_text(camera.get("looking_signal_state")) == "confirmed"
            )
            engaged_label = "ENGAGED"
            engaged_accent = "success"
            if not looking_confirmed:
                engaged_label = "ENGAGED_PROXY"
                engaged_accent = "info"
            signals.append(
                DisplayDebugSignal(
                    key="engaged_with_device",
                    label=engaged_label,
                    accent=engaged_accent,
                    priority=53,
                )
            )

        if _coerce_optional_bool(camera.get("looking_toward_device_unknown")) is not True and _coerce_optional_bool(
            camera.get("looking_toward_device")
        ) is True:
            looking_state = _normalize_text(camera.get("looking_signal_state"))
            looking_label = "LOOKING"
            if looking_state == "confirmed":
                looking_label = "LOOKING_CONFIRMED"
            elif looking_state == "proxy":
                looking_label = "LOOKING_PROXY"
            signals.append(
                DisplayDebugSignal(
                    key="looking_toward_device",
                    label=looking_label,
                    accent="info",
                    priority=50,
                )
            )

        pose_key = _normalize_text(camera.get("body_pose"))
        if pose_key and pose_key not in {"unknown", "none"} and _coerce_optional_bool(camera.get("body_pose_unknown")) is not True:
            pose_spec = _POSE_SIGNAL_MAP.get(pose_key)
            if pose_spec is not None:
                label, accent, priority = pose_spec
                signals.append(
                    DisplayDebugSignal(
                        key="body_pose",
                        label=label,
                        accent=accent,
                        priority=priority,
                    )
                )

        motion_key = _normalize_text(camera.get("motion_state"))
        if (
            motion_key
            and motion_key not in {"unknown", "none"}
            and _coerce_optional_bool(camera.get("motion_state_unknown")) is not True
        ):
            motion_spec = _MOTION_SIGNAL_MAP.get(motion_key)
            if motion_spec is not None:
                label, accent, priority = motion_spec
                signals.append(
                    DisplayDebugSignal(
                        key="motion_state",
                        label=label,
                        accent=accent,
                        priority=priority,
                    )
                )

        return tuple(signals)

    def _refresh_holds(
        self,
        *,
        now: datetime,
        event_names: Iterable[str],
        trigger_ids: Iterable[str],
    ) -> None:
        """Advance event/trigger hold windows and prune expired state."""

        active_holds: dict[str, tuple[DisplayDebugSignal, datetime]] = {}
        for key, (signal, expires_at) in self._held_signals.items():
            if expires_at > now:
                active_holds[key] = (signal, expires_at)
        self._held_signals = active_holds

        for event_name in {_normalize_text(name) for name in event_names if _normalize_text(name)}:
            mapped = _EVENT_SIGNAL_MAP.get(event_name)
            if mapped is None:
                continue
            signal, hold_seconds = mapped
            self._held_signals[signal.key] = (signal, now + timedelta(seconds=hold_seconds))

        for trigger_id in {_normalize_text(name) for name in trigger_ids if _normalize_text(name)}:
            mapped = _TRIGGER_SIGNAL_MAP.get(trigger_id)
            if mapped is None:
                continue
            signal, hold_seconds = mapped
            self._held_signals[signal.key] = (signal, now + timedelta(seconds=hold_seconds))

    def _active_held_signals(self, *, now: datetime) -> tuple[DisplayDebugSignal, ...]:
        """Return the currently unexpired event/trigger debug pills."""

        return tuple(
            signal
            for signal, expires_at in self._held_signals.values()
            if expires_at > now
        )

    def _max_remaining_hold_seconds(self, *, now: datetime) -> float:
        """Return the longest remaining hold among active transient signals."""

        remaining = [
            max(0.0, (expires_at - now).total_seconds())
            for _signal, expires_at in self._held_signals.values()
            if expires_at > now
        ]
        if not remaining:
            return 0.0
        return max(remaining)

    def _active_snapshot_needs_refresh(
        self,
        *,
        snapshot: DisplayDebugSignalSnapshot,
        hold_seconds: float,
        now: datetime,
    ) -> bool:
        """Return whether one unchanged active snapshot should renew its TTL."""

        expires_at = _parse_timestamp(snapshot.expires_at)
        if expires_at is None:
            return False
        seconds_left = (expires_at - now).total_seconds()
        threshold_s = min(float(hold_seconds), _MATCHING_SNAPSHOT_REFRESH_LOOKAHEAD_S)
        return seconds_left <= max(0.25, threshold_s)

    def _ordered_signals(
        self,
        signals: Iterable[DisplayDebugSignal],
    ) -> tuple[DisplayDebugSignal, ...]:
        """Deduplicate and sort signals into a deterministic header order."""

        deduped: dict[str, DisplayDebugSignal] = {}
        for signal in signals:
            current = deduped.get(signal.key)
            if current is None or signal.priority < current.priority:
                deduped[signal.key] = signal
        return tuple(
            sorted(
                deduped.values(),
                key=lambda item: (item.priority, item.label, item.key),
            )
        )


def _parse_timestamp(value: object | None) -> datetime | None:
    """Parse one optional ISO-8601 timestamp into aware UTC time."""

    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


__all__ = [
    "DisplayDebugSignalPublishResult",
    "DisplayDebugSignalPublisher",
]
