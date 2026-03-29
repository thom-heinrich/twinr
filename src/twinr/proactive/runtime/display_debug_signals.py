# CHANGELOG: 2026-03-29
# BUG-1: Replaced wall-clock hold expiry with a suspend-aware monotonic/boottime clock so transient pills do not stretch or vanish on clock jumps.
# BUG-2: Serialized publish/load/save and hold mutation with an RLock to remove real race conditions under concurrent runtime callers.
# BUG-3: Fixed falsey-token parsing so numeric 0/1 and single-string inputs no longer disappear or get iterated character-by-character.
# SEC-1: Added bounded, streaming parsing for event/trigger inputs plus text/TTL clamps to prevent easy CPU/RAM exhaustion on Raspberry Pi deployments.
# IMP-1: Upgraded the input API to accept richer mapping payloads (custom pills, explicit expiry/TTL, optional confidence gating) while remaining string-compatible.
# IMP-2: Added stale-camera detection, a CAMERA_STALE pill, richer publish telemetry, and bounded header output to match 2026 edge-runtime monitoring expectations.

"""Publish bounded HDMI header debug signals for camera and fusion inspection.

This module keeps short-lived operator debugging pills separate from the main
runtime snapshot. It mirrors current camera states such as ``POSE_UPRIGHT`` and
``MOTION_STILL`` plus brief trigger/event holds such as ``POSSIBLE_FALL`` into
the dedicated HDMI debug-signal store without bloating the monitor loop.

Compared with the original implementation, this version hardens the hot path
for real Raspberry Pi deployments: transient holds use a monotonic clock,
concurrent callers are serialized, oversized event payloads are bounded, and
newer upstream pipelines can pass richer mapping payloads with explicit TTLs,
absolute expiries, and optional confidence gating.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
import re
import threading
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.debug_signals import (
    DisplayDebugSignal,
    DisplayDebugSignalSnapshot,
    DisplayDebugSignalStore,
)


_SOURCE = "proactive_display_debug_signals"
_DEFAULT_EVENT_HOLD_S = 8.0
_DEFAULT_SAFETY_HOLD_S = 10.0
_DEFAULT_MAX_CUSTOM_HOLD_S = 120.0
_DEFAULT_MAX_INPUT_ITEMS = 64
_DEFAULT_MAX_TEXT_CHARS = 96
_DEFAULT_MAX_LABEL_CHARS = 48
_DEFAULT_MAX_SIGNALS = 12
_DEFAULT_MAX_CAMERA_FACT_AGE_S = 3.0
_MATCHING_SNAPSHOT_REFRESH_LOOKAHEAD_S = 1.5
_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})
_ALLOWED_ACCENTS = frozenset({"neutral", "info", "success", "warning", "alert"})
_EVENT_CUSTOM_PRIORITY = 45
_TRIGGER_CUSTOM_PRIORITY = 18
_TOKEN_SEPARATORS_RE = re.compile(r"[\s\-]+")
_LABEL_SEPARATORS_RE = re.compile(r"[\s\-.]+")

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

_CAMERA_STALE_SIGNAL = DisplayDebugSignal(
    key="camera_stale",
    label="CAMERA_STALE",
    accent="warning",
    priority=14,
)

_CAMERA_AGE_KEYS: tuple[str, ...] = (
    "camera_fact_age_s",
    "camera_facts_age_s",
    "camera_age_s",
    "frame_age_s",
    "age_s",
)

_CAMERA_TIMESTAMP_KEYS: tuple[str, ...] = (
    "camera_facts_at",
    "camera_updated_at",
    "frame_captured_at",
    "frame_observed_at",
    "observed_at",
    "updated_at",
)

_EXPLICIT_CAMERA_STALE_KEYS: tuple[str, ...] = (
    "camera_stale",
    "camera_facts_stale",
    "camera_pipeline_stale",
)


@dataclass(frozen=True, slots=True)
class _HeldSignalState:
    """One active transient signal plus its expiries."""

    signal: DisplayDebugSignal
    expires_at_utc: datetime
    expires_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class _TransientSignalInput:
    """One normalized transient debug pill to be held temporarily."""

    signal: DisplayDebugSignal
    hold_seconds: float


@dataclass(frozen=True, slots=True)
class DisplayDebugSignalPublishResult:
    """Summarize one debug-signal publish step for tests and telemetry."""

    action: str
    signal_keys: tuple[str, ...]
    hold_seconds: float
    signal_count: int = 0
    dropped_input_items: int = 0
    stale_camera_facts: bool = False
    input_was_truncated: bool = False


@dataclass(slots=True)
class DisplayDebugSignalPublisher:
    """Persist short-lived HDMI header debug pills from proactive camera facts."""

    store: DisplayDebugSignalStore
    source: str = _SOURCE
    max_input_items: int = _DEFAULT_MAX_INPUT_ITEMS
    max_input_text_chars: int = _DEFAULT_MAX_TEXT_CHARS
    max_signals: int = _DEFAULT_MAX_SIGNALS
    max_custom_hold_s: float = _DEFAULT_MAX_CUSTOM_HOLD_S
    max_camera_fact_age_s: float | None = _DEFAULT_MAX_CAMERA_FACT_AGE_S
    _held_signals: dict[str, _HeldSignalState] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayDebugSignalPublisher":
        """Build one publisher from the configured HDMI debug-signal store."""

        return cls(store=DisplayDebugSignalStore.from_config(config))

    def publish_from_camera_facts(
        self,
        *,
        camera_facts: Mapping[str, object] | object,
        event_names: Iterable[str | Mapping[str, object] | object] = (),
        trigger_ids: Iterable[str | Mapping[str, object] | object] = (),
        now: datetime | None = None,
        monotonic_ns: int | None = None,
    ) -> DisplayDebugSignalPublishResult:
        """Persist the current active debug pills from camera facts and recent triggers.

        ``event_names`` and ``trigger_ids`` remain drop-in compatible with the original
        string-based API, but may now also contain mapping payloads such as:

        ``{"id": "possible_fall", "hold_seconds": 12}``
        ``{"key": "fusion_recovery", "label": "FUSION_RECOVERY", "accent": "info", "hold_seconds": 5}``
        ``{"name": "camera.person_returned", "expires_at": "2026-03-29T12:00:00Z"}``

        # BREAKING: naive ``now`` values are treated as UTC instead of local time to
        # avoid locale-dependent TTL bugs in tests and headless deployments.
        """

        effective_now = _coerce_utc_datetime(now) or _utc_now()
        if monotonic_ns is not None:
            effective_monotonic_ns = monotonic_ns
        elif now is not None:
            # Keep explicit test/ops clocks internally consistent across calls.
            # When callers drive the wall clock manually, real monotonic time
            # does not advance with that synthetic timeline.
            effective_monotonic_ns = _seconds_to_ns(effective_now.timestamp())
        else:
            effective_monotonic_ns = _steady_now_ns()
        if effective_monotonic_ns < 0:
            effective_monotonic_ns = 0

        with self._lock:
            camera = _coerce_mapping(camera_facts)
            dropped_input_items, input_was_truncated = self._refresh_holds(
                now_utc=effective_now,
                now_monotonic_ns=effective_monotonic_ns,
                event_names=event_names,
                trigger_ids=trigger_ids,
            )
            stale_camera_facts = self._camera_facts_are_stale(camera, now=effective_now)
            signals = self._ordered_signals(
                (
                    *self._current_camera_signals(camera, stale_camera_facts=stale_camera_facts),
                    *self._active_held_signals(now_monotonic_ns=effective_monotonic_ns),
                )
            )
            hold_seconds = max(
                self._store_default_ttl_s(),
                self._max_remaining_hold_seconds(now_monotonic_ns=effective_monotonic_ns),
            )
            active_snapshot = self.store.load_active(now=effective_now)
            action = "updated"
            if (
                active_snapshot is not None
                and active_snapshot.source == self.source
                and active_snapshot.signature() == tuple(signal.signature() for signal in signals)
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
                        signal_count=len(signals),
                        dropped_input_items=dropped_input_items,
                        stale_camera_facts=stale_camera_facts,
                        input_was_truncated=input_was_truncated,
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
                signal_count=len(signals),
                dropped_input_items=dropped_input_items,
                stale_camera_facts=stale_camera_facts,
                input_was_truncated=input_was_truncated,
            )

    def _current_camera_signals(
        self,
        camera: Mapping[str, object],
        *,
        stale_camera_facts: bool,
    ) -> tuple[DisplayDebugSignal, ...]:
        """Return the currently active camera-derived debug pills."""

        if stale_camera_facts:
            return (_CAMERA_STALE_SIGNAL,)

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
                and _normalize_token(camera.get("looking_signal_state"), max_chars=self.max_input_text_chars) == "confirmed"
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
            looking_state = _normalize_token(camera.get("looking_signal_state"), max_chars=self.max_input_text_chars)
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

        pose_key = _normalize_token(camera.get("body_pose"), max_chars=self.max_input_text_chars)
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

        motion_key = _normalize_token(camera.get("motion_state"), max_chars=self.max_input_text_chars)
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
        now_utc: datetime,
        now_monotonic_ns: int,
        event_names: Iterable[str | Mapping[str, object] | object],
        trigger_ids: Iterable[str | Mapping[str, object] | object],
    ) -> tuple[int, bool]:
        """Advance event/trigger hold windows and prune expired state."""

        active_holds: dict[str, _HeldSignalState] = {}
        for key, state in self._held_signals.items():
            if state.expires_monotonic_ns > now_monotonic_ns:
                active_holds[key] = state
        self._held_signals = active_holds

        dropped_events, truncated_events = self._ingest_transient_items(
            items=event_names,
            catalog=_EVENT_SIGNAL_MAP,
            now_utc=now_utc,
            now_monotonic_ns=now_monotonic_ns,
            fallback_hold_seconds=_DEFAULT_EVENT_HOLD_S,
            fallback_accent="info",
            fallback_priority=_EVENT_CUSTOM_PRIORITY,
            id_keys=("event_name", "name", "id", "key"),
        )
        dropped_triggers, truncated_triggers = self._ingest_transient_items(
            items=trigger_ids,
            catalog=_TRIGGER_SIGNAL_MAP,
            now_utc=now_utc,
            now_monotonic_ns=now_monotonic_ns,
            fallback_hold_seconds=_DEFAULT_SAFETY_HOLD_S,
            fallback_accent="warning",
            fallback_priority=_TRIGGER_CUSTOM_PRIORITY,
            id_keys=("trigger_id", "id", "name", "key"),
        )
        return dropped_events + dropped_triggers, truncated_events or truncated_triggers

    def _ingest_transient_items(
        self,
        *,
        items: Iterable[str | Mapping[str, object] | object],
        catalog: Mapping[str, tuple[DisplayDebugSignal, float]],
        now_utc: datetime,
        now_monotonic_ns: int,
        fallback_hold_seconds: float,
        fallback_accent: str,
        fallback_priority: int,
        id_keys: tuple[str, ...],
    ) -> tuple[int, bool]:
        """Normalize, bound, and hold one batch of event/trigger inputs."""

        dropped_items = 0
        truncated = False
        seen_items = 0

        for raw_item in _as_input_items(items):
            if seen_items >= self.max_input_items:
                truncated = True
                break
            seen_items += 1
            transient = self._coerce_transient_signal_input(
                raw_item,
                catalog=catalog,
                now_utc=now_utc,
                fallback_hold_seconds=fallback_hold_seconds,
                fallback_accent=fallback_accent,
                fallback_priority=fallback_priority,
                id_keys=id_keys,
            )
            if transient is None:
                dropped_items += 1
                continue
            self._hold_transient_signal(
                transient,
                now_utc=now_utc,
                now_monotonic_ns=now_monotonic_ns,
            )
        return dropped_items, truncated

    def _coerce_transient_signal_input(
        self,
        raw_item: str | Mapping[str, object] | object,
        *,
        catalog: Mapping[str, tuple[DisplayDebugSignal, float]],
        now_utc: datetime,
        fallback_hold_seconds: float,
        fallback_accent: str,
        fallback_priority: int,
        id_keys: tuple[str, ...],
    ) -> _TransientSignalInput | None:
        """Normalize one transient input from either a legacy string or a richer mapping."""

        payload = _coerce_mapping(raw_item)
        if payload:
            if _coerce_optional_bool(payload.get("active")) is False:
                return None
            confidence = _coerce_optional_float(payload.get("confidence"))
            threshold = _coerce_optional_float(payload.get("confidence_threshold"))
            if confidence is not None and threshold is not None and confidence < threshold:
                return None

            token = ""
            for key in id_keys:
                token = _normalize_token(payload.get(key), max_chars=self.max_input_text_chars)
                if token:
                    break

            mapped = catalog.get(token)
            if mapped is not None:
                base_signal, default_hold_seconds = mapped
                hold_seconds = self._coerce_hold_seconds(
                    payload,
                    now_utc=now_utc,
                    default_hold_seconds=default_hold_seconds,
                )
                if hold_seconds is None:
                    return None
                return _TransientSignalInput(
                    signal=self._override_signal_from_payload(
                        payload,
                        base_signal=base_signal,
                    ),
                    hold_seconds=hold_seconds,
                )

            return self._build_custom_transient_signal(
                payload,
                now_utc=now_utc,
                fallback_hold_seconds=fallback_hold_seconds,
                fallback_accent=fallback_accent,
                fallback_priority=fallback_priority,
                seed_token=token,
            )

        token = _normalize_token(raw_item, max_chars=self.max_input_text_chars)
        if not token:
            return None
        mapped = catalog.get(token)
        if mapped is None:
            return None
        signal, hold_seconds = mapped
        return _TransientSignalInput(signal=signal, hold_seconds=hold_seconds)

    def _build_custom_transient_signal(
        self,
        payload: Mapping[str, object],
        *,
        now_utc: datetime,
        fallback_hold_seconds: float,
        fallback_accent: str,
        fallback_priority: int,
        seed_token: str,
    ) -> _TransientSignalInput | None:
        """Create one custom transient signal from a mapping payload."""

        key = _normalize_token(payload.get("key"), max_chars=self.max_input_text_chars) or seed_token
        label = _normalize_label(payload.get("label"), max_chars=_DEFAULT_MAX_LABEL_CHARS)
        if not key and label:
            key = _normalize_token(label, max_chars=self.max_input_text_chars)
        if key and not label:
            label = _normalize_label(key, max_chars=_DEFAULT_MAX_LABEL_CHARS)
        if not key or not label:
            return None

        hold_seconds = self._coerce_hold_seconds(
            payload,
            now_utc=now_utc,
            default_hold_seconds=fallback_hold_seconds,
        )
        if hold_seconds is None:
            return None

        return _TransientSignalInput(
            signal=DisplayDebugSignal(
                key=key,
                label=label,
                accent=_coerce_accent(payload.get("accent"), default=fallback_accent),
                priority=_coerce_priority(payload.get("priority"), default=fallback_priority),
            ),
            hold_seconds=hold_seconds,
        )

    def _override_signal_from_payload(
        self,
        payload: Mapping[str, object],
        *,
        base_signal: DisplayDebugSignal,
    ) -> DisplayDebugSignal:
        """Apply safe optional overrides to one catalog signal."""

        override_key = _normalize_token(payload.get("key"), max_chars=self.max_input_text_chars) or base_signal.key
        override_label = _normalize_label(payload.get("label"), max_chars=_DEFAULT_MAX_LABEL_CHARS) or base_signal.label
        override_accent = _coerce_accent(payload.get("accent"), default=base_signal.accent)
        override_priority = _coerce_priority(payload.get("priority"), default=base_signal.priority)
        return DisplayDebugSignal(
            key=override_key,
            label=override_label,
            accent=override_accent,
            priority=override_priority,
        )

    def _coerce_hold_seconds(
        self,
        payload: Mapping[str, object],
        *,
        now_utc: datetime,
        default_hold_seconds: float,
    ) -> float | None:
        """Resolve one hold duration from TTL-like fields or an absolute expiry."""

        expires_at = _parse_timestamp(payload.get("expires_at"))
        if expires_at is not None:
            resolved = max(0.0, (expires_at - now_utc).total_seconds())
        else:
            resolved = None
            for field_name in ("hold_seconds", "hold_s", "ttl_s", "expires_in_s"):
                parsed = _coerce_optional_float(payload.get(field_name))
                if parsed is not None:
                    resolved = parsed
                    break
            if resolved is None:
                resolved = default_hold_seconds
        clamped = _clamp_non_negative_float(
            resolved,
            maximum=self.max_custom_hold_s,
        )
        if clamped <= 0.0:
            return None
        return clamped

    def _hold_transient_signal(
        self,
        transient: _TransientSignalInput,
        *,
        now_utc: datetime,
        now_monotonic_ns: int,
    ) -> None:
        """Extend or insert one active transient hold."""

        signal = transient.signal
        expires_at_utc = now_utc + timedelta(seconds=transient.hold_seconds)
        expires_monotonic_ns = now_monotonic_ns + _seconds_to_ns(transient.hold_seconds)
        current = self._held_signals.get(signal.key)
        if current is None:
            self._held_signals[signal.key] = _HeldSignalState(
                signal=signal,
                expires_at_utc=expires_at_utc,
                expires_monotonic_ns=expires_monotonic_ns,
            )
            return

        chosen_signal = signal if signal.priority <= current.signal.priority else current.signal
        self._held_signals[signal.key] = _HeldSignalState(
            signal=chosen_signal,
            expires_at_utc=max(current.expires_at_utc, expires_at_utc),
            expires_monotonic_ns=max(current.expires_monotonic_ns, expires_monotonic_ns),
        )

    def _active_held_signals(self, *, now_monotonic_ns: int) -> tuple[DisplayDebugSignal, ...]:
        """Return the currently unexpired event/trigger debug pills."""

        return tuple(
            state.signal
            for state in self._held_signals.values()
            if state.expires_monotonic_ns > now_monotonic_ns
        )

    def _max_remaining_hold_seconds(self, *, now_monotonic_ns: int) -> float:
        """Return the longest remaining hold among active transient signals."""

        remaining = [
            max(0.0, _ns_to_seconds(state.expires_monotonic_ns - now_monotonic_ns))
            for state in self._held_signals.values()
            if state.expires_monotonic_ns > now_monotonic_ns
        ]
        if not remaining:
            return 0.0
        return max(remaining)

    def _camera_facts_are_stale(
        self,
        camera: Mapping[str, object],
        *,
        now: datetime,
    ) -> bool:
        """Return whether current-state camera facts should be suppressed as stale."""

        for key in _EXPLICIT_CAMERA_STALE_KEYS:
            if _coerce_optional_bool(camera.get(key)) is True:
                return True

        max_age_s = self.max_camera_fact_age_s
        if max_age_s is None:
            return False

        age_seconds = self._camera_fact_age_seconds(camera, now=now)
        return age_seconds is not None and age_seconds > max_age_s

    def _camera_fact_age_seconds(
        self,
        camera: Mapping[str, object],
        *,
        now: datetime,
    ) -> float | None:
        """Extract an optional age for the camera fact bundle."""

        for key in _CAMERA_AGE_KEYS:
            age = _coerce_optional_float(camera.get(key))
            if age is not None:
                return max(0.0, age)

        for key in _CAMERA_TIMESTAMP_KEYS:
            timestamp = _parse_timestamp(camera.get(key))
            if timestamp is None:
                continue
            return max(0.0, (now - timestamp).total_seconds())
        return None

    def _store_default_ttl_s(self) -> float:
        """Return the store default TTL as a safe non-negative float."""

        return _clamp_non_negative_float(getattr(self.store, "default_ttl_s", 0.0))

    def _active_snapshot_needs_refresh(
        self,
        *,
        snapshot: DisplayDebugSignalSnapshot,
        hold_seconds: float,
        now: datetime,
    ) -> bool:
        """Return whether one unchanged active snapshot should renew its TTL."""

        expires_at = _parse_timestamp(getattr(snapshot, "expires_at", None))
        if expires_at is None:
            return True
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
        ordered = tuple(
            sorted(
                deduped.values(),
                key=lambda item: (item.priority, item.label, item.key),
            )
        )
        if self.max_signals > 0:
            return ordered[: self.max_signals]
        return ordered


def _utc_now() -> datetime:
    """Return the current aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _steady_now_ns() -> int:
    """Return one suspend-aware monotonic timestamp in nanoseconds when available."""

    clock_gettime_ns = getattr(time, "clock_gettime_ns", None)
    clock_boottime = getattr(time, "CLOCK_BOOTTIME", None)
    if callable(clock_gettime_ns) and clock_boottime is not None:
        try:
            return int(clock_gettime_ns(clock_boottime))
        except OSError:
            pass
    return time.monotonic_ns()


def _coerce_mapping(value: object | None) -> Mapping[str, object]:
    """Return one mapping-like payload or an empty mapping."""

    return value if isinstance(value, Mapping) else {}


def _compact_text(value: object | None, *, max_chars: int = _DEFAULT_MAX_TEXT_CHARS) -> str:
    """Compact one optional text value and bound its size."""

    text = " ".join(("" if value is None else str(value)).split()).strip()
    if not text:
        return ""
    return text[:max_chars].rstrip()


def _normalize_text(value: object | None, *, max_chars: int = _DEFAULT_MAX_TEXT_CHARS) -> str:
    """Normalize one optional text value into compact lowercase text."""

    return _compact_text(value, max_chars=max_chars).lower()


def _normalize_token(value: object | None, *, max_chars: int = _DEFAULT_MAX_TEXT_CHARS) -> str:
    """Normalize one identifier-like value into a compact token."""

    text = _normalize_text(value, max_chars=max_chars)
    if not text:
        return ""
    return _TOKEN_SEPARATORS_RE.sub("_", text)


def _normalize_label(value: object | None, *, max_chars: int = _DEFAULT_MAX_LABEL_CHARS) -> str:
    """Normalize one label-like value into the HDMI pill style."""

    text = _compact_text(value, max_chars=max_chars).upper()
    if not text:
        return ""
    return _LABEL_SEPARATORS_RE.sub("_", text)


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


def _coerce_optional_float(value: object | None) -> float | None:
    """Parse one optional finite float-like runtime fact."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if value is None:
        return None
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _coerce_optional_int(value: object | None) -> int | None:
    """Parse one optional integer-like runtime fact."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        numeric = _coerce_optional_float(text)
        if numeric is None or not numeric.is_integer():
            return None
        return int(numeric)


def _coerce_accent(value: object | None, *, default: str) -> str:
    """Return one allowed HDMI accent token."""

    normalized = _normalize_token(value)
    if normalized in _ALLOWED_ACCENTS:
        return normalized
    return default


def _coerce_priority(value: object | None, *, default: int) -> int:
    """Parse and clamp one optional priority."""

    parsed = _coerce_optional_int(value)
    if parsed is None:
        return default
    return max(0, min(parsed, 999))


def _clamp_non_negative_float(value: object | None, *, maximum: float | None = None) -> float:
    """Return one safe non-negative float, optionally capped."""

    parsed = _coerce_optional_float(value)
    if parsed is None:
        return 0.0
    bounded = max(0.0, parsed)
    if maximum is not None:
        bounded = min(bounded, maximum)
    return bounded


def _coerce_utc_datetime(value: object | None) -> datetime | None:
    """Parse one optional datetime-like value into aware UTC time."""

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return _parse_timestamp(value)


def _parse_timestamp(value: object | None) -> datetime | None:
    """Parse one optional timestamp into aware UTC time."""

    if isinstance(value, datetime):
        return _coerce_utc_datetime(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        if abs(numeric) >= 100_000_000_000:
            numeric /= 1000.0
        try:
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = ("" if value is None else str(value)).strip()
    if not text:
        return None
    if text.isdecimal() or (text.startswith("-") and text[1:].isdecimal()):
        numeric = _coerce_optional_float(text)
        if numeric is None:
            return None
        if abs(numeric) >= 100_000_000_000:
            numeric /= 1000.0
        try:
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
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


def _as_input_items(value: Iterable[object] | Mapping[str, object] | str | object) -> Iterable[object]:
    """Treat one legacy string or mapping as a single input item."""

    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        return (value,)
    return value


def _seconds_to_ns(value: float) -> int:
    """Convert seconds to integer nanoseconds without negative output."""

    return max(0, int(round(value * 1_000_000_000)))


def _ns_to_seconds(value: int) -> float:
    """Convert nanoseconds to seconds."""

    return max(0.0, value / 1_000_000_000)


__all__ = [
    "DisplayDebugSignalPublishResult",
    "DisplayDebugSignalPublisher",
]
