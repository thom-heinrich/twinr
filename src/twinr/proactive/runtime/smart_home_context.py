"""Derive layered smart-home-aware runtime context from live facts.

This module keeps optional smart-home context interpretation out of the large
workflow/background files. It consumes merged live fact maps and maintains a
small bounded state so Twinr can expose three explicit runtime layers:

- ``near_device_presence`` for local PIR/camera/audio interaction truth
- ``room_context`` for optional explicitly configured same-room smart-home support
- ``home_context`` for broader home occupancy, alarm, and stream-health context

The tracker is intentionally conservative:

- same-room semantics require explicit entity-id configuration
- stale or unhealthy smart-home streams fail closed
- smart-home never manufactures local ``person_visible`` or voice-activation arming
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.claim_metadata import (
    RuntimeClaimMetadata,
    coerce_mapping,
    coerce_optional_bool,
    coerce_optional_float,
    mean_confidence,
    normalize_text,
)


_MAX_TRACKED_EVENT_IDS = 256


def _default_local_claim() -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(
        confidence=0.0,
        source="local_runtime_facts",
        requires_confirmation=False,
    )


def _default_room_claim() -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(
        confidence=0.0,
        source="smart_home_room_context",
        requires_confirmation=False,
    )


def _default_home_claim() -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(
        confidence=0.0,
        source="smart_home_home_context",
        requires_confirmation=False,
    )


@dataclass(frozen=True, slots=True)
class SmartHomeContextConfig:
    """Configure how optional smart-home facts map into runtime context."""

    same_room_entity_ids: tuple[str, ...] = ()
    same_room_motion_window_s: float = 90.0
    same_room_button_window_s: float = 30.0
    home_occupancy_window_s: float = 300.0
    stream_stale_after_s: float = 120.0

    def __post_init__(self) -> None:
        """Normalize one immutable smart-home context configuration."""

        object.__setattr__(
            self,
            "same_room_entity_ids",
            tuple(dict.fromkeys(_normalize_entity_id(value) for value in self.same_room_entity_ids if _normalize_entity_id(value))),
        )
        object.__setattr__(
            self,
            "same_room_motion_window_s",
            _coerce_non_negative_seconds(self.same_room_motion_window_s, default=90.0),
        )
        object.__setattr__(
            self,
            "same_room_button_window_s",
            _coerce_non_negative_seconds(self.same_room_button_window_s, default=30.0),
        )
        object.__setattr__(
            self,
            "home_occupancy_window_s",
            _coerce_non_negative_seconds(self.home_occupancy_window_s, default=300.0),
        )
        object.__setattr__(
            self,
            "stream_stale_after_s",
            _coerce_non_negative_seconds(self.stream_stale_after_s, default=120.0),
        )

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SmartHomeContextConfig":
        """Build one runtime config snapshot from ``TwinrConfig``."""

        return cls(
            same_room_entity_ids=tuple(getattr(config, "smart_home_same_room_entity_ids", ()) or ()),
            same_room_motion_window_s=float(
                getattr(config, "smart_home_same_room_motion_window_s", 90.0) or 90.0
            ),
            same_room_button_window_s=float(
                getattr(config, "smart_home_same_room_button_window_s", 30.0) or 30.0
            ),
            home_occupancy_window_s=float(
                getattr(config, "smart_home_home_occupancy_window_s", 300.0) or 300.0
            ),
            stream_stale_after_s=float(
                getattr(config, "smart_home_stream_stale_after_s", 120.0) or 120.0
            ),
        )


@dataclass(frozen=True, slots=True)
class NearDevicePresenceSnapshot:
    """Describe local interaction truth near the Twinr device."""

    observed_at: float | None = None
    occupied_likely: bool = False
    person_visible: bool = False
    person_recently_visible: bool = False
    room_motion_recent: bool = False
    speech_recent: bool = False
    voice_activation_armed: bool = False
    reason: str | None = None
    claim: RuntimeClaimMetadata = field(default_factory=_default_local_claim)

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the local presence layer into runtime facts."""

        payload = {
            "observed_at": self.observed_at,
            "occupied_likely": self.occupied_likely,
            "person_visible": self.person_visible,
            "person_recently_visible": self.person_recently_visible,
            "room_motion_recent": self.room_motion_recent,
            "speech_recent": self.speech_recent,
            "voice_activation_armed": self.voice_activation_armed,
            "reason": self.reason,
        }
        payload.update(self.claim.to_payload())
        return payload


@dataclass(frozen=True, slots=True)
class RoomContextSnapshot:
    """Describe optional explicitly configured same-room smart-home context."""

    observed_at: float | None = None
    configured: bool = False
    available: bool = False
    same_room_motion_recent: bool = False
    same_room_button_recent: bool = False
    secondary_activity_active: bool = False
    sensor_stale: bool = False
    context_ambiguous: bool = False
    reason: str | None = None
    matched_entity_ids: tuple[str, ...] = ()
    claim: RuntimeClaimMetadata = field(default_factory=_default_room_claim)

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the same-room context layer into runtime facts."""

        payload = {
            "observed_at": self.observed_at,
            "configured": self.configured,
            "available": self.available,
            "same_room_motion_recent": self.same_room_motion_recent,
            "same_room_button_recent": self.same_room_button_recent,
            "secondary_activity_active": self.secondary_activity_active,
            "sensor_stale": self.sensor_stale,
            "context_ambiguous": self.context_ambiguous,
            "reason": self.reason,
            "matched_entity_ids": list(self.matched_entity_ids),
        }
        payload.update(self.claim.to_payload())
        return payload


@dataclass(frozen=True, slots=True)
class HomeContextSnapshot:
    """Describe broader smart-home context for the surrounding home."""

    observed_at: float | None = None
    available: bool = False
    home_occupied_likely: bool = False
    other_room_motion_recent: bool = False
    alarm_active: bool = False
    device_offline: bool = False
    stream_live: bool | None = None
    stream_healthy: bool = False
    reason: str | None = None
    claim: RuntimeClaimMetadata = field(default_factory=_default_home_claim)

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the home-wide context layer into runtime facts."""

        payload = {
            "observed_at": self.observed_at,
            "available": self.available,
            "home_occupied_likely": self.home_occupied_likely,
            "other_room_motion_recent": self.other_room_motion_recent,
            "alarm_active": self.alarm_active,
            "device_offline": self.device_offline,
            "stream_live": self.stream_live,
            "stream_healthy": self.stream_healthy,
            "reason": self.reason,
        }
        payload.update(self.claim.to_payload())
        return payload


@dataclass(frozen=True, slots=True)
class SmartHomeRuntimeContextSnapshot:
    """Bundle the three runtime context layers into one immutable snapshot."""

    observed_at: float | None
    near_device_presence: NearDevicePresenceSnapshot
    room_context: RoomContextSnapshot
    home_context: HomeContextSnapshot

    def apply_to_facts(self, live_facts: Mapping[str, object] | object) -> dict[str, object]:
        """Return a cloned fact map augmented with the three context layers."""

        facts = coerce_mapping(live_facts)
        facts["near_device_presence"] = self.near_device_presence.to_automation_facts()
        facts["room_context"] = self.room_context.to_automation_facts()
        facts["home_context"] = self.home_context.to_automation_facts()
        return facts


@dataclass(frozen=True, slots=True)
class SmartHomeRuntimeContextUpdate:
    """Return one context snapshot plus the rising-edge event names it activated."""

    snapshot: SmartHomeRuntimeContextSnapshot
    event_names: tuple[str, ...] = ()


@dataclass(slots=True)
class SmartHomeContextTracker:
    """Track bounded smart-home context state across live observation updates."""

    config: SmartHomeContextConfig
    _last_stream_observed_at: float | None = field(default=None, init=False, repr=False)
    _last_motion_at_by_entity: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_button_at_by_entity: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _motion_active_by_entity: dict[str, bool] = field(default_factory=dict, init=False, repr=False)
    _processed_event_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _processed_event_id_set: set[str] = field(default_factory=set, init=False, repr=False)
    _last_event_flags: dict[str, bool] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SmartHomeContextTracker":
        """Build one tracker from the shared Twinr runtime configuration."""

        return cls(config=SmartHomeContextConfig.from_config(config))

    def observe(
        self,
        *,
        observed_at: float | None,
        live_facts: Mapping[str, object] | object,
        incoming_facts: Mapping[str, object] | object | None = None,
    ) -> SmartHomeRuntimeContextUpdate:
        """Update internal state and return the latest layered runtime context."""

        safe_now = _coerce_optional_monotonic(observed_at)
        facts = coerce_mapping(live_facts)
        incoming = coerce_mapping(incoming_facts)
        smart_home = coerce_mapping(facts.get("smart_home"))

        self._ingest_incoming_smart_home(now=safe_now, incoming_facts=incoming)
        self._sync_motion_state(now=safe_now, smart_home=smart_home)
        self._prune_history(now=safe_now)

        near_device_presence = _derive_near_device_presence(
            observed_at=safe_now,
            live_facts=facts,
        )
        room_context = self._derive_room_context(
            observed_at=safe_now,
            smart_home=smart_home,
        )
        home_context = self._derive_home_context(
            observed_at=safe_now,
            smart_home=smart_home,
            near_device_presence=near_device_presence,
            room_context=room_context,
        )
        snapshot = SmartHomeRuntimeContextSnapshot(
            observed_at=safe_now,
            near_device_presence=near_device_presence,
            room_context=room_context,
            home_context=home_context,
        )
        return SmartHomeRuntimeContextUpdate(
            snapshot=snapshot,
            event_names=self._derive_rising_edge_events(snapshot),
        )

    def _ingest_incoming_smart_home(
        self,
        *,
        now: float | None,
        incoming_facts: Mapping[str, object],
    ) -> None:
        """Update tracker history only when a new smart-home observation arrived."""

        smart_home = coerce_mapping(incoming_facts.get("smart_home"))
        if not smart_home or now is None:
            return

        self._last_stream_observed_at = now
        recent_events = smart_home.get("recent_events")
        if isinstance(recent_events, (str, bytes, bytearray)) or not isinstance(recent_events, Sequence):
            return
        for raw_event in recent_events:
            event = coerce_mapping(raw_event)
            entity_id = _normalize_entity_id(event.get("entity_id"))
            if not entity_id:
                continue
            event_kind = normalize_text(event.get("event_kind")).lower()
            event_id = normalize_text(event.get("event_id")) or f"{event_kind}:{entity_id}:{normalize_text(event.get('observed_at'))}"
            if not event_kind or not self._remember_event_id(event_id):
                continue
            if event_kind == "motion_detected":
                self._last_motion_at_by_entity[entity_id] = now
            elif event_kind == "button_pressed":
                self._last_button_at_by_entity[entity_id] = now

    def _sync_motion_state(
        self,
        *,
        now: float | None,
        smart_home: Mapping[str, object],
    ) -> None:
        """Track active-motion transitions without extending them forever."""

        motion_active_by_entity = coerce_mapping(smart_home.get("motion_active_by_entity"))
        for raw_entity_id, raw_active in motion_active_by_entity.items():
            entity_id = _normalize_entity_id(raw_entity_id)
            if not entity_id:
                continue
            active = coerce_optional_bool(raw_active) is True
            previous_active = self._motion_active_by_entity.get(entity_id)
            if active and previous_active is not True and now is not None:
                self._last_motion_at_by_entity[entity_id] = now
            self._motion_active_by_entity[entity_id] = active

    def _prune_history(self, *, now: float | None) -> None:
        """Keep the small bounded motion/button history from growing forever."""

        if now is None:
            return
        max_age = max(
            self.config.same_room_motion_window_s,
            self.config.same_room_button_window_s,
            self.config.home_occupancy_window_s,
            self.config.stream_stale_after_s,
            60.0,
        ) * 2.0
        self._last_motion_at_by_entity = {
            entity_id: timestamp
            for entity_id, timestamp in self._last_motion_at_by_entity.items()
            if (now - timestamp) <= max_age
        }
        self._last_button_at_by_entity = {
            entity_id: timestamp
            for entity_id, timestamp in self._last_button_at_by_entity.items()
            if (now - timestamp) <= max_age
        }
        self._motion_active_by_entity = {
            entity_id: active
            for entity_id, active in self._motion_active_by_entity.items()
            if active or entity_id in self._last_motion_at_by_entity
        }

    def _derive_room_context(
        self,
        *,
        observed_at: float | None,
        smart_home: Mapping[str, object],
    ) -> RoomContextSnapshot:
        """Return the current same-room smart-home context snapshot."""

        configured_ids = self.config.same_room_entity_ids
        configured = bool(configured_ids)
        available = configured and bool(smart_home)
        stream_live = coerce_optional_bool(smart_home.get("sensor_stream_live"))
        stream_stale = _stream_is_stale(
            now=observed_at,
            last_observed_at=self._last_stream_observed_at,
            stale_after_s=self.config.stream_stale_after_s,
        )
        matched_entity_ids = tuple(
            entity_id
            for entity_id in configured_ids
            if entity_id in self._motion_active_by_entity
            or entity_id in self._last_motion_at_by_entity
            or entity_id in self._last_button_at_by_entity
        )
        same_room_motion_recent = (
            available
            and stream_live is not False
            and not stream_stale
            and any(_age_within(observed_at, self._last_motion_at_by_entity.get(entity_id), self.config.same_room_motion_window_s) for entity_id in configured_ids)
        )
        same_room_button_recent = (
            available
            and stream_live is not False
            and not stream_stale
            and any(_age_within(observed_at, self._last_button_at_by_entity.get(entity_id), self.config.same_room_button_window_s) for entity_id in configured_ids)
        )
        secondary_activity_active = same_room_motion_recent or same_room_button_recent

        if not configured:
            reason = "not_configured"
            confidence = 0.0
        elif not smart_home:
            reason = "smart_home_unavailable"
            confidence = 0.0
        elif stream_live is False or stream_stale:
            reason = "stream_stale"
            confidence = 0.18
        elif same_room_motion_recent:
            reason = "same_room_motion_recent"
            confidence = 0.84
        elif same_room_button_recent:
            reason = "same_room_button_recent"
            confidence = 0.78
        else:
            reason = "same_room_quiet"
            confidence = 0.46 if matched_entity_ids else 0.32

        return RoomContextSnapshot(
            observed_at=observed_at,
            configured=configured,
            available=available,
            same_room_motion_recent=same_room_motion_recent,
            same_room_button_recent=same_room_button_recent,
            secondary_activity_active=secondary_activity_active,
            sensor_stale=available and (stream_live is False or stream_stale),
            context_ambiguous=False,
            reason=reason,
            matched_entity_ids=matched_entity_ids,
            claim=RuntimeClaimMetadata(
                confidence=confidence,
                source="smart_home_room_context",
                requires_confirmation=False,
            ),
        )

    def _derive_home_context(
        self,
        *,
        observed_at: float | None,
        smart_home: Mapping[str, object],
        near_device_presence: NearDevicePresenceSnapshot,
        room_context: RoomContextSnapshot,
    ) -> HomeContextSnapshot:
        """Return the current home-wide smart-home context snapshot."""

        stream_live = coerce_optional_bool(smart_home.get("sensor_stream_live"))
        stream_stale = _stream_is_stale(
            now=observed_at,
            last_observed_at=self._last_stream_observed_at,
            stale_after_s=self.config.stream_stale_after_s,
        )
        available = bool(smart_home)
        stream_healthy = available and stream_live is not False and not stream_stale
        same_room_ids = frozenset(self.config.same_room_entity_ids)
        other_room_motion_recent = stream_healthy and any(
            entity_id not in same_room_ids
            and _age_within(observed_at, timestamp, self.config.home_occupancy_window_s)
            for entity_id, timestamp in self._last_motion_at_by_entity.items()
        )
        alarm_active = coerce_optional_bool(smart_home.get("alarm_triggered")) is True
        device_offline = coerce_optional_bool(smart_home.get("device_offline")) is True
        home_occupied_likely = (
            near_device_presence.occupied_likely
            or room_context.secondary_activity_active
            or other_room_motion_recent
            or (stream_healthy and coerce_optional_bool(smart_home.get("motion_detected")) is True)
        )

        if not available:
            reason = "smart_home_unavailable"
            confidence = 0.0
        elif alarm_active:
            reason = "alarm_active"
            confidence = 0.95
        elif not stream_healthy:
            reason = "stream_unhealthy"
            confidence = 0.22
        elif other_room_motion_recent:
            reason = "other_room_motion_recent"
            confidence = 0.82
        elif home_occupied_likely:
            reason = "near_device_presence_active"
            confidence = 0.74
        elif device_offline:
            reason = "device_offline"
            confidence = 0.56
        else:
            reason = "home_quiet"
            confidence = 0.48

        return HomeContextSnapshot(
            observed_at=observed_at,
            available=available,
            home_occupied_likely=home_occupied_likely,
            other_room_motion_recent=other_room_motion_recent,
            alarm_active=alarm_active,
            device_offline=device_offline,
            stream_live=stream_live,
            stream_healthy=stream_healthy,
            reason=reason,
            claim=RuntimeClaimMetadata(
                confidence=confidence,
                source="smart_home_home_context",
                requires_confirmation=False,
            ),
        )

    def _derive_rising_edge_events(
        self,
        snapshot: SmartHomeRuntimeContextSnapshot,
    ) -> tuple[str, ...]:
        """Return bounded rising-edge events from the latest context snapshot."""

        current_flags = {
            "room_context.same_room_motion_recent": snapshot.room_context.same_room_motion_recent,
            "room_context.same_room_button_recent": snapshot.room_context.same_room_button_recent,
            "room_context.sensor_stale": snapshot.room_context.sensor_stale,
            "home_context.other_room_motion_recent": snapshot.home_context.other_room_motion_recent,
            "home_context.alarm_active": snapshot.home_context.alarm_active,
            "home_context.device_offline": snapshot.home_context.device_offline,
            "home_context.stream_unhealthy": not snapshot.home_context.stream_healthy and snapshot.home_context.available,
        }
        event_names: list[str] = []
        for event_name, active in current_flags.items():
            if active and self._last_event_flags.get(event_name) is not True:
                event_names.append(event_name)
        self._last_event_flags = current_flags
        return tuple(event_names)

    def _remember_event_id(self, event_id: str) -> bool:
        """Return whether one event id is new to the bounded tracker history."""

        if event_id in self._processed_event_id_set:
            return False
        self._processed_event_ids.append(event_id)
        self._processed_event_id_set.add(event_id)
        while len(self._processed_event_ids) > _MAX_TRACKED_EVENT_IDS:
            removed = self._processed_event_ids.pop(0)
            self._processed_event_id_set.discard(removed)
        return True


def _derive_near_device_presence(
    *,
    observed_at: float | None,
    live_facts: Mapping[str, object],
) -> NearDevicePresenceSnapshot:
    """Return the current local near-device presence snapshot."""

    camera = coerce_mapping(live_facts.get("camera"))
    pir = coerce_mapping(live_facts.get("pir"))
    vad = coerce_mapping(live_facts.get("vad"))
    audio_policy = coerce_mapping(live_facts.get("audio_policy"))
    sensor = coerce_mapping(live_facts.get("sensor"))

    person_visible = coerce_optional_bool(camera.get("person_visible")) is True
    person_visible_for_s = _coerce_non_negative_float(camera.get("person_visible_for_s"))
    voice_activation_armed = coerce_optional_bool(sensor.get("voice_activation_armed")) is True
    voice_activation_presence_reason = normalize_text(sensor.get("voice_activation_presence_reason")) or None
    room_motion_recent = (
        coerce_optional_bool(pir.get("motion_detected")) is True
        or voice_activation_presence_reason in {"pir_motion", "recent_pir_motion"}
    )
    person_recently_visible = (
        person_visible
        or (person_visible_for_s is not None and person_visible_for_s > 0.0)
        or voice_activation_presence_reason in {"person_visible", "recent_person_visible"}
    )
    speech_recent = any(
        (
            coerce_optional_bool(vad.get("speech_detected")) is True,
            coerce_optional_bool(audio_policy.get("presence_audio_active")) is True,
            coerce_optional_bool(audio_policy.get("recent_follow_up_speech")) is True,
            voice_activation_presence_reason in {"speech_while_recently_present", "recent_speech_while_present"},
        )
    )
    occupied_likely = any(
        (
            person_visible,
            person_recently_visible,
            room_motion_recent,
            speech_recent,
            voice_activation_armed,
        )
    )

    if person_visible:
        reason = "person_visible"
    elif person_recently_visible:
        reason = "recent_person_visible"
    elif room_motion_recent:
        reason = "room_motion_recent"
    elif speech_recent:
        reason = "speech_recent"
    elif voice_activation_armed:
        reason = "voice_activation_armed"
    else:
        reason = None

    confidence = mean_confidence(
        tuple(
            value
            for value in (
                0.94 if person_visible else None,
                0.86 if person_recently_visible and not person_visible else None,
                0.76 if room_motion_recent else None,
                0.72 if speech_recent else None,
                0.78 if voice_activation_armed else None,
            )
        )
    )
    return NearDevicePresenceSnapshot(
        observed_at=observed_at,
        occupied_likely=occupied_likely,
        person_visible=person_visible,
        person_recently_visible=person_recently_visible,
        room_motion_recent=room_motion_recent,
        speech_recent=speech_recent,
        voice_activation_armed=voice_activation_armed,
        reason=reason,
        claim=RuntimeClaimMetadata(
            confidence=(0.0 if confidence is None else confidence),
            source="local_runtime_facts",
            requires_confirmation=False,
        ),
    )


def _coerce_non_negative_seconds(value: object, *, default: float) -> float:
    """Return one finite non-negative duration."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(0.0, number)


def _coerce_non_negative_float(value: object | None) -> float | None:
    """Parse one optional non-negative float value."""

    numeric = coerce_optional_float(value)
    if numeric is None:
        return None
    return max(0.0, numeric)


def _coerce_optional_monotonic(value: object | None) -> float | None:
    """Parse one optional finite monotonic-like timestamp."""

    numeric = coerce_optional_float(value)
    if numeric is None:
        return None
    return max(0.0, numeric)


def _normalize_entity_id(value: object | None) -> str:
    """Normalize one entity id into compact text."""

    return normalize_text(value)


def _age_within(now: float | None, since: float | None, window_s: float) -> bool:
    """Return whether one tracked timestamp remains inside the requested window."""

    if now is None or since is None:
        return False
    return max(0.0, now - since) <= max(0.0, window_s)


def _stream_is_stale(
    *,
    now: float | None,
    last_observed_at: float | None,
    stale_after_s: float,
) -> bool:
    """Return whether the smart-home stream freshness window has expired."""

    if now is None or last_observed_at is None:
        return False
    return max(0.0, now - last_observed_at) > max(0.0, stale_after_s)


__all__ = [
    "HomeContextSnapshot",
    "NearDevicePresenceSnapshot",
    "RoomContextSnapshot",
    "SmartHomeContextConfig",
    "SmartHomeContextTracker",
    "SmartHomeRuntimeContextSnapshot",
    "SmartHomeRuntimeContextUpdate",
]
