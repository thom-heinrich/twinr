# CHANGELOG: 2026-03-29
# BUG-1: Switched smart-home history updates from processing-time semantics to event/snapshot-time semantics when timestamps are available, while preserving bounded fallbacks for untimestamped deltas.
# BUG-2: Unknown or never-confirmed smart-home freshness now fails closed as stale/unhealthy instead of being treated as healthy indefinitely.
# BUG-3: Active motion can now refresh from fresh snapshot/entity timestamps, preventing continuous motion from silently expiring while the stream is still live.
# BUG-4: recent_events are now ingested from merged live facts as well as incoming deltas, so the tracker works correctly even when callers only provide merged smart-home facts.
# SEC-1: Added hard caps for per-observation event ingestion, tracked entities, and identifier sizes to reduce practical CPU/memory abuse on Raspberry Pi-class edge gateways.
# SEC-2: Added timestamp plausibility guards and monotone state updates so replayed or future-dated smart-home events cannot extend occupancy windows when trustworthy event times are present.
# IMP-1: Added support for richer smart-home payloads, including snapshot timestamps, per-entity motion timestamps, mapping-form motion states, and event_type/id aliases.
# IMP-2: Exposed explicit stream freshness metadata in room/home snapshots and added tracker reset/inspection helpers for safer long-running edge runtimes.

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

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
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
_DEFAULT_MAX_EVENTS_PER_OBSERVATION = 128
_DEFAULT_MAX_TRACKED_ENTITIES = 512
_MAX_ENTITY_ID_LENGTH = 160
_MAX_EVENT_ID_LENGTH = 160
_MAX_EVENT_KIND_LENGTH = 64

_SMART_HOME_OBSERVED_AT_KEYS = (
    "stream_observed_at",
    "observed_at",
    "last_observed_at",
    "last_reported",
    "last_updated",
    "timestamp",
)
_EVENT_OBSERVED_AT_KEYS = (
    "observed_at",
    "event_time",
    "timestamp",
    "last_reported",
    "last_updated",
    "last_changed",
)
_MOTION_STATE_OBSERVED_AT_KEYS = (
    "last_motion_at",
    "observed_at",
    "timestamp",
    "last_reported",
    "last_updated",
    "last_changed",
)
_MOTION_STATE_VALUE_KEYS = (
    "active",
    "motion_detected",
    "is_active",
    "state",
    "value",
)
_EVENT_ID_KEYS = ("event_id", "id", "context_id")
_EVENT_KIND_KEYS = ("event_kind", "event_type", "type")
_EVENT_ENTITY_ID_KEYS = ("entity_id", "entity")
_STREAM_LIVE_KEYS = ("sensor_stream_live", "stream_live")
_ALARM_ACTIVE_KEYS = ("alarm_triggered", "alarm_active")
_DEVICE_OFFLINE_KEYS = ("device_offline", "bridge_offline")


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
    max_event_future_skew_s: float = 5.0
    max_events_per_observation: int = _DEFAULT_MAX_EVENTS_PER_OBSERVATION
    max_tracked_entities: int = _DEFAULT_MAX_TRACKED_ENTITIES

    def __post_init__(self) -> None:
        """Normalize one immutable smart-home context configuration."""

        object.__setattr__(
            self,
            "same_room_entity_ids",
            tuple(
                dict.fromkeys(
                    entity_id
                    for entity_id in (
                        _normalize_entity_id(value)
                        for value in self.same_room_entity_ids
                    )
                    if entity_id
                )
            ),
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
        object.__setattr__(
            self,
            "max_event_future_skew_s",
            _coerce_non_negative_seconds(self.max_event_future_skew_s, default=5.0),
        )
        object.__setattr__(
            self,
            "max_events_per_observation",
            _coerce_positive_int(
                self.max_events_per_observation,
                default=_DEFAULT_MAX_EVENTS_PER_OBSERVATION,
            ),
        )
        object.__setattr__(
            self,
            "max_tracked_entities",
            _coerce_positive_int(
                self.max_tracked_entities,
                default=_DEFAULT_MAX_TRACKED_ENTITIES,
            ),
        )

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SmartHomeContextConfig":
        """Build one runtime config snapshot from ``TwinrConfig``."""

        return cls(
            same_room_entity_ids=tuple(
                getattr(config, "smart_home_same_room_entity_ids", ()) or ()
            ),
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
            max_event_future_skew_s=float(
                getattr(config, "smart_home_max_event_future_skew_s", 5.0) or 5.0
            ),
            max_events_per_observation=int(
                getattr(
                    config,
                    "smart_home_max_events_per_observation",
                    _DEFAULT_MAX_EVENTS_PER_OBSERVATION,
                )
                or _DEFAULT_MAX_EVENTS_PER_OBSERVATION
            ),
            max_tracked_entities=int(
                getattr(
                    config,
                    "smart_home_max_tracked_entities",
                    _DEFAULT_MAX_TRACKED_ENTITIES,
                )
                or _DEFAULT_MAX_TRACKED_ENTITIES
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
    stream_observed_at: float | None = None
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
            "stream_observed_at": self.stream_observed_at,
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
    stream_stale: bool = False
    stream_observed_at: float | None = None
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
            "stream_stale": self.stream_stale,
            "stream_observed_at": self.stream_observed_at,
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

        facts = dict(coerce_mapping(live_facts))
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
    _processed_event_ids: deque[str] = field(default_factory=deque, init=False, repr=False)
    _processed_event_id_set: set[str] = field(default_factory=set, init=False, repr=False)
    _last_event_flags: dict[str, bool] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SmartHomeContextTracker":
        """Build one tracker from the shared Twinr runtime configuration."""

        return cls(config=SmartHomeContextConfig.from_config(config))

    @property
    def last_stream_observed_at(self) -> float | None:
        """Return the newest trusted smart-home snapshot timestamp."""

        return self._last_stream_observed_at

    def reset(self) -> None:
        """Drop all remembered smart-home state."""

        self._last_stream_observed_at = None
        self._last_motion_at_by_entity.clear()
        self._last_button_at_by_entity.clear()
        self._motion_active_by_entity.clear()
        self._processed_event_ids.clear()
        self._processed_event_id_set.clear()
        self._last_event_flags.clear()

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

        live_smart_home = coerce_mapping(facts.get("smart_home"))
        incoming_smart_home = coerce_mapping(incoming.get("smart_home"))
        effective_smart_home: Mapping[str, object] = live_smart_home
        if incoming_smart_home:
            merged_smart_home = dict(live_smart_home)
            merged_smart_home.update(incoming_smart_home)
            effective_smart_home = merged_smart_home

        self._ingest_smart_home_snapshot(
            now=safe_now,
            smart_home=live_smart_home,
            origin="live",
        )
        self._ingest_smart_home_snapshot(
            now=safe_now,
            smart_home=incoming_smart_home,
            origin="incoming",
        )
        self._sync_motion_state(
            now=safe_now,
            smart_home=live_smart_home,
            origin="live",
        )
        self._sync_motion_state(
            now=safe_now,
            smart_home=incoming_smart_home,
            origin="incoming",
        )

        effective_now = self._effective_observed_at(requested_now=safe_now)
        self._prune_history(now=effective_now)
        self._limit_entity_history()

        near_device_presence = _derive_near_device_presence(
            observed_at=effective_now,
            live_facts=facts,
        )
        room_context = self._derive_room_context(
            observed_at=effective_now,
            smart_home=effective_smart_home,
        )
        home_context = self._derive_home_context(
            observed_at=effective_now,
            smart_home=effective_smart_home,
            near_device_presence=near_device_presence,
            room_context=room_context,
        )
        snapshot = SmartHomeRuntimeContextSnapshot(
            observed_at=effective_now,
            near_device_presence=near_device_presence,
            room_context=room_context,
            home_context=home_context,
        )
        return SmartHomeRuntimeContextUpdate(
            snapshot=snapshot,
            event_names=self._derive_rising_edge_events(snapshot),
        )

    def _effective_observed_at(self, *, requested_now: float | None) -> float | None:
        """Return the best available reference time for recency calculations."""

        if requested_now is not None:
            return requested_now
        latest_known = self._last_stream_observed_at
        for timestamp in self._last_motion_at_by_entity.values():
            if latest_known is None or timestamp > latest_known:
                latest_known = timestamp
        for timestamp in self._last_button_at_by_entity.values():
            if latest_known is None or timestamp > latest_known:
                latest_known = timestamp
        return latest_known

    def _ingest_smart_home_snapshot(
        self,
        *,
        now: float | None,
        smart_home: Mapping[str, object],
        origin: str,
    ) -> None:
        """Update tracker history from one smart-home snapshot."""

        if not smart_home:
            return

        snapshot_observed_at = self._resolve_snapshot_observed_at(
            now=now,
            smart_home=smart_home,
            origin=origin,
        )
        if snapshot_observed_at is not None:
            self._update_last_stream_observed_at(snapshot_observed_at)

        recent_events = smart_home.get("recent_events")
        if isinstance(recent_events, (str, bytes, bytearray)) or not isinstance(recent_events, Sequence):
            return
        for index, raw_event in enumerate(recent_events):
            if index >= self.config.max_events_per_observation:
                break
            event = coerce_mapping(raw_event)
            entity_id = _first_normalized_entity_id(event, keys=_EVENT_ENTITY_ID_KEYS)
            if not entity_id:
                continue
            event_kind = _first_normalized_text(
                event,
                keys=_EVENT_KIND_KEYS,
                max_length=_MAX_EVENT_KIND_LENGTH,
            ).lower()
            if not event_kind:
                continue
            event_observed_at = self._resolve_event_observed_at(
                now=now,
                event=event,
                snapshot_observed_at=snapshot_observed_at,
                origin=origin,
            )
            if event_observed_at is None:
                continue
            event_id = self._build_event_id(
                event=event,
                event_kind=event_kind,
                entity_id=entity_id,
                event_observed_at=event_observed_at,
                origin=origin,
                index=index,
            )
            if event_id is not None and not self._remember_event_id(event_id):
                continue
            if event_kind == "motion_detected":
                self._record_motion_at(entity_id, event_observed_at)
            elif event_kind == "button_pressed":
                self._record_button_at(entity_id, event_observed_at)

    def _resolve_snapshot_observed_at(
        self,
        *,
        now: float | None,
        smart_home: Mapping[str, object],
        origin: str,
    ) -> float | None:
        """Resolve one trusted snapshot timestamp for a smart-home payload."""

        snapshot_observed_at = _first_plausible_timestamp(
            smart_home,
            keys=_SMART_HOME_OBSERVED_AT_KEYS,
            now=now,
            max_future_skew_s=self.config.max_event_future_skew_s,
        )
        if snapshot_observed_at is not None:
            return snapshot_observed_at
        if origin == "incoming":
            return now
        return None

    def _resolve_event_observed_at(
        self,
        *,
        now: float | None,
        event: Mapping[str, object],
        snapshot_observed_at: float | None,
        origin: str,
    ) -> float | None:
        """Resolve one trusted event timestamp for a smart-home event payload."""

        if _mapping_has_any_nonempty_key(event, keys=_EVENT_OBSERVED_AT_KEYS):
            event_observed_at = _first_plausible_timestamp(
                event,
                keys=_EVENT_OBSERVED_AT_KEYS,
                now=now,
                max_future_skew_s=self.config.max_event_future_skew_s,
            )
            if event_observed_at is not None:
                return event_observed_at
        if snapshot_observed_at is not None:
            return snapshot_observed_at
        if origin == "incoming":
            return now
        event_id = _first_normalized_text(event, keys=_EVENT_ID_KEYS, max_length=_MAX_EVENT_ID_LENGTH)
        if event_id and now is not None:
            return now
        return None

    def _build_event_id(
        self,
        *,
        event: Mapping[str, object],
        event_kind: str,
        entity_id: str,
        event_observed_at: float,
        origin: str,
        index: int,
    ) -> str | None:
        """Return one bounded event id for deduplication."""

        explicit_event_id = _first_normalized_text(
            event,
            keys=_EVENT_ID_KEYS,
            max_length=_MAX_EVENT_ID_LENGTH,
        )
        if explicit_event_id:
            return explicit_event_id
        event_fingerprint = _stable_event_fingerprint(event)
        if origin == "incoming":
            return f"{event_kind}:{entity_id}:{event_observed_at:.6f}:{index}:{event_fingerprint}"
        return f"{event_kind}:{entity_id}:{event_observed_at:.6f}:{event_fingerprint}"

    def _sync_motion_state(
        self,
        *,
        now: float | None,
        smart_home: Mapping[str, object],
        origin: str,
    ) -> None:
        """Track active-motion transitions without extending stale snapshots."""

        if not smart_home:
            return

        snapshot_observed_at = self._resolve_snapshot_observed_at(
            now=now,
            smart_home=smart_home,
            origin=origin,
        )
        if snapshot_observed_at is not None:
            self._update_last_stream_observed_at(snapshot_observed_at)

        motion_active_by_entity = coerce_mapping(smart_home.get("motion_active_by_entity"))
        for raw_entity_id, raw_state in motion_active_by_entity.items():
            entity_id = _normalize_entity_id(raw_entity_id)
            if not entity_id:
                continue
            active, entity_observed_at = _coerce_motion_state(
                raw_state,
                now=now,
                snapshot_observed_at=snapshot_observed_at,
                max_future_skew_s=self.config.max_event_future_skew_s,
            )
            previous_active = self._motion_active_by_entity.get(entity_id)
            if active:
                candidate_observed_at = entity_observed_at
                if candidate_observed_at is None and previous_active is not True:
                    candidate_observed_at = snapshot_observed_at or now
                if candidate_observed_at is not None:
                    self._record_motion_at(entity_id, candidate_observed_at)
            self._motion_active_by_entity[entity_id] = active

    def _update_last_stream_observed_at(self, observed_at: float) -> None:
        """Remember the newest trusted smart-home stream timestamp."""

        if self._last_stream_observed_at is None or observed_at > self._last_stream_observed_at:
            self._last_stream_observed_at = observed_at

    def _record_motion_at(self, entity_id: str, observed_at: float) -> None:
        """Remember the newest motion timestamp for one entity."""

        previous = self._last_motion_at_by_entity.get(entity_id)
        if previous is None or observed_at > previous:
            self._last_motion_at_by_entity[entity_id] = observed_at

    def _record_button_at(self, entity_id: str, observed_at: float) -> None:
        """Remember the newest button timestamp for one entity."""

        previous = self._last_button_at_by_entity.get(entity_id)
        if previous is None or observed_at > previous:
            self._last_button_at_by_entity[entity_id] = observed_at

    def _limit_entity_history(self) -> None:
        """Keep bounded per-entity history even under identifier floods."""

        all_entity_ids = (
            set(self._last_motion_at_by_entity)
            | set(self._last_button_at_by_entity)
            | set(self._motion_active_by_entity)
        )
        if len(all_entity_ids) <= self.config.max_tracked_entities:
            return

        ranked_entity_ids = sorted(
            all_entity_ids,
            key=lambda entity_id: (
                self._motion_active_by_entity.get(entity_id) is True,
                self._last_motion_at_by_entity.get(entity_id, -1.0),
                self._last_button_at_by_entity.get(entity_id, -1.0),
            ),
            reverse=True,
        )
        keep_entity_ids = set(ranked_entity_ids[: self.config.max_tracked_entities])
        self._last_motion_at_by_entity = {
            entity_id: timestamp
            for entity_id, timestamp in self._last_motion_at_by_entity.items()
            if entity_id in keep_entity_ids
        }
        self._last_button_at_by_entity = {
            entity_id: timestamp
            for entity_id, timestamp in self._last_button_at_by_entity.items()
            if entity_id in keep_entity_ids
        }
        self._motion_active_by_entity = {
            entity_id: active
            for entity_id, active in self._motion_active_by_entity.items()
            if entity_id in keep_entity_ids
        }

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
        available = configured and (bool(smart_home) or self._last_stream_observed_at is not None)
        stream_live = _first_optional_bool(smart_home, keys=_STREAM_LIVE_KEYS)
        stream_stale = _stream_is_stale(
            now=observed_at,
            last_observed_at=self._last_stream_observed_at,
            stale_after_s=self.config.stream_stale_after_s,
            treat_unknown_as_stale=available,
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
            and any(
                _age_within(
                    observed_at,
                    self._last_motion_at_by_entity.get(entity_id),
                    self.config.same_room_motion_window_s,
                )
                for entity_id in configured_ids
            )
        )
        same_room_button_recent = (
            available
            and stream_live is not False
            and not stream_stale
            and any(
                _age_within(
                    observed_at,
                    self._last_button_at_by_entity.get(entity_id),
                    self.config.same_room_button_window_s,
                )
                for entity_id in configured_ids
            )
        )
        secondary_activity_active = same_room_motion_recent or same_room_button_recent
        context_ambiguous = available and stream_live is None and not stream_stale

        if not configured:
            reason = "not_configured"
            confidence = 0.0
        elif not available:
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
        elif context_ambiguous:
            reason = "same_room_state_uncertain"
            confidence = 0.38 if matched_entity_ids else 0.28
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
            context_ambiguous=context_ambiguous,
            reason=reason,
            matched_entity_ids=matched_entity_ids,
            stream_observed_at=self._last_stream_observed_at,
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

        stream_live = _first_optional_bool(smart_home, keys=_STREAM_LIVE_KEYS)
        available = bool(smart_home) or self._last_stream_observed_at is not None
        stream_stale = _stream_is_stale(
            now=observed_at,
            last_observed_at=self._last_stream_observed_at,
            stale_after_s=self.config.stream_stale_after_s,
            treat_unknown_as_stale=available,
        )
        stream_healthy = available and stream_live is not False and not stream_stale
        same_room_ids = frozenset(self.config.same_room_entity_ids)
        other_room_motion_recent = stream_healthy and any(
            entity_id not in same_room_ids
            and _age_within(
                observed_at,
                timestamp,
                self.config.home_occupancy_window_s,
            )
            for entity_id, timestamp in self._last_motion_at_by_entity.items()
        )
        alarm_active = _first_optional_bool(smart_home, keys=_ALARM_ACTIVE_KEYS) is True
        device_offline = _first_optional_bool(smart_home, keys=_DEVICE_OFFLINE_KEYS) is True
        home_occupied_likely = (
            near_device_presence.occupied_likely
            or room_context.secondary_activity_active
            or other_room_motion_recent
            or (
                stream_healthy
                and coerce_optional_bool(smart_home.get("motion_detected")) is True
            )
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
            stream_stale=stream_stale,
            stream_observed_at=self._last_stream_observed_at,
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
            "home_context.stream_unhealthy": (
                not snapshot.home_context.stream_healthy and snapshot.home_context.available
            ),
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
            removed = self._processed_event_ids.popleft()
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


def _coerce_positive_int(value: object, *, default: int) -> int:
    """Return one finite positive integer."""

    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return default if numeric <= 0 else numeric


def _coerce_optional_monotonic(value: object | None) -> float | None:
    """Parse one optional finite monotonic-like timestamp."""

    numeric = coerce_optional_float(value)
    if numeric is None:
        return None
    if not math.isfinite(numeric):
        return None
    return max(0.0, numeric)


def _coerce_optional_time_value(
    value: object | None,
    *,
    reference_now: float | None,
) -> float | None:
    """Parse one optional runtime timestamp from numeric or ISO-8601 text input."""

    text = normalize_text(value)
    if text and any(marker in text for marker in ("T", ":", "Z", "+")):
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            timestamp = max(0.0, parsed.timestamp())
            if reference_now is not None and not _time_bases_compatible(reference_now, timestamp):
                return None
            return timestamp

    numeric = _coerce_optional_monotonic(value)
    if numeric is None:
        return None
    if reference_now is not None and not _time_bases_compatible(reference_now, numeric):
        return None
    return numeric


def _coerce_plausible_timestamp(
    value: object | None,
    *,
    now: float | None,
    max_future_skew_s: float,
) -> float | None:
    """Return one finite timestamp if it is not implausibly far in the future."""

    timestamp = _coerce_optional_time_value(value, reference_now=now)
    if timestamp is None:
        return None
    if now is not None and timestamp > (now + max(0.0, max_future_skew_s)):
        return None
    return timestamp


def _normalize_entity_id(value: object | None) -> str:
    """Normalize one entity id into compact bounded text."""

    entity_id = normalize_text(value)
    if not entity_id or len(entity_id) > _MAX_ENTITY_ID_LENGTH:
        return ""
    return entity_id


def _normalize_event_id(value: object | None) -> str:
    """Normalize one event id into bounded text."""

    event_id = normalize_text(value)
    if not event_id:
        return ""
    if len(event_id) <= _MAX_EVENT_ID_LENGTH:
        return event_id
    event_id_hash = hashlib.sha1(event_id.encode("utf-8")).hexdigest()
    return f"sha1:{event_id_hash}"


def _first_optional_bool(
    payload: Mapping[str, object],
    *,
    keys: Sequence[str],
) -> bool | None:
    """Return the first optionally-bool-like value found in one mapping."""

    for key in keys:
        value = payload.get(key)
        parsed = _coerce_optional_state_bool(value)
        if parsed is not None:
            return parsed
    return None


def _first_normalized_text(
    payload: Mapping[str, object],
    *,
    keys: Sequence[str],
    max_length: int,
) -> str:
    """Return the first bounded normalized text value found in one mapping."""

    for key in keys:
        text = normalize_text(payload.get(key))
        if not text:
            continue
        if max_length <= 0:
            return ""
        if len(text) <= max_length:
            return text
        text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"sha1:{text_hash}"
    return ""


def _first_normalized_entity_id(
    payload: Mapping[str, object],
    *,
    keys: Sequence[str],
) -> str:
    """Return the first bounded normalized entity id found in one mapping."""

    for key in keys:
        entity_id = _normalize_entity_id(payload.get(key))
        if entity_id:
            return entity_id
    return ""


def _first_plausible_timestamp(
    payload: Mapping[str, object],
    *,
    keys: Sequence[str],
    now: float | None,
    max_future_skew_s: float,
) -> float | None:
    """Return the first plausible timestamp found in one mapping."""

    for key in keys:
        timestamp = _coerce_plausible_timestamp(
            payload.get(key),
            now=now,
            max_future_skew_s=max_future_skew_s,
        )
        if timestamp is not None:
            return timestamp
    return None


def _mapping_has_any_nonempty_key(
    payload: Mapping[str, object],
    *,
    keys: Sequence[str],
) -> bool:
    """Return whether one mapping explicitly carries any non-empty value for the requested keys."""

    for key in keys:
        if normalize_text(payload.get(key)):
            return True
    return False


def _coerce_optional_state_bool(value: object | None) -> bool | None:
    """Parse one optional bool value, including common Home Assistant-like strings."""

    parsed = coerce_optional_bool(value)
    if parsed is not None:
        return parsed
    text = normalize_text(value).lower()
    if not text:
        return None
    if text in {"on", "true", "active", "open", "detected", "motion"}:
        return True
    if text in {"off", "false", "inactive", "closed", "clear", "idle"}:
        return False
    return None


def _coerce_motion_state(
    value: object,
    *,
    now: float | None,
    snapshot_observed_at: float | None,
    max_future_skew_s: float,
) -> tuple[bool, float | None]:
    """Return one normalized motion-active flag plus its freshest timestamp."""

    if isinstance(value, Mapping):
        payload = coerce_mapping(value)
        active: bool | None = None
        for key in _MOTION_STATE_VALUE_KEYS:
            active = _coerce_optional_state_bool(payload.get(key))
            if active is not None:
                break
        entity_observed_at = _first_plausible_timestamp(
            payload,
            keys=_MOTION_STATE_OBSERVED_AT_KEYS,
            now=now,
            max_future_skew_s=max_future_skew_s,
        )
        return (active is True), (entity_observed_at or snapshot_observed_at)
    return (_coerce_optional_state_bool(value) is True), snapshot_observed_at


def _stable_event_fingerprint(event: Mapping[str, object]) -> str:
    """Return one bounded stable fingerprint for an event mapping."""

    if not event:
        return "empty"
    fragments: list[str] = []
    for key in sorted(str(name) for name in event.keys()):
        fragments.append(key)
        fragments.append("=")
        fragments.append(normalize_text(event.get(key)) or repr(event.get(key)))
        fragments.append("|")
    raw_fingerprint = "".join(fragments)
    return hashlib.sha1(raw_fingerprint.encode("utf-8")).hexdigest()


def _time_bases_compatible(left: float, right: float) -> bool:
    """Return whether two timestamps likely share the same time basis."""

    left_looks_like_epoch = left >= 100_000_000.0
    right_looks_like_epoch = right >= 100_000_000.0
    return left_looks_like_epoch == right_looks_like_epoch


def _age_within(now: float | None, since: float | None, window_s: float) -> bool:
    """Return whether one tracked timestamp remains inside the requested window."""

    if now is None or since is None:
        return False
    if not _time_bases_compatible(now, since):
        return False
    return max(0.0, now - since) <= max(0.0, window_s)


def _stream_is_stale(
    *,
    now: float | None,
    last_observed_at: float | None,
    stale_after_s: float,
    treat_unknown_as_stale: bool,
) -> bool:
    """Return whether the smart-home stream freshness window has expired."""

    if now is None:
        return False
    if last_observed_at is None:
        return treat_unknown_as_stale
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
