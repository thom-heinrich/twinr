# CHANGELOG: 2026-03-28
# BUG-1: Fixed recency merge bug where older proactive observations could overwrite newer runtime-alert facts.
# BUG-2: Fixed observation detection bug that ignored valid ReSpeaker observations unless two narrow keys were present.
# BUG-3: Fixed silent parsing loss for common serialized bool/int/float forms and rejected non-finite / nonsensical numeric values.
# BUG-4: Fixed fail-open stale-telemetry behavior; transient listening/speech/direction/noise states now expire when observations stop updating.
# SEC-1: Sanitized operator-visible text against control / bidi characters to block display and terminal spoofing.
# SEC-2: Bounded JSON, text, and numeric parsing to avoid UI stalls from oversized payload fields.
# IMP-1: Added per-field recency merge, timestamp age tracking, and explicit stale-state surfacing.
# IMP-2: Made cache access thread-safe and refresh immediately when no cached state exists.
# IMP-3: Added alias-aware field extraction and bounded JSON-string payload decoding for schema evolution.

"""Surface calm ReSpeaker HCI state from authoritative ops events.

The display loop should not parse raw proactive event payloads inline. This
module reads the latest persisted ReSpeaker-related ops facts and condenses
them into calm operator-visible states such as muted microphone, blocked room
audio, or DFU mode.

2026 upgrade notes:
- Merge overlapping observation/alert fields by source recency instead of
  blindly preferring observation values.
- Treat live/transient cues as stale once authoritative observations stop
  updating.
- Sanitize operator-visible text before it reaches displays or terminal logs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import threading
import time
import unicodedata

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.respeaker.indicator_policy import resolve_respeaker_indicator_state
from twinr.ops.events import TwinrOpsEventStore, compact_text


_DEFAULT_EVENT_TAIL_LIMIT = 128
_DEFAULT_REFRESH_INTERVAL_S = 1.0
_DEFAULT_OBSERVATION_STALE_AFTER_S = 12.0
_DIRECTION_HINT_MIN_CONFIDENCE = 0.75
_RECENT_SPEECH_WINDOW_S = 1.5

_MAX_DISPLAY_TEXT_LEN = 160
_MAX_SAFE_TEXT_SCAN_LEN = 2048
_MAX_SAFE_JSON_BYTES = 8192
_MAX_NUMERIC_TEXT_LEN = 64

_OBSERVATION_EVENT_NAMES = frozenset({"proactive_observation", "proactive.observation"})
_ALERT_EVENT_NAMES = frozenset({"respeaker_runtime_alert", "respeaker.runtime_alert"})
_NESTED_PAYLOAD_KEYS = ("respeaker", "audio", "state", "observation", "snapshot", "facts")

_MISSING = object()
_INVALID = object()

_OBSERVATION_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "runtime_status": ("runtime_status", "respeaker_runtime_status"),
    "runtime_alert_code": ("respeaker_runtime_alert_code", "runtime_alert_code", "alert_code"),
    "device_runtime_mode": ("audio_device_runtime_mode", "device_runtime_mode", "runtime_mode"),
    "host_control_ready": ("audio_host_control_ready", "host_control_ready"),
    "transport_reason": ("audio_transport_reason", "transport_reason"),
    "mute_active": ("audio_mute_active", "mute_active"),
    "speech_detected": ("speech_detected", "audio_speech_detected"),
    "recent_speech_age_s": ("audio_recent_speech_age_s", "recent_speech_age_s"),
    "direction_confidence": ("audio_direction_confidence", "direction_confidence"),
    "azimuth_deg": ("audio_azimuth_deg", "azimuth_deg"),
    "room_busy_or_overlapping": ("room_busy_or_overlapping",),
    "quiet_window_open": ("quiet_window_open",),
    "non_speech_audio_likely": ("non_speech_audio_likely",),
    "background_media_likely": ("background_media_likely",),
    "resume_window_open": ("resume_window_open",),
    "voice_activation_armed": ("voice_activation_armed",),
    "initiative_block_reason": ("audio_initiative_block_reason", "initiative_block_reason"),
    "speech_delivery_defer_reason": ("audio_speech_delivery_defer_reason", "speech_delivery_defer_reason"),
}

_ALERT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "runtime_alert_code": ("alert_code", "runtime_alert_code", "respeaker_runtime_alert_code"),
    "device_runtime_mode": ("device_runtime_mode", "audio_device_runtime_mode", "runtime_mode"),
    "host_control_ready": ("host_control_ready", "audio_host_control_ready"),
    "transport_reason": ("transport_reason", "audio_transport_reason"),
    "mute_active": ("mute_active", "audio_mute_active"),
}

_ALERT_MESSAGE_ALIASES = ("alert_message", "message", "detail", "reason")

_RELEVANT_OBSERVATION_KEYS = frozenset(
    alias
    for aliases in _OBSERVATION_FIELD_ALIASES.values()
    for alias in aliases
)
_RELEVANT_ALERT_KEYS = frozenset(
    alias
    for aliases in _ALERT_FIELD_ALIASES.values()
    for alias in aliases
).union(_ALERT_MESSAGE_ALIASES)


@dataclass(frozen=True, slots=True)
class DisplayReSpeakerHciState:
    """Describe the latest operator-visible ReSpeaker runtime state."""

    observed_at: str | None = None
    runtime_status: str | None = None
    runtime_alert_code: str | None = None
    runtime_alert_message: str | None = None
    device_runtime_mode: str | None = None
    host_control_ready: bool | None = None
    transport_reason: str | None = None
    mute_active: bool | None = None
    speech_detected: bool | None = None
    recent_speech_age_s: float | None = None
    direction_confidence: float | None = None
    azimuth_deg: int | None = None
    room_busy_or_overlapping: bool | None = None
    quiet_window_open: bool | None = None
    non_speech_audio_likely: bool | None = None
    background_media_likely: bool | None = None
    resume_window_open: bool | None = None
    voice_activation_armed: bool | None = None
    initiative_block_reason: str | None = None
    speech_delivery_defer_reason: str | None = None
    observation_age_s: float | None = None
    snapshot_age_s: float | None = None

    @property
    def telemetry_stale(self) -> bool:
        """Return whether the live observation stream has gone stale."""

        return (
            self.observation_age_s is not None
            and self.observation_age_s > _DEFAULT_OBSERVATION_STALE_AFTER_S
        )

    @property
    def direction_hint_available(self) -> bool:
        """Return whether the latest azimuth hint is operator-usable."""

        return (
            not self.telemetry_stale
            and self.azimuth_deg is not None
            and self.direction_confidence is not None
            and self.direction_confidence >= _DIRECTION_HINT_MIN_CONFIDENCE
        )

    @property
    def listening(self) -> bool:
        """Return whether Twinr is explicitly in a live listening state."""

        return not self.telemetry_stale and self.runtime_status == "listening"

    @property
    def heard_speech(self) -> bool:
        """Return whether the ReSpeaker has a calm recent-speech signal."""

        if self.telemetry_stale:
            return False
        if self.speech_detected is True:
            return True
        if self.recent_speech_age_s is None:
            return False
        return self.recent_speech_age_s <= _RECENT_SPEECH_WINDOW_S

    @property
    def led_ring_mode(self) -> str:
        """Return the shared LED-ring mode contract for the current state."""

        return resolve_respeaker_indicator_state(
            runtime_status=self.runtime_status,
            runtime_alert_code=self.runtime_alert_code,
            mute_active=self.mute_active,
        ).mode

    @property
    def led_ring_semantics(self) -> str:
        """Return the current LED-ring semantics contract."""

        return resolve_respeaker_indicator_state(
            runtime_status=self.runtime_status,
            runtime_alert_code=self.runtime_alert_code,
            mute_active=self.mute_active,
        ).semantics

    @property
    def noise_blocked(self) -> bool:
        """Return whether current room audio blocks calm voice interaction."""

        if self.telemetry_stale:
            return False
        return (
            self.room_busy_or_overlapping is True
            or self.non_speech_audio_likely is True
            or self.background_media_likely is True
            or self.speech_delivery_defer_reason in {"background_media_active", "non_speech_audio_active"}
            or self.initiative_block_reason == "room_busy_or_overlapping"
        )

    def state_fields(self) -> tuple[tuple[str, str], ...]:
        """Render compact status-card fields for the general display surface."""

        fields: list[tuple[str, str]] = []
        respeaker_value = _respeaker_field_value(self)
        if respeaker_value:
            fields.append(("ReSpeaker", respeaker_value))
        mic_value = _microphone_field_value(self)
        if mic_value:
            fields.append(("Mikrofon", mic_value))
        audio_value = _audio_field_value(self)
        if audio_value:
            fields.append(("Audio", audio_value))
        direction_value = _direction_field_value(self)
        if direction_value:
            fields.append(("Richtung", direction_value))
        ring_value = _ring_field_value(self)
        if ring_value:
            fields.append(("Ring", ring_value))
        return tuple(fields)

    def hardware_log_lines(self) -> tuple[str, ...]:
        """Render short debug-log lines for operator hardware diagnostics."""

        lines: list[str] = []
        respeaker_line = _respeaker_log_line(self)
        if respeaker_line:
            lines.append(respeaker_line)
        if _microphone_field_value(self) == "stumm":
            lines.append("mic muted")
        if self.listening:
            lines.append("listening")
        if self.heard_speech:
            lines.append("heard speech")
        if self.noise_blocked:
            lines.append(_noise_blocked_log_line(self))
        if self.resume_window_open is True and not self.telemetry_stale:
            lines.append("resume window open")
        if self.direction_hint_available:
            azimuth = int(self.azimuth_deg) if self.azimuth_deg is not None else "?"
            confidence = "?" if self.direction_confidence is None else f"{self.direction_confidence:.2f}"
            lines.append(compact_text(f"direction {azimuth}deg conf {confidence}", limit=58))
        if not self.telemetry_stale and self.led_ring_mode not in {"idle", "off"}:
            lines.append(compact_text(f"ring {self.led_ring_mode}", limit=58))
        return tuple(lines[:4])


@dataclass(frozen=True, slots=True)
class _LatestOpsEvent:
    payload: Mapping[str, object]
    created_at_text: str | None
    created_at_ts: float | None
    message: str | None
    order_index: int


@dataclass(slots=True)
class DisplayReSpeakerHciStore:
    """Cache and load operator-facing ReSpeaker HCI state from ops events."""

    event_store: TwinrOpsEventStore
    refresh_interval_s: float = _DEFAULT_REFRESH_INTERVAL_S
    _cached_state: DisplayReSpeakerHciState | None = None
    _cached_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReSpeakerHciStore":
        """Build one cached store from Twinr config."""

        return cls(event_store=TwinrOpsEventStore.from_config(config))

    def load(self) -> DisplayReSpeakerHciState | None:
        """Return the latest cached or freshly parsed HCI state."""

        now = time.monotonic()
        refresh_interval_s = _normalize_refresh_interval(self.refresh_interval_s)
        with self._lock:
            if (
                self._cached_state is not None
                and (now - self._cached_at) < refresh_interval_s
            ):
                return self._cached_state
            entries = self.event_store.tail(limit=_DEFAULT_EVENT_TAIL_LIMIT)
            self._cached_state = parse_respeaker_hci_state(entries, now_ts=time.time())
            self._cached_at = now
            return self._cached_state


def parse_respeaker_hci_state(
    entries: Sequence[Mapping[str, object]],
    *,
    now_ts: float | None = None,
) -> DisplayReSpeakerHciState | None:
    """Parse the latest ReSpeaker display state from one ops-event tail."""

    latest_observation: _LatestOpsEvent | None = None
    latest_alert: _LatestOpsEvent | None = None

    for order_index in range(len(entries) - 1, -1, -1):
        entry = entries[order_index]
        if not isinstance(entry, Mapping):
            continue
        event_name = _normalize_token(entry.get("event"))
        payload = _mapping_from_event_data(entry.get("data"))
        if event_name in _OBSERVATION_EVENT_NAMES and latest_observation is None:
            if _payload_contains_any(payload, _RELEVANT_OBSERVATION_KEYS):
                latest_observation = _LatestOpsEvent(
                    payload=payload,
                    created_at_text=_normalize_text(entry.get("created_at")) or None,
                    created_at_ts=_parse_timestamp_s(entry.get("created_at")),
                    message=None,
                    order_index=order_index,
                )
        elif event_name in _ALERT_EVENT_NAMES and latest_alert is None:
            alert_message = _first_text(payload, _ALERT_MESSAGE_ALIASES)
            if alert_message is None:
                alert_message = _normalize_text(entry.get("message")) or None
            if _payload_contains_any(payload, _RELEVANT_ALERT_KEYS) or alert_message is not None:
                latest_alert = _LatestOpsEvent(
                    payload=payload,
                    created_at_text=_normalize_text(entry.get("created_at")) or None,
                    created_at_ts=_parse_timestamp_s(entry.get("created_at")),
                    message=alert_message,
                    order_index=order_index,
                )
        if latest_observation is not None and latest_alert is not None:
            break

    if latest_observation is None and latest_alert is None:
        return None

    runtime_alert_code, _ = _choose_token_value(
        latest_observation,
        latest_alert,
        observation_aliases=_OBSERVATION_FIELD_ALIASES["runtime_alert_code"],
        alert_aliases=_ALERT_FIELD_ALIASES["runtime_alert_code"],
    )
    alert_code_only, _ = _choose_token_value(
        None,
        latest_alert,
        observation_aliases=(),
        alert_aliases=_ALERT_FIELD_ALIASES["runtime_alert_code"],
    )

    snapshot_event = _newer_event(latest_observation, latest_alert)
    observation_age_s = _event_age_s(latest_observation, now_ts=now_ts)
    snapshot_age_s = _event_age_s(snapshot_event, now_ts=now_ts)

    runtime_alert_message: str | None = None
    if latest_alert is not None and alert_code_only is not None and alert_code_only == runtime_alert_code:
        runtime_alert_message = latest_alert.message

    return DisplayReSpeakerHciState(
        observed_at=snapshot_event.created_at_text if snapshot_event is not None else None,
        runtime_status=_choose_token_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["runtime_status"],
        ),
        runtime_alert_code=runtime_alert_code,
        runtime_alert_message=runtime_alert_message,
        device_runtime_mode=_choose_token_value(
            latest_observation,
            latest_alert,
            observation_aliases=_OBSERVATION_FIELD_ALIASES["device_runtime_mode"],
            alert_aliases=_ALERT_FIELD_ALIASES["device_runtime_mode"],
        )[0],
        host_control_ready=_choose_bool_value(
            latest_observation,
            latest_alert,
            observation_aliases=_OBSERVATION_FIELD_ALIASES["host_control_ready"],
            alert_aliases=_ALERT_FIELD_ALIASES["host_control_ready"],
        )[0],
        transport_reason=_choose_token_value(
            latest_observation,
            latest_alert,
            observation_aliases=_OBSERVATION_FIELD_ALIASES["transport_reason"],
            alert_aliases=_ALERT_FIELD_ALIASES["transport_reason"],
        )[0],
        mute_active=_choose_bool_value(
            latest_observation,
            latest_alert,
            observation_aliases=_OBSERVATION_FIELD_ALIASES["mute_active"],
            alert_aliases=_ALERT_FIELD_ALIASES["mute_active"],
        )[0],
        speech_detected=_choose_bool_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["speech_detected"],
        ),
        recent_speech_age_s=_choose_non_negative_float_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["recent_speech_age_s"],
        ),
        direction_confidence=_choose_probability_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["direction_confidence"],
        ),
        azimuth_deg=_choose_azimuth_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["azimuth_deg"],
        ),
        room_busy_or_overlapping=_choose_bool_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["room_busy_or_overlapping"],
        ),
        quiet_window_open=_choose_bool_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["quiet_window_open"],
        ),
        non_speech_audio_likely=_choose_bool_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["non_speech_audio_likely"],
        ),
        background_media_likely=_choose_bool_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["background_media_likely"],
        ),
        resume_window_open=_choose_bool_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["resume_window_open"],
        ),
        voice_activation_armed=_choose_bool_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["voice_activation_armed"],
        ),
        initiative_block_reason=_choose_token_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["initiative_block_reason"],
        ),
        speech_delivery_defer_reason=_choose_token_from_single(
            latest_observation,
            _OBSERVATION_FIELD_ALIASES["speech_delivery_defer_reason"],
        ),
        observation_age_s=observation_age_s,
        snapshot_age_s=snapshot_age_s,
    )


def _respeaker_field_value(state: DisplayReSpeakerHciState) -> str | None:
    code = state.runtime_alert_code
    if code == "dfu_mode":
        return "DFU"
    if code in {"disconnected", "probe_unavailable"}:
        return "fehlt"
    if code in {
        "capture_unknown",
        "host_control_unavailable",
        "transport_blocked",
        "signal_provider_error",
        "provider_lock_timeout",
    }:
        return "Achtung"
    if state.telemetry_stale:
        return "Daten alt"
    return None


def _microphone_field_value(state: DisplayReSpeakerHciState) -> str | None:
    if (
        state.runtime_alert_code == "mic_muted"
        or state.mute_active is True
        or state.initiative_block_reason == "mute_blocks_voice_capture"
    ):
        return "stumm"
    if state.listening:
        return "hört"
    return None


def _audio_field_value(state: DisplayReSpeakerHciState) -> str | None:
    if state.telemetry_stale:
        return None
    if state.background_media_likely is True or state.speech_delivery_defer_reason == "background_media_active":
        return "Medien"
    if state.non_speech_audio_likely is True or state.speech_delivery_defer_reason == "non_speech_audio_active":
        return "Geräusche"
    if state.room_busy_or_overlapping is True:
        return "laut"
    if state.resume_window_open is True:
        return "Pause"
    if state.heard_speech:
        return "Sprache"
    return None


def _direction_field_value(state: DisplayReSpeakerHciState) -> str | None:
    if not state.direction_hint_available:
        return None
    return f"{int(state.azimuth_deg)}°" if state.azimuth_deg is not None else None


def _ring_field_value(state: DisplayReSpeakerHciState) -> str | None:
    if state.telemetry_stale and state.led_ring_mode == "listening":
        return None
    if state.led_ring_mode == "listening":
        return "hört"
    if state.led_ring_mode == "muted":
        return "stumm"
    return None


def _respeaker_log_line(state: DisplayReSpeakerHciState) -> str | None:
    code = state.runtime_alert_code
    if code == "dfu_mode":
        return "respeaker dfu"
    if code in {"disconnected", "probe_unavailable"}:
        return "respeaker unavailable"
    if code in {
        "capture_unknown",
        "host_control_unavailable",
        "transport_blocked",
        "signal_provider_error",
        "provider_lock_timeout",
    }:
        return "respeaker degraded"
    if state.telemetry_stale:
        return "respeaker stale"
    if code == "ready":
        return "respeaker ready"
    return None


def _noise_blocked_log_line(state: DisplayReSpeakerHciState) -> str:
    if state.background_media_likely is True or state.speech_delivery_defer_reason == "background_media_active":
        return "noise blocked media"
    if state.non_speech_audio_likely is True or state.speech_delivery_defer_reason == "non_speech_audio_active":
        return "noise blocked non-speech"
    return "noise blocked overlap"


def _choose_token_value(
    observation: _LatestOpsEvent | None,
    alert: _LatestOpsEvent | None,
    *,
    observation_aliases: Sequence[str],
    alert_aliases: Sequence[str],
) -> tuple[str | None, str | None]:
    return _choose_value(
        observation,
        alert,
        observation_aliases=observation_aliases,
        alert_aliases=alert_aliases,
        parser=_parse_token,
    )


def _choose_bool_value(
    observation: _LatestOpsEvent | None,
    alert: _LatestOpsEvent | None,
    *,
    observation_aliases: Sequence[str],
    alert_aliases: Sequence[str],
) -> tuple[bool | None, str | None]:
    return _choose_value(
        observation,
        alert,
        observation_aliases=observation_aliases,
        alert_aliases=alert_aliases,
        parser=_parse_bool,
    )


def _choose_value(
    observation: _LatestOpsEvent | None,
    alert: _LatestOpsEvent | None,
    *,
    observation_aliases: Sequence[str],
    alert_aliases: Sequence[str],
    parser,
):
    if _event_order_key(alert) > _event_order_key(observation):
        newer_first = (
            ("alert", alert, alert_aliases),
            ("observation", observation, observation_aliases),
        )
    else:
        newer_first = (
            ("observation", observation, observation_aliases),
            ("alert", alert, alert_aliases),
        )
    for source_name, event, aliases in newer_first:
        if event is None or not aliases:
            continue
        raw = _extract_alias_value(event.payload, aliases)
        if raw is _MISSING:
            continue
        parsed = parser(raw)
        if parsed is _INVALID:
            continue
        return parsed, source_name
    return None, None


def _choose_token_from_single(
    event: _LatestOpsEvent | None,
    aliases: Sequence[str],
) -> str | None:
    return _choose_from_single(event, aliases, parser=_parse_token)


def _choose_bool_from_single(
    event: _LatestOpsEvent | None,
    aliases: Sequence[str],
) -> bool | None:
    return _choose_from_single(event, aliases, parser=_parse_bool)


def _choose_non_negative_float_from_single(
    event: _LatestOpsEvent | None,
    aliases: Sequence[str],
) -> float | None:
    return _choose_from_single(event, aliases, parser=_parse_non_negative_float)


def _choose_probability_from_single(
    event: _LatestOpsEvent | None,
    aliases: Sequence[str],
) -> float | None:
    return _choose_from_single(event, aliases, parser=_parse_probability)


def _choose_azimuth_from_single(
    event: _LatestOpsEvent | None,
    aliases: Sequence[str],
) -> int | None:
    return _choose_from_single(event, aliases, parser=_parse_azimuth)


def _choose_from_single(
    event: _LatestOpsEvent | None,
    aliases: Sequence[str],
    *,
    parser,
):
    if event is None or not aliases:
        return None
    raw = _extract_alias_value(event.payload, aliases)
    if raw is _MISSING:
        return None
    parsed = parser(raw)
    return None if parsed is _INVALID else parsed


def _mapping_from_event_data(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raw_bytes = _as_small_bytes(value, limit=_MAX_SAFE_JSON_BYTES)
        if raw_bytes:
            try:
                text = raw_bytes.decode("utf-8", "replace").strip()
            except Exception:
                return {}
            if text.startswith("{") and len(text) <= _MAX_SAFE_JSON_BYTES:
                try:
                    decoded = json.loads(text)
                except (TypeError, ValueError):
                    return {}
                return decoded if isinstance(decoded, Mapping) else {}
    return {}


def _payload_contains_any(payload: Mapping[str, object], keys: frozenset[str]) -> bool:
    for view in _iter_payload_views(payload):
        for key in keys:
            if key in view:
                return True
    return False


def _extract_alias_value(payload: Mapping[str, object], aliases: Sequence[str]):
    for view in _iter_payload_views(payload):
        for key in aliases:
            if key in view:
                return view[key]
    return _MISSING


def _first_text(payload: Mapping[str, object], aliases: Sequence[str]) -> str | None:
    for view in _iter_payload_views(payload):
        for key in aliases:
            if key in view:
                text = _normalize_text(view[key]) or None
                if text is not None:
                    return text
    return None


def _iter_payload_views(payload: Mapping[str, object]):
    yield payload
    for key in _NESTED_PAYLOAD_KEYS:
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            yield nested


def _event_order_key(event: _LatestOpsEvent | None) -> int:
    return -1 if event is None else event.order_index


def _newer_event(
    observation: _LatestOpsEvent | None,
    alert: _LatestOpsEvent | None,
) -> _LatestOpsEvent | None:
    return alert if _event_order_key(alert) > _event_order_key(observation) else observation


def _event_age_s(event: _LatestOpsEvent | None, *, now_ts: float | None = None) -> float | None:
    if event is None or event.created_at_ts is None:
        return None
    if now_ts is None:
        return None
    age_s = now_ts - event.created_at_ts
    if not math.isfinite(age_s):
        return None
    return 0.0 if age_s < 0.0 else age_s


def _normalize_refresh_interval(value: object) -> float:
    interval = _parse_non_negative_float(value)
    if interval is _INVALID or interval is None:
        return max(0.1, _DEFAULT_REFRESH_INTERVAL_S)
    return max(0.1, float(interval))


def _parse_bool(value: object):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        return _INVALID
    if isinstance(value, float):
        if not math.isfinite(value):
            return _INVALID
        if value in (0.0, 1.0):
            return bool(int(value))
        return _INVALID
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        token = _normalize_token(value)
        if token is None:
            return None
        if token in {"true", "1", "yes", "on"}:
            return True
        if token in {"false", "0", "no", "off"}:
            return False
        return _INVALID
    return _INVALID


def _parse_token(value: object):
    token = _normalize_token(value)
    return None if token is None else token


def _parse_int(value: object):
    if value is None:
        return None
    if isinstance(value, bool):
        return _INVALID
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return _INVALID
        return int(value)
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        text = _normalize_numeric_text(value)
        if text is None:
            return None
        try:
            return int(text)
        except (TypeError, ValueError):
            return _INVALID
    return _INVALID


def _parse_float(value: object):
    if value is None:
        return None
    if isinstance(value, bool):
        return _INVALID
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else _INVALID
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        text = _normalize_numeric_text(value)
        if text is None:
            return None
        try:
            number = float(text)
        except (TypeError, ValueError):
            return _INVALID
        return number if math.isfinite(number) else _INVALID
    return _INVALID


def _parse_non_negative_float(value: object):
    number = _parse_float(value)
    if number in {None, _INVALID}:
        return number
    if number < 0.0:
        return 0.0 if number > -0.5 else _INVALID
    return number


def _parse_probability(value: object):
    number = _parse_float(value)
    if number in {None, _INVALID}:
        return number
    if 0.0 <= number <= 1.0:
        return number
    return _INVALID


def _parse_azimuth(value: object):
    number = _parse_int(value)
    if number in {None, _INVALID}:
        return number
    if abs(number) > 360:
        return _INVALID
    return int(number)


def _normalize_numeric_text(value: object) -> str | None:
    text = _normalize_text(value, limit=_MAX_NUMERIC_TEXT_LEN) or None
    if text is None:
        return None
    return text if len(text) <= _MAX_NUMERIC_TEXT_LEN else None


def _parse_timestamp_s(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number if number > 0.0 else None
    text = _normalize_text(value, limit=_MAX_NUMERIC_TEXT_LEN) or None
    if text is None:
        return None
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None and math.isfinite(numeric) and numeric > 0.0:
        return numeric
    try:
        normalized = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo or timezone.utc
        parsed = parsed.replace(tzinfo=local_tz)
    return parsed.timestamp()


def _as_small_bytes(value: object, *, limit: int) -> bytes:
    if isinstance(value, bytes):
        return value[:limit]
    if isinstance(value, bytearray):
        return bytes(value[:limit])
    if isinstance(value, memoryview):
        return bytes(value[:limit])
    if isinstance(value, str):
        return value[:limit].encode("utf-8", "replace")
    return b""


def _safe_text_fragment(value: object, *, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _as_small_bytes(value, limit=limit).decode("utf-8", "replace")
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value)[:limit]


def _normalize_text(value: object, *, limit: int = _MAX_DISPLAY_TEXT_LEN) -> str:
    text = _safe_text_fragment(value, limit=_MAX_SAFE_TEXT_SCAN_LEN)
    if not text:
        return ""
    cleaned = "".join(
        ch
        for ch in text
        if ch in "\t\n\r" or unicodedata.category(ch)[0] != "C"
    )
    collapsed = " ".join(cleaned.split())
    return compact_text(collapsed, limit=limit)


def _normalize_token(value: object) -> str | None:
    text = _normalize_text(value, limit=_MAX_DISPLAY_TEXT_LEN)
    if not text:
        return None
    token = text.casefold()
    for separator in (" ", "-", ".", "/", "\\"):
        token = token.replace(separator, "_")
    while "__" in token:
        token = token.replace("__", "_")
    token = token.strip("_")
    return token or None


def _coerce_optional_bool(value: object, *, fallback: bool | None = None) -> bool | None:
    parsed = _parse_bool(value)
    return fallback if parsed is _INVALID else parsed


def _coerce_optional_int(value: object) -> int | None:
    parsed = _parse_int(value)
    return None if parsed is _INVALID else parsed


def _coerce_optional_float(value: object) -> float | None:
    parsed = _parse_float(value)
    return None if parsed is _INVALID else parsed


__all__ = [
    "DisplayReSpeakerHciState",
    "DisplayReSpeakerHciStore",
    "parse_respeaker_hci_state",
]
