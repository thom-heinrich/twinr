"""Surface calm ReSpeaker HCI state from authoritative ops events.

The display loop should not parse raw proactive event payloads inline. This
module reads the latest persisted ReSpeaker-related ops facts and condenses
them into calm operator-visible states such as muted microphone, blocked room
audio, or DFU mode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.respeaker.indicator_policy import resolve_respeaker_indicator_state
from twinr.ops.events import TwinrOpsEventStore, compact_text


_DEFAULT_EVENT_TAIL_LIMIT = 80
_DEFAULT_REFRESH_INTERVAL_S = 1.0


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
    wakeword_armed: bool | None = None
    initiative_block_reason: str | None = None
    speech_delivery_defer_reason: str | None = None

    @property
    def direction_hint_available(self) -> bool:
        """Return whether the latest azimuth hint is operator-usable."""

        return (
            self.azimuth_deg is not None
            and self.direction_confidence is not None
            and self.direction_confidence >= 0.75
        )

    @property
    def listening(self) -> bool:
        """Return whether Twinr is explicitly in a live listening state."""

        return self.runtime_status == "listening"

    @property
    def heard_speech(self) -> bool:
        """Return whether the ReSpeaker has a calm recent-speech signal."""

        if self.speech_detected is True:
            return True
        if self.recent_speech_age_s is None:
            return False
        return self.recent_speech_age_s <= 1.5

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
        if self.resume_window_open is True:
            lines.append("resume window open")
        if self.direction_hint_available:
            azimuth = int(self.azimuth_deg) if self.azimuth_deg is not None else "?"
            confidence = "?" if self.direction_confidence is None else f"{self.direction_confidence:.2f}"
            lines.append(compact_text(f"direction {azimuth}deg conf {confidence}", limit=58))
        if self.led_ring_mode not in {"idle", "off"}:
            lines.append(compact_text(f"ring {self.led_ring_mode}", limit=58))
        return tuple(lines[:4])


@dataclass(slots=True)
class DisplayReSpeakerHciStore:
    """Cache and load operator-facing ReSpeaker HCI state from ops events."""

    event_store: TwinrOpsEventStore
    refresh_interval_s: float = _DEFAULT_REFRESH_INTERVAL_S
    _cached_state: DisplayReSpeakerHciState | None = None
    _cached_at: float = 0.0

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReSpeakerHciStore":
        """Build one cached store from Twinr config."""

        return cls(event_store=TwinrOpsEventStore.from_config(config))

    def load(self) -> DisplayReSpeakerHciState | None:
        """Return the latest cached or freshly parsed HCI state."""

        now = time.monotonic()
        if (now - self._cached_at) < max(0.1, float(self.refresh_interval_s)):
            return self._cached_state
        entries = self.event_store.tail(limit=_DEFAULT_EVENT_TAIL_LIMIT)
        self._cached_state = parse_respeaker_hci_state(entries)
        self._cached_at = now
        return self._cached_state


def parse_respeaker_hci_state(
    entries: Sequence[Mapping[str, object]],
) -> DisplayReSpeakerHciState | None:
    """Parse the latest ReSpeaker display state from one ops-event tail."""

    latest_observation: Mapping[str, object] | None = None
    latest_observation_at: str | None = None
    latest_alert: Mapping[str, object] | None = None
    latest_alert_message: str | None = None
    latest_alert_at: str | None = None
    for entry in reversed(tuple(entries)):
        if not isinstance(entry, Mapping):
            continue
        event_name = _normalize_text(entry.get("event"))
        data = entry.get("data")
        payload = data if isinstance(data, Mapping) else {}
        if event_name == "proactive_observation" and latest_observation is None:
            if "audio_device_runtime_mode" in payload or "respeaker_runtime_alert_code" in payload:
                latest_observation = payload
                latest_observation_at = _normalize_text(entry.get("created_at")) or None
        elif event_name == "respeaker_runtime_alert" and latest_alert is None:
            latest_alert = payload
            latest_alert_message = _normalize_text(entry.get("message")) or None
            latest_alert_at = _normalize_text(entry.get("created_at")) or None
        if latest_observation is not None and latest_alert is not None:
            break
    if latest_observation is None and latest_alert is None:
        return None

    observation = latest_observation or {}
    alert = latest_alert or {}
    runtime_alert_code = _normalize_text(observation.get("respeaker_runtime_alert_code")) or _normalize_text(alert.get("alert_code")) or None
    return DisplayReSpeakerHciState(
        observed_at=latest_observation_at or latest_alert_at,
        runtime_status=_normalize_text(observation.get("runtime_status")) or None,
        runtime_alert_code=runtime_alert_code,
        runtime_alert_message=latest_alert_message,
        device_runtime_mode=_normalize_text(observation.get("audio_device_runtime_mode")) or _normalize_text(alert.get("device_runtime_mode")) or None,
        host_control_ready=_coerce_optional_bool(observation.get("audio_host_control_ready"), fallback=_coerce_optional_bool(alert.get("host_control_ready"))),
        transport_reason=_normalize_text(observation.get("audio_transport_reason")) or _normalize_text(alert.get("transport_reason")) or None,
        mute_active=_coerce_optional_bool(observation.get("audio_mute_active"), fallback=_coerce_optional_bool(alert.get("mute_active"))),
        speech_detected=_coerce_optional_bool(observation.get("speech_detected")),
        recent_speech_age_s=_coerce_optional_float(observation.get("audio_recent_speech_age_s")),
        direction_confidence=_coerce_optional_float(observation.get("audio_direction_confidence")),
        azimuth_deg=_coerce_optional_int(observation.get("audio_azimuth_deg")),
        room_busy_or_overlapping=_coerce_optional_bool(observation.get("room_busy_or_overlapping")),
        quiet_window_open=_coerce_optional_bool(observation.get("quiet_window_open")),
        non_speech_audio_likely=_coerce_optional_bool(observation.get("non_speech_audio_likely")),
        background_media_likely=_coerce_optional_bool(observation.get("background_media_likely")),
        resume_window_open=_coerce_optional_bool(observation.get("resume_window_open")),
        wakeword_armed=_coerce_optional_bool(observation.get("wakeword_armed")),
        initiative_block_reason=_normalize_text(observation.get("audio_initiative_block_reason")) or None,
        speech_delivery_defer_reason=_normalize_text(observation.get("audio_speech_delivery_defer_reason")) or None,
    )


def _respeaker_field_value(state: DisplayReSpeakerHciState) -> str | None:
    code = state.runtime_alert_code
    if code == "dfu_mode":
        return "DFU"
    if code in {"disconnected", "probe_unavailable"}:
        return "fehlt"
    if code in {"capture_unknown", "host_control_unavailable", "transport_blocked", "signal_provider_error", "provider_lock_timeout"}:
        return "Achtung"
    return None


def _microphone_field_value(state: DisplayReSpeakerHciState) -> str | None:
    if state.runtime_alert_code == "mic_muted" or state.mute_active is True or state.initiative_block_reason == "mute_blocks_voice_capture":
        return "stumm"
    if state.listening:
        return "hört"
    return None


def _audio_field_value(state: DisplayReSpeakerHciState) -> str | None:
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
    if code in {"capture_unknown", "host_control_unavailable", "transport_blocked", "signal_provider_error", "provider_lock_timeout"}:
        return "respeaker degraded"
    if code == "ready":
        return "respeaker ready"
    return None


def _noise_blocked_log_line(state: DisplayReSpeakerHciState) -> str:
    if state.background_media_likely is True or state.speech_delivery_defer_reason == "background_media_active":
        return "noise blocked media"
    if state.non_speech_audio_likely is True or state.speech_delivery_defer_reason == "non_speech_audio_active":
        return "noise blocked non-speech"
    return "noise blocked overlap"


def _coerce_optional_bool(value: object, *, fallback: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    return fallback


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _normalize_text(value: object) -> str:
    return compact_text(" ".join(str(value or "").split()), limit=160)


__all__ = [
    "DisplayReSpeakerHciState",
    "DisplayReSpeakerHciStore",
    "parse_respeaker_hci_state",
]
