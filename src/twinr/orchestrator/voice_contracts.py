"""Define websocket contracts for Twinr's streaming voice orchestrator path.

The text-turn orchestrator already owns bounded websocket transport for remote
tool turns. This module adds the audio-session contract used by the Alexa-like
server-backed voice path: the edge sends bounded PCM frames plus runtime-state
updates, and the server responds with explicit voice activation, transcript-commit,
follow-up-close, and barge-in decisions.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any


_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "off", ""})


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Return a plain dictionary for mapping-like payloads."""

    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        return {}


def _coerce_text(value: Any) -> str:
    """Normalize a payload field into stripped text."""

    return str(value or "").strip()


def _coerce_optional_text(value: Any) -> str | None:
    """Return stripped text or ``None`` for blank scalar payloads."""

    text = _coerce_text(value)
    return text or None


def _coerce_positive_int(value: Any, *, default: int) -> int:
    """Parse a positive integer and fall back to ``default`` when invalid."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Parse bounded websocket booleans from scalars and strings."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in _TRUE_STRINGS:
        return True
    if normalized in _FALSE_STRINGS:
        return False
    return default


def _decode_audio_bytes(value: Any) -> bytes:
    """Decode one base64 PCM payload into raw bytes."""

    if not isinstance(value, str) or not value:
        return b""
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception:
        return b""


def _encode_audio_bytes(audio_bytes: bytes) -> str:
    """Encode raw PCM bytes for websocket transport."""

    if not audio_bytes:
        return ""
    return base64.b64encode(bytes(audio_bytes)).decode("ascii")


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceHelloRequest:
    """Describe the opening metadata for one streaming voice session."""

    session_id: str
    sample_rate: int
    channels: int
    chunk_ms: int
    trace_id: str | None = None
    initial_state: str = "waiting"
    detail: str | None = None
    follow_up_allowed: bool = False
    attention_state: str | None = None
    interaction_intent_state: str | None = None
    person_visible: bool | None = None
    presence_active: bool | None = None
    interaction_ready: bool | None = None
    targeted_inference_blocked: bool | None = None
    recommended_channel: str | None = None
    state_attested: bool = True

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "voice_hello",
            "session_id": self.session_id,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_ms": self.chunk_ms,
            "initial_state": self.initial_state,
            "follow_up_allowed": self.follow_up_allowed,
            "state_attested": self.state_attested,
        }
        if self.trace_id is not None:
            payload["trace_id"] = self.trace_id
        if self.detail is not None:
            payload["detail"] = self.detail
        if self.attention_state is not None:
            payload["attention_state"] = self.attention_state
        if self.interaction_intent_state is not None:
            payload["interaction_intent_state"] = self.interaction_intent_state
        if self.person_visible is not None:
            payload["person_visible"] = self.person_visible
        if self.presence_active is not None:
            payload["presence_active"] = self.presence_active
        if self.interaction_ready is not None:
            payload["interaction_ready"] = self.interaction_ready
        if self.targeted_inference_blocked is not None:
            payload["targeted_inference_blocked"] = self.targeted_inference_blocked
        if self.recommended_channel is not None:
            payload["recommended_channel"] = self.recommended_channel
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorVoiceHelloRequest:
        payload_dict = _coerce_dict(payload)
        return cls(
            session_id=_coerce_text(payload_dict.get("session_id")),
            trace_id=_coerce_optional_text(payload_dict.get("trace_id")),
            sample_rate=_coerce_positive_int(payload_dict.get("sample_rate"), default=16_000),
            channels=_coerce_positive_int(payload_dict.get("channels"), default=1),
            chunk_ms=_coerce_positive_int(payload_dict.get("chunk_ms"), default=100),
            initial_state=_coerce_text(payload_dict.get("initial_state")) or "waiting",
            detail=_coerce_optional_text(payload_dict.get("detail")),
            follow_up_allowed=_coerce_bool(payload_dict.get("follow_up_allowed"), default=False),
            attention_state=_coerce_optional_text(payload_dict.get("attention_state")),
            interaction_intent_state=_coerce_optional_text(
                payload_dict.get("interaction_intent_state")
            ),
            person_visible=(
                _coerce_bool(payload_dict.get("person_visible"))
                if payload_dict.get("person_visible") is not None
                else None
            ),
            presence_active=(
                _coerce_bool(payload_dict.get("presence_active"))
                if payload_dict.get("presence_active") is not None
                else None
            ),
            interaction_ready=(
                _coerce_bool(payload_dict.get("interaction_ready"))
                if payload_dict.get("interaction_ready") is not None
                else None
            ),
            targeted_inference_blocked=(
                _coerce_bool(payload_dict.get("targeted_inference_blocked"))
                if payload_dict.get("targeted_inference_blocked") is not None
                else None
            ),
            recommended_channel=_coerce_optional_text(payload_dict.get("recommended_channel")),
            state_attested=_coerce_bool(payload_dict.get("state_attested"), default=False),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceAudioFrame:
    """Represent one bounded PCM frame sent from edge to server."""

    sequence: int
    pcm_bytes: bytes
    runtime_state: OrchestratorVoiceRuntimeStateEvent | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "type": "voice_audio_frame",
            "sequence": self.sequence,
            "pcm_s16le_b64": _encode_audio_bytes(self.pcm_bytes),
        }
        if self.runtime_state is not None:
            payload["runtime_state"] = self.runtime_state.to_payload()
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorVoiceAudioFrame:
        payload_dict = _coerce_dict(payload)
        runtime_state_payload = payload_dict.get("runtime_state")
        return cls(
            sequence=max(0, _coerce_positive_int(payload_dict.get("sequence"), default=0)),
            pcm_bytes=_decode_audio_bytes(payload_dict.get("pcm_s16le_b64")),
            runtime_state=(
                OrchestratorVoiceRuntimeStateEvent.from_payload(runtime_state_payload)
                if isinstance(runtime_state_payload, dict)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceRuntimeStateEvent:
    """Represent one explicit edge-side voice runtime state update."""

    state: str
    detail: str | None = None
    follow_up_allowed: bool = False
    attention_state: str | None = None
    interaction_intent_state: str | None = None
    person_visible: bool | None = None
    presence_active: bool | None = None
    interaction_ready: bool | None = None
    targeted_inference_blocked: bool | None = None
    recommended_channel: str | None = None

    def to_payload(self, *, include_type: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": self.state,
            "follow_up_allowed": self.follow_up_allowed,
        }
        if include_type:
            payload["type"] = "voice_runtime_state"
        if self.detail is not None:
            payload["detail"] = self.detail
        if self.attention_state is not None:
            payload["attention_state"] = self.attention_state
        if self.interaction_intent_state is not None:
            payload["interaction_intent_state"] = self.interaction_intent_state
        if self.person_visible is not None:
            payload["person_visible"] = self.person_visible
        if self.presence_active is not None:
            payload["presence_active"] = self.presence_active
        if self.interaction_ready is not None:
            payload["interaction_ready"] = self.interaction_ready
        if self.targeted_inference_blocked is not None:
            payload["targeted_inference_blocked"] = self.targeted_inference_blocked
        if self.recommended_channel is not None:
            payload["recommended_channel"] = self.recommended_channel
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorVoiceRuntimeStateEvent:
        payload_dict = _coerce_dict(payload)
        return cls(
            state=_coerce_text(payload_dict.get("state")) or "waiting",
            detail=_coerce_optional_text(payload_dict.get("detail")),
            follow_up_allowed=_coerce_bool(payload_dict.get("follow_up_allowed"), default=False),
            attention_state=_coerce_optional_text(payload_dict.get("attention_state")),
            interaction_intent_state=_coerce_optional_text(
                payload_dict.get("interaction_intent_state")
            ),
            person_visible=(
                _coerce_bool(payload_dict.get("person_visible"))
                if payload_dict.get("person_visible") is not None
                else None
            ),
            presence_active=(
                _coerce_bool(payload_dict.get("presence_active"))
                if payload_dict.get("presence_active") is not None
                else None
            ),
            interaction_ready=(
                _coerce_bool(payload_dict.get("interaction_ready"))
                if payload_dict.get("interaction_ready") is not None
                else None
            ),
            targeted_inference_blocked=(
                _coerce_bool(payload_dict.get("targeted_inference_blocked"))
                if payload_dict.get("targeted_inference_blocked") is not None
                else None
            ),
            recommended_channel=_coerce_optional_text(payload_dict.get("recommended_channel")),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceReadyEvent:
    """Acknowledge that the server accepted one voice websocket session."""

    session_id: str
    backend: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "voice_ready",
            "session_id": self.session_id,
            "backend": self.backend,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorVoiceReadyEvent:
        payload_dict = _coerce_dict(payload)
        return cls(
            session_id=_coerce_text(payload_dict.get("session_id")),
            backend=_coerce_text(payload_dict.get("backend")) or "unknown",
        )


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceWakeConfirmedEvent:
    """Represent one confirmed remote voice activation match."""

    matched_phrase: str | None
    remaining_text: str
    backend: str
    detector_label: str | None = None
    score: float | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "wake_confirmed",
            "remaining_text": self.remaining_text,
            "backend": self.backend,
        }
        if self.matched_phrase is not None:
            payload["matched_phrase"] = self.matched_phrase
        if self.detector_label is not None:
            payload["detector_label"] = self.detector_label
        if self.score is not None:
            payload["score"] = float(self.score)
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorVoiceWakeConfirmedEvent:
        payload_dict = _coerce_dict(payload)
        raw_score = payload_dict.get("score")
        score: float | None = None
        try:
            if raw_score is not None:
                score = float(raw_score)
        except (TypeError, ValueError):
            score = None
        return cls(
            matched_phrase=_coerce_optional_text(payload_dict.get("matched_phrase")),
            remaining_text=_coerce_text(payload_dict.get("remaining_text")),
            backend=_coerce_text(payload_dict.get("backend")) or "unknown",
            detector_label=_coerce_optional_text(payload_dict.get("detector_label")),
            score=score,
        )


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceTranscriptCommittedEvent:
    """Commit one transcribed user utterance from the live remote stream."""

    transcript: str
    source: str = "listening"

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "transcript_committed",
            "transcript": self.transcript,
            "source": self.source,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorVoiceTranscriptCommittedEvent:
        payload_dict = _coerce_dict(payload)
        return cls(
            transcript=_coerce_text(payload_dict.get("transcript")),
            source=_coerce_text(payload_dict.get("source")) or "listening",
        )


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceFollowUpClosedEvent:
    """Signal that the server closed the current remote follow-up window."""

    reason: str

    def to_payload(self) -> dict[str, Any]:
        return {"type": "follow_up_closed", "reason": self.reason}

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorVoiceFollowUpClosedEvent:
        payload_dict = _coerce_dict(payload)
        return cls(reason=_coerce_text(payload_dict.get("reason")) or "timeout")


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceBargeInInterruptEvent:
    """Tell the edge to stop the current answer because the user interrupted."""

    transcript_preview: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {"type": "barge_in_interrupt"}
        if self.transcript_preview is not None:
            payload["transcript_preview"] = self.transcript_preview
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorVoiceBargeInInterruptEvent:
        payload_dict = _coerce_dict(payload)
        return cls(transcript_preview=_coerce_optional_text(payload_dict.get("transcript_preview")))


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceErrorEvent:
    """Represent one sanitized voice-session transport or runtime failure."""

    error: str

    def to_payload(self) -> dict[str, Any]:
        return {"type": "voice_error", "error": self.error}

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorVoiceErrorEvent:
        payload_dict = _coerce_dict(payload)
        return cls(error=_coerce_text(payload_dict.get("error")) or "Voice session failed")


VoiceServerEvent = (
    OrchestratorVoiceReadyEvent
    | OrchestratorVoiceWakeConfirmedEvent
    | OrchestratorVoiceTranscriptCommittedEvent
    | OrchestratorVoiceFollowUpClosedEvent
    | OrchestratorVoiceBargeInInterruptEvent
    | OrchestratorVoiceErrorEvent
)


def decode_voice_server_event(payload: Any) -> VoiceServerEvent:
    """Decode one server-originated voice websocket event."""

    payload_dict = _coerce_dict(payload)
    message_type = _coerce_text(payload_dict.get("type"))
    if message_type == "voice_ready":
        return OrchestratorVoiceReadyEvent.from_payload(payload_dict)
    if message_type == "wake_confirmed":
        return OrchestratorVoiceWakeConfirmedEvent.from_payload(payload_dict)
    if message_type == "transcript_committed":
        return OrchestratorVoiceTranscriptCommittedEvent.from_payload(payload_dict)
    if message_type == "follow_up_closed":
        return OrchestratorVoiceFollowUpClosedEvent.from_payload(payload_dict)
    if message_type == "barge_in_interrupt":
        return OrchestratorVoiceBargeInInterruptEvent.from_payload(payload_dict)
    if message_type == "voice_error":
        return OrchestratorVoiceErrorEvent.from_payload(payload_dict)
    raise RuntimeError(f"Unsupported voice server event type: {message_type or '<empty>'}")


__all__ = [
    "OrchestratorVoiceAudioFrame",
    "OrchestratorVoiceBargeInInterruptEvent",
    "OrchestratorVoiceErrorEvent",
    "OrchestratorVoiceFollowUpClosedEvent",
    "OrchestratorVoiceHelloRequest",
    "OrchestratorVoiceReadyEvent",
    "OrchestratorVoiceRuntimeStateEvent",
    "OrchestratorVoiceTranscriptCommittedEvent",
    "OrchestratorVoiceWakeConfirmedEvent",
    "VoiceServerEvent",
    "decode_voice_server_event",
]
