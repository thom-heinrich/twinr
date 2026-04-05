"""Define websocket contracts for Twinr's streaming voice orchestrator path.

The text-turn orchestrator already owns bounded websocket transport for remote
tool turns. This module adds the audio-session contract used by the Alexa-like
server-backed voice path: the edge sends bounded PCM frames plus runtime-state
updates, and the server responds with explicit voice activation, transcript
commit, follow-up-close, barge-in, keepalive, and session-control decisions.

This revision hardens payload validation, adds protocol negotiation and
resumption hooks, and exposes symmetric client/server decoders so mixed-version
rollouts fail predictably instead of silently mis-parsing malformed messages.
"""

# CHANGELOG: 2026-03-29
# BUG-1: Reject malformed required payload fields instead of silently defaulting
#        to empty session IDs, blank transcripts, and mismatched session metadata.
# BUG-2: Validate PCM s16le alignment, finite confidence values, and embedding
#        dimensions so corrupted audio and invalid scores do not flow downstream.
# SEC-1: Bound base64 PCM, text, profile counts, and embedding sizes to close a
#        practical memory/CPU DoS path on Raspberry Pi 4 deployments.
# IMP-1: Add protocol negotiation, keepalive/go-away/resume hooks, and symmetric
#        client/server decoders for resilient realtime session management.
# IMP-2: Add timing/sample-offset and embedding-space metadata for micro-turn
#        orchestration, semantic turn-taking, and safe speaker-model upgrades.
# IMP-3: Carry optional per-frame speech-probability side-channel metadata so
#        the host-side utterance scanner can use Pi speech evidence instead of
#        raw RMS fallback when the edge provides it.

from __future__ import annotations

import base64
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "off", ""})

VOICE_PROTOCOL_VERSION = 2
VOICE_SUBPROTOCOL = "twinr.voice.v2"
AUDIO_ENCODING_PCM_S16LE = "pcm_s16le"
AUDIO_TRANSPORT_JSON_B64 = "json_b64"
PCM_SAMPLE_WIDTH_BYTES = 2

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK_MS = 100

MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 48_000
ALLOWED_CHANNELS = frozenset({1, 2})
MIN_CHUNK_MS = 10
MAX_CHUNK_MS = 500

MAX_PCM_FRAME_BYTES = 96_000
MAX_EVENT_ID_LENGTH = 128
MAX_SESSION_ID_LENGTH = 128
MAX_TRACE_ID_LENGTH = 128
MAX_BACKEND_LENGTH = 64
MAX_MESSAGE_TYPE_LENGTH = 64
MAX_STATE_LENGTH = 64
MAX_CHANNEL_NAME_LENGTH = 64
MAX_REASON_LENGTH = 256
MAX_TEXT_LENGTH = 8_192
MAX_DETAIL_LENGTH = 2_048
MAX_CAPABILITIES = 32
MAX_CAPABILITY_LENGTH = 64
MAX_EMBEDDING_DIM = 4_096
MAX_PROFILES = 32

ALLOWED_AUDIO_ENCODINGS = frozenset({AUDIO_ENCODING_PCM_S16LE})
ALLOWED_AUDIO_TRANSPORTS = frozenset({AUDIO_TRANSPORT_JSON_B64})
ALLOWED_TURN_DETECTION_MODES = frozenset({"manual", "server_vad", "semantic_vad"})
ALLOWED_NOISE_REDUCTION_PROFILES = frozenset({"off", "near_field", "far_field"})
ALLOWED_EMBEDDING_METRICS = frozenset({"cosine", "dot", "l2"})


class VoiceContractError(ValueError):
    """Raised when a websocket payload violates the voice contract."""


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Return a plain dictionary for mapping-like payloads."""

    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_text(value: Any) -> str:
    """Normalize a payload field into stripped text."""

    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return bytes(value).decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
    return str(value).strip()


def _coerce_optional_text(value: Any, *, max_length: int | None = None) -> str | None:
    """Return stripped text or ``None`` for blank scalar payloads."""

    text = _coerce_text(value)
    if not text:
        return None
    if max_length is not None and len(text) > max_length:
        raise VoiceContractError(f"Text field exceeds max length {max_length}")
    return text


def _require_text(value: Any, *, field: str, max_length: int) -> str:
    """Return one required text field or raise a contract error."""

    text = _coerce_text(value)
    if not text:
        raise VoiceContractError(f"{field} is required")
    if len(text) > max_length:
        raise VoiceContractError(f"{field} exceeds max length {max_length}")
    return text


def _coerce_optional_ratio(value: Any) -> float | None:
    """Parse one optional ratio in ``[0.0, 1.0]``."""

    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def _coerce_optional_non_negative_int(value: Any) -> int | None:
    """Parse one optional non-negative integer."""

    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _coerce_positive_int(value: Any, *, default: int) -> int:
    """Parse a positive integer and fall back to ``default`` when invalid."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed


def _coerce_non_negative_int(value: Any, *, default: int) -> int:
    """Parse a non-negative integer and fall back to ``default`` when invalid."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0:
        return default
    return parsed


def _parse_positive_int(
    value: Any,
    *,
    field: str,
    default: int | None = None,
) -> int:
    """Parse one required-or-default positive integer."""

    if value is None:
        if default is None:
            raise VoiceContractError(f"{field} is required")
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise VoiceContractError(f"{field} must be an integer") from exc
    if parsed <= 0:
        raise VoiceContractError(f"{field} must be positive")
    return parsed


def _parse_non_negative_int(
    value: Any,
    *,
    field: str,
    default: int | None = None,
) -> int:
    """Parse one required-or-default non-negative integer."""

    if value is None:
        if default is None:
            raise VoiceContractError(f"{field} is required")
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise VoiceContractError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise VoiceContractError(f"{field} must be non-negative")
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


def _normalize_string_list(
    value: Any,
    *,
    max_items: int,
    max_item_length: int,
) -> tuple[str, ...]:
    """Normalize one list of short strings while preserving order."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, memoryview)) or not isinstance(
        value, Sequence
    ):
        raise VoiceContractError("Expected a sequence of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _coerce_text(item)
        if not text:
            continue
        if len(text) > max_item_length:
            raise VoiceContractError(
                f"List item exceeds max length {max_item_length}: {text!r}"
            )
        if text in seen:
            continue
        seen.add(text)
        normalized.append(text)
        if len(normalized) > max_items:
            raise VoiceContractError(f"List exceeds max size {max_items}")
    return tuple(normalized)


def _coerce_embedding(value: Any) -> tuple[float, ...]:
    """Normalize one embedding payload into a finite float tuple."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, memoryview)) or not isinstance(
        value, Sequence
    ):
        return ()
    embedding: list[float] = []
    for item in value:
        try:
            normalized = float(item)
        except (TypeError, ValueError):
            return ()
        if not math.isfinite(normalized):
            return ()
        embedding.append(normalized)
        if len(embedding) > MAX_EMBEDDING_DIM:
            raise VoiceContractError(
                f"Embedding exceeds max dimension {MAX_EMBEDDING_DIM}"
            )
    return tuple(embedding)


def _decoded_audio_length_from_b64(value: str) -> int:
    """Return the decoded byte length for one base64 string."""

    padding = value.count("=")
    return (len(value) // 4) * 3 - padding


def _decode_audio_bytes(value: Any) -> bytes:
    """Decode one base64 PCM payload into bounded raw bytes."""

    if value is None:
        return b""
    if not isinstance(value, str) or not value:
        raise VoiceContractError("pcm_s16le_b64 must be a non-empty base64 string")
    normalized = value.strip()
    if len(normalized) % 4 != 0:
        raise VoiceContractError("pcm_s16le_b64 is not valid base64")
    if _decoded_audio_length_from_b64(normalized) > MAX_PCM_FRAME_BYTES:
        raise VoiceContractError(
            f"PCM frame exceeds max decoded size {MAX_PCM_FRAME_BYTES} bytes"
        )
    try:
        decoded = base64.b64decode(normalized.encode("ascii"), validate=True)
    except Exception as exc:
        raise VoiceContractError("pcm_s16le_b64 is not valid base64") from exc
    _validate_pcm_bytes(decoded, allow_empty=False)
    return decoded


def _encode_audio_bytes(audio_bytes: bytes) -> str:
    """Encode raw PCM bytes for websocket transport."""

    _validate_pcm_bytes(audio_bytes, allow_empty=False)
    return base64.b64encode(bytes(audio_bytes)).decode("ascii")


def _validate_pcm_bytes(audio_bytes: bytes, *, allow_empty: bool) -> None:
    """Validate bounded PCM s16le payload bytes."""

    if not isinstance(audio_bytes, (bytes, bytearray, memoryview)):
        raise VoiceContractError("pcm_bytes must be bytes-like")
    audio_length = len(audio_bytes)
    if audio_length == 0:
        if allow_empty:
            return
        raise VoiceContractError("pcm_bytes must not be empty")
    if audio_length > MAX_PCM_FRAME_BYTES:
        raise VoiceContractError(
            f"PCM frame exceeds max size {MAX_PCM_FRAME_BYTES} bytes"
        )
    if audio_length % PCM_SAMPLE_WIDTH_BYTES != 0:
        raise VoiceContractError(
            "pcm_bytes length must be aligned to 16-bit PCM sample width"
        )


def _validate_short_text(value: str | None, *, field: str, max_length: int) -> None:
    """Validate one optional text field length."""

    if value is None:
        return
    if len(value) > max_length:
        raise VoiceContractError(f"{field} exceeds max length {max_length}")


def _validate_timestamp_ms(value: int | None, *, field: str) -> None:
    """Validate one optional non-negative timestamp in milliseconds."""

    if value is None:
        return
    if value < 0:
        raise VoiceContractError(f"{field} must be non-negative")


def _validate_ratio(value: float | None, *, field: str) -> None:
    """Validate one optional ratio in ``[0.0, 1.0]``."""

    if value is None:
        return
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise VoiceContractError(f"{field} must be a finite ratio in [0.0, 1.0]")


def _validate_enum(
    value: str | None,
    *,
    field: str,
    allowed: frozenset[str],
) -> None:
    """Validate one optional short enum value."""

    if value is None:
        return
    if value not in allowed:
        raise VoiceContractError(f"{field} must be one of {sorted(allowed)!r}")


def _with_common_metadata(
    payload: dict[str, Any],
    *,
    event_id: str | None,
    sent_at_unix_ms: int | None,
) -> dict[str, Any]:
    """Attach optional event metadata to one payload."""

    if event_id is not None:
        payload["event_id"] = event_id
    if sent_at_unix_ms is not None:
        payload["sent_at_unix_ms"] = sent_at_unix_ms
    return payload


def _parse_common_metadata(payload_dict: Mapping[str, Any]) -> tuple[str | None, int | None]:
    """Parse optional event metadata from one payload mapping."""

    return (
        _coerce_optional_text(payload_dict.get("event_id"), max_length=MAX_EVENT_ID_LENGTH),
        _coerce_optional_non_negative_int(payload_dict.get("sent_at_unix_ms")),
    )


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
    speaker_associated: bool | None = None
    speaker_association_confidence: float | None = None
    background_media_likely: bool | None = None
    speech_overlap_likely: bool | None = None
    voice_quiet_until_utc: str | None = None
    state_attested: bool = True
    protocol_version: int = VOICE_PROTOCOL_VERSION
    subprotocol: str = VOICE_SUBPROTOCOL
    audio_encoding: str = AUDIO_ENCODING_PCM_S16LE
    audio_transport: str = AUDIO_TRANSPORT_JSON_B64
    turn_detection_mode: str | None = None
    noise_reduction_profile: str | None = None
    resume_token: str | None = None
    capabilities: tuple[str, ...] = ()
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    def validate(self) -> None:
        _require_text(
            self.session_id, field="session_id", max_length=MAX_SESSION_ID_LENGTH
        )
        if self.sample_rate < MIN_SAMPLE_RATE or self.sample_rate > MAX_SAMPLE_RATE:
            raise VoiceContractError(
                f"sample_rate must be within [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]"
            )
        if self.channels not in ALLOWED_CHANNELS:
            raise VoiceContractError(f"channels must be one of {sorted(ALLOWED_CHANNELS)!r}")
        if self.chunk_ms < MIN_CHUNK_MS or self.chunk_ms > MAX_CHUNK_MS:
            raise VoiceContractError(
                f"chunk_ms must be within [{MIN_CHUNK_MS}, {MAX_CHUNK_MS}]"
            )
        if self.protocol_version < 1:
            raise VoiceContractError("protocol_version must be >= 1")
        _require_text(
            self.initial_state, field="initial_state", max_length=MAX_STATE_LENGTH
        )
        _validate_short_text(self.trace_id, field="trace_id", max_length=MAX_TRACE_ID_LENGTH)
        _validate_short_text(self.detail, field="detail", max_length=MAX_DETAIL_LENGTH)
        _validate_short_text(
            self.attention_state, field="attention_state", max_length=MAX_STATE_LENGTH
        )
        _validate_short_text(
            self.interaction_intent_state,
            field="interaction_intent_state",
            max_length=MAX_STATE_LENGTH,
        )
        _validate_short_text(
            self.recommended_channel,
            field="recommended_channel",
            max_length=MAX_CHANNEL_NAME_LENGTH,
        )
        _validate_short_text(
            self.voice_quiet_until_utc,
            field="voice_quiet_until_utc",
            max_length=64,
        )
        _validate_short_text(
            self.subprotocol, field="subprotocol", max_length=MAX_MESSAGE_TYPE_LENGTH
        )
        _validate_short_text(
            self.resume_token, field="resume_token", max_length=MAX_TEXT_LENGTH
        )
        _validate_ratio(
            self.speaker_association_confidence,
            field="speaker_association_confidence",
        )
        _validate_enum(
            self.audio_encoding, field="audio_encoding", allowed=ALLOWED_AUDIO_ENCODINGS
        )
        _validate_enum(
            self.audio_transport,
            field="audio_transport",
            allowed=ALLOWED_AUDIO_TRANSPORTS,
        )
        _validate_enum(
            self.turn_detection_mode,
            field="turn_detection_mode",
            allowed=ALLOWED_TURN_DETECTION_MODES,
        )
        _validate_enum(
            self.noise_reduction_profile,
            field="noise_reduction_profile",
            allowed=ALLOWED_NOISE_REDUCTION_PROFILES,
        )
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")
        if len(self.capabilities) > MAX_CAPABILITIES:
            raise VoiceContractError(f"capabilities exceeds max size {MAX_CAPABILITIES}")
        for capability in self.capabilities:
            _validate_short_text(
                capability,
                field="capability",
                max_length=MAX_CAPABILITY_LENGTH,
            )

    def expected_pcm_bytes(self) -> int:
        """Return the nominal PCM byte size for one frame."""

        return (
            self.sample_rate
            * self.channels
            * PCM_SAMPLE_WIDTH_BYTES
            * self.chunk_ms
        ) // 1000

    def max_pcm_bytes(self) -> int:
        """Return the largest PCM frame size tolerated for this session."""

        expected = max(self.expected_pcm_bytes(), PCM_SAMPLE_WIDTH_BYTES * self.channels)
        return min(MAX_PCM_FRAME_BYTES, expected * 2)

    def validate_audio_frame(self, frame: "OrchestratorVoiceAudioFrame") -> None:
        """Validate one audio frame against the negotiated hello session."""

        frame.validate()
        if frame.pcm_bytes:
            audio_length = len(frame.pcm_bytes)
            if audio_length % (PCM_SAMPLE_WIDTH_BYTES * self.channels) != 0:
                raise VoiceContractError(
                    "pcm_bytes length is not aligned to the negotiated channel count"
                )
            if audio_length > self.max_pcm_bytes():
                raise VoiceContractError(
                    "pcm_bytes exceeds the negotiated session frame size budget"
                )

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "type": "voice_hello",
            "session_id": self.session_id,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_ms": self.chunk_ms,
            "initial_state": self.initial_state,
            "follow_up_allowed": self.follow_up_allowed,
            "state_attested": self.state_attested,
            "protocol_version": self.protocol_version,
            "subprotocol": self.subprotocol,
            "audio_encoding": self.audio_encoding,
            "audio_transport": self.audio_transport,
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
        if self.speaker_associated is not None:
            payload["speaker_associated"] = self.speaker_associated
        if self.speaker_association_confidence is not None:
            payload["speaker_association_confidence"] = self.speaker_association_confidence
        if self.background_media_likely is not None:
            payload["background_media_likely"] = self.background_media_likely
        if self.speech_overlap_likely is not None:
            payload["speech_overlap_likely"] = self.speech_overlap_likely
        if self.voice_quiet_until_utc is not None:
            payload["voice_quiet_until_utc"] = self.voice_quiet_until_utc
        if self.turn_detection_mode is not None:
            payload["turn_detection_mode"] = self.turn_detection_mode
        if self.noise_reduction_profile is not None:
            payload["noise_reduction_profile"] = self.noise_reduction_profile
        if self.resume_token is not None:
            payload["resume_token"] = self.resume_token
        if self.capabilities:
            payload["capabilities"] = list(self.capabilities)
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceHelloRequest":
        # BREAKING: malformed required hello payloads now raise VoiceContractError
        # instead of silently substituting empty or default values.
        payload_dict = _coerce_dict(payload)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        request = cls(
            session_id=_require_text(
                payload_dict.get("session_id"),
                field="session_id",
                max_length=MAX_SESSION_ID_LENGTH,
            ),
            trace_id=_coerce_optional_text(
                payload_dict.get("trace_id"), max_length=MAX_TRACE_ID_LENGTH
            ),
            sample_rate=_parse_positive_int(
                payload_dict.get("sample_rate"), field="sample_rate", default=DEFAULT_SAMPLE_RATE
            ),
            channels=_parse_positive_int(
                payload_dict.get("channels"), field="channels", default=DEFAULT_CHANNELS
            ),
            chunk_ms=_parse_positive_int(
                payload_dict.get("chunk_ms"), field="chunk_ms", default=DEFAULT_CHUNK_MS
            ),
            initial_state=_coerce_text(payload_dict.get("initial_state")) or "waiting",
            detail=_coerce_optional_text(
                payload_dict.get("detail"), max_length=MAX_DETAIL_LENGTH
            ),
            follow_up_allowed=_coerce_bool(
                payload_dict.get("follow_up_allowed"), default=False
            ),
            attention_state=_coerce_optional_text(
                payload_dict.get("attention_state"), max_length=MAX_STATE_LENGTH
            ),
            interaction_intent_state=_coerce_optional_text(
                payload_dict.get("interaction_intent_state"),
                max_length=MAX_STATE_LENGTH,
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
            recommended_channel=_coerce_optional_text(
                payload_dict.get("recommended_channel"),
                max_length=MAX_CHANNEL_NAME_LENGTH,
            ),
            speaker_associated=(
                _coerce_bool(payload_dict.get("speaker_associated"))
                if payload_dict.get("speaker_associated") is not None
                else None
            ),
            speaker_association_confidence=_coerce_optional_ratio(
                payload_dict.get("speaker_association_confidence")
            ),
            background_media_likely=(
                _coerce_bool(payload_dict.get("background_media_likely"))
                if payload_dict.get("background_media_likely") is not None
                else None
            ),
            speech_overlap_likely=(
                _coerce_bool(payload_dict.get("speech_overlap_likely"))
                if payload_dict.get("speech_overlap_likely") is not None
                else None
            ),
            voice_quiet_until_utc=_coerce_optional_text(
                payload_dict.get("voice_quiet_until_utc"),
                max_length=64,
            ),
            state_attested=_coerce_bool(payload_dict.get("state_attested"), default=False),
            protocol_version=_parse_positive_int(
                payload_dict.get("protocol_version"), field="protocol_version", default=VOICE_PROTOCOL_VERSION
            ),
            subprotocol=_coerce_text(payload_dict.get("subprotocol")) or VOICE_SUBPROTOCOL,
            audio_encoding=_coerce_text(payload_dict.get("audio_encoding"))
            or AUDIO_ENCODING_PCM_S16LE,
            audio_transport=_coerce_text(payload_dict.get("audio_transport"))
            or AUDIO_TRANSPORT_JSON_B64,
            turn_detection_mode=_coerce_optional_text(
                payload_dict.get("turn_detection_mode"),
                max_length=MAX_MESSAGE_TYPE_LENGTH,
            ),
            noise_reduction_profile=_coerce_optional_text(
                payload_dict.get("noise_reduction_profile"),
                max_length=MAX_MESSAGE_TYPE_LENGTH,
            ),
            resume_token=_coerce_optional_text(
                payload_dict.get("resume_token"), max_length=MAX_TEXT_LENGTH
            ),
            capabilities=_normalize_string_list(
                payload_dict.get("capabilities"),
                max_items=MAX_CAPABILITIES,
                max_item_length=MAX_CAPABILITY_LENGTH,
            )
            if payload_dict.get("capabilities") is not None
            else (),
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        request.validate()
        return request


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
    speaker_associated: bool | None = None
    speaker_association_confidence: float | None = None
    background_media_likely: bool | None = None
    speech_overlap_likely: bool | None = None
    voice_quiet_until_utc: str | None = None
    turn_detection_mode: str | None = None
    end_of_turn_probability: float | None = None
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    def validate(self) -> None:
        _require_text(self.state, field="state", max_length=MAX_STATE_LENGTH)
        _validate_short_text(self.detail, field="detail", max_length=MAX_DETAIL_LENGTH)
        _validate_short_text(
            self.attention_state, field="attention_state", max_length=MAX_STATE_LENGTH
        )
        _validate_short_text(
            self.interaction_intent_state,
            field="interaction_intent_state",
            max_length=MAX_STATE_LENGTH,
        )
        _validate_short_text(
            self.recommended_channel,
            field="recommended_channel",
            max_length=MAX_CHANNEL_NAME_LENGTH,
        )
        _validate_short_text(
            self.voice_quiet_until_utc,
            field="voice_quiet_until_utc",
            max_length=64,
        )
        _validate_ratio(
            self.speaker_association_confidence,
            field="speaker_association_confidence",
        )
        _validate_ratio(
            self.end_of_turn_probability,
            field="end_of_turn_probability",
        )
        _validate_enum(
            self.turn_detection_mode,
            field="turn_detection_mode",
            allowed=ALLOWED_TURN_DETECTION_MODES,
        )
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")

    def to_payload(self, *, include_type: bool = True) -> dict[str, Any]:
        self.validate()
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
        if self.speaker_associated is not None:
            payload["speaker_associated"] = self.speaker_associated
        if self.speaker_association_confidence is not None:
            payload["speaker_association_confidence"] = self.speaker_association_confidence
        if self.background_media_likely is not None:
            payload["background_media_likely"] = self.background_media_likely
        if self.speech_overlap_likely is not None:
            payload["speech_overlap_likely"] = self.speech_overlap_likely
        if self.voice_quiet_until_utc is not None:
            payload["voice_quiet_until_utc"] = self.voice_quiet_until_utc
        if self.turn_detection_mode is not None:
            payload["turn_detection_mode"] = self.turn_detection_mode
        if self.end_of_turn_probability is not None:
            payload["end_of_turn_probability"] = self.end_of_turn_probability
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceRuntimeStateEvent":
        payload_dict = _coerce_dict(payload)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        event = cls(
            state=_coerce_text(payload_dict.get("state")) or "waiting",
            detail=_coerce_optional_text(
                payload_dict.get("detail"), max_length=MAX_DETAIL_LENGTH
            ),
            follow_up_allowed=_coerce_bool(
                payload_dict.get("follow_up_allowed"), default=False
            ),
            attention_state=_coerce_optional_text(
                payload_dict.get("attention_state"), max_length=MAX_STATE_LENGTH
            ),
            interaction_intent_state=_coerce_optional_text(
                payload_dict.get("interaction_intent_state"),
                max_length=MAX_STATE_LENGTH,
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
            recommended_channel=_coerce_optional_text(
                payload_dict.get("recommended_channel"),
                max_length=MAX_CHANNEL_NAME_LENGTH,
            ),
            speaker_associated=(
                _coerce_bool(payload_dict.get("speaker_associated"))
                if payload_dict.get("speaker_associated") is not None
                else None
            ),
            speaker_association_confidence=_coerce_optional_ratio(
                payload_dict.get("speaker_association_confidence")
            ),
            background_media_likely=(
                _coerce_bool(payload_dict.get("background_media_likely"))
                if payload_dict.get("background_media_likely") is not None
                else None
            ),
            speech_overlap_likely=(
                _coerce_bool(payload_dict.get("speech_overlap_likely"))
                if payload_dict.get("speech_overlap_likely") is not None
                else None
            ),
            voice_quiet_until_utc=_coerce_optional_text(
                payload_dict.get("voice_quiet_until_utc"),
                max_length=64,
            ),
            turn_detection_mode=_coerce_optional_text(
                payload_dict.get("turn_detection_mode"),
                max_length=MAX_MESSAGE_TYPE_LENGTH,
            ),
            end_of_turn_probability=_coerce_optional_ratio(
                payload_dict.get("end_of_turn_probability")
            ),
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceAudioFrame:
    """Represent one bounded PCM frame sent from edge to server."""

    sequence: int
    pcm_bytes: bytes
    runtime_state: OrchestratorVoiceRuntimeStateEvent | None = None
    start_sample: int | None = None
    stream_ended: bool = False
    speech_probability: float | None = None
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    @property
    def sample_count(self) -> int:
        """Return the number of mono-equivalent PCM samples in this frame."""

        return len(self.pcm_bytes) // PCM_SAMPLE_WIDTH_BYTES

    def validate(self) -> None:
        if self.sequence < 0:
            raise VoiceContractError("sequence must be non-negative")
        _validate_pcm_bytes(self.pcm_bytes, allow_empty=self.stream_ended)
        if self.start_sample is not None and self.start_sample < 0:
            raise VoiceContractError("start_sample must be non-negative")
        _validate_ratio(self.speech_probability, field="speech_probability")
        if self.runtime_state is not None:
            self.runtime_state.validate()
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "type": "voice_audio_frame",
            "sequence": self.sequence,
            "pcm_s16le_b64": (
                _encode_audio_bytes(self.pcm_bytes) if self.pcm_bytes else ""
            ),
        }
        if self.runtime_state is not None:
            payload["runtime_state"] = self.runtime_state.to_payload()
        if self.start_sample is not None:
            payload["start_sample"] = self.start_sample
        if self.stream_ended:
            payload["stream_ended"] = True
        if self.speech_probability is not None:
            payload["speech_probability"] = self.speech_probability
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceAudioFrame":
        # BREAKING: invalid or oversized PCM payloads now raise VoiceContractError
        # instead of decoding to empty bytes and continuing silently.
        payload_dict = _coerce_dict(payload)
        runtime_state_payload = payload_dict.get("runtime_state")
        stream_ended = _coerce_bool(payload_dict.get("stream_ended"), default=False)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        pcm_value = payload_dict.get("pcm_s16le_b64")
        if stream_ended and not pcm_value:
            pcm_bytes = b""
        else:
            pcm_bytes = _decode_audio_bytes(pcm_value)
        event = cls(
            sequence=_parse_non_negative_int(payload_dict.get("sequence"), field="sequence", default=0),
            pcm_bytes=pcm_bytes,
            runtime_state=(
                OrchestratorVoiceRuntimeStateEvent.from_payload(runtime_state_payload)
                if isinstance(runtime_state_payload, Mapping)
                else None
            ),
            start_sample=_coerce_optional_non_negative_int(payload_dict.get("start_sample")),
            stream_ended=stream_ended,
            speech_probability=_coerce_optional_ratio(payload_dict.get("speech_probability")),
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceIdentityProfile:
    """Describe one read-only household voice profile snapshot."""

    user_id: str
    embedding: tuple[float, ...]
    display_name: str | None = None
    primary_user: bool = False
    sample_count: int = 1
    average_duration_ms: int = 0
    updated_at: str | None = None
    embedding_space: str | None = None
    embedding_metric: str | None = None
    embedding_normalized: bool | None = None

    def validate(self) -> None:
        _require_text(self.user_id, field="user_id", max_length=MAX_SESSION_ID_LENGTH)
        if not self.embedding:
            raise VoiceContractError("embedding must not be empty")
        if len(self.embedding) > MAX_EMBEDDING_DIM:
            raise VoiceContractError(
                f"embedding exceeds max dimension {MAX_EMBEDDING_DIM}"
            )
        for item in self.embedding:
            if not math.isfinite(item):
                raise VoiceContractError("embedding contains non-finite values")
        if self.sample_count <= 0:
            raise VoiceContractError("sample_count must be positive")
        if self.average_duration_ms < 0:
            raise VoiceContractError("average_duration_ms must be non-negative")
        _validate_short_text(
            self.display_name, field="display_name", max_length=MAX_CHANNEL_NAME_LENGTH
        )
        _validate_short_text(self.updated_at, field="updated_at", max_length=64)
        _validate_short_text(
            self.embedding_space,
            field="embedding_space",
            max_length=MAX_CHANNEL_NAME_LENGTH,
        )
        _validate_enum(
            self.embedding_metric,
            field="embedding_metric",
            allowed=ALLOWED_EMBEDDING_METRICS,
        )

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "user_id": self.user_id,
            "embedding": list(self.embedding),
            "primary_user": self.primary_user,
            "sample_count": self.sample_count,
            "average_duration_ms": self.average_duration_ms,
        }
        if self.display_name is not None:
            payload["display_name"] = self.display_name
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.embedding_space is not None:
            payload["embedding_space"] = self.embedding_space
        if self.embedding_metric is not None:
            payload["embedding_metric"] = self.embedding_metric
        if self.embedding_normalized is not None:
            payload["embedding_normalized"] = self.embedding_normalized
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceIdentityProfile | None":
        payload_dict = _coerce_dict(payload)
        embedding = _coerce_embedding(payload_dict.get("embedding"))
        if not embedding:
            return None
        user_id = _coerce_text(payload_dict.get("user_id"))
        if not user_id:
            return None
        try:
            profile = cls(
                user_id=user_id,
                embedding=embedding,
                display_name=_coerce_optional_text(
                    payload_dict.get("display_name"),
                    max_length=MAX_CHANNEL_NAME_LENGTH,
                ),
                primary_user=_coerce_bool(payload_dict.get("primary_user"), default=False),
                sample_count=_coerce_positive_int(
                    payload_dict.get("sample_count"), default=1
                ),
                average_duration_ms=_coerce_non_negative_int(
                    payload_dict.get("average_duration_ms"),
                    default=0,
                ),
                updated_at=_coerce_optional_text(
                    payload_dict.get("updated_at"), max_length=64
                ),
                embedding_space=_coerce_optional_text(
                    payload_dict.get("embedding_space"),
                    max_length=MAX_CHANNEL_NAME_LENGTH,
                ),
                embedding_metric=_coerce_optional_text(
                    payload_dict.get("embedding_metric"),
                    max_length=MAX_CHANNEL_NAME_LENGTH,
                ),
                embedding_normalized=(
                    _coerce_bool(payload_dict.get("embedding_normalized"))
                    if payload_dict.get("embedding_normalized") is not None
                    else None
                ),
            )
            profile.validate()
            return profile
        except VoiceContractError:
            return None


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceIdentityProfilesEvent:
    """Sync the current household voice profile snapshot to the voice gateway."""

    revision: str | None
    profiles: tuple[OrchestratorVoiceIdentityProfile, ...]
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    def validate(self) -> None:
        _validate_short_text(self.revision, field="revision", max_length=MAX_TRACE_ID_LENGTH)
        if len(self.profiles) > MAX_PROFILES:
            raise VoiceContractError(f"profiles exceeds max size {MAX_PROFILES}")
        for profile in self.profiles:
            profile.validate()
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "type": "voice_identity_profiles",
            "profiles": [profile.to_payload() for profile in self.profiles],
        }
        if self.revision is not None:
            payload["revision"] = self.revision
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceIdentityProfilesEvent":
        payload_dict = _coerce_dict(payload)
        profiles_raw = payload_dict.get("profiles")
        if profiles_raw is not None and (
            isinstance(profiles_raw, (str, bytes, bytearray, memoryview))
            or not isinstance(profiles_raw, Sequence)
        ):
            raise VoiceContractError("profiles must be a list")
        profiles: list[OrchestratorVoiceIdentityProfile] = []
        if isinstance(profiles_raw, Sequence):
            if len(profiles_raw) > MAX_PROFILES:
                raise VoiceContractError(f"profiles exceeds max size {MAX_PROFILES}")
            for item in profiles_raw:
                profile = OrchestratorVoiceIdentityProfile.from_payload(item)
                if profile is not None:
                    profiles.append(profile)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        event = cls(
            revision=_coerce_optional_text(
                payload_dict.get("revision"), max_length=MAX_TRACE_ID_LENGTH
            ),
            profiles=tuple(profiles),
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceKeepAliveEvent:
    """Keep one long-lived voice websocket session warm."""

    session_id: str | None = None
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    def validate(self) -> None:
        _validate_short_text(
            self.session_id, field="session_id", max_length=MAX_SESSION_ID_LENGTH
        )
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {"type": "voice_keepalive"}
        if self.session_id is not None:
            payload["session_id"] = self.session_id
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceKeepAliveEvent":
        payload_dict = _coerce_dict(payload)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        event = cls(
            session_id=_coerce_optional_text(
                payload_dict.get("session_id"), max_length=MAX_SESSION_ID_LENGTH
            ),
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceReadyEvent:
    """Acknowledge that the server accepted one voice websocket session."""

    session_id: str
    backend: str
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    def validate(self) -> None:
        _require_text(
            self.session_id, field="session_id", max_length=MAX_SESSION_ID_LENGTH
        )
        _require_text(self.backend, field="backend", max_length=MAX_BACKEND_LENGTH)
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "type": "voice_ready",
            "session_id": self.session_id,
            "backend": self.backend,
        }
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceReadyEvent":
        payload_dict = _coerce_dict(payload)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        event = cls(
            session_id=_require_text(
                payload_dict.get("session_id"),
                field="session_id",
                max_length=MAX_SESSION_ID_LENGTH,
            ),
            backend=_coerce_text(payload_dict.get("backend")) or "unknown",
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceWakeConfirmedEvent:
    """Represent one confirmed remote voice activation match."""

    matched_phrase: str | None
    remaining_text: str
    backend: str
    detector_label: str | None = None
    score: float | None = None
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    def validate(self) -> None:
        _validate_short_text(
            self.matched_phrase, field="matched_phrase", max_length=MAX_TEXT_LENGTH
        )
        _validate_short_text(
            self.detector_label, field="detector_label", max_length=MAX_CHANNEL_NAME_LENGTH
        )
        _validate_short_text(
            self.remaining_text, field="remaining_text", max_length=MAX_TEXT_LENGTH
        )
        _require_text(self.backend, field="backend", max_length=MAX_BACKEND_LENGTH)
        _validate_ratio(self.score, field="score")
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
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
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceWakeConfirmedEvent":
        payload_dict = _coerce_dict(payload)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        event = cls(
            matched_phrase=_coerce_optional_text(
                payload_dict.get("matched_phrase"),
                max_length=MAX_TEXT_LENGTH,
            ),
            remaining_text=_coerce_text(payload_dict.get("remaining_text")),
            backend=_coerce_text(payload_dict.get("backend")) or "unknown",
            detector_label=_coerce_optional_text(
                payload_dict.get("detector_label"),
                max_length=MAX_CHANNEL_NAME_LENGTH,
            ),
            score=_coerce_optional_ratio(payload_dict.get("score")),
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceTranscriptCommittedEvent:
    """Commit one transcribed user utterance from the live remote stream."""

    transcript: str
    source: str = "listening"
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    def validate(self) -> None:
        _require_text(self.transcript, field="transcript", max_length=MAX_TEXT_LENGTH)
        _require_text(self.source, field="source", max_length=MAX_STATE_LENGTH)
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "type": "transcript_committed",
            "transcript": self.transcript,
            "source": self.source,
        }
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceTranscriptCommittedEvent":
        payload_dict = _coerce_dict(payload)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        event = cls(
            transcript=_require_text(
                payload_dict.get("transcript"),
                field="transcript",
                max_length=MAX_TEXT_LENGTH,
            ),
            source=_coerce_text(payload_dict.get("source")) or "listening",
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceFollowUpClosedEvent:
    """Signal that the server closed the current remote follow-up window."""

    reason: str
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    def validate(self) -> None:
        _require_text(self.reason, field="reason", max_length=MAX_REASON_LENGTH)
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload = {"type": "follow_up_closed", "reason": self.reason}
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceFollowUpClosedEvent":
        payload_dict = _coerce_dict(payload)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        event = cls(
            reason=_coerce_text(payload_dict.get("reason")) or "timeout",
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceBargeInInterruptEvent:
    """Tell the edge to stop the current answer because the user interrupted."""

    transcript_preview: str | None = None
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    def validate(self) -> None:
        _validate_short_text(
            self.transcript_preview,
            field="transcript_preview",
            max_length=MAX_TEXT_LENGTH,
        )
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload = {"type": "barge_in_interrupt"}
        if self.transcript_preview is not None:
            payload["transcript_preview"] = self.transcript_preview
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceBargeInInterruptEvent":
        payload_dict = _coerce_dict(payload)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        event = cls(
            transcript_preview=_coerce_optional_text(
                payload_dict.get("transcript_preview"),
                max_length=MAX_TEXT_LENGTH,
            ),
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceGoAwayEvent:
    """Tell the edge the session will be closed soon and may be resumed."""

    reason: str = "server_restart"
    time_left_ms: int = 0
    resume_token: str | None = None
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    def validate(self) -> None:
        _require_text(self.reason, field="reason", max_length=MAX_REASON_LENGTH)
        if self.time_left_ms < 0:
            raise VoiceContractError("time_left_ms must be non-negative")
        _validate_short_text(
            self.resume_token, field="resume_token", max_length=MAX_TEXT_LENGTH
        )
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "type": "voice_go_away",
            "reason": self.reason,
            "time_left_ms": self.time_left_ms,
        }
        if self.resume_token is not None:
            payload["resume_token"] = self.resume_token
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceGoAwayEvent":
        payload_dict = _coerce_dict(payload)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        event = cls(
            reason=_coerce_text(payload_dict.get("reason")) or "server_restart",
            time_left_ms=_parse_non_negative_int(
                payload_dict.get("time_left_ms"), field="time_left_ms", default=0
            ),
            resume_token=_coerce_optional_text(
                payload_dict.get("resume_token"), max_length=MAX_TEXT_LENGTH
            ),
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceErrorEvent:
    """Represent one sanitized voice-session transport or runtime failure."""

    error: str
    event_id: str | None = None
    sent_at_unix_ms: int | None = None

    def validate(self) -> None:
        _require_text(self.error, field="error", max_length=MAX_TEXT_LENGTH)
        _validate_short_text(self.event_id, field="event_id", max_length=MAX_EVENT_ID_LENGTH)
        _validate_timestamp_ms(self.sent_at_unix_ms, field="sent_at_unix_ms")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload = {"type": "voice_error", "error": self.error}
        return _with_common_metadata(
            payload,
            event_id=self.event_id,
            sent_at_unix_ms=self.sent_at_unix_ms,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceErrorEvent":
        payload_dict = _coerce_dict(payload)
        event_id, sent_at_unix_ms = _parse_common_metadata(payload_dict)
        event = cls(
            error=_coerce_text(payload_dict.get("error")) or "Voice session failed",
            event_id=event_id,
            sent_at_unix_ms=sent_at_unix_ms,
        )
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceUnknownServerEvent:
    """Preserve unsupported server events for forward compatibility."""

    message_type: str
    payload: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.payload)
        if "type" not in payload:
            payload["type"] = self.message_type
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceUnknownServerEvent":
        payload_dict = _coerce_dict(payload)
        message_type = _coerce_text(payload_dict.get("type")) or "unknown"
        if len(message_type) > MAX_MESSAGE_TYPE_LENGTH:
            raise VoiceContractError("message type exceeds max length")
        return cls(message_type=message_type, payload=dict(payload_dict))


@dataclass(frozen=True, slots=True)
class OrchestratorVoiceUnknownClientEvent:
    """Preserve unsupported client events for forward compatibility."""

    message_type: str
    payload: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.payload)
        if "type" not in payload:
            payload["type"] = self.message_type
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorVoiceUnknownClientEvent":
        payload_dict = _coerce_dict(payload)
        message_type = _coerce_text(payload_dict.get("type")) or "unknown"
        if len(message_type) > MAX_MESSAGE_TYPE_LENGTH:
            raise VoiceContractError("message type exceeds max length")
        return cls(message_type=message_type, payload=dict(payload_dict))


VoiceClientEvent = (
    OrchestratorVoiceHelloRequest
    | OrchestratorVoiceAudioFrame
    | OrchestratorVoiceRuntimeStateEvent
    | OrchestratorVoiceIdentityProfilesEvent
    | OrchestratorVoiceKeepAliveEvent
    | OrchestratorVoiceUnknownClientEvent
)

VoiceServerEvent = (
    OrchestratorVoiceReadyEvent
    | OrchestratorVoiceWakeConfirmedEvent
    | OrchestratorVoiceTranscriptCommittedEvent
    | OrchestratorVoiceFollowUpClosedEvent
    | OrchestratorVoiceBargeInInterruptEvent
    | OrchestratorVoiceGoAwayEvent
    | OrchestratorVoiceKeepAliveEvent
    | OrchestratorVoiceErrorEvent
    | OrchestratorVoiceUnknownServerEvent
)


def decode_voice_client_event(
    payload: Any,
    *,
    strict: bool = False,
) -> VoiceClientEvent:
    """Decode one client-originated voice websocket event.

    # BREAKING:
    # Older code had no symmetric client decoder and often dispatched manually.
    # Unknown events now round-trip as OrchestratorVoiceUnknownClientEvent unless
    # strict=True, which is safer for mixed-version deploys.
    """

    payload_dict = _coerce_dict(payload)
    message_type = _coerce_text(payload_dict.get("type"))
    if message_type == "voice_hello":
        return OrchestratorVoiceHelloRequest.from_payload(payload_dict)
    if message_type == "voice_audio_frame":
        return OrchestratorVoiceAudioFrame.from_payload(payload_dict)
    if message_type == "voice_runtime_state":
        return OrchestratorVoiceRuntimeStateEvent.from_payload(payload_dict)
    if message_type == "voice_identity_profiles":
        return OrchestratorVoiceIdentityProfilesEvent.from_payload(payload_dict)
    if message_type == "voice_keepalive":
        return OrchestratorVoiceKeepAliveEvent.from_payload(payload_dict)
    if strict:
        raise VoiceContractError(
            f"Unsupported voice client event type: {message_type or '<empty>'}"
        )
    return OrchestratorVoiceUnknownClientEvent.from_payload(payload_dict)


def decode_voice_server_event(
    payload: Any,
    *,
    strict: bool = False,
) -> VoiceServerEvent:
    """Decode one server-originated voice websocket event.

    # BREAKING:
    # Unknown server events now round-trip as OrchestratorVoiceUnknownServerEvent
    # unless strict=True. This avoids crashing older edges during mixed-version
    # rollouts while still supporting strict callers.
    """

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
    if message_type == "voice_go_away":
        return OrchestratorVoiceGoAwayEvent.from_payload(payload_dict)
    if message_type == "voice_keepalive":
        return OrchestratorVoiceKeepAliveEvent.from_payload(payload_dict)
    if message_type == "voice_error":
        return OrchestratorVoiceErrorEvent.from_payload(payload_dict)
    if strict:
        raise VoiceContractError(
            f"Unsupported voice server event type: {message_type or '<empty>'}"
        )
    return OrchestratorVoiceUnknownServerEvent.from_payload(payload_dict)


__all__ = [
    "AUDIO_ENCODING_PCM_S16LE",
    "AUDIO_TRANSPORT_JSON_B64",
    "DEFAULT_CHANNELS",
    "DEFAULT_CHUNK_MS",
    "DEFAULT_SAMPLE_RATE",
    "OrchestratorVoiceAudioFrame",
    "OrchestratorVoiceBargeInInterruptEvent",
    "OrchestratorVoiceErrorEvent",
    "OrchestratorVoiceFollowUpClosedEvent",
    "OrchestratorVoiceGoAwayEvent",
    "OrchestratorVoiceHelloRequest",
    "OrchestratorVoiceIdentityProfile",
    "OrchestratorVoiceIdentityProfilesEvent",
    "OrchestratorVoiceKeepAliveEvent",
    "OrchestratorVoiceReadyEvent",
    "OrchestratorVoiceRuntimeStateEvent",
    "OrchestratorVoiceTranscriptCommittedEvent",
    "OrchestratorVoiceUnknownClientEvent",
    "OrchestratorVoiceUnknownServerEvent",
    "OrchestratorVoiceWakeConfirmedEvent",
    "VOICE_PROTOCOL_VERSION",
    "VOICE_SUBPROTOCOL",
    "VoiceClientEvent",
    "VoiceContractError",
    "VoiceServerEvent",
    "decode_voice_client_event",
    "decode_voice_server_event",
]
