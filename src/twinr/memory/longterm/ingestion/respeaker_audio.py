"""Extract structured ReSpeaker audio routine seeds from sensor observations.

This helper keeps XVF3800-specific long-term memory extraction separate from
the generic multimodal event extractor. It reads only allowlisted structured
automation facts and emits low-confidence routine-seed patterns that later
sensor-memory compilation can aggregate across days. It never consumes or
persists raw PCM payloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.ingestion.respeaker_contract import (
    RESPEAKER_SESSION_MEMORY,
    ReSpeakerClaimEvidence,
    claim_confidence_summary,
    claim_contract_payload_subset,
    claim_requires_confirmation,
    claim_session_id,
    claim_sources,
    coerce_respeaker_claim_evidence_map,
)


_RESPEAKER_SOURCE = "respeaker_xvf3800"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})
_MAX_EVENT_NAMES = 12
_MAX_SIGNAL_AGE_S = 1.5
_CONVERSATION_START_CLAIMS = (
    "speech_detected",
    "recent_speech_age_s",
    "direction_confidence",
    "azimuth_deg",
)
_QUIET_WINDOW_CLAIMS = (
    "room_quiet",
    "non_speech_audio_likely",
    "background_media_likely",
)
_FRICTION_CLAIMS = (
    "speech_overlap_likely",
    "barge_in_detected",
    "background_media_likely",
)
_RESUME_CLAIMS = (
    "recent_speech_age_s",
    "speech_detected",
    "direction_confidence",
)


def _normalize_text(value: object | None) -> str:
    """Collapse one arbitrary value into single-line text."""

    return " ".join(str(value or "").split()).strip()


def _coerce_mapping(value: object | None) -> dict[str, object]:
    """Coerce one optional mapping-like payload into a plain dictionary."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    try:
        return dict(value or {})
    except (TypeError, ValueError):
        return {}


def _coerce_bool(value: object | None) -> bool | None:
    """Parse one optional boolean-like payload conservatively."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value != 0
    text = _normalize_text(value).lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def _coerce_optional_float(value: object | None) -> float | None:
    """Parse one optional finite float value."""

    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_optional_int(value: object | None) -> int | None:
    """Parse one optional integer value conservatively."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_tuple(value: object | None) -> tuple[str, ...]:
    """Normalize an optional scalar or sequence into bounded event names."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raw_items = (value,)
    else:
        try:
            raw_items = tuple(value)
        except TypeError:
            raw_items = (value,)
    normalized: list[str] = []
    for item in raw_items[:_MAX_EVENT_NAMES]:
        text = _normalize_text(item)
        if text:
            normalized.append(text)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class _ReSpeakerAudioContext:
    """Hold the allowlisted structured ReSpeaker audio facts for one event."""

    signal_source: str | None
    captured_at: float | None
    presence_session_id: int | None
    speech_detected: bool | None
    recent_speech_age_s: float | None
    assistant_output_active: bool | None
    direction_confidence: float | None
    room_quiet: bool | None
    non_speech_audio_likely: bool | None
    background_media_likely: bool | None
    room_busy_or_overlapping: bool | None
    quiet_window_open: bool | None
    presence_audio_active: bool | None
    recent_follow_up_speech: bool | None
    barge_in_recent: bool | None
    resume_window_open: bool | None
    mute_blocks_voice_capture: bool | None
    event_names: tuple[str, ...]
    claim_evidence_map: Mapping[str, ReSpeakerClaimEvidence]

    @classmethod
    def from_fact_map(cls, fact_map: Mapping[str, object], *, event_names: tuple[str, ...]) -> "_ReSpeakerAudioContext":
        """Build one normalized context from automation facts."""

        sensor = _coerce_mapping(fact_map.get("sensor"))
        vad = _coerce_mapping(fact_map.get("vad"))
        respeaker = _coerce_mapping(fact_map.get("respeaker"))
        audio_policy = _coerce_mapping(fact_map.get("audio_policy"))
        claim_evidence_map = coerce_respeaker_claim_evidence_map(respeaker.get("claim_contract"))
        return cls(
            signal_source=_normalize_text(vad.get("signal_source")) or None,
            captured_at=_coerce_optional_float(sensor.get("captured_at") or sensor.get("observed_at")),
            presence_session_id=_coerce_optional_int(sensor.get("presence_session_id")),
            speech_detected=_coerce_bool(vad.get("speech_detected")),
            recent_speech_age_s=_coerce_optional_float(vad.get("recent_speech_age_s")),
            assistant_output_active=_coerce_bool(vad.get("assistant_output_active")),
            direction_confidence=_coerce_optional_float(respeaker.get("direction_confidence")),
            room_quiet=_coerce_bool(vad.get("room_quiet")),
            non_speech_audio_likely=_coerce_bool(respeaker.get("non_speech_audio_likely")),
            background_media_likely=_coerce_bool(respeaker.get("background_media_likely")),
            room_busy_or_overlapping=_coerce_bool(audio_policy.get("room_busy_or_overlapping")),
            quiet_window_open=_coerce_bool(audio_policy.get("quiet_window_open")),
            presence_audio_active=_coerce_bool(audio_policy.get("presence_audio_active")),
            recent_follow_up_speech=_coerce_bool(audio_policy.get("recent_follow_up_speech")),
            barge_in_recent=_coerce_bool(audio_policy.get("barge_in_recent")),
            resume_window_open=_coerce_bool(audio_policy.get("resume_window_open")),
            mute_blocks_voice_capture=_coerce_bool(audio_policy.get("mute_blocks_voice_capture")),
            event_names=event_names,
            claim_evidence_map=claim_evidence_map,
        )

    @property
    def is_respeaker(self) -> bool:
        """Return whether the current event is backed by a ReSpeaker signal path."""

        return self.signal_source == _RESPEAKER_SOURCE


def extract_respeaker_audio_patterns(
    *,
    fact_map: Mapping[str, object],
    source_ref: LongTermSourceRefV1,
    date_key: str,
    daypart: str,
    event_names: tuple[str, ...],
) -> tuple[LongTermMemoryObjectV1, ...]:
    """Extract low-confidence ReSpeaker audio routine seeds from one sensor event.

    Args:
        fact_map: Structured automation facts for one sensor observation.
        source_ref: Shared source reference for the surrounding multimodal event.
        date_key: Local observation date used for bounded validity windows.
        daypart: Shared daypart bucket already resolved by the multimodal extractor.
        event_names: Rising-edge event names derived from the automation facts.

    Returns:
        A tuple of low-confidence pattern memories derived from allowlisted
        ReSpeaker audio facts. Returns an empty tuple when the observation is
        not backed by the XVF3800 path or when no relevant audio event fired.
    """

    normalized_event_names = _string_tuple(event_names)
    context = _ReSpeakerAudioContext.from_fact_map(fact_map, event_names=normalized_event_names)
    if not context.is_respeaker:
        return ()

    created: list[LongTermMemoryObjectV1] = []
    if _should_capture_conversation_start(context):
        created.append(
            _build_pattern(
                source_ref=source_ref,
                date_key=date_key,
                daypart=daypart,
                interaction_type="conversation_start_audio",
                summary=f"Voice conversation start was observed in the {daypart}.",
                details="Low-confidence ReSpeaker pattern derived from qualifying speech near the device.",
                confidence=_conversation_start_confidence(context),
                context=context,
                claim_names=_CONVERSATION_START_CLAIMS,
            )
        )
    if _should_capture_quiet_window(context):
        created.append(
            _build_pattern(
                source_ref=source_ref,
                date_key=date_key,
                daypart=daypart,
                interaction_type="quiet_window",
                summary=f"A quiet window around the device was observed in the {daypart}.",
                details="Low-confidence ReSpeaker pattern derived from a calm room state suitable for interaction.",
                confidence=_quiet_window_confidence(context),
                context=context,
                claim_names=_QUIET_WINDOW_CLAIMS,
            )
        )
    if _should_capture_friction(context):
        created.append(
            _build_pattern(
                source_ref=source_ref,
                date_key=date_key,
                daypart=daypart,
                interaction_type="friction_overlap",
                summary=f"Audio interruption or overlap was observed in the {daypart}.",
                details="Low-confidence ReSpeaker pattern derived from bounded overlap or interruption facts.",
                confidence=_friction_confidence(context),
                context=context,
                claim_names=_FRICTION_CLAIMS,
            )
        )
    if _should_capture_resume(context):
        created.append(
            _build_pattern(
                source_ref=source_ref,
                date_key=date_key,
                daypart=daypart,
                interaction_type="resume_follow_up",
                summary=f"A short spoken follow-up was observed in the {daypart}.",
                details="Low-confidence ReSpeaker pattern derived from a bounded resume window after recent speech.",
                confidence=_resume_confidence(context),
                context=context,
                claim_names=_RESUME_CLAIMS,
            )
        )
    return tuple(created)


def _build_pattern(
    *,
    source_ref: LongTermSourceRefV1,
    date_key: str,
    daypart: str,
    interaction_type: str,
    summary: str,
    details: str,
    confidence: float,
    context: _ReSpeakerAudioContext,
    claim_names: tuple[str, ...],
) -> LongTermMemoryObjectV1:
    """Build one bounded pattern object from normalized ReSpeaker context."""

    claim_confidence = claim_confidence_summary(
        context.claim_evidence_map,
        claim_names=claim_names,
    )
    claim_source, claim_source_type = claim_sources(
        context.claim_evidence_map,
        claim_names=claim_names,
    )
    claim_session = claim_session_id(
        context.claim_evidence_map,
        claim_names=claim_names,
        fallback_session_id=context.presence_session_id,
    )
    attributes = {
        "memory_domain": "respeaker_audio_routine",
        "memory_class": RESPEAKER_SESSION_MEMORY,
        "pattern_type": "interaction",
        "interaction_type": interaction_type,
        "daypart": daypart,
        "captured_at": None if context.captured_at is None else round(context.captured_at, 3),
        "presence_session_id": claim_session,
        "source_sensor": _RESPEAKER_SOURCE,
        "source_type": claim_source_type or "observed",
        "requires_confirmation": claim_requires_confirmation(
            context.claim_evidence_map,
            claim_names=claim_names,
            default=True,
        ),
        "claim_confidence": None if claim_confidence is None else round(claim_confidence, 4),
        "claim_source": claim_source,
        "claim_names": claim_names,
        "claim_contract": claim_contract_payload_subset(
            context.claim_evidence_map,
            claim_names=claim_names,
        ),
        "event_names": context.event_names,
        "recent_speech_age_s": None if context.recent_speech_age_s is None else round(context.recent_speech_age_s, 3),
        "direction_confidence": None if context.direction_confidence is None else round(context.direction_confidence, 4),
        "assistant_output_active": context.assistant_output_active,
        "room_quiet": context.room_quiet,
        "room_busy_or_overlapping": context.room_busy_or_overlapping,
        "barge_in_recent": context.barge_in_recent,
        "resume_window_open": context.resume_window_open,
        "non_speech_audio_likely": context.non_speech_audio_likely,
        "background_media_likely": context.background_media_likely,
        "mute_blocks_voice_capture": context.mute_blocks_voice_capture,
    }
    return LongTermMemoryObjectV1(
        memory_id=f"pattern:audio_interaction:{interaction_type}:{daypart}",
        kind="pattern",
        summary=summary,
        details=details,
        source=source_ref,
        confidence=confidence,
        sensitivity="low",
        slot_key=f"pattern:audio_interaction:{interaction_type}:{daypart}",
        value_key="audio_interaction",
        valid_from=date_key,
        valid_to=date_key,
        attributes=attributes,
    )


def _event_fired(context: _ReSpeakerAudioContext, event_name: str) -> bool:
    """Return whether one normalized event name fired for the current payload."""

    return event_name in context.event_names


def _recent_speech(context: _ReSpeakerAudioContext) -> bool:
    """Return whether speech happened recently enough for start/resume logic."""

    if context.speech_detected is True:
        return True
    if context.recent_speech_age_s is None:
        return False
    return context.recent_speech_age_s <= _MAX_SIGNAL_AGE_S


def _should_capture_conversation_start(context: _ReSpeakerAudioContext) -> bool:
    """Return whether the current event should seed a voice-start routine."""

    return (
        _event_fired(context, "audio_policy.presence_audio_active")
        and context.presence_audio_active is True
        and context.resume_window_open is not True
        and context.room_busy_or_overlapping is not True
        and context.background_media_likely is not True
        and context.mute_blocks_voice_capture is not True
        and _recent_speech(context)
    )


def _should_capture_quiet_window(context: _ReSpeakerAudioContext) -> bool:
    """Return whether the current event should seed a quiet-window routine."""

    return (
        _event_fired(context, "audio_policy.quiet_window_open")
        and context.quiet_window_open is True
        and context.presence_audio_active is not True
        and context.background_media_likely is not True
        and context.non_speech_audio_likely is not True
        and context.mute_blocks_voice_capture is not True
    )


def _should_capture_friction(context: _ReSpeakerAudioContext) -> bool:
    """Return whether the current event should seed an audio-friction routine."""

    return (
        (_event_fired(context, "audio_policy.room_busy_or_overlapping") or _event_fired(context, "audio_policy.barge_in_recent"))
        and (context.room_busy_or_overlapping is True or context.barge_in_recent is True)
        and context.background_media_likely is not True
    )


def _should_capture_resume(context: _ReSpeakerAudioContext) -> bool:
    """Return whether the current event should seed a short-resume routine."""

    return (
        _event_fired(context, "audio_policy.resume_window_open")
        and context.resume_window_open is True
        and context.recent_follow_up_speech is True
        and context.background_media_likely is not True
        and context.room_busy_or_overlapping is not True
        and _recent_speech(context)
    )


def _conversation_start_confidence(context: _ReSpeakerAudioContext) -> float:
    """Return one bounded confidence score for a voice-start seed."""

    confidence = 0.58
    if context.direction_confidence is not None:
        confidence += min(0.12, max(0.0, context.direction_confidence) * 0.12)
    if context.presence_session_id is not None:
        confidence += 0.05
    if context.recent_speech_age_s is not None and context.recent_speech_age_s <= 0.75:
        confidence += 0.05
    return min(0.84, confidence)


def _quiet_window_confidence(context: _ReSpeakerAudioContext) -> float:
    """Return one bounded confidence score for a quiet-window seed."""

    confidence = 0.57
    if context.room_quiet is True:
        confidence += 0.1
    if context.presence_session_id is not None:
        confidence += 0.04
    return min(0.82, confidence)


def _friction_confidence(context: _ReSpeakerAudioContext) -> float:
    """Return one bounded confidence score for an audio-friction seed."""

    confidence = 0.55
    if context.barge_in_recent is True:
        confidence += 0.12
    if context.assistant_output_active is True:
        confidence += 0.07
    if context.presence_session_id is not None:
        confidence += 0.03
    return min(0.84, confidence)


def _resume_confidence(context: _ReSpeakerAudioContext) -> float:
    """Return one bounded confidence score for a short-resume seed."""

    confidence = 0.57
    if context.recent_follow_up_speech is True:
        confidence += 0.1
    if context.direction_confidence is not None:
        confidence += min(0.08, max(0.0, context.direction_confidence) * 0.08)
    if context.presence_session_id is not None:
        confidence += 0.04
    return min(0.84, confidence)
