"""Derive conservative non-speech audio facts from ReSpeaker capture activity.

The XVF3800 exposes reliable speech-oriented host-control facts such as DoA,
speech detection, and beam speech energies, but it does not expose a direct
"music" or "background media" classifier. This module therefore combines the
bounded ambient capture window from the same microphone path with XVF3800
speech flags to produce conservative runtime facts that are safe to use only
for suppression or visual-first delivery.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.hardware.audio import AmbientAudioLevelSample
from twinr.hardware.respeaker.models import ReSpeakerSignalSnapshot


_MIN_ACTIVE_CHUNKS = 3
_NON_SPEECH_ACTIVE_RATIO = 0.45
_BACKGROUND_MEDIA_ACTIVE_RATIO = 0.72


@dataclass(frozen=True, slots=True)
class ReSpeakerAmbientClassification:
    """Store conservative activity facts derived from one capture window."""

    audio_activity_detected: bool | None = None
    non_speech_audio_likely: bool | None = None
    background_media_likely: bool | None = None


def classify_respeaker_ambient_audio(
    *,
    signal_snapshot: ReSpeakerSignalSnapshot,
    sample: AmbientAudioLevelSample | None,
) -> ReSpeakerAmbientClassification:
    """Classify one ReSpeaker capture window into bounded non-speech facts.

    The classification intentionally fails closed. It only reports
    ``non_speech_audio_likely`` when the ambient sample shows sustained audio
    activity while XVF3800 host-control confidently says that speech is not
    currently present. ``background_media_likely`` is a stricter subset for
    more continuous non-speech audio such as music or TV.
    """

    audio_activity_detected = _audio_activity_detected(sample)
    if audio_activity_detected is False:
        return ReSpeakerAmbientClassification(
            audio_activity_detected=False,
            non_speech_audio_likely=False,
            background_media_likely=False,
        )
    if audio_activity_detected is None:
        return ReSpeakerAmbientClassification()
    if signal_snapshot.assistant_output_active is True:
        return ReSpeakerAmbientClassification(
            audio_activity_detected=False,
            non_speech_audio_likely=False,
            background_media_likely=False,
        )
    if signal_snapshot.host_control_ready is not True:
        return ReSpeakerAmbientClassification(audio_activity_detected=True)
    if signal_snapshot.speech_detected is True:
        return ReSpeakerAmbientClassification(
            audio_activity_detected=True,
            non_speech_audio_likely=False,
            background_media_likely=False,
        )
    if signal_snapshot.speech_detected is not False:
        return ReSpeakerAmbientClassification(audio_activity_detected=True)
    active_ratio = _active_ratio(sample)
    non_speech_audio_likely = True
    background_media_likely = active_ratio is not None and active_ratio >= _BACKGROUND_MEDIA_ACTIVE_RATIO
    return ReSpeakerAmbientClassification(
        audio_activity_detected=True,
        non_speech_audio_likely=non_speech_audio_likely,
        background_media_likely=background_media_likely,
    )


def _audio_activity_detected(sample: AmbientAudioLevelSample | None) -> bool | None:
    """Return whether one ambient capture window shows sustained activity."""

    if sample is None:
        return None
    if _active_ratio(sample) is None:
        return None
    active_chunk_count = _coerce_non_negative_int(getattr(sample, "active_chunk_count", None))
    if active_chunk_count is None:
        return None
    return active_chunk_count >= _MIN_ACTIVE_CHUNKS and _active_ratio(sample) >= _NON_SPEECH_ACTIVE_RATIO


def _active_ratio(sample: AmbientAudioLevelSample | None) -> float | None:
    """Return one bounded active-ratio value when present."""

    if sample is None:
        return None
    value = getattr(sample, "active_ratio", None)
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    if ratio != ratio:
        return None
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return ratio


def _coerce_non_negative_int(value: object) -> int | None:
    """Return one non-negative integer or ``None``."""

    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number
