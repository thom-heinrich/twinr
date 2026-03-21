"""Derive conservative non-speech audio facts from ReSpeaker capture activity.

The XVF3800 exposes reliable speech-oriented host-control facts such as DoA,
speech detection, and beam speech energies, but it does not expose a direct
"music" or "background media" classifier. This module therefore combines the
bounded ambient capture window from the same microphone path with XVF3800
speech flags to produce conservative runtime facts that are safe to use only
for suppression or visual-first delivery.
"""

from __future__ import annotations

import logging  # AUDIT-FIX(#1): Keep malformed hardware-model reads observable without crashing the caller.
import math  # AUDIT-FIX(#2): Reject non-finite numeric inputs and bound only small floating-point drift.
from dataclasses import dataclass

from twinr.hardware.audio import AmbientAudioLevelSample
from twinr.hardware.respeaker.pcm_content_classifier import classify_pcm_speech_likeness
from twinr.hardware.respeaker.models import ReSpeakerSignalSnapshot


logger = logging.getLogger(__name__)  # AUDIT-FIX(#1): Boundary errors should remain diagnosable while the classifier fails closed.

_MIN_ACTIVE_CHUNKS = 3
_NON_SPEECH_ACTIVE_RATIO = 0.45
_BACKGROUND_MEDIA_ACTIVE_RATIO = 0.72
_NON_SPEECH_AVERAGE_RMS = 350
_NON_SPEECH_PEAK_RMS = 500
_BACKGROUND_MEDIA_PEAK_TO_AVERAGE_MAX = 1.8
_CORROBORATED_DIRECTION_CONFIDENCE_MIN = 0.75
_RATIO_BOUND_TOLERANCE = 1e-6  # AUDIT-FIX(#2): Clamp tiny overshoot only; reject clearly corrupted ratios.


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
    pcm_bytes: bytes | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> ReSpeakerAmbientClassification:
    """Classify one ReSpeaker capture window into bounded non-speech facts.

    The classification intentionally fails closed. It only reports
    ``non_speech_audio_likely`` when the ambient sample shows sustained audio
    activity while XVF3800 host-control confidently says that speech is not
    currently present. ``background_media_likely`` is a stricter subset for
    more continuous non-speech audio such as music or TV.
    """

    active_ratio = _active_ratio(sample)  # AUDIT-FIX(#3): Snapshot sample-derived values once to avoid mixed-window classification.
    active_chunk_count = _active_chunk_count(sample)  # AUDIT-FIX(#3): Snapshot sample-derived values once to avoid mixed-window classification.
    average_rms = _average_rms(sample)
    peak_rms = _peak_rms(sample)
    pcm_speech_evidence = classify_pcm_speech_likeness(
        pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
    )
    audio_activity_detected = _audio_activity_detected_from_values(
        active_chunk_count=active_chunk_count,
        active_ratio=active_ratio,
        average_rms=average_rms,
        peak_rms=peak_rms,
    )
    if audio_activity_detected is False and pcm_speech_evidence.strong_non_speech is True:
        audio_activity_detected = True
    if audio_activity_detected is False:
        return ReSpeakerAmbientClassification(
            audio_activity_detected=False,
            non_speech_audio_likely=False,
            background_media_likely=False,
        )
    if audio_activity_detected is None:
        return ReSpeakerAmbientClassification()

    assistant_output_active = _snapshot_flag(signal_snapshot, "assistant_output_active")  # AUDIT-FIX(#1): Guard malformed snapshots instead of raising.
    host_control_ready = _snapshot_flag(signal_snapshot, "host_control_ready")  # AUDIT-FIX(#1): Guard malformed snapshots instead of raising.
    speech_detected = _snapshot_flag(signal_snapshot, "speech_detected")  # AUDIT-FIX(#1): Guard malformed snapshots instead of raising.

    if assistant_output_active is True:
        return ReSpeakerAmbientClassification(
            audio_activity_detected=False,
            non_speech_audio_likely=False,
            background_media_likely=False,
        )
    if host_control_ready is not True:
        return ReSpeakerAmbientClassification(audio_activity_detected=True)
    corroborated_speech = _speech_signal_corroborated(
        signal_snapshot=signal_snapshot,
        speech_detected=speech_detected,
    )
    if pcm_speech_evidence.strong_non_speech is True:
        return ReSpeakerAmbientClassification(
            audio_activity_detected=True,
            non_speech_audio_likely=True,
            background_media_likely=_background_media_likely_from_values(
                active_ratio=active_ratio,
                average_rms=average_rms,
                peak_rms=peak_rms,
            ),
        )
    if corroborated_speech:
        return ReSpeakerAmbientClassification(
            audio_activity_detected=True,
            non_speech_audio_likely=False,
            background_media_likely=False,
        )
    if speech_detected is True:
        if pcm_speech_evidence.speech_likely is True:
            return ReSpeakerAmbientClassification(
                audio_activity_detected=True,
                non_speech_audio_likely=False,
                background_media_likely=False,
            )
        non_speech_audio_likely = True
        background_media_likely = _background_media_likely_from_values(
            active_ratio=active_ratio,
            average_rms=average_rms,
            peak_rms=peak_rms,
        )
        return ReSpeakerAmbientClassification(
            audio_activity_detected=True,
            non_speech_audio_likely=non_speech_audio_likely,
            background_media_likely=background_media_likely,
        )
    if speech_detected is not False:
        return ReSpeakerAmbientClassification(audio_activity_detected=True)
    non_speech_audio_likely = True
    background_media_likely = _background_media_likely_from_values(
        active_ratio=active_ratio,
        average_rms=average_rms,
        peak_rms=peak_rms,
    )
    return ReSpeakerAmbientClassification(
        audio_activity_detected=True,
        non_speech_audio_likely=non_speech_audio_likely,
        background_media_likely=background_media_likely,
    )


def _speech_signal_corroborated(
    *,
    signal_snapshot: ReSpeakerSignalSnapshot,
    speech_detected: bool | None,
) -> bool:
    """Return whether XVF3800 speech is backed by directional beam evidence.

    The raw DOA speech flag can overfire on some bounded non-speech stimuli. For
    suppression decisions we therefore only treat speech as authoritative when
    the snapshot also carries corroborating beam or direction evidence.
    """

    if speech_detected is not True:
        return False
    speech_overlap_likely = _snapshot_flag(signal_snapshot, "speech_overlap_likely")
    direction_confidence = _safe_optional_float(
        _safe_getattr(signal_snapshot, "direction_confidence")
    )
    if direction_confidence is not None:
        return (
            speech_overlap_likely is not True
            and direction_confidence >= _CORROBORATED_DIRECTION_CONFIDENCE_MIN
        )
    beam_activity = _safe_float_tuple(_safe_getattr(signal_snapshot, "beam_activity"))
    if beam_activity is None:
        return False
    fixed_beam_speech_count = sum(1 for value in beam_activity[:2] if value > 0.0)
    if fixed_beam_speech_count <= 0:
        return False
    if speech_overlap_likely is True:
        return False
    return fixed_beam_speech_count == 1


def _audio_activity_detected(sample: AmbientAudioLevelSample | None) -> bool | None:
    """Return whether one ambient capture window shows sustained activity."""

    return _audio_activity_detected_from_values(  # AUDIT-FIX(#3): Reuse normalized helpers so direct callers get the same deterministic logic.
        active_chunk_count=_active_chunk_count(sample),
        active_ratio=_active_ratio(sample),
        average_rms=_average_rms(sample),
        peak_rms=_peak_rms(sample),
    )


def _audio_activity_detected_from_values(  # AUDIT-FIX(#3): Keep one-window classification based on a single normalized sample snapshot.
    *,
    active_chunk_count: int | None,
    active_ratio: float | None,
    average_rms: int | None,
    peak_rms: int | None,
) -> bool | None:
    """Return whether normalized ambient values show sustained activity."""

    ratio_activity_known = active_ratio is not None and active_chunk_count is not None
    ratio_activity = (
        ratio_activity_known
        and active_chunk_count >= _MIN_ACTIVE_CHUNKS
        and active_ratio >= _NON_SPEECH_ACTIVE_RATIO
    )
    rms_activity_known = average_rms is not None and peak_rms is not None
    rms_activity = (
        rms_activity_known
        and average_rms >= _NON_SPEECH_AVERAGE_RMS
        and peak_rms >= _NON_SPEECH_PEAK_RMS
    )
    if ratio_activity_known or rms_activity_known:
        return bool(ratio_activity or rms_activity)
    return None


def _background_media_likely_from_values(
    *,
    active_ratio: float | None,
    average_rms: int | None,
    peak_rms: int | None,
) -> bool:
    """Return whether one ambient window looks more like steady media than speech."""

    if active_ratio is not None and active_ratio >= _BACKGROUND_MEDIA_ACTIVE_RATIO:
        return True
    if average_rms is None or peak_rms is None or average_rms <= 0:
        return False
    if average_rms < _NON_SPEECH_AVERAGE_RMS or peak_rms < _NON_SPEECH_PEAK_RMS:
        return False
    return (peak_rms / average_rms) <= _BACKGROUND_MEDIA_PEAK_TO_AVERAGE_MAX


def _active_chunk_count(sample: AmbientAudioLevelSample | None) -> int | None:
    """Return one normalized active-chunk count when present."""

    if sample is None:
        return None
    return _coerce_non_negative_int(_safe_getattr(sample, "active_chunk_count"))  # AUDIT-FIX(#1): Centralize safe extraction from possibly malformed sample objects.


def _active_ratio(sample: AmbientAudioLevelSample | None) -> float | None:
    """Return one bounded active-ratio value when present."""

    if sample is None:
        return None
    value = _safe_getattr(sample, "active_ratio")  # AUDIT-FIX(#1): Guard attribute access from malformed ambient sample objects.
    if value is None or isinstance(value, bool):  # AUDIT-FIX(#2): Reject booleans so True/False cannot masquerade as 1.0/0.0.
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError):  # AUDIT-FIX(#2): Treat overflow/corruption as unknown, not active audio.
        return None
    if not math.isfinite(ratio):  # AUDIT-FIX(#2): Fail closed on NaN/inf instead of converting them into valid activity.
        return None
    if ratio < -_RATIO_BOUND_TOLERANCE or ratio > 1.0 + _RATIO_BOUND_TOLERANCE:  # AUDIT-FIX(#2): Reject grossly invalid ratios instead of clamping them to 0/1.
        return None
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return ratio


def _average_rms(sample: AmbientAudioLevelSample | None) -> int | None:
    """Return one normalized average RMS value when present."""

    if sample is None:
        return None
    return _coerce_non_negative_int(_safe_getattr(sample, "average_rms"))


def _peak_rms(sample: AmbientAudioLevelSample | None) -> int | None:
    """Return one normalized peak RMS value when present."""

    if sample is None:
        return None
    return _coerce_non_negative_int(_safe_getattr(sample, "peak_rms"))


def _coerce_non_negative_int(value: object) -> int | None:
    """Return one non-negative integer or ``None``."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):  # AUDIT-FIX(#4): Reject lossy float-to-int truncation unless the value is exactly integral.
        if not math.isfinite(value) or not value.is_integer():
            return None
        number = int(value)
        return number if number >= 0 else None
    if isinstance(value, str):  # AUDIT-FIX(#4): Accept only canonical integer strings, not arbitrary coercions.
        text = value.strip()
        if not text:
            return None
        if text[0] in "+-":
            if not text[1:].isdigit():
                return None
        elif not text.isdigit():
            return None
        number = int(text, 10)
        return number if number >= 0 else None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number < 0:
        return None
    try:
        if value != number:  # AUDIT-FIX(#4): Reject coercions that would lose fractional information.
            return None
    except Exception:
        return None
    return number


def _safe_optional_float(value: object) -> float | None:
    """Return one finite float or ``None``."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_float_tuple(value: object) -> tuple[float, ...] | None:
    """Return one finite float tuple or ``None`` when malformed."""

    if not isinstance(value, tuple):
        return None
    normalized: list[float] = []
    for item in value:
        number = _safe_optional_float(item)
        if number is None:
            return None
        normalized.append(number)
    return tuple(normalized)


def _snapshot_flag(  # AUDIT-FIX(#1): Normalize snapshot flags into strict True/False/None values.
    signal_snapshot: ReSpeakerSignalSnapshot | None,
    attribute_name: str,
) -> bool | None:
    """Return one tri-state snapshot flag or ``None`` when malformed."""

    value = _safe_getattr(signal_snapshot, attribute_name)
    if value is None or isinstance(value, bool):
        return value
    return None


def _safe_getattr(  # AUDIT-FIX(#1): Fail closed when model properties are missing or raise during access.
    instance: object | None,
    attribute_name: str,
) -> object | None:
    """Read one attribute from an object and degrade to ``None`` on failure."""

    if instance is None:
        return None
    try:
        return getattr(instance, attribute_name, None)
    except Exception:
        logger.warning(
            "Ignoring malformed %s.%s during ambient classification",
            type(instance).__name__,
            attribute_name,
            exc_info=True,
        )
        return None
