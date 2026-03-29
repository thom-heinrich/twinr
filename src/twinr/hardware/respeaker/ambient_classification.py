"""Derive conservative non-speech audio facts from ReSpeaker capture activity.

The XVF3800 exposes reliable speech-oriented host-control facts such as DoA,
speech detection, and beam speech energies, but it does not expose a direct
"music" or "background media" classifier. This module therefore combines the
bounded ambient capture window from the same microphone path with XVF3800
speech flags to produce conservative runtime facts that are safe to use only
for suppression or visual-first delivery.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Fixed false "no audio activity" results when ambient RMS heuristics miss a window but XVF3800 or PCM evidence still indicates real activity.
# BUG-2: Preserve the corroboration contract for raw XVF3800 speech flags; uncorroborated
#        `speech_detected=True` no longer forces activity or suppresses non-speech classification.
# BUG-3: Guarded downstream PCM speech-likeness classification against malformed buffers and classifier exceptions so the caller no longer crashes instead of failing closed.
# SEC-1: Added strict PCM window validation and hard size bounds to block practical CPU/memory denial-of-service via oversized or misaligned PCM buffers on Raspberry Pi deployments.
# IMP-1: Added low-cost temporal PCM features (frame continuity, energy variation, crest factor, zero-crossing rate) so background-media decisions are not driven by RMS/active-ratio alone.
# IMP-2: Added an optional semantic non-speech hook for local edge audio classifiers (for example YAMNet / EfficientAT wrappers) while keeping this module dependency-free and drop-in compatible.
# IMP-3: Moved expensive PCM work behind earlier hardware gates to reduce steady-state CPU and thermal load during assistant playback and other already-decided paths.

from __future__ import annotations

import importlib
import logging
import math
import sys
from array import array
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping

from twinr.hardware.audio import AmbientAudioLevelSample
from twinr.hardware.respeaker.pcm_content_classifier import classify_pcm_speech_likeness
from twinr.hardware.respeaker.models import ReSpeakerSignalSnapshot


logger = logging.getLogger(__name__)

_MIN_ACTIVE_CHUNKS = 3
_NON_SPEECH_ACTIVE_RATIO = 0.45
_BACKGROUND_MEDIA_ACTIVE_RATIO = 0.72
_NON_SPEECH_AVERAGE_RMS = 350
_NON_SPEECH_PEAK_RMS = 500
_BACKGROUND_MEDIA_PEAK_TO_AVERAGE_MAX = 1.8
_CORROBORATED_DIRECTION_CONFIDENCE_MIN = 0.75
_RATIO_BOUND_TOLERANCE = 1e-6

# Conservative PCM guardrails for one bounded ambient window.
_PCM_MIN_SAMPLE_RATE = 8_000
_PCM_MAX_SAMPLE_RATE = 96_000
_PCM_MIN_CHANNELS = 1
_PCM_MAX_CHANNELS = 8
_PCM_SAMPLE_WIDTH_BYTES = 2
_MAX_PCM_WINDOW_SECONDS = 2.0
_MAX_PCM_BYTES_ABSOLUTE = 1_048_576  # 1 MiB hard stop.

# Low-cost per-window temporal features used only as corroborating evidence.
_PCM_FRAME_MS = 20
_PCM_ACTIVE_FRAME_RMS_MIN = 220.0
_PCM_STEADY_ACTIVE_RATIO_MIN = 0.78
_PCM_STEADY_LONGEST_RUN_RATIO_MIN = 0.55
_PCM_STEADY_ENERGY_CV_MAX = 0.75
_PCM_MEDIA_ACTIVE_RATIO_MIN = 0.85
_PCM_MEDIA_LONGEST_RUN_RATIO_MIN = 0.72
_PCM_MEDIA_ENERGY_CV_MAX = 0.55
_PCM_MEDIA_CREST_FACTOR_MAX = 6.0
_PCM_MEDIA_ZCR_MIN = 0.005
_PCM_MEDIA_ZCR_MAX = 0.35
_SEMANTIC_MEDIA_CONFIDENCE_MIN = 0.80


@dataclass(frozen=True, slots=True)
class ReSpeakerAmbientClassification:
    """Store conservative activity facts derived from one capture window."""

    audio_activity_detected: bool | None = None
    non_speech_audio_likely: bool | None = None
    background_media_likely: bool | None = None


@dataclass(frozen=True, slots=True)
class _PcmWindow:
    """Store one validated PCM window."""

    pcm_bytes: bytes
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class _PcmFrameFeatures:
    """Store lightweight temporal facts derived from one PCM window."""

    average_rms: int | None = None
    peak_rms: int | None = None
    active_ratio: float | None = None
    longest_active_run_ratio: float | None = None
    energy_cv: float | None = None
    crest_factor: float | None = None
    zero_crossing_rate: float | None = None
    steady_non_speech: bool | None = None
    background_media_likely: bool | None = None


@dataclass(frozen=True, slots=True)
class _SemanticAmbientEvidence:
    """Store optional semantic evidence from a local edge classifier."""

    non_speech_audio_likely: bool | None = None
    background_media_likely: bool | None = None
    speech_likely: bool | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class _PcmAmbientEvidence:
    """Store combined PCM-derived corroborating evidence."""

    speech_likely: bool | None = None
    strong_non_speech: bool | None = None
    steady_non_speech: bool | None = None
    background_media_likely: bool | None = None
    audio_activity_detected: bool | None = None


def classify_respeaker_ambient_audio(
    *,
    signal_snapshot: ReSpeakerSignalSnapshot,
    sample: AmbientAudioLevelSample | None,
    pcm_bytes: bytes | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    semantic_classifier: Callable[..., object] | None = None,
) -> ReSpeakerAmbientClassification:
    """Classify one ReSpeaker capture window into bounded non-speech facts.

    The classification intentionally fails closed. It only reports
    ``non_speech_audio_likely`` when ambient or PCM evidence shows sustained
    activity while XVF3800 host-control does not currently provide reliable
    speech evidence. ``background_media_likely`` is a stricter subset for more
    continuous non-speech audio such as music or TV.
    """

    active_ratio = _active_ratio(sample)
    active_chunk_count = _active_chunk_count(sample)
    average_rms = _average_rms(sample)
    peak_rms = _peak_rms(sample)

    assistant_output_active = _snapshot_flag(signal_snapshot, "assistant_output_active")
    host_control_ready = _snapshot_flag(signal_snapshot, "host_control_ready")
    speech_detected = _snapshot_flag(signal_snapshot, "speech_detected")

    if assistant_output_active is True:
        return ReSpeakerAmbientClassification(
            audio_activity_detected=False,
            non_speech_audio_likely=False,
            background_media_likely=False,
        )

    corroborated_speech = False
    if host_control_ready is True:
        corroborated_speech = _speech_signal_corroborated(
            signal_snapshot=signal_snapshot,
            speech_detected=speech_detected,
        )

    sample_activity = _audio_activity_detected_from_values(
        active_chunk_count=active_chunk_count,
        active_ratio=active_ratio,
        average_rms=average_rms,
        peak_rms=peak_rms,
    )

    pcm_window = _normalize_pcm_window(
        pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
    )
    pcm_evidence = _classify_pcm_ambient_evidence(
        pcm_window,
        semantic_classifier=semantic_classifier,
    )

    audio_activity_detected = _merge_audio_activity(
        sample_activity=sample_activity,
        corroborated_speech=corroborated_speech,
        pcm_evidence=pcm_evidence,
    )
    if audio_activity_detected is False:
        return ReSpeakerAmbientClassification(
            audio_activity_detected=False,
            non_speech_audio_likely=False,
            background_media_likely=False,
        )
    if audio_activity_detected is None:
        return ReSpeakerAmbientClassification()

    if host_control_ready is not True:
        return ReSpeakerAmbientClassification(audio_activity_detected=True)

    background_media_likely = _background_media_likely_from_evidence(
        active_ratio=active_ratio,
        average_rms=average_rms,
        peak_rms=peak_rms,
        pcm_evidence=pcm_evidence,
    )

    if pcm_evidence.strong_non_speech is True and pcm_evidence.speech_likely is not True:
        return ReSpeakerAmbientClassification(
            audio_activity_detected=True,
            non_speech_audio_likely=True,
            background_media_likely=background_media_likely,
        )

    if corroborated_speech:
        return ReSpeakerAmbientClassification(
            audio_activity_detected=True,
            non_speech_audio_likely=False,
            background_media_likely=False,
        )

    if speech_detected is False:
        if pcm_evidence.speech_likely is True and pcm_evidence.strong_non_speech is not True:
            return ReSpeakerAmbientClassification(audio_activity_detected=True)
        return ReSpeakerAmbientClassification(
            audio_activity_detected=True,
            non_speech_audio_likely=True,
            background_media_likely=background_media_likely,
        )

    if speech_detected is True:
        if pcm_evidence.speech_likely is True and pcm_evidence.strong_non_speech is not True:
            return ReSpeakerAmbientClassification(
                audio_activity_detected=True,
                non_speech_audio_likely=False,
                background_media_likely=False,
            )
        return ReSpeakerAmbientClassification(
            audio_activity_detected=True,
            non_speech_audio_likely=True,
            background_media_likely=background_media_likely,
        )

    return ReSpeakerAmbientClassification(audio_activity_detected=True)


def _merge_audio_activity(
    *,
    sample_activity: bool | None,
    corroborated_speech: bool,
    pcm_evidence: _PcmAmbientEvidence,
) -> bool | None:
    """Merge one-window activity evidence conservatively.

    Raw ``speech_detected`` alone is intentionally not enough here. The XVF3800
    speech flag can overfire on bounded non-speech stimuli, so activity should
    only rise from the sample window, corroborated speech, or explicit
    PCM-derived evidence.
    """

    if sample_activity is True:
        return True
    if corroborated_speech:
        return True
    if pcm_evidence.audio_activity_detected is True:
        return True
    if sample_activity is False:
        return False
    return None


def _classify_pcm_ambient_evidence(
    pcm_window: _PcmWindow | None,
    *,
    semantic_classifier: Callable[..., object] | None = None,
) -> _PcmAmbientEvidence:
    """Combine strict PCM speech-likeness, temporal DSP, and optional semantics."""

    if pcm_window is None:
        return _PcmAmbientEvidence()

    speech_likely: bool | None = None
    strong_non_speech: bool | None = None

    try:
        speech_evidence = classify_pcm_speech_likeness(
            pcm_window.pcm_bytes,
            sample_rate=pcm_window.sample_rate,
            channels=pcm_window.channels,
        )
    except Exception:
        logger.warning(
            "Ignoring PCM speech-likeness failure during ambient classification",
            exc_info=True,
        )
        speech_evidence = None

    if speech_evidence is not None:
        speech_likely = _strict_optional_bool(_safe_getattr(speech_evidence, "speech_likely"))
        strong_non_speech = _strict_optional_bool(
            _safe_getattr(speech_evidence, "strong_non_speech")
        )

    frame_features = _extract_pcm_frame_features(pcm_window)
    semantic_evidence = _classify_pcm_ambient_semantics(
        pcm_window,
        semantic_classifier=semantic_classifier,
    )

    if semantic_evidence.speech_likely is True:
        speech_likely = True
    if semantic_evidence.non_speech_audio_likely is True and semantic_evidence.speech_likely is not True:
        semantic_confident = (
            semantic_evidence.confidence is None
            or semantic_evidence.confidence >= _SEMANTIC_MEDIA_CONFIDENCE_MIN
        )
        if semantic_confident:
            strong_non_speech = True

    steady_non_speech = bool(
        frame_features.steady_non_speech is True
        or (strong_non_speech is True and speech_likely is not True)
    )
    background_media_likely = bool(
        semantic_evidence.background_media_likely is True
        or (
            frame_features.background_media_likely is True
            and speech_likely is not True
        )
    )
    if background_media_likely:
        steady_non_speech = True
    if background_media_likely and speech_likely is not True:
        strong_non_speech = True

    return _PcmAmbientEvidence(
        speech_likely=speech_likely,
        strong_non_speech=strong_non_speech,
        steady_non_speech=steady_non_speech or None,
        background_media_likely=background_media_likely or None,
        audio_activity_detected=(
            True
            if (
                speech_likely is True
                or strong_non_speech is True
                or steady_non_speech is True
                or background_media_likely is True
            )
            else None
        ),
    )


def _classify_pcm_ambient_semantics(
    pcm_window: _PcmWindow,
    *,
    semantic_classifier: Callable[..., object] | None = None,
) -> _SemanticAmbientEvidence:
    """Return optional semantic evidence from a local on-device classifier."""

    classifier = semantic_classifier or _optional_semantic_classifier()
    if classifier is None:
        return _SemanticAmbientEvidence()
    try:
        result = classifier(
            pcm_window.pcm_bytes,
            sample_rate=pcm_window.sample_rate,
            channels=pcm_window.channels,
        )
    except TypeError:
        try:
            result = classifier(
                pcm_bytes=pcm_window.pcm_bytes,
                sample_rate=pcm_window.sample_rate,
                channels=pcm_window.channels,
            )
        except Exception:
            logger.warning(
                "Ignoring optional semantic ambient classifier failure",
                exc_info=True,
            )
            return _SemanticAmbientEvidence()
    except Exception:
        logger.warning(
            "Ignoring optional semantic ambient classifier failure",
            exc_info=True,
        )
        return _SemanticAmbientEvidence()
    return _coerce_semantic_ambient_evidence(result)


@lru_cache(maxsize=1)
def _optional_semantic_classifier() -> Callable[..., object] | None:
    """Load one optional Twinr-local semantic ambient classifier when available."""

    try:
        module = importlib.import_module(
            "twinr.hardware.respeaker.ambient_semantic_classifier"
        )
    except ModuleNotFoundError:
        return None
    except Exception:
        logger.warning(
            "Ignoring broken optional ambient semantic classifier import",
            exc_info=True,
        )
        return None

    classifier = getattr(module, "classify_pcm_ambient_semantics", None)
    return classifier if callable(classifier) else None


def _coerce_semantic_ambient_evidence(value: object) -> _SemanticAmbientEvidence:
    """Coerce one semantic evidence object into a strict internal form."""

    if value is None:
        return _SemanticAmbientEvidence()

    if isinstance(value, _SemanticAmbientEvidence):
        return value

    if isinstance(value, Mapping):
        non_speech_audio_likely = _strict_optional_bool(value.get("non_speech_audio_likely"))
        background_media_likely = _strict_optional_bool(value.get("background_media_likely"))
        speech_likely = _strict_optional_bool(value.get("speech_likely"))
        confidence = _safe_optional_float(value.get("confidence"))
        return _SemanticAmbientEvidence(
            non_speech_audio_likely=non_speech_audio_likely,
            background_media_likely=background_media_likely,
            speech_likely=speech_likely,
            confidence=confidence,
        )

    return _SemanticAmbientEvidence(
        non_speech_audio_likely=_strict_optional_bool(
            _safe_getattr(value, "non_speech_audio_likely")
        ),
        background_media_likely=_strict_optional_bool(
            _safe_getattr(value, "background_media_likely")
        ),
        speech_likely=_strict_optional_bool(_safe_getattr(value, "speech_likely")),
        confidence=_safe_optional_float(_safe_getattr(value, "confidence")),
    )


def _extract_pcm_frame_features(pcm_window: _PcmWindow) -> _PcmFrameFeatures:
    """Extract cheap temporal PCM facts that distinguish steady media from speech."""

    samples = array("h")
    samples.frombytes(pcm_window.pcm_bytes)
    if sys.byteorder != "little":
        samples.byteswap()

    if pcm_window.channels > 1:
        mono_samples = [samples[index] for index in range(0, len(samples), pcm_window.channels)]
    else:
        mono_samples = samples

    if not mono_samples:
        return _PcmFrameFeatures()

    frame_size = max(1, int(round(pcm_window.sample_rate * (_PCM_FRAME_MS / 1000.0))))
    total_sample_count = len(mono_samples)

    peak_abs = 0
    zero_crossings = 0
    previous_sign = 0
    for raw_sample in mono_samples:
        sample_value = int(raw_sample)
        absolute_value = abs(sample_value)
        if absolute_value > peak_abs:
            peak_abs = absolute_value
        current_sign = 1 if sample_value > 0 else (-1 if sample_value < 0 else 0)
        if previous_sign and current_sign and current_sign != previous_sign:
            zero_crossings += 1
        if current_sign:
            previous_sign = current_sign

    frame_rms_values: list[float] = []
    active_frame_count = 0
    longest_active_run = 0
    current_active_run = 0

    minimum_tail = max(1, frame_size // 2)
    for start_index in range(0, total_sample_count, frame_size):
        frame = mono_samples[start_index : start_index + frame_size]
        frame_length = len(frame)
        if frame_length < minimum_tail:
            break

        square_sum = 0
        for raw_sample in frame:
            sample_value = int(raw_sample)
            square_sum += sample_value * sample_value
        frame_rms = math.sqrt(square_sum / frame_length)
        frame_rms_values.append(frame_rms)

        if frame_rms >= _PCM_ACTIVE_FRAME_RMS_MIN:
            active_frame_count += 1
            current_active_run += 1
            if current_active_run > longest_active_run:
                longest_active_run = current_active_run
        else:
            current_active_run = 0

    if not frame_rms_values:
        return _PcmFrameFeatures()

    mean_rms = sum(frame_rms_values) / len(frame_rms_values)
    variance = 0.0
    if len(frame_rms_values) > 1:
        variance = sum(
            (frame_rms - mean_rms) * (frame_rms - mean_rms)
            for frame_rms in frame_rms_values
        ) / len(frame_rms_values)
    standard_deviation = math.sqrt(variance)
    energy_cv = (standard_deviation / mean_rms) if mean_rms > 0.0 else None
    active_ratio = active_frame_count / len(frame_rms_values)
    longest_active_run_ratio = longest_active_run / len(frame_rms_values)
    crest_factor = (peak_abs / mean_rms) if mean_rms > 0.0 else None
    zero_crossing_rate = zero_crossings / max(1, total_sample_count - 1)

    average_rms = int(round(mean_rms)) if math.isfinite(mean_rms) else None
    peak_rms = int(peak_abs) if peak_abs >= 0 else None

    steady_non_speech = (
        average_rms is not None
        and peak_rms is not None
        and average_rms >= _NON_SPEECH_AVERAGE_RMS
        and peak_rms >= _NON_SPEECH_PEAK_RMS
        and active_ratio >= _PCM_STEADY_ACTIVE_RATIO_MIN
        and longest_active_run_ratio >= _PCM_STEADY_LONGEST_RUN_RATIO_MIN
        and energy_cv is not None
        and energy_cv <= _PCM_STEADY_ENERGY_CV_MAX
    )
    background_media_likely = (
        average_rms is not None
        and peak_rms is not None
        and average_rms >= _NON_SPEECH_AVERAGE_RMS
        and peak_rms >= _NON_SPEECH_PEAK_RMS
        and active_ratio >= _PCM_MEDIA_ACTIVE_RATIO_MIN
        and longest_active_run_ratio >= _PCM_MEDIA_LONGEST_RUN_RATIO_MIN
        and energy_cv is not None
        and energy_cv <= _PCM_MEDIA_ENERGY_CV_MAX
        and crest_factor is not None
        and crest_factor <= _PCM_MEDIA_CREST_FACTOR_MAX
        and _PCM_MEDIA_ZCR_MIN <= zero_crossing_rate <= _PCM_MEDIA_ZCR_MAX
    )

    return _PcmFrameFeatures(
        average_rms=average_rms,
        peak_rms=peak_rms,
        active_ratio=active_ratio,
        longest_active_run_ratio=longest_active_run_ratio,
        energy_cv=energy_cv,
        crest_factor=crest_factor,
        zero_crossing_rate=zero_crossing_rate,
        steady_non_speech=steady_non_speech or None,
        background_media_likely=background_media_likely or None,
    )


def _normalize_pcm_window(
    pcm_bytes: bytes | None,
    *,
    sample_rate: int | None,
    channels: int | None,
) -> _PcmWindow | None:
    """Return one validated PCM window or ``None`` when malformed."""

    if pcm_bytes is None:
        return None
    if not isinstance(pcm_bytes, (bytes, bytearray, memoryview)):
        return None

    normalized_sample_rate = _coerce_int_in_range(
        sample_rate,
        minimum=_PCM_MIN_SAMPLE_RATE,
        maximum=_PCM_MAX_SAMPLE_RATE,
    )
    normalized_channels = _coerce_int_in_range(
        channels,
        minimum=_PCM_MIN_CHANNELS,
        maximum=_PCM_MAX_CHANNELS,
    )
    if normalized_sample_rate is None or normalized_channels is None:
        return None

    try:
        payload = bytes(pcm_bytes)
    except Exception:
        return None

    if not payload:
        return None
    if len(payload) > _MAX_PCM_BYTES_ABSOLUTE:
        return None

    bytes_per_frame = _PCM_SAMPLE_WIDTH_BYTES * normalized_channels
    if bytes_per_frame <= 0:
        return None
    if len(payload) % _PCM_SAMPLE_WIDTH_BYTES != 0:
        return None
    if len(payload) % bytes_per_frame != 0:
        return None

    max_bytes_for_duration = int(
        normalized_sample_rate
        * normalized_channels
        * _PCM_SAMPLE_WIDTH_BYTES
        * _MAX_PCM_WINDOW_SECONDS
    )
    if len(payload) > max_bytes_for_duration:
        return None

    return _PcmWindow(
        pcm_bytes=payload,
        sample_rate=normalized_sample_rate,
        channels=normalized_channels,
    )


def _coerce_int_in_range(
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    """Return one bounded integer or ``None``."""

    number = _coerce_non_negative_int(value)
    if number is None:
        return None
    if number < minimum or number > maximum:
        return None
    return number


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

    return _audio_activity_detected_from_values(
        active_chunk_count=_active_chunk_count(sample),
        active_ratio=_active_ratio(sample),
        average_rms=_average_rms(sample),
        peak_rms=_peak_rms(sample),
    )


def _audio_activity_detected_from_values(
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


def _background_media_likely_from_evidence(
    *,
    active_ratio: float | None,
    average_rms: int | None,
    peak_rms: int | None,
    pcm_evidence: _PcmAmbientEvidence,
) -> bool:
    """Return whether one ambient window is a steady background-media candidate."""

    if pcm_evidence.background_media_likely is True:
        return True
    return _background_media_likely_from_values(
        active_ratio=active_ratio,
        average_rms=average_rms,
        peak_rms=peak_rms,
    )


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
    return _coerce_non_negative_int(_safe_getattr(sample, "active_chunk_count"))


def _active_ratio(sample: AmbientAudioLevelSample | None) -> float | None:
    """Return one bounded active-ratio value when present."""

    if sample is None:
        return None
    value = _safe_getattr(sample, "active_ratio")
    if value is None or isinstance(value, bool):
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(ratio):
        return None
    if ratio < -_RATIO_BOUND_TOLERANCE or ratio > 1.0 + _RATIO_BOUND_TOLERANCE:
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
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        number = int(value)
        return number if number >= 0 else None
    if isinstance(value, str):
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
        if value != number:
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

    if not isinstance(value, (tuple, list)):
        return None
    normalized: list[float] = []
    for item in value:
        number = _safe_optional_float(item)
        if number is None:
            return None
        normalized.append(number)
    return tuple(normalized)


def _strict_optional_bool(value: object) -> bool | None:
    """Return strict tri-state booleans only."""

    if value is None or isinstance(value, bool):
        return value
    return None


def _snapshot_flag(
    signal_snapshot: ReSpeakerSignalSnapshot | None,
    attribute_name: str,
) -> bool | None:
    """Return one tri-state snapshot flag or ``None`` when malformed."""

    return _strict_optional_bool(_safe_getattr(signal_snapshot, attribute_name))


def _safe_getattr(
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
