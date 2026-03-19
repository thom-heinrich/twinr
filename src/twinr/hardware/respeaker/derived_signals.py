"""Derive conservative higher-level XVF3800 signals from direct primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

from twinr.hardware.respeaker.models import ReSpeakerDirectionSnapshot


@dataclass(frozen=True, slots=True)
class ReSpeakerDerivedSignalState:
    """Store conservative interpreted signals derived from XVF3800 primitives."""

    fixed_beam_speech_count: int | None
    near_end_speech_detected: bool | None
    direction_confidence: float | None
    speech_overlap_likely: bool | None
    barge_in_detected: bool | None


def derive_respeaker_signal_state(
    direction: ReSpeakerDirectionSnapshot,
    *,
    assistant_output_active: bool | None,
) -> ReSpeakerDerivedSignalState:
    """Interpret one direction snapshot into conservative runtime signals.

    The XVF3800 exposes direct speech/no-speech, beam energies, fixed-beam
    azimuths, and selected azimuths, but not a first-class near-end/double-talk
    state. Twinr therefore treats fixed-beam speech energy as the strongest
    evidence of near-end speech and only emits derived overlap/barge-in facts
    when those direct cues are present.
    """

    fixed_beam_speech_count = _count_fixed_beam_speech(direction.beam_speech_energies)
    near_end_speech_detected = _near_end_speech_detected(
        speech_detected=direction.speech_detected,
        fixed_beam_speech_count=fixed_beam_speech_count,
    )
    direction_confidence = _estimate_direction_confidence(
        direction=direction,
        fixed_beam_speech_count=fixed_beam_speech_count,
    )
    speech_overlap_likely = _speech_overlap_likely(
        speech_detected=direction.speech_detected,
        fixed_beam_speech_count=fixed_beam_speech_count,
    )
    barge_in_detected = _barge_in_detected(
        assistant_output_active=assistant_output_active,
        speech_overlap_likely=speech_overlap_likely,
    )
    return ReSpeakerDerivedSignalState(
        fixed_beam_speech_count=fixed_beam_speech_count,
        near_end_speech_detected=near_end_speech_detected,
        direction_confidence=direction_confidence,
        speech_overlap_likely=speech_overlap_likely,
        barge_in_detected=barge_in_detected,
    )


def _count_fixed_beam_speech(
    beam_speech_energies: tuple[float | None, ...] | None,
) -> int | None:
    """Count fixed beams whose official speech-energy values indicate speech."""

    fixed_beams = _fixed_beam_energies(beam_speech_energies)
    if fixed_beams is None:
        return None
    return sum(1 for energy in fixed_beams if energy > 0.0)


def _near_end_speech_detected(
    *,
    speech_detected: bool | None,
    fixed_beam_speech_count: int | None,
) -> bool | None:
    """Return whether fixed-beam evidence suggests near-end speech is present."""

    if speech_detected is False:
        return False
    if speech_detected is None or fixed_beam_speech_count is None:
        return None
    return speech_detected and fixed_beam_speech_count >= 1


def _speech_overlap_likely(
    *,
    speech_detected: bool | None,
    fixed_beam_speech_count: int | None,
) -> bool | None:
    """Return whether multiple fixed beams currently report speech."""

    if speech_detected is False:
        return False
    if speech_detected is None or fixed_beam_speech_count is None:
        return None
    return speech_detected and fixed_beam_speech_count >= 2


def _barge_in_detected(
    *,
    assistant_output_active: bool | None,
    speech_overlap_likely: bool | None,
) -> bool | None:
    """Return whether bounded runtime state indicates a likely interruption."""

    if assistant_output_active is False:
        return False
    if assistant_output_active is None or speech_overlap_likely is None:
        return None
    return assistant_output_active and speech_overlap_likely


def _estimate_direction_confidence(
    *,
    direction: ReSpeakerDirectionSnapshot,
    fixed_beam_speech_count: int | None,
) -> float | None:
    """Estimate one conservative confidence score for the current DoA reading."""

    if direction.speech_detected is not True:
        return None
    if fixed_beam_speech_count is None or fixed_beam_speech_count < 1:
        return None
    if direction.doa_degrees is None:
        return None

    fixed_beam_energies = _fixed_beam_energies(direction.beam_speech_energies)
    fixed_beam_azimuths = _fixed_beam_azimuths(direction.beam_azimuth_degrees)
    strongest_beam_azimuth = _strongest_fixed_beam_azimuth(
        fixed_beam_energies=fixed_beam_energies,
        fixed_beam_azimuths=fixed_beam_azimuths,
    )
    processed_selected_azimuth = _processed_selected_azimuth(direction.selected_azimuth_degrees)

    alignment_scores: list[float] = []
    if strongest_beam_azimuth is not None:
        alignment_scores.append(_alignment_score(direction.doa_degrees, strongest_beam_azimuth))
    if processed_selected_azimuth is not None:
        alignment_scores.append(_alignment_score(direction.doa_degrees, processed_selected_azimuth))
    if not alignment_scores:
        return None

    ambiguity_factor = 1.0 / float(max(1, fixed_beam_speech_count))
    confidence = (sum(alignment_scores) / len(alignment_scores)) * ambiguity_factor
    return round(max(0.0, min(1.0, confidence)), 3)


def _fixed_beam_energies(
    beam_speech_energies: tuple[float | None, ...] | None,
) -> tuple[float, float] | None:
    """Return the two fixed-beam speech energies when available."""

    if beam_speech_energies is None or len(beam_speech_energies) < 2:
        return None
    values: list[float] = []
    for energy in beam_speech_energies[:2]:
        if energy is None:
            return None
        try:
            normalized = float(energy)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(normalized) or normalized < 0.0:
            return None
        values.append(normalized)
    return values[0], values[1]


def _fixed_beam_azimuths(
    beam_azimuth_degrees: tuple[float | None, ...] | None,
) -> tuple[float | None, float | None] | None:
    """Return the two fixed-beam azimuths when available."""

    if beam_azimuth_degrees is None or len(beam_azimuth_degrees) < 2:
        return None
    return beam_azimuth_degrees[0], beam_azimuth_degrees[1]


def _strongest_fixed_beam_azimuth(
    *,
    fixed_beam_energies: tuple[float, float] | None,
    fixed_beam_azimuths: tuple[float | None, float | None] | None,
) -> float | None:
    """Return the azimuth of the fixed beam with the highest speech energy."""

    if fixed_beam_energies is None or fixed_beam_azimuths is None:
        return None
    strongest_index = 0 if fixed_beam_energies[0] >= fixed_beam_energies[1] else 1
    azimuth = fixed_beam_azimuths[strongest_index]
    if azimuth is None:
        return None
    return _normalize_azimuth(azimuth)


def _processed_selected_azimuth(
    selected_azimuth_degrees: tuple[float | None, ...] | None,
) -> float | None:
    """Return the processed selected azimuth when XVF3800 provided one."""

    if selected_azimuth_degrees is None or not selected_azimuth_degrees:
        return None
    azimuth = selected_azimuth_degrees[0]
    if azimuth is None:
        return None
    return _normalize_azimuth(azimuth)


def _alignment_score(reference_azimuth: int, candidate_azimuth: float) -> float:
    """Return one bounded confidence score from two azimuths."""

    delta = _circular_distance(float(reference_azimuth), candidate_azimuth)
    return max(0.0, 1.0 - (min(delta, 90.0) / 90.0))


def _normalize_azimuth(value: float) -> float | None:
    """Normalize one azimuth-like value into ``0..360``."""

    if not math.isfinite(value):
        return None
    return float(value) % 360.0


def _circular_distance(left: float, right: float) -> float:
    """Return the absolute circular distance in degrees."""

    raw_delta = abs((left - right) % 360.0)
    return min(raw_delta, 360.0 - raw_delta)
