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

    # AUDIT-FIX(#2): Normalize tri-state flags once so malformed upstream values
    # never leak through as integer 0/1 outputs via Python short-circuit rules.
    speech_detected = _coerce_optional_bool(direction.speech_detected)
    # AUDIT-FIX(#2): Apply the same normalization to assistant playback state so
    # invalid values degrade conservatively to None instead of truthy/falsy ints.
    assistant_output_active = _coerce_optional_bool(assistant_output_active)

    fixed_beam_speech_count = _count_fixed_beam_speech(direction.beam_speech_energies)
    near_end_speech_detected = _near_end_speech_detected(
        speech_detected=speech_detected,
        fixed_beam_speech_count=fixed_beam_speech_count,
    )
    direction_confidence = _estimate_direction_confidence(
        direction=direction,
        speech_detected=speech_detected,
        fixed_beam_speech_count=fixed_beam_speech_count,
    )
    speech_overlap_likely = _speech_overlap_likely(
        speech_detected=speech_detected,
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
    # AUDIT-FIX(#2): Return an explicit bool after normalization instead of
    # relying on "and", which can propagate non-bool operands on bad inputs.
    return fixed_beam_speech_count >= 1


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
    # AUDIT-FIX(#2): Keep the helper strictly bool-typed for downstream callers.
    return fixed_beam_speech_count >= 2


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
    # AUDIT-FIX(#2): Return the validated bool directly to avoid accidental int
    # outputs if malformed values ever bypass the caller-side normalization.
    return speech_overlap_likely


def _estimate_direction_confidence(
    *,
    direction: ReSpeakerDirectionSnapshot,
    speech_detected: bool | None,
    fixed_beam_speech_count: int | None,
) -> float | None:
    """Estimate one conservative confidence score for the current DoA reading."""

    if speech_detected is not True:
        return None
    if fixed_beam_speech_count is None or fixed_beam_speech_count < 1:
        return None

    # AUDIT-FIX(#1): Sanitize the direct DoA primitive so malformed, NaN, or
    # infinite values degrade to None instead of crashing or producing a fake
    # low-confidence numeric output.
    doa_degrees = _normalize_azimuth(direction.doa_degrees)
    if doa_degrees is None:
        return None

    fixed_beam_energies = _fixed_beam_energies(direction.beam_speech_energies)
    fixed_beam_azimuths = _fixed_beam_azimuths(direction.beam_azimuth_degrees)
    strongest_beam_azimuth = _strongest_fixed_beam_azimuth(
        fixed_beam_energies=fixed_beam_energies,
        fixed_beam_azimuths=fixed_beam_azimuths,
    )
    processed_selected_azimuth = _processed_selected_azimuth(
        direction.selected_azimuth_degrees
    )

    alignment_scores: list[float] = []
    if strongest_beam_azimuth is not None:
        alignment_scores.append(_alignment_score(doa_degrees, strongest_beam_azimuth))
    if processed_selected_azimuth is not None:
        alignment_scores.append(_alignment_score(doa_degrees, processed_selected_azimuth))
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
        # AUDIT-FIX(#3): Reject bool/NaN/Inf and string-like values at one choke
        # point so invalid sensor payloads cannot synthesize speech evidence.
        normalized = _coerce_finite_float(energy, allow_negative=False)
        if normalized is None:
            return None
        values.append(normalized)
    return values[0], values[1]


def _fixed_beam_azimuths(
    beam_azimuth_degrees: tuple[float | None, ...] | None,
) -> tuple[float | None, float | None] | None:
    """Return the two fixed-beam azimuths when available."""

    if beam_azimuth_degrees is None or len(beam_azimuth_degrees) < 2:
        return None
    # AUDIT-FIX(#1): Normalize azimuth-like primitives at the edge so later math
    # never sees raw external values that could raise TypeError.
    return (
        _normalize_optional_azimuth(beam_azimuth_degrees[0]),
        _normalize_optional_azimuth(beam_azimuth_degrees[1]),
    )


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
    # AUDIT-FIX(#1): The azimuth tuple is already normalized; return it directly
    # to avoid double-processing and to keep invalid values out of this path.
    return azimuth


def _processed_selected_azimuth(
    selected_azimuth_degrees: tuple[float | None, ...] | None,
) -> float | None:
    """Return the processed selected azimuth when XVF3800 provided one."""

    if selected_azimuth_degrees is None or not selected_azimuth_degrees:
        return None
    # AUDIT-FIX(#1): Normalize the selected azimuth with the same guarded parser
    # used for all other angle primitives.
    return _normalize_optional_azimuth(selected_azimuth_degrees[0])


# AUDIT-FIX(#4): Use float typing here because the confidence path operates on
# normalized floating-point azimuths, not an int-only contract.
def _alignment_score(reference_azimuth: float, candidate_azimuth: float) -> float:
    """Return one bounded confidence score from two azimuths."""

    # AUDIT-FIX(#1): Re-validate both inputs defensively so any future callsite
    # also degrades safely instead of raising on malformed numeric values.
    normalized_reference = _normalize_azimuth(reference_azimuth)
    normalized_candidate = _normalize_azimuth(candidate_azimuth)
    if normalized_reference is None or normalized_candidate is None:
        return 0.0

    delta = _circular_distance(normalized_reference, normalized_candidate)
    return max(0.0, 1.0 - (min(delta, 90.0) / 90.0))


def _normalize_optional_azimuth(value: object | None) -> float | None:
    """Normalize one optional azimuth-like value into ``[0, 360)``."""

    # AUDIT-FIX(#1): Centralize optional-angle handling so every azimuth source
    # gets the same crash-safe normalization behavior.
    if value is None:
        return None
    return _normalize_azimuth(value)


def _normalize_azimuth(value: object) -> float | None:
    """Normalize one azimuth-like value into ``[0, 360)``."""

    # AUDIT-FIX(#1): Route all azimuth normalization through a safe coercion
    # helper so bad payloads never crash this pure-derivation layer.
    normalized = _coerce_finite_float(value)
    if normalized is None:
        return None
    return normalized % 360.0


def _coerce_optional_bool(value: object | None) -> bool | None:
    """Return a strict tri-state bool, degrading invalid values to ``None``."""

    # AUDIT-FIX(#2): Keep tri-state inputs strict; this module should never
    # reinterpret integers or other truthy objects as valid runtime booleans.
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _coerce_finite_float(value: object, *, allow_negative: bool = True) -> float | None:
    """Coerce one external numeric payload into a finite float."""

    # AUDIT-FIX(#3): Reject bool and string-like payloads explicitly because
    # Python would otherwise coerce them into fabricated numeric evidence.
    if isinstance(value, (bool, str, bytes, bytearray)):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized):
        return None
    if not allow_negative and normalized < 0.0:
        return None
    return normalized


def _circular_distance(left: float, right: float) -> float:
    """Return the absolute circular distance in degrees."""

    raw_delta = abs((left - right) % 360.0)
    return min(raw_delta, 360.0 - raw_delta)