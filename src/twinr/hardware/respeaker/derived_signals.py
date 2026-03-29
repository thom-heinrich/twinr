# CHANGELOG: 2026-03-28
# BUG-1: Fixed false negatives where positive fixed-beam speech evidence was overridden by a lagging/false global speech flag.
# BUG-2: Keep busy-state overlap/barge-in conservative by requiring stronger
#        evidence than a single fixed-beam speech bit during playback; optional
#        pVAD/primary-speaker hooks can still assert real interruptions earlier.
# BUG-3: Fixed direction confidence hard-failing on missing raw DoA even when XVF3800 still provides a valid processed selected azimuth.
# SEC-1: Added bounded parsing of external beam/azimuth sequences to prevent hot-loop CPU abuse from oversized/corrupted sensor or IPC payloads on Raspberry Pi deployments.
# IMP-1: Added ReSpeakerSignalDeriver with temporal hysteresis and confidence smoothing for production use while keeping the old stateless function drop-in compatible.
# IMP-2: Added optional external evidence hooks (acoustic VAD / primary-speaker probability) so the module can fuse XVF3800 primitives with 2026-style pVAD / semantic-turn-taking stacks.

"""Derive conservative higher-level XVF3800 signals from direct primitives.

For production use, prefer ``ReSpeakerSignalDeriver`` over the stateless
``derive_respeaker_signal_state()`` wrapper. The class adds temporal hysteresis
and confidence smoothing while remaining stdlib-only and Raspberry-Pi-safe.

The stateless function remains drop-in compatible with the previous API and adds
two optional keyword-only hooks:
    * ``acoustic_vad_probability`` for an external low-latency acoustic VAD
    * ``primary_speaker_probability`` for target-speaker / pVAD gating
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import math

from twinr.hardware.respeaker.models import ReSpeakerDirectionSnapshot


_EXPECTED_FIXED_BEAM_COUNT = 2
_EXPECTED_SELECTED_AZIMUTH_VALUES = 2


@dataclass(frozen=True, slots=True)
class ReSpeakerDerivedSignalState:
    """Store interpreted signals derived from XVF3800 primitives."""

    fixed_beam_speech_count: int | None
    near_end_speech_detected: bool | None
    direction_confidence: float | None
    speech_overlap_likely: bool | None
    barge_in_detected: bool | None


@dataclass(frozen=True, slots=True)
class ReSpeakerSignalDeriverConfig:
    """Runtime tuning for the stateful derivation path."""

    near_end_assert_frames: int = 1
    near_end_release_frames: int = 3
    overlap_assert_frames: int = 1
    overlap_release_frames: int = 2
    barge_in_assert_frames: int = 1
    barge_in_release_frames: int = 2
    direction_confidence_ema_alpha: float = 0.4
    direction_confidence_hold_frames: int = 2
    acoustic_vad_on_threshold: float = 0.6
    primary_speaker_on_threshold: float = 0.6

    def __post_init__(self) -> None:
        for name in (
            "near_end_assert_frames",
            "near_end_release_frames",
            "overlap_assert_frames",
            "overlap_release_frames",
            "barge_in_assert_frames",
            "barge_in_release_frames",
            "direction_confidence_hold_frames",
        ):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1")

        for name in (
            "direction_confidence_ema_alpha",
            "acoustic_vad_on_threshold",
            "primary_speaker_on_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0.0, 1.0]")


class _TriStateHysteresis:
    """Minimal hot-loop-friendly hysteresis for bool|None evidence."""

    __slots__ = ("_state", "_true_streak", "_false_streak")

    def __init__(self) -> None:
        self._state: bool | None = None
        self._true_streak = 0
        self._false_streak = 0

    def reset(self) -> None:
        self._state = None
        self._true_streak = 0
        self._false_streak = 0

    def update(
        self,
        evidence: bool | None,
        *,
        assert_frames: int,
        release_frames: int,
    ) -> bool | None:
        if evidence is True:
            self._true_streak += 1
            self._false_streak = 0
            if self._state is not True and self._true_streak >= assert_frames:
                self._state = True
            return self._state

        if evidence is False:
            self._false_streak += 1
            self._true_streak = 0
            if self._state is None and self._false_streak >= assert_frames:
                self._state = False
            elif self._state is True and self._false_streak >= release_frames:
                self._state = False
            return self._state

        self._true_streak = 0
        self._false_streak = 0
        return self._state


class ReSpeakerSignalDeriver:
    """Stateful signal fusion for production full-duplex control.

    This class is the recommended 2026 path because it adds the minimum temporal
    stabilization expected from modern edge voice stacks while staying light
    enough for a Raspberry Pi 4.
    """

    def __init__(self, config: ReSpeakerSignalDeriverConfig | None = None) -> None:
        self._config = config or ReSpeakerSignalDeriverConfig()
        self._near_end = _TriStateHysteresis()
        self._overlap = _TriStateHysteresis()
        self._barge_in = _TriStateHysteresis()
        self._direction_confidence_ema: float | None = None
        self._direction_confidence_missing_frames = 0

    def reset(self) -> None:
        """Reset all temporal state."""

        self._near_end.reset()
        self._overlap.reset()
        self._barge_in.reset()
        self._direction_confidence_ema = None
        self._direction_confidence_missing_frames = 0

    def update(
        self,
        direction: ReSpeakerDirectionSnapshot,
        *,
        assistant_output_active: bool | None,
        acoustic_vad_probability: float | None = None,
        primary_speaker_probability: float | None = None,
    ) -> ReSpeakerDerivedSignalState:
        """Fuse one snapshot into temporally stabilized runtime signals."""

        frame = _derive_frame_signal_state(
            direction,
            assistant_output_active=assistant_output_active,
            acoustic_vad_probability=acoustic_vad_probability,
            primary_speaker_probability=primary_speaker_probability,
            acoustic_vad_on_threshold=self._config.acoustic_vad_on_threshold,
            primary_speaker_on_threshold=self._config.primary_speaker_on_threshold,
        )

        near_end = self._near_end.update(
            frame.near_end_speech_detected,
            assert_frames=self._config.near_end_assert_frames,
            release_frames=self._config.near_end_release_frames,
        )

        overlap_frame = _speech_overlap_likely(
            assistant_output_active=_coerce_optional_bool(assistant_output_active),
            near_end_speech_detected=near_end,
            fixed_beam_speech_count=frame.fixed_beam_speech_count,
            primary_speaker_probability=primary_speaker_probability,
            primary_speaker_on_threshold=self._config.primary_speaker_on_threshold,
        )
        overlap = self._overlap.update(
            overlap_frame,
            assert_frames=self._config.overlap_assert_frames,
            release_frames=self._config.overlap_release_frames,
        )

        barge_frame = _barge_in_detected(
            assistant_output_active=_coerce_optional_bool(assistant_output_active),
            near_end_speech_detected=near_end,
            fixed_beam_speech_count=frame.fixed_beam_speech_count,
            primary_speaker_probability=primary_speaker_probability,
            primary_speaker_on_threshold=self._config.primary_speaker_on_threshold,
        )
        barge_in = self._barge_in.update(
            barge_frame,
            assert_frames=self._config.barge_in_assert_frames,
            release_frames=self._config.barge_in_release_frames,
        )

        direction_confidence = self._smooth_direction_confidence(
            frame.direction_confidence if near_end is True else None
        )

        return ReSpeakerDerivedSignalState(
            fixed_beam_speech_count=frame.fixed_beam_speech_count,
            near_end_speech_detected=near_end,
            direction_confidence=direction_confidence,
            speech_overlap_likely=overlap,
            barge_in_detected=barge_in,
        )

    def _smooth_direction_confidence(self, current: float | None) -> float | None:
        if current is None:
            if self._direction_confidence_ema is None:
                return None
            self._direction_confidence_missing_frames += 1
            if self._direction_confidence_missing_frames > self._config.direction_confidence_hold_frames:
                self._direction_confidence_ema = None
                self._direction_confidence_missing_frames = 0
                return None
            return round(self._direction_confidence_ema, 3)

        self._direction_confidence_missing_frames = 0
        alpha = self._config.direction_confidence_ema_alpha
        if self._direction_confidence_ema is None:
            self._direction_confidence_ema = current
        else:
            self._direction_confidence_ema = (
                (alpha * current)
                + ((1.0 - alpha) * self._direction_confidence_ema)
            )
        return round(max(0.0, min(1.0, self._direction_confidence_ema)), 3)


def derive_respeaker_signal_state(
    direction: ReSpeakerDirectionSnapshot,
    *,
    assistant_output_active: bool | None,
    acoustic_vad_probability: float | None = None,
    primary_speaker_probability: float | None = None,
) -> ReSpeakerDerivedSignalState:
    """Interpret one snapshot into conservative runtime signals.

    The XVF3800 exposes direct speech/no-speech, beam energies, focused-beam
    azimuths, and selected azimuths, but not a first-class near-end/double-talk
    state. This stateless helper keeps the original call pattern intact while
    correcting the old false-negative and false-barge-in logic.

    ``acoustic_vad_probability`` and ``primary_speaker_probability`` are
    optional frontier hooks for external low-latency VAD / pVAD modules.
    """

    return _derive_frame_signal_state(
        direction,
        assistant_output_active=assistant_output_active,
        acoustic_vad_probability=acoustic_vad_probability,
        primary_speaker_probability=primary_speaker_probability,
        acoustic_vad_on_threshold=0.6,
        primary_speaker_on_threshold=0.6,
    )


def _derive_frame_signal_state(
    direction: ReSpeakerDirectionSnapshot,
    *,
    assistant_output_active: bool | None,
    acoustic_vad_probability: float | None,
    primary_speaker_probability: float | None,
    acoustic_vad_on_threshold: float,
    primary_speaker_on_threshold: float,
) -> ReSpeakerDerivedSignalState:
    """Derive one instantaneous signal state without temporal smoothing."""

    speech_detected = _coerce_optional_bool(direction.speech_detected)
    assistant_output_active = _coerce_optional_bool(assistant_output_active)

    fixed_beam_speech_count = _count_fixed_beam_speech(direction.beam_speech_energies)

    near_end_speech_detected = _near_end_speech_detected(
        speech_detected=speech_detected,
        fixed_beam_speech_count=fixed_beam_speech_count,
        acoustic_vad_probability=acoustic_vad_probability,
        acoustic_vad_on_threshold=acoustic_vad_on_threshold,
    )

    direction_confidence = _estimate_direction_confidence(
        direction=direction,
        speech_detected=speech_detected,
        fixed_beam_speech_count=fixed_beam_speech_count,
    )

    speech_overlap_likely = _speech_overlap_likely(
        assistant_output_active=assistant_output_active,
        near_end_speech_detected=near_end_speech_detected,
        fixed_beam_speech_count=fixed_beam_speech_count,
        primary_speaker_probability=primary_speaker_probability,
        primary_speaker_on_threshold=primary_speaker_on_threshold,
    )

    barge_in_detected = _barge_in_detected(
        assistant_output_active=assistant_output_active,
        near_end_speech_detected=near_end_speech_detected,
        fixed_beam_speech_count=fixed_beam_speech_count,
        primary_speaker_probability=primary_speaker_probability,
        primary_speaker_on_threshold=primary_speaker_on_threshold,
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
    """Count focused beams whose official speech-energy values indicate speech."""

    fixed_beams = _fixed_beam_energies(beam_speech_energies)
    if fixed_beams is None:
        return None
    return sum(1 for energy in fixed_beams if energy > 0.0)


def _near_end_speech_detected(
    *,
    speech_detected: bool | None,
    fixed_beam_speech_count: int | None,
    acoustic_vad_probability: float | None,
    acoustic_vad_on_threshold: float,
) -> bool | None:
    """Return whether near-end speech is currently present.

    Positive focused-beam speech energy is treated as the strongest direct cue.
    This fixes the prior false-negative path where ``speech_detected=False``
    could wrongly suppress positive beam speech evidence.
    """

    if fixed_beam_speech_count is not None:
        if fixed_beam_speech_count >= 1:
            return True
        if speech_detected is False:
            return False

        probability = _coerce_optional_probability(acoustic_vad_probability)
        if probability is not None and probability >= acoustic_vad_on_threshold:
            return None
        return False

    probability = _coerce_optional_probability(acoustic_vad_probability)
    if probability is not None:
        return probability >= acoustic_vad_on_threshold
    if speech_detected is False:
        return False
    return None


# BREAKING: ``speech_overlap_likely`` now means assistant/user double-talk
# instead of "two fixed beams are simultaneously hot". The old meaning was not a
# useful barge-in signal and breaks once XVF3800 fixed-beam gating is enabled.
def _speech_overlap_likely(
    *,
    assistant_output_active: bool | None,
    near_end_speech_detected: bool | None,
    fixed_beam_speech_count: int | None,
    primary_speaker_probability: float | None,
    primary_speaker_on_threshold: float,
) -> bool | None:
    """Return whether assistant playback and near-end speech overlap.

    Current V1 intentionally stays conservative during playback: a single
    fixed-beam speech bit is not sufficient by itself because assistant output
    leakage can satisfy that cue. Stronger overlap evidence comes either from
    multiple active fixed beams or an explicit external primary-speaker score.
    """

    if near_end_speech_detected is False:
        return False
    if assistant_output_active is True:
        probability = _coerce_optional_probability(primary_speaker_probability)
        if probability is not None:
            return probability >= primary_speaker_on_threshold
        if fixed_beam_speech_count is None:
            return None
        return fixed_beam_speech_count >= 2
    if assistant_output_active is False:
        if fixed_beam_speech_count is None:
            return False
        return fixed_beam_speech_count >= 2
    return None


def _barge_in_detected(
    *,
    assistant_output_active: bool | None,
    near_end_speech_detected: bool | None,
    fixed_beam_speech_count: int | None,
    primary_speaker_probability: float | None,
    primary_speaker_on_threshold: float,
) -> bool | None:
    """Return whether the active user is likely interrupting assistant output."""

    overlap = _speech_overlap_likely(
        assistant_output_active=assistant_output_active,
        near_end_speech_detected=near_end_speech_detected,
        fixed_beam_speech_count=fixed_beam_speech_count,
        primary_speaker_probability=primary_speaker_probability,
        primary_speaker_on_threshold=primary_speaker_on_threshold,
    )
    if overlap is not True:
        return overlap

    probability = _coerce_optional_probability(primary_speaker_probability)
    if probability is None:
        return True
    return probability >= primary_speaker_on_threshold


def _estimate_direction_confidence(
    *,
    direction: ReSpeakerDirectionSnapshot,
    speech_detected: bool | None,
    fixed_beam_speech_count: int | None,
) -> float | None:
    """Estimate one conservative confidence score for the current direction."""

    if speech_detected is False and (fixed_beam_speech_count is None or fixed_beam_speech_count < 1):
        return None
    if fixed_beam_speech_count is not None and fixed_beam_speech_count < 1:
        return None

    doa_degrees = _normalize_optional_azimuth(direction.doa_degrees)
    fixed_beam_energies = _fixed_beam_energies(direction.beam_speech_energies)
    fixed_beam_azimuths = _fixed_beam_azimuths(direction.beam_azimuth_degrees)
    strongest_beam_azimuth = _strongest_fixed_beam_azimuth(
        fixed_beam_energies=fixed_beam_energies,
        fixed_beam_azimuths=fixed_beam_azimuths,
    )
    processed_selected_azimuth, auto_selected_azimuth = _selected_azimuth_pair(
        direction.selected_azimuth_degrees
    )

    cues = tuple(
        value
        for value in (
            doa_degrees,
            processed_selected_azimuth,
            auto_selected_azimuth,
            strongest_beam_azimuth,
        )
        if value is not None
    )
    if not cues:
        return None

    cue_agreement = _pairwise_azimuth_agreement(cues)
    beam_dominance = _beam_dominance_score(fixed_beam_energies)
    selected_bonus = _selected_azimuth_bonus(
        processed_selected_azimuth=processed_selected_azimuth,
        auto_selected_azimuth=auto_selected_azimuth,
    )
    ambiguity_factor = _ambiguity_factor(
        fixed_beam_speech_count=fixed_beam_speech_count,
        beam_dominance=beam_dominance,
    )

    confidence = (
        (0.60 * cue_agreement)
        + (0.25 * beam_dominance)
        + (0.15 * selected_bonus)
    ) * ambiguity_factor
    return round(max(0.0, min(1.0, confidence)), 3)


def _fixed_beam_energies(
    beam_speech_energies: tuple[float | None, ...] | None,
) -> tuple[float, float] | None:
    """Return the first two focused-beam speech energies when available."""

    values = _bounded_prefix_tuple(
        beam_speech_energies,
        limit=_EXPECTED_FIXED_BEAM_COUNT,
    )
    if values is None or len(values) < _EXPECTED_FIXED_BEAM_COUNT:
        return None

    normalized_values: list[float] = []
    for energy in values[:_EXPECTED_FIXED_BEAM_COUNT]:
        normalized = _coerce_finite_float(energy, allow_negative=False)
        if normalized is None:
            return None
        normalized_values.append(normalized)
    return normalized_values[0], normalized_values[1]


def _fixed_beam_azimuths(
    beam_azimuth_degrees: tuple[float | None, ...] | None,
) -> tuple[float | None, float | None] | None:
    """Return the first two focused-beam azimuths when available."""

    values = _bounded_prefix_tuple(
        beam_azimuth_degrees,
        limit=_EXPECTED_FIXED_BEAM_COUNT,
    )
    if values is None or len(values) < _EXPECTED_FIXED_BEAM_COUNT:
        return None

    return (
        _normalize_optional_azimuth(values[0]),
        _normalize_optional_azimuth(values[1]),
    )


def _strongest_fixed_beam_azimuth(
    *,
    fixed_beam_energies: tuple[float, float] | None,
    fixed_beam_azimuths: tuple[float | None, float | None] | None,
) -> float | None:
    """Return the azimuth of the strongest focused beam."""

    if fixed_beam_energies is None or fixed_beam_azimuths is None:
        return None
    strongest_index = 0 if fixed_beam_energies[0] >= fixed_beam_energies[1] else 1
    return fixed_beam_azimuths[strongest_index]


def _selected_azimuth_pair(
    selected_azimuth_degrees: tuple[float | None, ...] | None,
) -> tuple[float | None, float | None]:
    """Return XVF3800 processed speaker azimuth and auto-select beam azimuth.

    XMOS documents ``AUDIO_MGR_SELECTED_AZIMUTHS`` as returning two values:
    index 0 = processed DoA for the current speaker, index 1 = auto-select beam.
    """

    values = _bounded_prefix_tuple(
        selected_azimuth_degrees,
        limit=_EXPECTED_SELECTED_AZIMUTH_VALUES,
    )
    if values is None:
        return (None, None)

    processed = _normalize_optional_azimuth(values[0]) if len(values) >= 1 else None
    auto_selected = _normalize_optional_azimuth(values[1]) if len(values) >= 2 else None
    return processed, auto_selected


def _pairwise_azimuth_agreement(azimuths: tuple[float, ...]) -> float:
    """Return one bounded agreement score across all available azimuth cues."""

    if not azimuths:
        return 0.0
    if len(azimuths) == 1:
        return 0.5

    scores: list[float] = []
    for index, left in enumerate(azimuths):
        for right in azimuths[index + 1 :]:
            scores.append(_alignment_score(left, right))
    if not scores:
        return 0.5
    return sum(scores) / len(scores)


def _beam_dominance_score(fixed_beam_energies: tuple[float, float] | None) -> float:
    """Return how strongly one focused beam dominates the other."""

    if fixed_beam_energies is None:
        return 0.0
    total = fixed_beam_energies[0] + fixed_beam_energies[1]
    if total <= 0.0:
        return 0.0
    return abs(fixed_beam_energies[0] - fixed_beam_energies[1]) / total


def _selected_azimuth_bonus(
    *,
    processed_selected_azimuth: float | None,
    auto_selected_azimuth: float | None,
) -> float:
    """Prefer XVF3800 processed selected azimuth over weaker fallback cues."""

    if processed_selected_azimuth is not None:
        return 1.0
    if auto_selected_azimuth is not None:
        return 0.7
    return 0.35


def _ambiguity_factor(
    *,
    fixed_beam_speech_count: int | None,
    beam_dominance: float,
) -> float:
    """Reduce confidence when speech evidence is spatially ambiguous."""

    if fixed_beam_speech_count is None:
        return 0.75
    if fixed_beam_speech_count <= 1:
        return 1.0
    return 0.55 + (0.25 * beam_dominance)


def _alignment_score(reference_azimuth: float, candidate_azimuth: float) -> float:
    """Return one bounded confidence score from two azimuths."""

    normalized_reference = _normalize_azimuth(reference_azimuth)
    normalized_candidate = _normalize_azimuth(candidate_azimuth)
    if normalized_reference is None or normalized_candidate is None:
        return 0.0

    delta = _circular_distance(normalized_reference, normalized_candidate)
    return max(0.0, 1.0 - (min(delta, 90.0) / 90.0))


def _normalize_optional_azimuth(value: object | None) -> float | None:
    """Normalize one optional azimuth-like value into ``[0, 360)``."""

    if value is None:
        return None
    return _normalize_azimuth(value)


def _normalize_azimuth(value: object) -> float | None:
    """Normalize one azimuth-like value into ``[0, 360)``."""

    normalized = _coerce_finite_float(value)
    if normalized is None:
        return None
    return normalized % 360.0


def _coerce_optional_bool(value: object | None) -> bool | None:
    """Return a strict tri-state bool, degrading invalid values to ``None``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _coerce_optional_probability(value: object | None) -> float | None:
    """Normalize one optional probability into ``[0, 1]``."""

    if value is None:
        return None
    normalized = _coerce_finite_float(value)
    if normalized is None:
        return None
    return max(0.0, min(1.0, normalized))


def _coerce_finite_float(value: object, *, allow_negative: bool = True) -> float | None:
    """Coerce one external numeric payload into a finite float."""

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


def _bounded_prefix_tuple(values: object | None, *, limit: int) -> tuple[object, ...] | None:
    """Return at most ``limit`` items from an external iterable-like payload."""

    if values is None:
        return None
    if limit < 1:
        return tuple()
    if isinstance(values, (str, bytes, bytearray)):
        return None
    if isinstance(values, tuple):
        return values[:limit]
    if isinstance(values, list):
        return tuple(values[:limit])
    try:
        iterator = iter(values)
    except TypeError:
        return None
    return tuple(islice(iterator, limit))


def _circular_distance(left: float, right: float) -> float:
    """Return the absolute circular distance in degrees."""

    raw_delta = abs((left - right) % 360.0)
    return min(raw_delta, 360.0 - raw_delta)
