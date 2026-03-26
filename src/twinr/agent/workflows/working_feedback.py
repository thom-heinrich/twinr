"""Play bounded audio cues while Twinr is processing, answering, or printing."""

from __future__ import annotations

import audioop
import io
import math
import time
import wave
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from dataclasses import replace
import logging
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Literal, SupportsFloat, SupportsInt, cast
from weakref import WeakKeyDictionary

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator, PlaybackPriority
from twinr.agent.workflows.rendered_audio_clip import (
    RenderedAudioClipError,
    RenderedAudioClipSpec,
    build_rendered_audio_clip_wav_bytes,
    iter_wav_bytes_chunks,
)

_LOGGER = logging.getLogger(__name__)

WorkingFeedbackKind = Literal["processing", "answering", "printing"]


@dataclass(frozen=True, slots=True)
class WorkingFeedbackProfile:
    """Describe the tone timing and patterns for one feedback kind."""

    delay_ms: int
    pause_ms: int
    volume: float
    gap_ms: int
    patterns: tuple[tuple[tuple[int, int], ...], ...]


@dataclass(frozen=True, slots=True)
class WorkingFeedbackMediaSpec:
    """Describe one optional rendered media loop for a feedback kind."""

    clip: RenderedAudioClipSpec
    pause_ms: int = 80
    attenuation_start_s: float | None = None
    attenuation_reach_floor_s: float | None = None
    minimum_output_gain: float | None = None
    attenuation_step_ms: int = 1000


@dataclass(frozen=True, slots=True)
class _ResolvedWorkingFeedbackMedia:
    """Carry one rendered media clip plus optional PCM metadata."""

    spec: WorkingFeedbackMediaSpec
    wav_bytes: bytes
    pcm_bytes: bytes | None = None
    sample_rate: int | None = None
    channels: int | None = None


_DEFAULT_WORKING_FEEDBACK_PROFILES: dict[WorkingFeedbackKind, WorkingFeedbackProfile] = {
    "processing": WorkingFeedbackProfile(
        delay_ms=450,
        pause_ms=620,
        volume=0.06,
        gap_ms=28,
        patterns=(
            ((659, 52), (880, 38)),
            ((740, 44), (988, 34), (831, 40)),
            ((698, 46), (932, 36)),
            ((784, 42), (1046, 32), (932, 34)),
        ),
    ),
    "answering": WorkingFeedbackProfile(
        delay_ms=320,
        pause_ms=560,
        volume=0.13,
        gap_ms=24,
        patterns=(
            ((988, 42), (1318, 32)),
            ((1046, 38), (1396, 28), (1175, 32)),
            ((1175, 36), (1568, 26)),
            ((932, 40), (1244, 30), (1396, 28)),
        ),
    ),
    "printing": WorkingFeedbackProfile(
        delay_ms=180,
        pause_ms=500,
        volume=0.14,
        gap_ms=0,
        patterns=(
            ((988, 58),),
            ((1046, 54),),
            ((932, 60),),
            ((1108, 52),),
        ),
    ),
}

_DEFAULT_WORKING_FEEDBACK_MEDIA_SPECS: dict[WorkingFeedbackKind, WorkingFeedbackMediaSpec] = {
    "processing": WorkingFeedbackMediaSpec(
        clip=RenderedAudioClipSpec(
            relative_path=Path("media") / "dragon-studio-computer-startup-sound-effect-312870.mp3",
            clip_start_s=0.0,
            clip_duration_s=0.8,
            fade_in_duration_s=0.09,
            fade_out_start_s=1.08,
            fade_out_duration_s=0.15,
            output_gain=0.105,
            playback_speed=0.65,
            normalize_max_gain=1.0,
        ),
        pause_ms=0,
        attenuation_start_s=4.0,
        attenuation_reach_floor_s=30.0,
        minimum_output_gain=0.15,
        attenuation_step_ms=1000,
    )
}


_VALID_WORKING_FEEDBACK_KINDS = tuple(_DEFAULT_WORKING_FEEDBACK_PROFILES)


@dataclass(slots=True)
class _PlayerRuntimeState:
    """Track per-player locks and the currently active feedback generation."""

    playback_lock: Lock
    lifecycle_lock: Lock
    active_stop_event: Event | None = None
    generation: int = 0


# AUDIT-FIX(#1): Coordinate concurrent loops per player to prevent overlapping playback and stale workers.
_PLAYER_STATES_LOCK = Lock()
_PLAYER_STATES: WeakKeyDictionary[object, _PlayerRuntimeState] = WeakKeyDictionary()
_PLAYER_STATES_FALLBACK: dict[int, _PlayerRuntimeState] = {}


# AUDIT-FIX(#5): Keep background-thread telemetry best-effort and sanitized.
def _safe_emit(emit: Callable[[str], None] | None, message: str) -> None:
    """Emit a telemetry line best-effort from a worker thread."""
    if emit is None:
        return
    try:
        emit(message)
    except Exception:
        _LOGGER.warning(
            "Working feedback emit failed for message %s.",
            message,
            exc_info=True,
        )
        return


# AUDIT-FIX(#4): Reject invalid/bool coercions up front instead of failing later in the worker thread.
def _coerce_non_negative_int(value: object, *, field_name: str) -> int:
    """Coerce a runtime value to a non-negative integer."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must not be bool")
    if isinstance(value, int):
        coerced = value
    else:
        int_like = cast(SupportsInt | str | bytes | bytearray, value)
        coerced = int(int_like)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{field_name} must be an integer value")
    if coerced < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return coerced


# AUDIT-FIX(#4): Audio parameters must stay within a safe range for speaker output.
def _coerce_positive_int(value: object, *, field_name: str) -> int:
    """Coerce a runtime value to a strictly positive integer."""
    coerced = _coerce_non_negative_int(value, field_name=field_name)
    if coerced <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return coerced


# AUDIT-FIX(#4): Clamp/sanitize volume before playback so invalid config cannot produce unsafe output.
def _coerce_volume(value: object, *, field_name: str) -> float:
    """Coerce a runtime value to a bounded playback volume."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must not be bool")
    float_like = cast(SupportsFloat | str | bytes | bytearray, value)
    coerced = float(float_like)
    if not math.isfinite(coerced):
        raise ValueError(f"{field_name} must be finite")
    if not 0.0 <= coerced <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return coerced


# AUDIT-FIX(#4): Validate all notes eagerly so malformed profiles cannot trigger latent worker crashes.
def _normalize_patterns(
    patterns: object,
    *,
    field_name: str,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Validate and normalize tone-pattern tuples for playback."""
    if isinstance(patterns, (str, bytes, bytearray)) or not isinstance(patterns, Iterable):
        raise ValueError(f"{field_name} must be an iterable of tone sequences")
    normalized_patterns: list[tuple[tuple[int, int], ...]] = []
    for pattern_index, sequence in enumerate(patterns):
        if isinstance(sequence, (str, bytes, bytearray)) or not isinstance(sequence, Iterable):
            raise ValueError(f"{field_name}[{pattern_index}] must be an iterable of note tuples")
        normalized_sequence: list[tuple[int, int]] = []
        for tone_index, note in enumerate(sequence):
            try:
                frequency_hz, duration_ms = note
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field_name}[{pattern_index}][{tone_index}] must be a 2-item note tuple"
                ) from exc
            normalized_sequence.append(
                (
                    _coerce_positive_int(
                        frequency_hz,
                        field_name=f"{field_name}[{pattern_index}][{tone_index}].frequency_hz",
                    ),
                    _coerce_positive_int(
                        duration_ms,
                        field_name=f"{field_name}[{pattern_index}][{tone_index}].duration_ms",
                    ),
                )
            )
        if not normalized_sequence:
            raise ValueError(f"{field_name}[{pattern_index}] must not be empty")
        normalized_patterns.append(tuple(normalized_sequence))
    if not normalized_patterns:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(normalized_patterns)


# AUDIT-FIX(#4): Normalize runtime-provided profiles before starting the thread.
def _normalize_profile(profile: WorkingFeedbackProfile) -> WorkingFeedbackProfile:
    """Validate a working-feedback profile before worker startup."""
    return WorkingFeedbackProfile(
        delay_ms=_coerce_non_negative_int(profile.delay_ms, field_name="delay_ms"),
        pause_ms=_coerce_non_negative_int(profile.pause_ms, field_name="pause_ms"),
        volume=_coerce_volume(profile.volume, field_name="volume"),
        gap_ms=_coerce_non_negative_int(profile.gap_ms, field_name="gap_ms"),
        patterns=_normalize_patterns(profile.patterns, field_name="patterns"),
    )


# AUDIT-FIX(#4): Resolve invalid kind/profile input to a safe fallback instead of crashing callers or workers.
def _resolve_profile(
    kind: WorkingFeedbackKind,
    profiles: Mapping[WorkingFeedbackKind, WorkingFeedbackProfile] | None,
    delay_override_ms: int | None,
    emit: Callable[[str], None] | None,
) -> WorkingFeedbackProfile:
    """Resolve the effective feedback profile with safe fallbacks."""
    default_profile = _normalize_profile(_DEFAULT_WORKING_FEEDBACK_PROFILES[kind])
    candidate_profile = default_profile

    if profiles is not None:
        try:
            profile = profiles[kind]
        except KeyError:
            _safe_emit(emit, f"working_feedback_profile_fallback={kind}:missing")
        else:
            try:
                candidate_profile = _normalize_profile(profile)
            except (AttributeError, TypeError, ValueError) as exc:
                _safe_emit(
                    emit,
                    f"working_feedback_profile_fallback={kind}:{exc.__class__.__name__}",
                )
                candidate_profile = default_profile

    if delay_override_ms is not None:
        try:
            candidate_profile = replace(
                candidate_profile,
                delay_ms=_coerce_non_negative_int(
                    delay_override_ms,
                    field_name="delay_override_ms",
                ),
            )
        except (TypeError, ValueError) as exc:
            _safe_emit(
                emit,
                f"working_feedback_delay_override_ignored={kind}:{exc.__class__.__name__}",
            )

    return candidate_profile


# AUDIT-FIX(#1): Shared player instances need per-player state; WeakKeyDictionary avoids leaking normal objects.
def _get_player_runtime_state(player) -> _PlayerRuntimeState:
    """Return or create the shared runtime state for a player instance."""
    with _PLAYER_STATES_LOCK:
        try:
            state = _PLAYER_STATES.get(player)
        except TypeError:
            state = _PLAYER_STATES_FALLBACK.get(id(player))
            if state is None:
                state = _PlayerRuntimeState(
                    playback_lock=Lock(),
                    lifecycle_lock=Lock(),
                )
                _PLAYER_STATES_FALLBACK[id(player)] = state
            return state
        if state is None:
            state = _PlayerRuntimeState(
                playback_lock=Lock(),
                lifecycle_lock=Lock(),
            )
            _PLAYER_STATES[player] = state
        return state


# AUDIT-FIX(#1): New loops preempt prior loops for the same player instead of overlapping with them.
def _activate_player_loop(
    state: _PlayerRuntimeState,
    stop_event: Event,
) -> tuple[int, Event | None]:
    """Register a new active loop and return the superseded stop event."""
    with state.lifecycle_lock:
        previous_stop_event = state.active_stop_event
        state.active_stop_event = stop_event
        state.generation += 1
        generation = state.generation
    return generation, previous_stop_event


# AUDIT-FIX(#1): Stale workers must exit once another loop supersedes them.
def _is_active_player_loop(
    state: _PlayerRuntimeState,
    generation: int,
    stop_event: Event,
) -> bool:
    """Report whether a worker generation still owns the player."""
    with state.lifecycle_lock:
        return state.generation == generation and state.active_stop_event is stop_event


# AUDIT-FIX(#1): Do not clear the active loop marker if a newer worker has already replaced this one.
def _release_player_loop(
    state: _PlayerRuntimeState,
    generation: int,
    stop_event: Event,
) -> None:
    """Clear the active loop marker if this worker still owns it."""
    with state.lifecycle_lock:
        if state.generation == generation and state.active_stop_event is stop_event:
            state.active_stop_event = None


# AUDIT-FIX(#2): Best-effort backend stop hooks reduce the chance of lingering tones after cancellation.
def _stop_player_playback(player) -> None:
    """Stop tone playback best-effort on the configured player."""
    for method_name in ("stop_tone_sequence", "stop_tone", "stop_playback"):
        method = getattr(player, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                _LOGGER.warning(
                    "Working-feedback stop hook %s failed; trying the next fallback.",
                    method_name,
                    exc_info=True,
                )
                continue


# AUDIT-FIX(#4): Fail closed if the player cannot actually render tones.
def _player_supports_playback(player) -> bool:
    """Report whether the player exposes a supported tone playback API."""
    return (
        callable(getattr(player, "play_tone_sequence", None))
        or callable(getattr(player, "play_tone", None))
        or callable(getattr(player, "play_wav_chunks", None))
        or callable(getattr(player, "play_pcm16_chunks", None))
    )


def _play_sequence(
    player,
    sequence: tuple[tuple[int, int], ...],
    *,
    volume: float,
    sample_rate: int,
    gap_ms: int,
    stop_event: Event,
) -> None:
    """Play one normalized tone sequence through the configured player."""
    play_tone_sequence = getattr(player, "play_tone_sequence", None)
    if callable(play_tone_sequence):
        play_tone_sequence(
            sequence,
            volume=volume,
            sample_rate=sample_rate,
            gap_ms=gap_ms,
        )
        return

    play_tone = getattr(player, "play_tone", None)
    if not callable(play_tone):
        raise AttributeError("player must implement play_tone_sequence() or play_tone()")

    # AUDIT-FIX(#3): Match gap behavior and cancellation semantics even on the per-tone fallback path.
    for tone_index, (frequency_hz, duration_ms) in enumerate(sequence):
        if stop_event.is_set():
            return
        play_tone(
            frequency_hz=frequency_hz,
            duration_ms=duration_ms,
            volume=volume,
            sample_rate=sample_rate,
        )
        if stop_event.is_set():
            return
        if gap_ms > 0 and tone_index + 1 < len(sequence):
            if stop_event.wait(gap_ms / 1000.0):
                return


def _play_wav_clip(
    player,
    wav_bytes: bytes,
    *,
    stop_event: Event,
) -> None:
    """Play one rendered WAV clip through chunked, interruptible playback."""

    play_wav_chunks = getattr(player, "play_wav_chunks", None)
    if not callable(play_wav_chunks):
        raise AttributeError("player must implement play_wav_chunks()")
    play_wav_chunks(
        iter_wav_bytes_chunks(wav_bytes),
        should_stop=stop_event.is_set,
    )


def _play_pcm16_clip(
    player,
    *,
    pcm_chunks,
    sample_rate: int,
    channels: int,
    stop_event: Event,
) -> None:
    """Play one PCM16 clip through interruptible chunked playback."""

    play_pcm16_chunks = getattr(player, "play_pcm16_chunks", None)
    if not callable(play_pcm16_chunks):
        raise AttributeError("player must implement play_pcm16_chunks()")
    try:
        play_pcm16_chunks(
            pcm_chunks,
            sample_rate=sample_rate,
            channels=channels,
            should_stop=stop_event.is_set,
        )
    except TypeError as exc:
        if "should_stop" not in str(exc):
            raise
        play_pcm16_chunks(
            pcm_chunks,
            sample_rate=sample_rate,
            channels=channels,
        )


def _resolve_media_spec(kind: WorkingFeedbackKind) -> WorkingFeedbackMediaSpec | None:
    return _DEFAULT_WORKING_FEEDBACK_MEDIA_SPECS.get(kind)


def _media_uses_long_think_attenuation(media_spec: WorkingFeedbackMediaSpec) -> bool:
    return (
        media_spec.minimum_output_gain is not None
        and media_spec.attenuation_start_s is not None
        and media_spec.attenuation_reach_floor_s is not None
        and float(media_spec.clip.output_gain) > 0.0
    )


def _player_supports_pcm16_chunks(player) -> bool:
    return callable(getattr(player, "play_pcm16_chunks", None))


def _decode_pcm16_wav_bytes(wav_bytes: bytes) -> tuple[bytes, int, int] | None:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wave_reader:
            channels = int(wave_reader.getnchannels())
            sample_width = int(wave_reader.getsampwidth())
            sample_rate = int(wave_reader.getframerate())
            pcm_bytes = wave_reader.readframes(wave_reader.getnframes())
    except (wave.Error, EOFError):
        return None
    if sample_width != 2 or channels <= 0 or sample_rate <= 0 or not pcm_bytes:
        return None
    return pcm_bytes, sample_rate, channels


def _media_output_gain_for_elapsed_s(
    media_spec: WorkingFeedbackMediaSpec,
    *,
    elapsed_s: float,
) -> float:
    base_output_gain = max(0.0, float(media_spec.clip.output_gain))
    if not _media_uses_long_think_attenuation(media_spec):
        return base_output_gain
    floor_output_gain = max(
        0.0,
        min(base_output_gain, float(media_spec.minimum_output_gain or base_output_gain)),
    )
    start_s = max(0.0, float(media_spec.attenuation_start_s or 0.0))
    reach_floor_s = max(start_s, float(media_spec.attenuation_reach_floor_s or start_s))
    if elapsed_s <= start_s:
        return base_output_gain
    if reach_floor_s <= start_s or elapsed_s >= reach_floor_s:
        return floor_output_gain
    progress = (elapsed_s - start_s) / max(0.001, reach_floor_s - start_s)
    return base_output_gain + ((floor_output_gain - base_output_gain) * progress)


def _iter_attenuated_pcm16_chunks(
    media: _ResolvedWorkingFeedbackMedia,
    *,
    loop_elapsed_s: float,
):
    pcm_bytes = media.pcm_bytes or b""
    sample_rate = int(media.sample_rate or 0)
    channels = int(media.channels or 0)
    if not pcm_bytes or sample_rate <= 0 or channels <= 0:
        return
    step_ms = max(1000, int(media.spec.attenuation_step_ms))
    bytes_per_chunk = max(2 * channels, int(sample_rate * (step_ms / 1000.0)) * channels * 2)
    base_output_gain = max(0.001, float(media.spec.clip.output_gain))
    for chunk_index, start in enumerate(range(0, len(pcm_bytes), bytes_per_chunk)):
        chunk = pcm_bytes[start : start + bytes_per_chunk]
        if not chunk:
            continue
        chunk_elapsed_s = loop_elapsed_s + ((chunk_index * step_ms) / 1000.0)
        effective_gain = _media_output_gain_for_elapsed_s(
            media.spec,
            elapsed_s=chunk_elapsed_s,
        )
        scale = max(0.0, min(1.0, effective_gain / base_output_gain))
        if scale >= 0.999:
            yield chunk
            continue
        yield audioop.mul(chunk, 2, scale)


def _resolve_media_clip(
    *,
    player,
    kind: WorkingFeedbackKind,
    sample_rate: int,
    config: TwinrConfig | None,
    emit: Callable[[str], None] | None,
) -> _ResolvedWorkingFeedbackMedia | None:
    media_spec = _resolve_media_spec(kind)
    if media_spec is None:
        return None
    if config is None:
        return None
    try:
        wav_bytes = build_rendered_audio_clip_wav_bytes(config, media_spec.clip)
    except RenderedAudioClipError as exc:
        _safe_emit(
            emit,
            f"working_feedback_media_fallback={kind}:{exc.__class__.__name__}",
        )
        return None
    if wav_bytes is None:
        _safe_emit(
            emit,
            f"working_feedback_media_fallback={kind}:missing_asset",
        )
        return None
    if not _media_uses_long_think_attenuation(media_spec):
        return _ResolvedWorkingFeedbackMedia(spec=media_spec, wav_bytes=wav_bytes)
    if not _player_supports_pcm16_chunks(player):
        _safe_emit(
            emit,
            f"working_feedback_media_attenuation_fallback={kind}:missing_player_api",
        )
        return _ResolvedWorkingFeedbackMedia(spec=media_spec, wav_bytes=wav_bytes)
    decoded = _decode_pcm16_wav_bytes(wav_bytes)
    if decoded is None:
        _safe_emit(
            emit,
            f"working_feedback_media_attenuation_fallback={kind}:invalid_wav",
        )
        return _ResolvedWorkingFeedbackMedia(spec=media_spec, wav_bytes=wav_bytes)
    pcm_bytes, sample_rate, channels = decoded
    return _ResolvedWorkingFeedbackMedia(
        spec=media_spec,
        wav_bytes=wav_bytes,
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
    )


def start_working_feedback_loop(
    player,
    *,
    kind: WorkingFeedbackKind,
    sample_rate: int,
    emit: Callable[[str], None] | None = None,
    profiles: Mapping[WorkingFeedbackKind, WorkingFeedbackProfile] | None = None,
    delay_override_ms: int | None = None,
    playback_coordinator: PlaybackCoordinator | None = None,
    config: TwinrConfig | None = None,
) -> Callable[[], None]:
    """Start a bounded feedback loop and return its stop callback.

    Args:
        player: Audio player exposing ``play_tone_sequence()`` or ``play_tone()``.
        kind: Feedback profile family to use.
        sample_rate: Playback sample rate in hertz.
        emit: Optional telemetry sink for fallback and timeout notices.
        profiles: Optional profile overrides keyed by feedback kind.
        delay_override_ms: Optional replacement startup delay in milliseconds.

    Returns:
        A no-argument callback that stops the active feedback worker.
    """
    # AUDIT-FIX(#4): Runtime values can still be invalid even if typing says otherwise.
    if kind not in _VALID_WORKING_FEEDBACK_KINDS:
        _safe_emit(emit, "working_feedback_disabled=invalid_kind")
        return lambda: None

    try:
        normalized_sample_rate = _coerce_positive_int(
            sample_rate,
            field_name="sample_rate",
        )
    except (TypeError, ValueError) as exc:
        _safe_emit(
            emit,
            f"working_feedback_disabled={kind}:invalid_sample_rate:{exc.__class__.__name__}",
        )
        return lambda: None

    if not _player_supports_playback(player):
        _safe_emit(emit, f"working_feedback_disabled={kind}:missing_player_api")
        return lambda: None

    # AUDIT-FIX(#6): Preserve the distinction between "no profile override" and "misconfigured empty mapping".
    profile = _resolve_profile(
        kind,
        profiles if profiles is not None else None,
        delay_override_ms,
        emit,
    )
    resolved_media = _resolve_media_clip(
        player=player,
        kind=kind,
        sample_rate=normalized_sample_rate,
        config=config,
        emit=emit,
    )
    stop_event = Event()
    player_state = _get_player_runtime_state(player)
    generation, previous_stop_event = _activate_player_loop(player_state, stop_event)
    if previous_stop_event is not None and previous_stop_event is not stop_event:
        previous_stop_event.set()

    def wait_for_pause() -> bool:
        pause_ms = resolved_media.spec.pause_ms if resolved_media is not None else profile.pause_ms
        if pause_ms <= 0:
            return stop_event.is_set()
        return stop_event.wait(pause_ms / 1000.0)

    def worker() -> None:
        try:
            if stop_event.wait(profile.delay_ms / 1000.0):
                return
            feedback_started_at = time.monotonic()
            pattern_index = 0
            while not stop_event.is_set():
                if not _is_active_player_loop(player_state, generation, stop_event):
                    return
                sequence = profile.patterns[pattern_index % len(profile.patterns)]
                pattern_index += 1

                acquired = False
                while not stop_event.is_set():
                    if not _is_active_player_loop(player_state, generation, stop_event):
                        return
                    if playback_coordinator is not None:
                        break
                    acquired = player_state.playback_lock.acquire(timeout=0.1)
                    if acquired:
                        break
                if not acquired and playback_coordinator is None:
                    return

                try:
                    if stop_event.is_set() or not _is_active_player_loop(
                        player_state,
                        generation,
                        stop_event,
                    ):
                        return
                    if (
                        resolved_media is not None
                        and resolved_media.pcm_bytes is not None
                        and resolved_media.sample_rate is not None
                        and resolved_media.channels is not None
                        and (
                            playback_coordinator is not None
                            or _player_supports_pcm16_chunks(player)
                        )
                    ):
                        loop_elapsed_s = max(0.0, time.monotonic() - feedback_started_at)
                        pcm_chunks = _iter_attenuated_pcm16_chunks(
                            resolved_media,
                            loop_elapsed_s=loop_elapsed_s,
                        )
                        if playback_coordinator is None:
                            _play_pcm16_clip(
                                player,
                                pcm_chunks=pcm_chunks,
                                sample_rate=resolved_media.sample_rate,
                                channels=resolved_media.channels,
                                stop_event=stop_event,
                            )
                        else:
                            playback_coordinator.play_pcm16_chunks(
                                owner=f"working_feedback:{kind}",
                                priority=PlaybackPriority.FEEDBACK,
                                chunks=pcm_chunks,
                                sample_rate=resolved_media.sample_rate,
                                channels=resolved_media.channels,
                                should_stop=stop_event.is_set,
                            )
                    elif resolved_media is not None and (
                        playback_coordinator is not None or callable(getattr(player, "play_wav_chunks", None))
                    ):
                        if playback_coordinator is None:
                            _play_wav_clip(
                                player,
                                resolved_media.wav_bytes,
                                stop_event=stop_event,
                            )
                        else:
                            playback_coordinator.play_wav_chunks(
                                owner=f"working_feedback:{kind}",
                                priority=PlaybackPriority.FEEDBACK,
                                chunks=iter_wav_bytes_chunks(resolved_media.wav_bytes),
                                should_stop=stop_event.is_set,
                            )
                    elif playback_coordinator is None:
                        _play_sequence(
                            player,
                            sequence,
                            volume=profile.volume,
                            sample_rate=normalized_sample_rate,
                            gap_ms=profile.gap_ms,
                            stop_event=stop_event,
                        )
                    else:
                        playback_coordinator.play_tone_sequence(
                            owner=f"working_feedback:{kind}",
                            priority=PlaybackPriority.FEEDBACK,
                            sequence=sequence,
                            volume=profile.volume,
                            sample_rate=normalized_sample_rate,
                            gap_ms=profile.gap_ms,
                            should_stop=stop_event.is_set,
                        )
                except Exception as exc:
                    if stop_event.is_set() or not _is_active_player_loop(
                        player_state,
                        generation,
                        stop_event,
                    ):
                        return
                    # AUDIT-FIX(#5): Never emit raw exception payloads from the audio backend.
                    _safe_emit(
                        emit,
                        f"working_feedback_error={kind}:{exc.__class__.__name__}",
                    )
                    return
                finally:
                    if acquired:
                        player_state.playback_lock.release()

                if wait_for_pause():
                    return
        finally:
            _release_player_loop(player_state, generation, stop_event)

    # AUDIT-FIX(#5): Thread naming materially improves production diagnosis without changing behavior.
    thread = Thread(
        target=worker,
        name=f"working-feedback-{kind}",
        daemon=True,
    )
    thread.start()

    stop_called = Event()
    owner = f"working_feedback:{kind}"

    def stop() -> None:
        # AUDIT-FIX(#2): Make stop idempotent and warn if the worker could not be drained in time.
        if stop_called.is_set():
            return
        stop_called.set()
        stop_event.set()
        if _is_active_player_loop(player_state, generation, stop_event):
            if playback_coordinator is not None:
                playback_coordinator.stop_owner(owner)
            else:
                _stop_player_playback(player)
        if thread.is_alive() and thread is not current_thread():
            thread.join(timeout=1.0)
        if thread.is_alive():
            _safe_emit(emit, f"working_feedback_stop_timeout={kind}")
        else:
            _release_player_loop(player_state, generation, stop_event)

    return stop
