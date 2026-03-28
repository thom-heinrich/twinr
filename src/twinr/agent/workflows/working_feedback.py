# CHANGELOG: 2026-03-28
# BUG-1: Removed stdlib audioop usage; the module now imports and runs on Python 3.13+.
# BUG-2: Fixed tone playback on chunk-only backends; answering/printing no longer fail when a player exposes play_pcm16_chunks()/play_wav_chunks() but not play_tone*().
# BUG-3: Fixed long-think attenuation; the default processing media now actually ducks over time instead of staying at full configured gain.
# BUG-4: Fixed playback-coordinator owner collisions by making owners unique per loop instance.
# SEC-1: Replaced the unbounded id()-fallback registry with attached-state / weakref.finalize cleanup plus a bounded last-resort cache to prevent practical local memory-DoS on Raspberry Pi deployments.
# IMP-1: Pre-render and cache click-free PCM tone assets, then prefer raw-buffer playback for lower jitter, cleaner cancellation, and better compatibility with modern Pi audio backends.
# IMP-2: Retuned the default cue palette and added default runtime ceilings so senior-facing devices avoid shrill feedback and endless loops when upstream workflows hang.

"""Play bounded audio cues while Twinr is processing, answering, or printing."""

from __future__ import annotations

import io
import inspect
import math
import sys
import time
import wave
import weakref
from array import array
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
import logging
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Literal, SupportsFloat, SupportsInt, cast

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
    max_runtime_s: float | None = None


@dataclass(frozen=True, slots=True)
class WorkingFeedbackMediaSpec:
    """Describe one optional rendered media loop for a feedback kind."""

    clip: RenderedAudioClipSpec
    pause_ms: int = 80
    attenuation_start_s: float | None = None
    attenuation_reach_floor_s: float | None = None
    minimum_output_gain: float | None = None
    minimum_output_gain_ratio: float | None = None
    attenuation_step_ms: int = 160


@dataclass(frozen=True, slots=True)
class _ResolvedWorkingFeedbackMedia:
    """Carry one rendered media clip plus optional PCM metadata."""

    spec: WorkingFeedbackMediaSpec
    wav_bytes: bytes
    pcm_bytes: bytes | None = None
    sample_rate: int | None = None
    channels: int | None = None


@dataclass(frozen=True, slots=True)
class _RenderedToneSequence:
    """Carry one pre-rendered tone sequence for raw-buffer playback."""

    sequence: tuple[tuple[int, int], ...]
    pcm_bytes: bytes
    wav_bytes: bytes
    sample_rate: int
    channels: int = 1


# BREAKING: Default tones are now centered below ~1 kHz and all default profiles
# enforce a safety runtime ceiling. This is intentional for senior-facing devices:
# the old palette was noticeably shriller and the old loop could run forever if an
# upstream workflow forgot to stop it.
_DEFAULT_WORKING_FEEDBACK_PROFILES: dict[WorkingFeedbackKind, WorkingFeedbackProfile] = {
    "processing": WorkingFeedbackProfile(
        delay_ms=450,
        pause_ms=620,
        volume=0.065,
        gap_ms=24,
        patterns=(
            ((523, 72), (659, 52)),
            ((587, 64), (740, 46), (659, 52)),
            ((554, 68), (698, 48)),
            ((622, 62), (784, 44), (698, 46)),
        ),
        max_runtime_s=120.0,
    ),
    "answering": WorkingFeedbackProfile(
        delay_ms=320,
        pause_ms=520,
        volume=0.12,
        gap_ms=20,
        patterns=(
            ((740, 48), (932, 34)),
            ((784, 44), (988, 30), (880, 34)),
            ((880, 42), (988, 30)),
            ((698, 48), (880, 34), (988, 30)),
        ),
        max_runtime_s=90.0,
    ),
    "printing": WorkingFeedbackProfile(
        delay_ms=180,
        pause_ms=420,
        volume=0.13,
        gap_ms=0,
        patterns=(
            ((698, 64),),
            ((740, 60),),
            ((659, 66),),
            ((784, 58),),
        ),
        max_runtime_s=45.0,
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
        minimum_output_gain_ratio=0.15,
        attenuation_step_ms=160,
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
    last_used_monotonic: float = 0.0


_PLAYER_STATE_ATTR = "__twinr_working_feedback_runtime_state__"
_PLAYER_STATES_LOCK = Lock()
_PLAYER_STATES: weakref.WeakKeyDictionary[object, _PlayerRuntimeState] = weakref.WeakKeyDictionary()
_PLAYER_STATES_BY_ID: dict[int, _PlayerRuntimeState] = {}
_PLAYER_STATE_FINALIZERS: dict[int, weakref.finalize] = {}
_PLAYER_STATES_LAST_RESORT: OrderedDict[int, _PlayerRuntimeState] = OrderedDict()
_PLAYER_STATES_LAST_RESORT_MAX = 16
_PLAYER_STATES_LAST_RESORT_MAX_AGE_S = 300.0

_RENDERED_MEDIA_CACHE_LOCK = Lock()
_RENDERED_MEDIA_CACHE: OrderedDict[tuple[int, str], _ResolvedWorkingFeedbackMedia] = OrderedDict()
_RENDERED_MEDIA_CACHE_FINALIZERS: dict[int, weakref.finalize] = {}
_RENDERED_MEDIA_CACHE_MAX = 12

_PCM_CHUNK_MS = 40
_NOTE_FADE_MS = 5


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


def _coerce_positive_int(value: object, *, field_name: str) -> int:
    """Coerce a runtime value to a strictly positive integer."""
    coerced = _coerce_non_negative_int(value, field_name=field_name)
    if coerced <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return coerced


def _coerce_finite_float(
    value: object,
    *,
    field_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Coerce a runtime value to a bounded finite float."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must not be bool")
    float_like = cast(SupportsFloat | str | bytes | bytearray, value)
    coerced = float(float_like)
    if not math.isfinite(coerced):
        raise ValueError(f"{field_name} must be finite")
    if min_value is not None and coerced < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    if max_value is not None and coerced > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}")
    return coerced


def _coerce_volume(value: object, *, field_name: str) -> float:
    """Coerce a runtime value to a bounded playback volume."""
    return _coerce_finite_float(
        value,
        field_name=field_name,
        min_value=0.0,
        max_value=1.0,
    )


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


def _normalize_profile(profile: WorkingFeedbackProfile) -> WorkingFeedbackProfile:
    """Validate a working-feedback profile before worker startup."""
    max_runtime_s = profile.max_runtime_s
    normalized_max_runtime_s: float | None = None
    if max_runtime_s is not None:
        normalized_max_runtime_s = _coerce_finite_float(
            max_runtime_s,
            field_name="max_runtime_s",
            min_value=0.1,
        )
    return WorkingFeedbackProfile(
        delay_ms=_coerce_non_negative_int(profile.delay_ms, field_name="delay_ms"),
        pause_ms=_coerce_non_negative_int(profile.pause_ms, field_name="pause_ms"),
        volume=_coerce_volume(profile.volume, field_name="volume"),
        gap_ms=_coerce_non_negative_int(profile.gap_ms, field_name="gap_ms"),
        patterns=_normalize_patterns(profile.patterns, field_name="patterns"),
        max_runtime_s=normalized_max_runtime_s,
    )


def _normalize_media_spec(media_spec: WorkingFeedbackMediaSpec) -> WorkingFeedbackMediaSpec:
    """Validate one optional media spec before worker startup."""
    pause_ms = _coerce_non_negative_int(media_spec.pause_ms, field_name="pause_ms")
    attenuation_step_ms = _coerce_positive_int(
        media_spec.attenuation_step_ms,
        field_name="attenuation_step_ms",
    )
    attenuation_start_s = (
        None
        if media_spec.attenuation_start_s is None
        else _coerce_finite_float(
            media_spec.attenuation_start_s,
            field_name="attenuation_start_s",
            min_value=0.0,
        )
    )
    attenuation_reach_floor_s = (
        None
        if media_spec.attenuation_reach_floor_s is None
        else _coerce_finite_float(
            media_spec.attenuation_reach_floor_s,
            field_name="attenuation_reach_floor_s",
            min_value=0.0,
        )
    )
    minimum_output_gain = (
        None
        if media_spec.minimum_output_gain is None
        else _coerce_finite_float(
            media_spec.minimum_output_gain,
            field_name="minimum_output_gain",
            min_value=0.0,
            max_value=1.0,
        )
    )
    minimum_output_gain_ratio = (
        None
        if media_spec.minimum_output_gain_ratio is None
        else _coerce_finite_float(
            media_spec.minimum_output_gain_ratio,
            field_name="minimum_output_gain_ratio",
            min_value=0.0,
            max_value=1.0,
        )
    )
    return WorkingFeedbackMediaSpec(
        clip=media_spec.clip,
        pause_ms=pause_ms,
        attenuation_start_s=attenuation_start_s,
        attenuation_reach_floor_s=attenuation_reach_floor_s,
        minimum_output_gain=minimum_output_gain,
        minimum_output_gain_ratio=minimum_output_gain_ratio,
        attenuation_step_ms=attenuation_step_ms,
    )


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
        except Exception as exc:
            _safe_emit(
                emit,
                f"working_feedback_profile_fallback={kind}:{exc.__class__.__name__}",
            )
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


def _new_player_runtime_state() -> _PlayerRuntimeState:
    return _PlayerRuntimeState(
        playback_lock=Lock(),
        lifecycle_lock=Lock(),
        last_used_monotonic=time.monotonic(),
    )


def _try_get_attached_player_state(player) -> _PlayerRuntimeState | None:
    try:
        state = getattr(player, _PLAYER_STATE_ATTR)
    except AttributeError:
        return None
    except Exception:
        return None
    if isinstance(state, _PlayerRuntimeState):
        state.last_used_monotonic = time.monotonic()
        return state
    return None


def _try_attach_player_state(player, state: _PlayerRuntimeState) -> bool:
    try:
        setattr(player, _PLAYER_STATE_ATTR, state)
    except Exception:
        return False
    return True


def _cleanup_player_state_by_id(player_id: int) -> None:
    with _PLAYER_STATES_LOCK:
        _PLAYER_STATES_BY_ID.pop(player_id, None)
        _PLAYER_STATE_FINALIZERS.pop(player_id, None)


def _prune_last_resort_player_states(now: float) -> None:
    stale_ids = [
        player_id
        for player_id, state in _PLAYER_STATES_LAST_RESORT.items()
        if state.active_stop_event is None
        and (now - state.last_used_monotonic) >= _PLAYER_STATES_LAST_RESORT_MAX_AGE_S
    ]
    for player_id in stale_ids:
        _PLAYER_STATES_LAST_RESORT.pop(player_id, None)
    while len(_PLAYER_STATES_LAST_RESORT) > _PLAYER_STATES_LAST_RESORT_MAX:
        _PLAYER_STATES_LAST_RESORT.popitem(last=False)


def _get_player_runtime_state(player) -> _PlayerRuntimeState:
    """Return or create the shared runtime state for a player instance."""
    state = _try_get_attached_player_state(player)
    if state is not None:
        return state

    with _PLAYER_STATES_LOCK:
        try:
            state = _PLAYER_STATES.get(player)
        except TypeError:
            state = None
        else:
            if state is None:
                state = _new_player_runtime_state()
                _PLAYER_STATES[player] = state
            state.last_used_monotonic = time.monotonic()
            return state

        state = _new_player_runtime_state()
        if _try_attach_player_state(player, state):
            return state

        try:
            weakref.ref(player)
        except TypeError:
            now = time.monotonic()
            player_id = id(player)
            _prune_last_resort_player_states(now)
            state = _PLAYER_STATES_LAST_RESORT.get(player_id)
            if state is None or (
                state.active_stop_event is None
                and (now - state.last_used_monotonic) >= _PLAYER_STATES_LAST_RESORT_MAX_AGE_S
            ):
                state = _new_player_runtime_state()
                _PLAYER_STATES_LAST_RESORT[player_id] = state
            else:
                _PLAYER_STATES_LAST_RESORT.move_to_end(player_id)
                state.last_used_monotonic = now
            _prune_last_resort_player_states(now)
            return state

        player_id = id(player)
        state = _PLAYER_STATES_BY_ID.get(player_id)
        if state is None:
            state = _new_player_runtime_state()
            _PLAYER_STATES_BY_ID[player_id] = state
        else:
            state.last_used_monotonic = time.monotonic()
        finalizer = _PLAYER_STATE_FINALIZERS.get(player_id)
        if finalizer is None or not finalizer.alive:
            _PLAYER_STATE_FINALIZERS[player_id] = weakref.finalize(
                player,
                _cleanup_player_state_by_id,
                player_id,
            )
        return state


def _activate_player_loop(
    state: _PlayerRuntimeState,
    stop_event: Event,
) -> tuple[int, Event | None]:
    """Register a new active loop and return the superseded stop event."""
    with state.lifecycle_lock:
        previous_stop_event = state.active_stop_event
        state.active_stop_event = stop_event
        state.generation += 1
        state.last_used_monotonic = time.monotonic()
        generation = state.generation
    return generation, previous_stop_event


def _is_active_player_loop(
    state: _PlayerRuntimeState,
    generation: int,
    stop_event: Event,
) -> bool:
    """Report whether a worker generation still owns the player."""
    with state.lifecycle_lock:
        return state.generation == generation and state.active_stop_event is stop_event


def _release_player_loop(
    state: _PlayerRuntimeState,
    generation: int,
    stop_event: Event,
) -> None:
    """Clear the active loop marker if this worker still owns it."""
    with state.lifecycle_lock:
        if state.generation == generation and state.active_stop_event is stop_event:
            state.active_stop_event = None
        state.last_used_monotonic = time.monotonic()


def _stop_player_playback(player) -> None:
    """Stop tone playback best-effort on the configured player."""
    for method_name in ("stop_tone_sequence", "stop_tone", "stop_playback"):
        method = getattr(player, method_name, None)
        if callable(method):
            try:
                method()
                return
            except Exception:
                _LOGGER.warning(
                    "Working-feedback stop hook %s failed; trying the next fallback.",
                    method_name,
                    exc_info=True,
                )


def _player_supports_playback(player) -> bool:
    """Report whether the player exposes a supported playback API."""
    return (
        callable(getattr(player, "play_tone_sequence", None))
        or callable(getattr(player, "play_tone", None))
        or callable(getattr(player, "play_wav_chunks", None))
        or callable(getattr(player, "play_pcm16_chunks", None))
    )


def _player_supports_pcm16_chunks(player) -> bool:
    return callable(getattr(player, "play_pcm16_chunks", None))


def _player_supports_wav_chunks(player) -> bool:
    return callable(getattr(player, "play_wav_chunks", None))


def _callable_accepts_keyword(func, keyword: str) -> bool:
    """Return whether ``func`` accepts the named keyword argument."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters


def _chunk_player_supports_interruptible_pcm16(player) -> bool:
    method = getattr(player, "play_pcm16_chunks", None)
    return callable(method) and _callable_accepts_keyword(method, "should_stop")


def _resolved_chunk_player(
    *,
    player,
    playback_coordinator: PlaybackCoordinator | None,
):
    """Return the real chunk-playback backend behind the optional coordinator."""

    return playback_coordinator.player if playback_coordinator is not None else player


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
    should_stop: Callable[[], bool],
) -> None:
    """Play one rendered WAV clip through chunked, interruptible playback."""
    play_wav_chunks = getattr(player, "play_wav_chunks", None)
    if not callable(play_wav_chunks):
        raise AttributeError("player must implement play_wav_chunks()")
    try:
        play_wav_chunks(
            iter_wav_bytes_chunks(wav_bytes),
            should_stop=should_stop,
        )
    except TypeError as exc:
        if "should_stop" not in str(exc):
            raise
        play_wav_chunks(iter_wav_bytes_chunks(wav_bytes))


def _play_pcm16_clip(
    player,
    *,
    pcm_chunks: Iterable[bytes],
    sample_rate: int,
    channels: int,
    should_stop: Callable[[], bool],
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
            should_stop=should_stop,
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
    media_spec = _DEFAULT_WORKING_FEEDBACK_MEDIA_SPECS.get(kind)
    if media_spec is None:
        return None
    return _normalize_media_spec(media_spec)


def _media_uses_long_think_attenuation(media_spec: WorkingFeedbackMediaSpec) -> bool:
    return (
        media_spec.attenuation_start_s is not None
        and media_spec.attenuation_reach_floor_s is not None
        and (
            media_spec.minimum_output_gain is not None
            or media_spec.minimum_output_gain_ratio is not None
        )
        and float(media_spec.clip.output_gain) > 0.0
    )


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


def _effective_minimum_output_gain(
    media_spec: WorkingFeedbackMediaSpec,
    *,
    base_output_gain: float,
) -> float:
    ratio = media_spec.minimum_output_gain_ratio
    if ratio is not None:
        return max(0.0, min(base_output_gain, base_output_gain * float(ratio)))

    floor = media_spec.minimum_output_gain
    if floor is None:
        return base_output_gain

    floor_value = float(floor)
    if floor_value <= base_output_gain:
        return max(0.0, floor_value)

    # Backward-compatibility for the old spec shape: values in [0, 1] above the
    # base clip gain are treated as relative floor ratios instead of absolutes.
    if 0.0 <= floor_value <= 1.0:
        return max(0.0, min(base_output_gain, base_output_gain * floor_value))

    return base_output_gain


def _media_output_gain_for_elapsed_s(
    media_spec: WorkingFeedbackMediaSpec,
    *,
    elapsed_s: float,
) -> float:
    base_output_gain = max(0.0, float(media_spec.clip.output_gain))
    if not _media_uses_long_think_attenuation(media_spec):
        return base_output_gain

    floor_output_gain = _effective_minimum_output_gain(
        media_spec,
        base_output_gain=base_output_gain,
    )
    start_s = max(0.0, float(media_spec.attenuation_start_s or 0.0))
    reach_floor_s = max(start_s, float(media_spec.attenuation_reach_floor_s or start_s))
    if elapsed_s <= start_s:
        return base_output_gain
    if reach_floor_s <= start_s or elapsed_s >= reach_floor_s:
        return floor_output_gain
    progress = (elapsed_s - start_s) / max(0.001, reach_floor_s - start_s)
    return base_output_gain + ((floor_output_gain - base_output_gain) * progress)


def _scale_pcm16le_chunk(chunk: bytes, scale: float) -> bytes:
    """Scale little-endian PCM16 bytes without audioop (removed in Python 3.13)."""
    if scale <= 0.0:
        return b"\\x00" * len(chunk)
    if scale >= 0.9999:
        return chunk

    samples = array("h")
    samples.frombytes(chunk)
    if sys.byteorder != "little":
        samples.byteswap()

    scaled = array("h")
    scaled.extend(
        max(-32768, min(32767, int(round(sample * scale))))
        for sample in samples
    )

    if sys.byteorder != "little":
        scaled.byteswap()
    return scaled.tobytes()


def _iter_pcm16_bytes_chunks(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    chunk_ms: int = _PCM_CHUNK_MS,
):
    bytes_per_frame = channels * 2
    frames_per_chunk = max(1, int(sample_rate * (chunk_ms / 1000.0)))
    bytes_per_chunk = max(bytes_per_frame, frames_per_chunk * bytes_per_frame)
    for start in range(0, len(pcm_bytes), bytes_per_chunk):
        chunk = pcm_bytes[start : start + bytes_per_chunk]
        if chunk:
            yield chunk


def _pcm_chunk_duration_s(
    chunk: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> float:
    bytes_per_frame = max(1, channels * 2)
    frame_count = len(chunk) / bytes_per_frame
    return frame_count / max(1, sample_rate)


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
    step_ms = max(20, int(media.spec.attenuation_step_ms))
    bytes_per_frame = channels * 2
    frames_per_chunk = max(1, int(sample_rate * (step_ms / 1000.0)))
    bytes_per_chunk = max(bytes_per_frame, frames_per_chunk * bytes_per_frame)
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
        yield _scale_pcm16le_chunk(chunk, scale)


def _iter_repeating_attenuated_pcm16_chunks(
    media: _ResolvedWorkingFeedbackMedia,
    *,
    stop_event: Event,
):
    """Loop one rendered media clip as a single long-lived PCM stream."""

    pcm_bytes = media.pcm_bytes or b""
    sample_rate = int(media.sample_rate or 0)
    channels = int(media.channels or 0)
    if not pcm_bytes or sample_rate <= 0 or channels <= 0:
        return

    loop_elapsed_s = 0.0
    while not stop_event.is_set():
        emitted_any_chunk = False
        for chunk in _iter_attenuated_pcm16_chunks(media, loop_elapsed_s=loop_elapsed_s):
            if stop_event.is_set():
                return
            emitted_any_chunk = True
            yield chunk
            loop_elapsed_s += _pcm_chunk_duration_s(
                chunk,
                sample_rate=sample_rate,
                channels=channels,
            )
        if not emitted_any_chunk:
            return


def _iter_repeating_pcm16_clip_chunks(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    stop_event: Event,
):
    """Loop one PCM16 clip through a single interruptible playback request."""

    if not pcm_bytes or sample_rate <= 0 or channels <= 0:
        return
    while not stop_event.is_set():
        emitted_any_chunk = False
        for chunk in _iter_pcm16_bytes_chunks(
            pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
        ):
            if stop_event.is_set():
                return
            emitted_any_chunk = True
            yield chunk
        if not emitted_any_chunk:
            return


def _rendered_audio_clip_cache_key(media_spec: WorkingFeedbackMediaSpec) -> str:
    return repr(media_spec)


def _cleanup_rendered_media_cache_for_config(config_id: int) -> None:
    with _RENDERED_MEDIA_CACHE_LOCK:
        doomed_keys = [key for key in _RENDERED_MEDIA_CACHE if key[0] == config_id]
        for key in doomed_keys:
            _RENDERED_MEDIA_CACHE.pop(key, None)
        _RENDERED_MEDIA_CACHE_FINALIZERS.pop(config_id, None)


def _prune_rendered_media_cache() -> None:
    while len(_RENDERED_MEDIA_CACHE) > _RENDERED_MEDIA_CACHE_MAX:
        _RENDERED_MEDIA_CACHE.popitem(last=False)


def _resolve_media_clip(
    *,
    kind: WorkingFeedbackKind,
    config: TwinrConfig | None,
    allow_pcm16: bool,
    emit: Callable[[str], None] | None,
) -> _ResolvedWorkingFeedbackMedia | None:
    media_spec = _resolve_media_spec(kind)
    if media_spec is None or config is None:
        return None

    cache_key = (id(config), _rendered_audio_clip_cache_key(media_spec))
    with _RENDERED_MEDIA_CACHE_LOCK:
        cached = _RENDERED_MEDIA_CACHE.get(cache_key)
        if cached is not None:
            _RENDERED_MEDIA_CACHE.move_to_end(cache_key)
            if _media_uses_long_think_attenuation(media_spec) and not allow_pcm16:
                return _ResolvedWorkingFeedbackMedia(spec=media_spec, wav_bytes=cached.wav_bytes)
            return _ResolvedWorkingFeedbackMedia(
                spec=media_spec,
                wav_bytes=cached.wav_bytes,
                pcm_bytes=cached.pcm_bytes,
                sample_rate=cached.sample_rate,
                channels=cached.channels,
            )

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

    decoded = _decode_pcm16_wav_bytes(wav_bytes)
    resolved = (
        _ResolvedWorkingFeedbackMedia(
            spec=media_spec,
            wav_bytes=wav_bytes,
            pcm_bytes=decoded[0],
            sample_rate=decoded[1],
            channels=decoded[2],
        )
        if decoded is not None
        else _ResolvedWorkingFeedbackMedia(spec=media_spec, wav_bytes=wav_bytes)
    )

    with _RENDERED_MEDIA_CACHE_LOCK:
        _RENDERED_MEDIA_CACHE[cache_key] = resolved
        _RENDERED_MEDIA_CACHE.move_to_end(cache_key)
        _prune_rendered_media_cache()
        try:
            finalizer = _RENDERED_MEDIA_CACHE_FINALIZERS.get(id(config))
            if finalizer is None or not finalizer.alive:
                _RENDERED_MEDIA_CACHE_FINALIZERS[id(config)] = weakref.finalize(
                    config,
                    _cleanup_rendered_media_cache_for_config,
                    id(config),
                )
        except TypeError:
            pass

    if _media_uses_long_think_attenuation(media_spec) and decoded is None:
        _safe_emit(
            emit,
            f"working_feedback_media_attenuation_fallback={kind}:invalid_wav",
        )
    elif _media_uses_long_think_attenuation(media_spec) and not allow_pcm16:
        _safe_emit(
            emit,
            f"working_feedback_media_attenuation_fallback={kind}:missing_pcm_api",
        )
        return _ResolvedWorkingFeedbackMedia(spec=media_spec, wav_bytes=wav_bytes)

    return resolved


def _pcm16_bytes_to_wav_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> bytes:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as raw_writer:
            writer = cast(wave.Wave_write, raw_writer)
            # Pylint resolves ``wave.open(..., "wb")`` as ``Wave_read`` here even
            # though the runtime object is a writer for ``wb`` mode.
            # pylint: disable=no-member
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(pcm_bytes)
        return buffer.getvalue()


def _quantized_amplitude(volume: float) -> int:
    return max(0, min(32767, int(round(volume * 32767.0))))


@lru_cache(maxsize=512)
def _render_note_pcm16_bytes(
    *,
    frequency_hz: int,
    duration_ms: int,
    sample_rate: int,
    amplitude: int,
) -> bytes:
    total_frames = max(1, int(round(sample_rate * (duration_ms / 1000.0))))
    fade_frames = min(
        max(1, int(round(sample_rate * (_NOTE_FADE_MS / 1000.0)))),
        max(1, total_frames // 4),
    )
    note = array("h")
    for frame_index in range(total_frames):
        envelope = 1.0
        if frame_index < fade_frames:
            envelope = frame_index / fade_frames
        elif frame_index >= total_frames - fade_frames:
            envelope = (total_frames - frame_index - 1) / fade_frames
        envelope = max(0.0, min(1.0, envelope))
        sample = int(
            round(
                amplitude
                * envelope
                * math.sin((2.0 * math.pi * frequency_hz * frame_index) / sample_rate)
            )
        )
        note.append(max(-32768, min(32767, sample)))
    if sys.byteorder != "little":
        note.byteswap()
    return note.tobytes()


@lru_cache(maxsize=128)
def _render_tone_sequence_cached(
    sequence: tuple[tuple[int, int], ...],
    *,
    amplitude: int,
    sample_rate: int,
    gap_ms: int,
) -> tuple[bytes, bytes]:
    frames = array("h")
    silence_frames = max(0, int(round(sample_rate * (gap_ms / 1000.0))))
    for note_index, (frequency_hz, duration_ms) in enumerate(sequence):
        note_bytes = _render_note_pcm16_bytes(
            frequency_hz=frequency_hz,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            amplitude=amplitude,
        )
        note = array("h")
        note.frombytes(note_bytes)
        if sys.byteorder != "little":
            note.byteswap()
        frames.extend(note)
        if silence_frames > 0 and note_index + 1 < len(sequence):
            frames.extend(0 for _ in range(silence_frames))

    pcm_little_endian = array("h", frames)
    if sys.byteorder != "little":
        pcm_little_endian.byteswap()
    pcm_bytes = pcm_little_endian.tobytes()
    return (
        pcm_bytes,
        _pcm16_bytes_to_wav_bytes(
            pcm_bytes,
            sample_rate=sample_rate,
            channels=1,
        ),
    )


def _render_tone_sequence(
    sequence: tuple[tuple[int, int], ...],
    *,
    volume: float,
    sample_rate: int,
    gap_ms: int,
) -> _RenderedToneSequence:
    amplitude = _quantized_amplitude(volume)
    pcm_bytes, wav_bytes = _render_tone_sequence_cached(
        sequence,
        amplitude=amplitude,
        sample_rate=sample_rate,
        gap_ms=gap_ms,
    )
    return _RenderedToneSequence(
        sequence=sequence,
        pcm_bytes=pcm_bytes,
        wav_bytes=wav_bytes,
        sample_rate=sample_rate,
        channels=1,
    )


def _resolve_tone_renders(
    profile: WorkingFeedbackProfile,
    *,
    sample_rate: int,
) -> tuple[_RenderedToneSequence, ...]:
    return tuple(
        _render_tone_sequence(
            sequence,
            volume=profile.volume,
            sample_rate=sample_rate,
            gap_ms=profile.gap_ms,
        )
        for sequence in profile.patterns
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
    """Start a bounded feedback loop and return its stop callback."""
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

    if playback_coordinator is None and not _player_supports_playback(player):
        _safe_emit(emit, f"working_feedback_disabled={kind}:missing_player_api")
        return lambda: None

    profile = _resolve_profile(
        kind,
        profiles if profiles is not None else None,
        delay_override_ms,
        emit,
    )
    resolved_media = _resolve_media_clip(
        kind=kind,
        config=config,
        allow_pcm16=(playback_coordinator is not None or _player_supports_pcm16_chunks(player)),
        emit=emit,
    )
    resolved_tones: tuple[_RenderedToneSequence, ...] = ()
    if (
        playback_coordinator is not None
        or _player_supports_pcm16_chunks(player)
        or _player_supports_wav_chunks(player)
    ):
        resolved_tones = _resolve_tone_renders(
            profile,
            sample_rate=normalized_sample_rate,
        )

    stop_event = Event()
    player_state = _get_player_runtime_state(player)
    generation, previous_stop_event = _activate_player_loop(player_state, stop_event)
    if previous_stop_event is not None and previous_stop_event is not stop_event:
        previous_stop_event.set()

    owner = f"working_feedback:{kind}:{id(player)}:{generation}"

    def wait_for_pause() -> bool:
        pause_ms = resolved_media.spec.pause_ms if resolved_media is not None else profile.pause_ms
        if pause_ms <= 0:
            return stop_event.is_set()
        return stop_event.wait(pause_ms / 1000.0)

    def should_stop_continuous_playback(
        *,
        feedback_started_at: float,
        timed_out: Event,
    ) -> bool:
        if stop_event.is_set():
            return True
        if profile.max_runtime_s is None:
            return False
        if max(0.0, time.monotonic() - feedback_started_at) < profile.max_runtime_s:
            return False
        timed_out.set()
        stop_event.set()
        return True

    def worker() -> None:
        try:
            if stop_event.wait(profile.delay_ms / 1000.0):
                return
            feedback_started_at = time.monotonic()
            pattern_index = 0
            chunk_player = _resolved_chunk_player(
                player=player,
                playback_coordinator=playback_coordinator,
            )
            continuous_pcm_supported = _chunk_player_supports_interruptible_pcm16(chunk_player)

            if (
                resolved_media is not None
                and resolved_media.pcm_bytes is not None
                and resolved_media.sample_rate is not None
                and resolved_media.channels is not None
                and continuous_pcm_supported
            ):
                timed_out = Event()

                def _should_stop_media_playback() -> bool:
                    return should_stop_continuous_playback(
                        feedback_started_at=feedback_started_at,
                        timed_out=timed_out,
                    )

                pcm_chunks = _iter_repeating_attenuated_pcm16_chunks(
                    resolved_media,
                    stop_event=stop_event,
                )
                try:
                    if playback_coordinator is None:
                        _play_pcm16_clip(
                            player,
                            pcm_chunks=pcm_chunks,
                            sample_rate=resolved_media.sample_rate,
                            channels=resolved_media.channels,
                            should_stop=_should_stop_media_playback,
                        )
                    else:
                        playback_coordinator.play_pcm16_chunks(
                            owner=owner,
                            priority=PlaybackPriority.FEEDBACK,
                            chunks=pcm_chunks,
                            sample_rate=resolved_media.sample_rate,
                            channels=resolved_media.channels,
                            should_stop=_should_stop_media_playback,
                        )
                finally:
                    if timed_out.is_set():
                        _safe_emit(emit, f"working_feedback_timeout={kind}")
                return

            if resolved_tones and continuous_pcm_supported:
                rendered_tone = resolved_tones[0]
                timed_out = Event()

                def _should_stop_tone_playback() -> bool:
                    return should_stop_continuous_playback(
                        feedback_started_at=feedback_started_at,
                        timed_out=timed_out,
                    )

                tone_chunks = _iter_repeating_pcm16_clip_chunks(
                    rendered_tone.pcm_bytes,
                    sample_rate=rendered_tone.sample_rate,
                    channels=rendered_tone.channels,
                    stop_event=stop_event,
                )
                try:
                    if playback_coordinator is None:
                        _play_pcm16_clip(
                            player,
                            pcm_chunks=tone_chunks,
                            sample_rate=rendered_tone.sample_rate,
                            channels=rendered_tone.channels,
                            should_stop=_should_stop_tone_playback,
                        )
                    else:
                        playback_coordinator.play_pcm16_chunks(
                            owner=owner,
                            priority=PlaybackPriority.FEEDBACK,
                            chunks=tone_chunks,
                            sample_rate=rendered_tone.sample_rate,
                            channels=rendered_tone.channels,
                            should_stop=_should_stop_tone_playback,
                        )
                finally:
                    if timed_out.is_set():
                        _safe_emit(emit, f"working_feedback_timeout={kind}")
                return

            while not stop_event.is_set():
                if not _is_active_player_loop(player_state, generation, stop_event):
                    return

                loop_elapsed_s = max(0.0, time.monotonic() - feedback_started_at)
                if profile.max_runtime_s is not None and loop_elapsed_s >= profile.max_runtime_s:
                    _safe_emit(emit, f"working_feedback_timeout={kind}")
                    stop_event.set()
                    if playback_coordinator is not None:
                        playback_coordinator.stop_owner(owner)
                    else:
                        _stop_player_playback(player)
                    return

                sequence = profile.patterns[pattern_index % len(profile.patterns)]
                rendered_tone = (
                    resolved_tones[pattern_index % len(resolved_tones)]
                    if resolved_tones
                    else None
                )
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
                                should_stop=stop_event.is_set,
                            )
                        else:
                            playback_coordinator.play_pcm16_chunks(
                                owner=owner,
                                priority=PlaybackPriority.FEEDBACK,
                                chunks=pcm_chunks,
                                sample_rate=resolved_media.sample_rate,
                                channels=resolved_media.channels,
                                should_stop=stop_event.is_set,
                            )
                    elif resolved_media is not None and (
                        playback_coordinator is not None or _player_supports_wav_chunks(player)
                    ):
                        if playback_coordinator is None:
                            _play_wav_clip(
                                player,
                                resolved_media.wav_bytes,
                                should_stop=stop_event.is_set,
                            )
                        else:
                            playback_coordinator.play_wav_chunks(
                                owner=owner,
                                priority=PlaybackPriority.FEEDBACK,
                                chunks=iter_wav_bytes_chunks(resolved_media.wav_bytes),
                                should_stop=stop_event.is_set,
                            )
                    elif rendered_tone is not None and (
                        playback_coordinator is not None or _player_supports_pcm16_chunks(player)
                    ):
                        tone_chunks = _iter_pcm16_bytes_chunks(
                            rendered_tone.pcm_bytes,
                            sample_rate=rendered_tone.sample_rate,
                            channels=rendered_tone.channels,
                        )
                        if playback_coordinator is None:
                            _play_pcm16_clip(
                                player,
                                pcm_chunks=tone_chunks,
                                sample_rate=rendered_tone.sample_rate,
                                channels=rendered_tone.channels,
                                should_stop=stop_event.is_set,
                            )
                        else:
                            playback_coordinator.play_pcm16_chunks(
                                owner=owner,
                                priority=PlaybackPriority.FEEDBACK,
                                chunks=tone_chunks,
                                sample_rate=rendered_tone.sample_rate,
                                channels=rendered_tone.channels,
                                should_stop=stop_event.is_set,
                            )
                    elif rendered_tone is not None and (
                        playback_coordinator is not None or _player_supports_wav_chunks(player)
                    ):
                        if playback_coordinator is None:
                            _play_wav_clip(
                                player,
                                rendered_tone.wav_bytes,
                                should_stop=stop_event.is_set,
                            )
                        else:
                            playback_coordinator.play_wav_chunks(
                                owner=owner,
                                priority=PlaybackPriority.FEEDBACK,
                                chunks=iter_wav_bytes_chunks(rendered_tone.wav_bytes),
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
                            owner=owner,
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

    thread = Thread(
        target=worker,
        name=f"working-feedback-{kind}",
        daemon=True,
    )
    thread.start()

    stop_called = Event()

    def stop() -> None:
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
