"""Play bounded audio cues while Twinr is processing, answering, or printing."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import replace
import logging
from threading import Event, Lock, Thread, current_thread
from typing import Literal
from weakref import WeakKeyDictionary

from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator, PlaybackPriority

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


_DEFAULT_WORKING_FEEDBACK_PROFILES: dict[WorkingFeedbackKind, WorkingFeedbackProfile] = {
    "processing": WorkingFeedbackProfile(
        delay_ms=450,
        pause_ms=620,
        volume=0.12,
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
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{field_name} must be an integer value")
    coerced = int(value)
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
    coerced = float(value)
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
    normalized_patterns: list[tuple[tuple[int, int], ...]] = []
    for pattern_index, sequence in enumerate(patterns):
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
    return callable(getattr(player, "play_tone_sequence", None)) or callable(
        getattr(player, "play_tone", None)
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


def start_working_feedback_loop(
    player,
    *,
    kind: WorkingFeedbackKind,
    sample_rate: int,
    emit: Callable[[str], None] | None = None,
    profiles: Mapping[WorkingFeedbackKind, WorkingFeedbackProfile] | None = None,
    delay_override_ms: int | None = None,
    playback_coordinator: PlaybackCoordinator | None = None,
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
    stop_event = Event()
    player_state = _get_player_runtime_state(player)
    generation, previous_stop_event = _activate_player_loop(player_state, stop_event)
    if previous_stop_event is not None and previous_stop_event is not stop_event:
        previous_stop_event.set()

    def worker() -> None:
        try:
            if stop_event.wait(profile.delay_ms / 1000.0):
                return
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
                    if playback_coordinator is None:
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
                    # AUDIT-FIX(#5): Never emit raw exception payloads from the audio backend.
                    _safe_emit(
                        emit,
                        f"working_feedback_error={kind}:{exc.__class__.__name__}",
                    )
                    return
                finally:
                    if acquired:
                        player_state.playback_lock.release()

                if stop_event.wait(max(0.05, profile.pause_ms / 1000.0)):
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
