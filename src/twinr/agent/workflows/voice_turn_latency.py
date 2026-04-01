# CHANGELOG: 2026-03-30
# IMP-1: Track per-turn voice milestones across wake, commit, supervisor, remote-memory, and
#        first-audio TTS so Pi journal timings show where live latency accumulated.
# IMP-2: Keep the timing state in one focused helper instead of growing the large voice/runtime
#        orchestration files with more cross-cutting bookkeeping.

"""Track live voice-turn latency milestones for Pi journal diagnostics.

The transcript-first voice path crosses several modules before Twinr speaks:
the edge wake/commit bridge, session startup, streaming lane planning,
remote-memory context assembly, and finally streamed TTS. This helper keeps the
cross-cutting timing state out of those orchestration files while still letting
them stamp one shared per-turn breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
import time
from typing import Callable

from twinr.agent.workflows.forensics import current_workflow_trace_id

_PENDING_ATTR = "_voice_turn_latency_pending"


@dataclass(slots=True)
class _VoiceTurnLatencyState:
    """Capture one in-flight voice turn across wake, memory, and TTS milestones."""

    source: str
    created_at: float
    initial_source: str | None = None
    trace_id: str | None = None
    wake_confirmed_at: float | None = None
    transcript_committed_at: float | None = None
    supervisor_ready_at: float | None = None
    remote_memory_ready_at: float | None = None
    tts_first_audio_at: float | None = None
    remote_memory_reads: int = 0
    remote_memory_total_ms: int = 0


@dataclass(frozen=True, slots=True)
class VoiceTurnLatencySnapshot:
    """Frozen per-turn timing breakdown ready for journal emission."""

    trace_id: str | None
    source: str
    initial_source: str | None
    wake_to_commit_ms: int | None
    commit_to_supervisor_ms: int | None
    supervisor_to_remote_memory_ms: int | None
    remote_memory_to_tts_ms: int | None
    remote_memory_reads: int
    remote_memory_total_ms: int
    missing_stages: tuple[str, ...]


class _VoiceTurnLatencyRegistry:
    """Bind active workflow traces to their voice-turn latency state."""

    def __init__(self) -> None:
        self._states_by_trace_id: dict[str, _VoiceTurnLatencyState] = {}
        self._lock = RLock()

    def bind(
        self,
        *,
        trace_id: str,
        initial_source: str,
        state: _VoiceTurnLatencyState,
    ) -> None:
        with self._lock:
            state.trace_id = trace_id
            state.initial_source = initial_source
            self._states_by_trace_id[trace_id] = state

    def get(self, trace_id: str | None) -> _VoiceTurnLatencyState | None:
        if not trace_id:
            return None
        with self._lock:
            return self._states_by_trace_id.get(trace_id)

    def pop(self, trace_id: str | None) -> _VoiceTurnLatencyState | None:
        if not trace_id:
            return None
        with self._lock:
            return self._states_by_trace_id.pop(trace_id, None)


_REGISTRY = _VoiceTurnLatencyRegistry()


def _now() -> float:
    """Return the monotonic timestamp used for all stage math."""

    return time.monotonic()


def _duration_ms(started_at: float | None, finished_at: float | None) -> int | None:
    """Return one bounded integer duration when both endpoints exist."""

    if started_at is None or finished_at is None:
        return None
    return max(0, int(round((finished_at - started_at) * 1000.0)))


def _current_state(trace_id: str | None = None) -> _VoiceTurnLatencyState | None:
    """Return the traced state for one explicit or current workflow trace."""

    resolved_trace_id = trace_id or current_workflow_trace_id()
    return _REGISTRY.get(resolved_trace_id)


def _pending_state(loop: object) -> _VoiceTurnLatencyState | None:
    state = getattr(loop, _PENDING_ATTR, None)
    return state if isinstance(state, _VoiceTurnLatencyState) else None


def _set_pending_state(loop: object, state: _VoiceTurnLatencyState | None) -> None:
    setattr(loop, _PENDING_ATTR, state)


def mark_voice_turn_wake_confirmed(loop: object, *, source: str) -> None:
    """Start one fresh pending voice turn when wake confirmation is accepted."""

    state = _VoiceTurnLatencyState(source=str(source or "voice"), created_at=_now())
    state.wake_confirmed_at = state.created_at
    _set_pending_state(loop, state)


def mark_voice_turn_commit(loop: object, *, source: str) -> None:
    """Stamp transcript commit on the pending or active traced voice turn."""

    state = _current_state()
    if state is None:
        state = _pending_state(loop)
        if state is None:
            state = _VoiceTurnLatencyState(source=str(source or "voice"), created_at=_now())
            _set_pending_state(loop, state)
    if not state.source:
        state.source = str(source or "voice")
    if state.transcript_committed_at is None:
        state.transcript_committed_at = _now()


def bind_voice_turn_trace(loop: object, *, trace_id: str, initial_source: str) -> None:
    """Attach the current pending voice turn to the new workflow trace."""

    state = _pending_state(loop)
    if state is None:
        return
    _REGISTRY.bind(trace_id=trace_id, initial_source=initial_source, state=state)
    _set_pending_state(loop, None)


def clear_voice_turn_latency(loop: object, *, trace_id: str | None = None) -> None:
    """Drop pending/traced voice latency state when a turn aborts before speaking."""

    _set_pending_state(loop, None)
    resolved_trace_id = trace_id or current_workflow_trace_id()
    if resolved_trace_id:
        _REGISTRY.pop(resolved_trace_id)


def mark_voice_turn_supervisor_ready(*, trace_id: str | None = None) -> None:
    """Stamp the point where the supervisor or lane-plan stage finished."""

    state = _current_state(trace_id)
    if state is None or state.supervisor_ready_at is not None:
        return
    state.supervisor_ready_at = _now()


def record_voice_turn_remote_memory_ready(
    *,
    duration_ms: float,
    trace_id: str | None = None,
) -> None:
    """Aggregate one remote-memory stage completion for the active voice trace."""

    state = _current_state(trace_id)
    if state is None:
        return
    state.remote_memory_reads += 1
    state.remote_memory_total_ms += max(0, int(round(float(duration_ms))))
    state.remote_memory_ready_at = _now()


def mark_voice_turn_tts_started(*, trace_id: str | None = None) -> None:
    """Stamp the first audible TTS point for the active voice trace."""

    state = _current_state(trace_id)
    if state is None or state.tts_first_audio_at is not None:
        return
    state.tts_first_audio_at = _now()


def snapshot_voice_turn_latency(*, trace_id: str | None = None) -> VoiceTurnLatencySnapshot | None:
    """Return the current immutable breakdown for one voice trace."""

    state = _current_state(trace_id)
    if state is None:
        return None
    wake_to_commit_ms = _duration_ms(state.wake_confirmed_at, state.transcript_committed_at)
    commit_to_supervisor_ms = _duration_ms(state.transcript_committed_at, state.supervisor_ready_at)
    supervisor_to_remote_memory_ms = _duration_ms(
        state.supervisor_ready_at,
        state.remote_memory_ready_at,
    )
    remote_memory_to_tts_ms = _duration_ms(
        state.remote_memory_ready_at,
        state.tts_first_audio_at,
    )
    missing_stages: list[str] = []
    if wake_to_commit_ms is None:
        missing_stages.append("wake_to_commit")
    if commit_to_supervisor_ms is None:
        missing_stages.append("commit_to_supervisor")
    if supervisor_to_remote_memory_ms is None:
        missing_stages.append("supervisor_to_remote_memory")
    if remote_memory_to_tts_ms is None:
        missing_stages.append("remote_memory_to_tts")
    return VoiceTurnLatencySnapshot(
        trace_id=state.trace_id,
        source=state.source,
        initial_source=state.initial_source,
        wake_to_commit_ms=wake_to_commit_ms,
        commit_to_supervisor_ms=commit_to_supervisor_ms,
        supervisor_to_remote_memory_ms=supervisor_to_remote_memory_ms,
        remote_memory_to_tts_ms=remote_memory_to_tts_ms,
        remote_memory_reads=state.remote_memory_reads,
        remote_memory_total_ms=state.remote_memory_total_ms,
        missing_stages=tuple(missing_stages),
    )


def emit_voice_turn_latency_breakdown(
    *,
    emit: Callable[[str], None],
    trace_event: Callable[..., None] | None = None,
    trace_id: str | None = None,
) -> None:
    """Emit stable key=value timing lines for the completed voice trace."""

    snapshot = snapshot_voice_turn_latency(trace_id=trace_id)
    if snapshot is None:
        return
    if snapshot.wake_to_commit_ms is not None:
        emit(f"timing_wake_to_commit_ms={snapshot.wake_to_commit_ms}")
    if snapshot.commit_to_supervisor_ms is not None:
        emit(f"timing_commit_to_supervisor_ms={snapshot.commit_to_supervisor_ms}")
    if snapshot.supervisor_to_remote_memory_ms is not None:
        emit(f"timing_supervisor_to_remote_memory_ms={snapshot.supervisor_to_remote_memory_ms}")
    if snapshot.remote_memory_to_tts_ms is not None:
        emit(f"timing_remote_memory_to_tts_ms={snapshot.remote_memory_to_tts_ms}")
    if snapshot.remote_memory_reads > 0:
        emit(f"timing_remote_memory_reads={snapshot.remote_memory_reads}")
        emit(f"timing_remote_memory_total_ms={snapshot.remote_memory_total_ms}")
    if callable(trace_event):
        trace_event(
            "voice_turn_latency_breakdown",
            kind="metric",
            details={
                "source": snapshot.source,
                "initial_source": snapshot.initial_source,
                "wake_to_commit_ms": snapshot.wake_to_commit_ms,
                "commit_to_supervisor_ms": snapshot.commit_to_supervisor_ms,
                "supervisor_to_remote_memory_ms": snapshot.supervisor_to_remote_memory_ms,
                "remote_memory_to_tts_ms": snapshot.remote_memory_to_tts_ms,
                "remote_memory_reads": snapshot.remote_memory_reads,
                "remote_memory_total_ms": snapshot.remote_memory_total_ms,
                "missing_stages": list(snapshot.missing_stages),
            },
        )
    _REGISTRY.pop(snapshot.trace_id or trace_id)
