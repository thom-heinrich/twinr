"""Runtime-side voice-orchestrator state and same-stream transcript helpers."""

from __future__ import annotations

import time
from typing import Any

from twinr.agent.workflows.remote_transcript_commit import (
    RemoteTranscriptCommit,
    RemoteTranscriptWaitHandle,
)
from twinr.agent.workflows.voice_identity_runtime import (
    sync_voice_orchestrator_identity_profiles,
)
from twinr.orchestrator.voice_runtime_intent import VoiceRuntimeIntentContext


def _voice_quiet_until_utc(loop: Any) -> str | None:
    """Return the current temporary voice-quiet deadline, if active."""

    runtime = getattr(loop, "runtime", None)
    getter = getattr(runtime, "voice_quiet_until_utc", None)
    if not callable(getter):
        return None
    value = getter()
    return str(value or "").strip() or None


def _effective_follow_up_allowed(loop: Any, *, follow_up_allowed: bool) -> bool:
    """Disable automatic follow-up while temporary quiet mode is active."""

    if not follow_up_allowed:
        return False
    return _voice_quiet_until_utc(loop) is None


def voice_orchestrator_handles_follow_up(loop: Any, *, initial_source: str) -> bool:
    """Report whether the current voice path reopens follow-up remotely."""

    if loop.voice_orchestrator is None:
        return False
    return voice_orchestrator_follow_up_mode(loop, initial_source=initial_source) == "remote"


def voice_orchestrator_follow_up_mode(loop: Any, *, initial_source: str) -> str:
    """Return how Twinr should reopen continuation after a finished answer."""

    if loop.voice_orchestrator is None:
        return "disabled"
    if not loop._follow_up_allowed_for_source(initial_source=initial_source):
        return "disabled"
    return "remote"


def notify_voice_orchestrator_state(
    loop: Any,
    state: str,
    *,
    detail: str | None = None,
    follow_up_allowed: bool = False,
) -> None:
    """Push one runtime-state snapshot plus compact intent context to the bridge."""

    if loop.voice_orchestrator is None:
        return
    intent_context = VoiceRuntimeIntentContext.from_sensor_facts(
        getattr(loop, "_latest_sensor_observation_facts", None)
    )
    quiet_until_utc = _voice_quiet_until_utc(loop)
    try:
        with loop._voice_orchestrator_runtime_state_lock:
            loop._last_voice_orchestrator_runtime_state = (state, detail, follow_up_allowed)
            loop._last_voice_orchestrator_intent_context = intent_context
            loop._last_voice_orchestrator_quiet_until_utc = quiet_until_utc
            loop.voice_orchestrator.notify_runtime_state(
                state=state,
                detail=detail,
                follow_up_allowed=_effective_follow_up_allowed(loop, follow_up_allowed=follow_up_allowed),
                voice_quiet_until_utc=quiet_until_utc,
                **intent_context.to_event_fields(),
            )
        sync_voice_orchestrator_identity_profiles(loop)
    except Exception as exc:
        loop.emit(f"voice_orchestrator_notify_failed={type(exc).__name__}")


def prime_voice_orchestrator_waiting_state(loop: Any) -> None:
    """Seed one explicit idle state so later sensor refreshes can replay it."""

    if loop.voice_orchestrator is None:
        return
    if loop._last_voice_orchestrator_runtime_state is not None:
        return
    sensor_facts = getattr(loop, "_latest_sensor_observation_facts", None)
    intent_context = (
        VoiceRuntimeIntentContext.from_sensor_facts(sensor_facts)
        if isinstance(sensor_facts, dict) and sensor_facts
        else None
    )
    with loop._voice_orchestrator_runtime_state_lock:
        loop._last_voice_orchestrator_runtime_state = ("waiting", None, False)
        loop._last_voice_orchestrator_intent_context = intent_context
        loop._last_voice_orchestrator_quiet_until_utc = _voice_quiet_until_utc(loop)
        if intent_context is None:
            loop._trace_event(
                "voice_orchestrator_waiting_state_seed_deferred",
                kind="branch",
                details={"reason": "missing_sensor_context"},
            )
            return
        seed_runtime_state = getattr(loop.voice_orchestrator, "seed_runtime_state", None)
        if callable(seed_runtime_state):
            seed_runtime_state(
                state="waiting",
                detail=None,
                follow_up_allowed=False,
                voice_quiet_until_utc=loop._last_voice_orchestrator_quiet_until_utc,
                **intent_context.to_event_fields(),
            )
        else:
            loop.voice_orchestrator.notify_runtime_state(
                state="waiting",
                detail=None,
                follow_up_allowed=False,
                voice_quiet_until_utc=loop._last_voice_orchestrator_quiet_until_utc,
                **intent_context.to_event_fields(),
            )
        sync_voice_orchestrator_identity_profiles(loop, force=True)


def refresh_voice_orchestrator_sensor_context(loop: Any) -> None:
    """Push changed multimodal intent context into the live voice gateway."""

    if loop.voice_orchestrator is None:
        return
    intent_context = VoiceRuntimeIntentContext.from_sensor_facts(
        getattr(loop, "_latest_sensor_observation_facts", None)
    )
    quiet_until_utc = _voice_quiet_until_utc(loop)
    try:
        with loop._voice_orchestrator_runtime_state_lock:
            last_state = loop._last_voice_orchestrator_runtime_state
            if last_state is None:
                return
            if (
                intent_context == loop._last_voice_orchestrator_intent_context
                and quiet_until_utc == getattr(loop, "_last_voice_orchestrator_quiet_until_utc", None)
            ):
                return
            state, detail, follow_up_allowed = last_state
            loop._last_voice_orchestrator_intent_context = intent_context
            loop._last_voice_orchestrator_quiet_until_utc = quiet_until_utc
            loop.voice_orchestrator.notify_runtime_state(
                state=state,
                detail=detail,
                follow_up_allowed=_effective_follow_up_allowed(loop, follow_up_allowed=follow_up_allowed),
                voice_quiet_until_utc=quiet_until_utc,
                **intent_context.to_event_fields(),
            )
        sync_voice_orchestrator_identity_profiles(loop)
    except Exception as exc:
        loop.emit(f"voice_orchestrator_context_refresh_failed={type(exc).__name__}")


def handle_remote_transcript_committed(loop: Any, transcript: str, source: str) -> bool:
    """Consume one transcript committed by the live remote voice stream."""

    try:
        committed_transcript = transcript.strip()
        committed_source = str(source or "").strip().lower() or "listening"
        if not committed_transcript:
            loop.emit("voice_orchestrator_transcript_ignored=empty")
            return False
        if loop._remote_transcript_commits.commit(
            source=committed_source,
            transcript=committed_transcript,
        ):
            loop.emit(f"voice_orchestrator_transcript_delivered={committed_source}")
            return True
        if committed_source != "follow_up":
            loop.emit(f"voice_orchestrator_transcript_ignored={committed_source}")
            return False
        if not loop._follow_up_allowed_for_source(initial_source="follow_up"):
            loop.emit("voice_orchestrator_follow_up_skipped=disabled")
            return False
        cached_state = getattr(loop, "_last_voice_orchestrator_runtime_state", None)
        follow_up_open = (
            not isinstance(cached_state, tuple)
            or len(cached_state) < 3
            or (
                cached_state[0] == "follow_up_open"
                and bool(cached_state[2])
            )
        )
        if not follow_up_open:
            loop.emit("voice_orchestrator_follow_up_skipped=closed")
            return False
        if not loop._required_remote_dependency_current_ready():
            loop._request_required_remote_dependency_refresh()
            return False
        loop.emit("voice_orchestrator_follow_up=true")
        loop._trace_event(
            "voice_orchestrator_follow_up_transcript_committed",
            kind="decision",
            details={
                "runtime_status": loop.runtime.status.value,
                "transcript_chars": len(committed_transcript),
            },
        )
        return loop._run_conversation_session(
            initial_source="follow_up",
            seed_transcript=committed_transcript,
            play_initial_beep=False,
        )
    except Exception as exc:
        loop._handle_error(exc)
        return False


def handle_remote_follow_up_closed(loop: Any, reason: str) -> None:
    """Return the local runtime to waiting when a remote follow-up window closes.

    The server owns the transcript-first follow-up timeout window, but the Pi
    still re-arms its local runtime snapshot to ``listening`` so display and
    operator cues stay in sync. When the server later emits
    ``follow_up_closed``, the edge must collapse that local follow-up state back
    to ``waiting``. Otherwise the runtime snapshot remains stuck in
    ``listening`` until the supervisor recycles the streaming loop for
    ``runtime_snapshot_stale``.
    """

    try:
        cached_state = getattr(loop, "_last_voice_orchestrator_runtime_state", None)
        cached_mode = cached_state[0] if isinstance(cached_state, tuple) and cached_state else None
        if cached_mode != "follow_up_open":
            loop.emit("voice_orchestrator_follow_up_closed_ignored=stale_state")
            return
        if bool(getattr(loop, "_conversation_session_active", False)):
            loop.emit("voice_orchestrator_follow_up_closed_ignored=session_active")
            return
        runtime_status = getattr(getattr(loop, "runtime", None), "status", None)
        if getattr(runtime_status, "value", None) != "listening":
            loop.emit("voice_orchestrator_follow_up_closed_ignored=runtime_not_listening")
            loop._notify_voice_orchestrator_state("waiting", detail=reason)
            return
        loop.runtime.cancel_listening()
        loop._emit_status(force=True)
        loop._notify_voice_orchestrator_state("waiting", detail=reason)
        loop.emit(f"voice_orchestrator_follow_up_closed_local_waiting={reason}")
        loop._trace_event(
            "voice_orchestrator_follow_up_closed_local_waiting",
            kind="mutation",
            details={"reason": reason},
        )
    except Exception as exc:
        loop._handle_error(exc)


def voice_orchestrator_owns_live_listening(loop: Any) -> bool:
    """Return whether the current voice path is server-owned after wake."""

    return loop.voice_orchestrator is not None


def pause_voice_orchestrator_capture(loop: Any, *, reason: str) -> None:
    """Delegate legacy capture pauses to the edge voice bridge when present."""

    if loop.voice_orchestrator is None:
        return
    loop.voice_orchestrator.pause_capture(reason=reason)


def resume_voice_orchestrator_capture(loop: Any, *, reason: str) -> None:
    """Delegate legacy capture resumes to the edge voice bridge when present."""

    if loop.voice_orchestrator is None:
        return
    loop.voice_orchestrator.resume_capture(reason=reason)


def begin_remote_transcript_wait(loop: Any, *, source: str) -> RemoteTranscriptWaitHandle | None:
    """Open one bounded wait for a remote transcript commit."""

    try:
        return loop._remote_transcript_commits.begin_wait(source=source)
    except RuntimeError as exc:
        loop.emit(f"voice_orchestrator_transcript_wait_failed={type(exc).__name__}")
        loop._trace_event(
            "voice_orchestrator_transcript_wait_failed",
            kind="error",
            level="ERROR",
            details={"source": source, "error": str(exc)},
        )
        return None


def wait_for_remote_transcript_commit(
    loop: Any,
    *,
    wait_handle: RemoteTranscriptWaitHandle,
    timeout_s: float,
    initial_source: str,
    follow_up: bool,
    listen_source: str,
    timeout_emit_key: str,
    timeout_message: str,
) -> RemoteTranscriptCommit | None:
    """Wait until the remote stream commits one transcript or times out."""

    deadline_at = time.monotonic() + max(0.1, float(timeout_s))
    while True:
        if loop._active_turn_stop_requested():
            loop._cancel_interrupted_turn()
            return None
        if wait_handle.commit is not None:
            return wait_handle.commit
        if wait_handle.close is not None:
            loop.runtime.remember_listen_timeout(
                initial_source=initial_source,
                follow_up=follow_up,
            )
            loop.runtime.cancel_listening()
            loop._emit_status(force=True)
            loop._notify_voice_orchestrator_state("waiting", detail=wait_handle.close.reason)
            loop.emit(f"{timeout_emit_key}=true")
            loop.emit(f"voice_orchestrator_transcript_closed={wait_handle.close.reason}")
            loop._record_event(
                "listen_timeout",
                timeout_message,
                request_source=listen_source,
                reason=wait_handle.close.reason,
            )
            return None
        remaining_s = deadline_at - time.monotonic()
        if remaining_s <= 0:
            loop.runtime.remember_listen_timeout(
                initial_source=initial_source,
                follow_up=follow_up,
            )
            loop.runtime.cancel_listening()
            loop._emit_status(force=True)
            loop._notify_voice_orchestrator_state("waiting", detail="listen_timeout")
            loop.emit(f"{timeout_emit_key}=true")
            loop._record_event("listen_timeout", timeout_message, request_source=listen_source)
            return None
        wait_handle.wait(timeout_s=min(0.05, remaining_s))
