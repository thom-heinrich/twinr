# CHANGELOG: 2026-03-28
# BUG-1: Fixed inconsistent follow-up gating. Temporary voice-quiet mode and stale/malformed cached state
#        could still allow remote follow-up reopen/session start.
# BUG-2: Fixed timeout accounting. Non-timeout remote closes were previously recorded as listen timeouts.
# BUG-3: Fixed state-sync desync. Runtime-state cache was updated before successful bridge delivery, so a
#        transient notify failure could leave the local cache "sent" while the remote bridge never saw it.
# BUG-4: Fixed possible bridge deadlock / long critical sections by moving bridge I/O outside the runtime-state
#        lock while preserving send order with a dedicated transport lock.
# SEC-1: Sanitized untrusted remote text used in emits/details to reduce practical log-injection risk.
# SEC-2: Added bounded remote transcript normalization to reduce transcript-flood DoS risk on Raspberry Pi 4.
# IMP-1: Added dirty/requires-context replay semantics so deferred/failed runtime snapshots can be retried safely.
# IMP-2: Added resilient intent-context handling and same-module notifier fallback for more reliable edge-runtime behavior.

"""Runtime-side voice-orchestrator state and same-stream transcript helpers."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from typing import Any

from twinr.agent.base_agent.conversation.follow_up_context import (
    clear_pending_conversation_follow_up_hint,
    pending_conversation_follow_up_hint_trace_details,
)
from twinr.agent.workflows.remote_transcript_commit import (
    RemoteTranscriptCommit,
    RemoteTranscriptWaitHandle,
)
from twinr.agent.workflows.voice_identity_runtime import (
    sync_voice_orchestrator_identity_profiles,
)
from twinr.orchestrator.voice_runtime_intent import VoiceRuntimeIntentContext

_DEFAULT_REMOTE_TRANSCRIPT_SOURCE = "listening"
# BREAKING: Remote transcripts are now bounded before entering the local runtime to
# protect the Pi from transcript-flood DoS and runaway downstream compute/cost.
_MAX_REMOTE_TRANSCRIPT_CHARS = 8192
_MAX_DETAIL_CHARS = 160
_MAX_EMIT_TEXT_CHARS = 96
_MAX_SOURCE_CHARS = 64
_REMOTE_TRANSCRIPT_WAIT_POLL_S = 0.05
_INTENT_CONTEXT_FIELDS: tuple[str, ...] = (
    "attention_state",
    "interaction_intent_state",
    "person_visible",
    "presence_active",
    "interaction_ready",
    "targeted_inference_blocked",
    "recommended_channel",
    "speaker_associated",
    "speaker_association_confidence",
    "background_media_likely",
    "speech_overlap_likely",
)


def _sanitize_untrusted_text(value: Any, *, max_chars: int) -> str | None:
    """Return one compact, control-character-safe text fragment."""

    # BREAKING: Untrusted bridge strings are normalized before they hit emits and
    # runtime-state details so downstream logs/observers cannot be forged with CRLF.
    if value is None:
        return None
    text = str(value)
    text = (
        text.replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\x00", " ")
        .strip()
    )
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text or None


def _sanitize_emit_text(value: Any, *, default: str = "unknown") -> str:
    """Return one safe text fragment for emit/trace strings."""

    return _sanitize_untrusted_text(value, max_chars=_MAX_EMIT_TEXT_CHARS) or default


def _normalize_remote_source(source: Any) -> str:
    """Normalize one remote transcript source token."""

    return (
        _sanitize_untrusted_text(source, max_chars=_MAX_SOURCE_CHARS)
        or _DEFAULT_REMOTE_TRANSCRIPT_SOURCE
    ).lower()


def _normalize_remote_transcript(transcript: Any) -> str:
    """Normalize one committed remote transcript to a bounded text payload."""

    text = str(transcript or "").replace("\x00", " ").strip()
    if not text:
        return ""
    if len(text) > _MAX_REMOTE_TRANSCRIPT_CHARS:
        text = text[:_MAX_REMOTE_TRANSCRIPT_CHARS].rstrip()
    return text


def _transcript_trace_details(transcript: str) -> dict[str, object]:
    """Return privacy-safe trace fields for one committed transcript."""

    normalized = str(transcript or "").strip()
    if not normalized:
        return {"present": False, "chars": 0, "words": 0, "sha256_12": ""}
    return {
        "present": True,
        "chars": len(normalized),
        "words": len(normalized.split()),
        "sha256_12": hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12],
    }


def _is_timeout_reason(reason: str | None) -> bool:
    """Return whether one close reason should count as a listen timeout."""

    normalized = (reason or "").strip().lower()
    return "timeout" in normalized


def _bounded_timeout_s(timeout_s: Any, *, minimum: float = 0.1) -> float:
    """Return one finite, positive timeout."""

    try:
        value = float(timeout_s)
    except (TypeError, ValueError):
        return minimum
    if not math.isfinite(value):
        return minimum
    return max(minimum, value)


def _notify_voice_state(
    loop: Any,
    state: str,
    *,
    detail: str | None = None,
    follow_up_allowed: bool = False,
) -> None:
    """Use the loop-bound notifier when present, otherwise call the module helper."""

    notifier = getattr(loop, "_notify_voice_orchestrator_state", None)
    if callable(notifier):
        notifier(state, detail=detail, follow_up_allowed=follow_up_allowed)
        return
    notify_voice_orchestrator_state(
        loop,
        state,
        detail=detail,
        follow_up_allowed=follow_up_allowed,
    )


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


def _follow_up_reopen_allowed(loop: Any, *, initial_source: str) -> bool:
    """Return whether follow-up is allowed for the source and current quiet mode."""

    if loop.voice_orchestrator is None:
        return False
    allowed = loop._follow_up_allowed_for_source(initial_source=initial_source)
    return _effective_follow_up_allowed(loop, follow_up_allowed=bool(allowed))


def _cached_follow_up_window_open(loop: Any) -> bool:
    """Return whether the local cached runtime state explicitly has an open follow-up window."""

    cached_state = getattr(loop, "_last_voice_orchestrator_runtime_state", None)
    if not (isinstance(cached_state, tuple) and len(cached_state) >= 3):
        return False
    state, _detail, follow_up_allowed = cached_state[:3]
    return state == "follow_up_open" and _effective_follow_up_allowed(
        loop,
        follow_up_allowed=bool(follow_up_allowed),
    )


def _intent_context_from_loop(
    loop: Any,
    *,
    prefer_cached: bool,
) -> VoiceRuntimeIntentContext | None:
    """Build one intent context, optionally falling back to the last good cached context."""

    try:
        context = VoiceRuntimeIntentContext.from_sensor_facts(
            getattr(loop, "_latest_sensor_observation_facts", None)
        )
    except Exception as exc:
        loop.emit(f"voice_orchestrator_intent_context_failed={type(exc).__name__}")
        loop._trace_event(
            "voice_orchestrator_intent_context_failed",
            kind="error",
            level="ERROR",
            details={"error": type(exc).__name__},
        )
        context = None
    if context is not None and _intent_context_has_signal(context):
        return context
    if prefer_cached:
        cached_context = getattr(loop, "_last_voice_orchestrator_intent_context", None)
        if cached_context is not None:
            return cached_context
    return None


def _intent_context_has_signal(intent_context: VoiceRuntimeIntentContext) -> bool:
    """Return whether the compact context contains any explicit signal."""

    return any(getattr(intent_context, field_name) is not None for field_name in _INTENT_CONTEXT_FIELDS)


def _intent_event_fields(
    loop: Any,
    intent_context: VoiceRuntimeIntentContext | None,
) -> dict[str, Any]:
    """Serialize one intent context to event fields without letting serialization failures break runtime flow."""

    if intent_context is None:
        return {}
    try:
        return intent_context.to_event_fields()
    except Exception as exc:
        loop.emit(f"voice_orchestrator_intent_context_serialize_failed={type(exc).__name__}")
        loop._trace_event(
            "voice_orchestrator_intent_context_serialize_failed",
            kind="error",
            level="ERROR",
            details={"error": type(exc).__name__},
        )
        return {}


def _ensure_voice_orchestrator_transport_lock(loop: Any) -> Any:
    """Return one lock that serializes bridge notifications without blocking the state lock."""

    lock = getattr(loop, "_voice_orchestrator_transport_lock", None)
    if lock is not None:
        return lock
    lock = threading.RLock()
    setattr(loop, "_voice_orchestrator_transport_lock", lock)
    return lock


def _cache_voice_orchestrator_runtime_state(
    loop: Any,
    *,
    state: str,
    detail: str | None,
    follow_up_allowed: bool,
    intent_context: VoiceRuntimeIntentContext | None,
    quiet_until_utc: str | None,
    dirty: bool,
    requires_context: bool,
) -> None:
    """Cache the latest desired runtime snapshot locally."""

    with loop._voice_orchestrator_runtime_state_lock:
        loop._last_voice_orchestrator_runtime_state = (
            state,
            detail,
            follow_up_allowed,
        )
        loop._last_voice_orchestrator_intent_context = intent_context
        loop._last_voice_orchestrator_quiet_until_utc = quiet_until_utc
        loop._last_voice_orchestrator_runtime_state_dirty = bool(dirty)
        loop._last_voice_orchestrator_runtime_state_requires_context = bool(
            requires_context
        )
        loop._voice_orchestrator_runtime_state_inflight_payload = None


def _send_voice_orchestrator_runtime_state(
    loop: Any,
    *,
    state: str,
    detail: str | None,
    follow_up_allowed: bool,
    intent_context: VoiceRuntimeIntentContext | None,
    quiet_until_utc: str | None,
    failure_emit_key: str,
    use_seed_method: bool = False,
    force_identity_sync: bool = False,
) -> bool:
    """Send one runtime snapshot to the bridge without holding the state lock across bridge I/O."""

    if loop.voice_orchestrator is None:
        return False

    normalized_detail = _sanitize_untrusted_text(detail, max_chars=_MAX_DETAIL_CHARS)
    effective_follow_up = _effective_follow_up_allowed(
        loop,
        follow_up_allowed=follow_up_allowed,
    )
    event_fields = _intent_event_fields(loop, intent_context)
    state_tuple = (state, normalized_detail, follow_up_allowed)
    transport_lock = _ensure_voice_orchestrator_transport_lock(loop)
    payload_key = (
        state,
        normalized_detail,
        follow_up_allowed,
        intent_context,
        quiet_until_utc,
    )
    send_succeeded = False

    with loop._voice_orchestrator_runtime_state_lock:
        generation = int(
            getattr(loop, "_voice_orchestrator_runtime_state_generation", 0)
        ) + 1
        loop._voice_orchestrator_runtime_state_generation = generation
        loop._last_voice_orchestrator_runtime_state = state_tuple
        loop._last_voice_orchestrator_intent_context = intent_context
        loop._last_voice_orchestrator_quiet_until_utc = quiet_until_utc
        loop._last_voice_orchestrator_runtime_state_dirty = True
        loop._last_voice_orchestrator_runtime_state_requires_context = False
        loop._voice_orchestrator_runtime_state_inflight_payload = payload_key
        transport_lock.acquire()

    try:
        seed_runtime_state = getattr(loop.voice_orchestrator, "seed_runtime_state", None)
        if use_seed_method and callable(seed_runtime_state):
            seed_runtime_state(
                state=state,
                detail=normalized_detail,
                follow_up_allowed=effective_follow_up,
                voice_quiet_until_utc=quiet_until_utc,
                **event_fields,
            )
        else:
            loop.voice_orchestrator.notify_runtime_state(
                state=state,
                detail=normalized_detail,
                follow_up_allowed=effective_follow_up,
                voice_quiet_until_utc=quiet_until_utc,
                **event_fields,
            )
    except Exception as exc:
        loop.emit(f"{failure_emit_key}={type(exc).__name__}")
        loop._trace_event(
            failure_emit_key,
            kind="error",
            level="ERROR",
            details={
                "state": state,
                "error": type(exc).__name__,
            },
        )
        return False
    finally:
        transport_lock.release()
        with loop._voice_orchestrator_runtime_state_lock:
            if getattr(loop, "_voice_orchestrator_runtime_state_generation", 0) == generation:
                loop._voice_orchestrator_runtime_state_inflight_payload = None

    send_succeeded = True

    with loop._voice_orchestrator_runtime_state_lock:
        if (
            send_succeeded
            and getattr(loop, "_voice_orchestrator_runtime_state_generation", 0) == generation
            and getattr(loop, "_last_voice_orchestrator_runtime_state", None) == state_tuple
            and getattr(loop, "_last_voice_orchestrator_intent_context", None) == intent_context
            and getattr(loop, "_last_voice_orchestrator_quiet_until_utc", None)
            == quiet_until_utc
        ):
            loop._last_voice_orchestrator_runtime_state_dirty = False
            loop._last_voice_orchestrator_runtime_state_requires_context = False

    try:
        sync_voice_orchestrator_identity_profiles(loop, force=force_identity_sync)
    except Exception as exc:
        loop.emit(f"{failure_emit_key}={type(exc).__name__}")
        loop._trace_event(
            f"{failure_emit_key}_identity_sync",
            kind="error",
            level="ERROR",
            details={
                "state": state,
                "error": type(exc).__name__,
            },
        )
        return False

    return True


def voice_orchestrator_handles_follow_up(loop: Any, *, initial_source: str) -> bool:
    """Report whether the current voice path reopens follow-up remotely."""

    if loop.voice_orchestrator is None:
        return False
    return voice_orchestrator_follow_up_mode(loop, initial_source=initial_source) == "remote"


def voice_orchestrator_follow_up_mode(loop: Any, *, initial_source: str) -> str:
    """Return how Twinr should reopen continuation after a finished answer."""

    if loop.voice_orchestrator is None:
        return "disabled"
    if not _follow_up_reopen_allowed(loop, initial_source=initial_source):
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

    intent_context = _intent_context_from_loop(loop, prefer_cached=True)
    quiet_until_utc = _voice_quiet_until_utc(loop)
    _send_voice_orchestrator_runtime_state(
        loop,
        state=state,
        detail=detail,
        follow_up_allowed=follow_up_allowed,
        intent_context=intent_context,
        quiet_until_utc=quiet_until_utc,
        failure_emit_key="voice_orchestrator_notify_failed",
    )


def prime_voice_orchestrator_waiting_state(loop: Any) -> None:
    """Seed one explicit idle state so later sensor refreshes can replay it."""

    if loop.voice_orchestrator is None:
        return
    if loop._last_voice_orchestrator_runtime_state is not None:
        return

    quiet_until_utc = _voice_quiet_until_utc(loop)
    intent_context = _intent_context_from_loop(loop, prefer_cached=False)
    if intent_context is None:
        _cache_voice_orchestrator_runtime_state(
            loop,
            state="waiting",
            detail=None,
            follow_up_allowed=False,
            intent_context=None,
            quiet_until_utc=quiet_until_utc,
            dirty=True,
            requires_context=True,
        )
        loop._trace_event(
            "voice_orchestrator_waiting_state_seed_deferred",
            kind="branch",
            details={"reason": "missing_sensor_context"},
        )
        return

    _send_voice_orchestrator_runtime_state(
        loop,
        state="waiting",
        detail=None,
        follow_up_allowed=False,
        intent_context=intent_context,
        quiet_until_utc=quiet_until_utc,
        failure_emit_key="voice_orchestrator_waiting_state_seed_failed",
        use_seed_method=True,
        force_identity_sync=True,
    )


def refresh_voice_orchestrator_sensor_context(loop: Any) -> None:
    """Push changed multimodal intent context into the live voice gateway."""

    if loop.voice_orchestrator is None:
        return

    requires_context = bool(
        getattr(loop, "_last_voice_orchestrator_runtime_state_requires_context", False)
    )
    intent_context = _intent_context_from_loop(
        loop,
        prefer_cached=not requires_context,
    )
    quiet_until_utc = _voice_quiet_until_utc(loop)

    with loop._voice_orchestrator_runtime_state_lock:
        last_state = loop._last_voice_orchestrator_runtime_state
        if last_state is None:
            return
        if requires_context and intent_context is None:
            return

        state, detail, follow_up_allowed = last_state
        dirty = bool(getattr(loop, "_last_voice_orchestrator_runtime_state_dirty", False))
        last_intent_context = loop._last_voice_orchestrator_intent_context
        last_quiet_until_utc = getattr(
            loop,
            "_last_voice_orchestrator_quiet_until_utc",
            None,
        )
        inflight_payload = getattr(
            loop,
            "_voice_orchestrator_runtime_state_inflight_payload",
            None,
        )
        payload_key = (
            state,
            detail,
            follow_up_allowed,
            intent_context,
            quiet_until_utc,
        )
        if inflight_payload == payload_key:
            return
        if (
            not dirty
            and intent_context == last_intent_context
            and quiet_until_utc == last_quiet_until_utc
        ):
            return

    _send_voice_orchestrator_runtime_state(
        loop,
        state=state,
        detail=detail,
        follow_up_allowed=follow_up_allowed,
        intent_context=intent_context,
        quiet_until_utc=quiet_until_utc,
        failure_emit_key="voice_orchestrator_context_refresh_failed",
    )


def handle_remote_transcript_committed(loop: Any, transcript: str, source: str) -> bool:
    """Consume one transcript committed by the live remote voice stream."""

    try:
        raw_transcript = str(transcript or "")
        committed_source = _normalize_remote_source(source)
        committed_transcript = _normalize_remote_transcript(raw_transcript)

        if not committed_transcript:
            loop.emit("voice_orchestrator_transcript_ignored=empty")
            return False

        if len(raw_transcript.strip()) > _MAX_REMOTE_TRANSCRIPT_CHARS:
            loop.emit("voice_orchestrator_transcript_truncated=true")
        loop._trace_event(
            "voice_orchestrator_transcript_committed_payload",
            kind="workflow",
            details={
                "source": committed_source,
                "raw_chars": len(raw_transcript.strip()),
                "normalized": _transcript_trace_details(committed_transcript),
                "follow_up_hint": pending_conversation_follow_up_hint_trace_details(
                    getattr(loop, "runtime", None)
                ),
            },
        )

        if committed_source == "follow_up":
            if not _follow_up_reopen_allowed(loop, initial_source="follow_up"):
                clear_pending_conversation_follow_up_hint(getattr(loop, "runtime", None))
                loop.emit("voice_orchestrator_follow_up_skipped=disabled")
                return False
            if not _cached_follow_up_window_open(loop):
                clear_pending_conversation_follow_up_hint(getattr(loop, "runtime", None))
                loop.emit("voice_orchestrator_follow_up_skipped=closed")
                return False
            if bool(getattr(loop, "_conversation_session_active", False)):
                loop.emit("voice_orchestrator_follow_up_skipped=session_active")
                return False

        if loop._remote_transcript_commits.commit(
            source=committed_source,
            transcript=committed_transcript,
        ):
            loop.emit(
                f"voice_orchestrator_transcript_delivered="
                f"{_sanitize_emit_text(committed_source, default=_DEFAULT_REMOTE_TRANSCRIPT_SOURCE)}"
            )
            return True

        if committed_source != "follow_up":
            loop.emit(
                f"voice_orchestrator_transcript_ignored="
                f"{_sanitize_emit_text(committed_source, default=_DEFAULT_REMOTE_TRANSCRIPT_SOURCE)}"
            )
            return False

        if not loop._required_remote_dependency_current_ready():
            loop._request_required_remote_dependency_refresh()
            clear_pending_conversation_follow_up_hint(getattr(loop, "runtime", None))
            loop.emit("voice_orchestrator_follow_up_skipped=dependency_unready")
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
        normalized_reason = _sanitize_untrusted_text(reason, max_chars=_MAX_DETAIL_CHARS) or "closed"
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
            clear_pending_conversation_follow_up_hint(getattr(loop, "runtime", None))
            loop.emit("voice_orchestrator_follow_up_closed_ignored=runtime_not_listening")
            _notify_voice_state(loop, "waiting", detail=normalized_reason)
            return

        clear_pending_conversation_follow_up_hint(getattr(loop, "runtime", None))
        loop.runtime.cancel_listening()
        loop._emit_status(force=True)
        _notify_voice_state(loop, "waiting", detail=normalized_reason)
        loop.emit(
            "voice_orchestrator_follow_up_closed_local_waiting="
            f"{_sanitize_emit_text(normalized_reason, default='closed')}"
        )
        loop._trace_event(
            "voice_orchestrator_follow_up_closed_local_waiting",
            kind="mutation",
            details={"reason": normalized_reason},
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

    deadline_at = time.monotonic() + _bounded_timeout_s(timeout_s)
    while True:
        if loop._active_turn_stop_requested():
            loop._cancel_interrupted_turn()
            return None

        if wait_handle.commit is not None:
            return wait_handle.commit

        if wait_handle.close is not None:
            close_reason = (
                _sanitize_untrusted_text(wait_handle.close.reason, max_chars=_MAX_DETAIL_CHARS)
                or "closed"
            )
            loop.runtime.cancel_listening()
            loop._emit_status(force=True)
            _notify_voice_state(loop, "waiting", detail=close_reason)
            loop.emit(
                "voice_orchestrator_transcript_closed="
                f"{_sanitize_emit_text(close_reason, default='closed')}"
            )

            if _is_timeout_reason(close_reason):
                loop.runtime.remember_listen_timeout(
                    initial_source=initial_source,
                    follow_up=follow_up,
                )
                loop.emit(f"{timeout_emit_key}=true")
                loop._record_event(
                    "listen_timeout",
                    timeout_message,
                    request_source=listen_source,
                    reason=close_reason,
                )
            else:
                loop._record_event(
                    "listen_closed",
                    "Remote listen window closed before transcript commit.",
                    request_source=listen_source,
                    reason=close_reason,
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
            _notify_voice_state(loop, "waiting", detail="listen_timeout")
            loop.emit(f"{timeout_emit_key}=true")
            loop._record_event("listen_timeout", timeout_message, request_source=listen_source)
            return None

        wait_handle.wait(timeout_s=min(_REMOTE_TRANSCRIPT_WAIT_POLL_S, remaining_s))
