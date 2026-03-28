"""Coordinate parallel bridge and final lanes for streaming turns.

This helper keeps the streaming hardware loop thin by owning the concurrency,
deadline handling, and bridge/final-lane coordination for the dual-lane speech
path. It runs the short bridge lane and the slower final tool/search lane in
parallel, emits a watchdog fallback when the bridge lane stalls, propagates
cooperative stop signals into both lanes, and fails closed unless a recovery
callback is explicitly provided.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Final replies were needlessly blocked by a stalled bridge lane until the
#        bridge timeout expired, even when the final lane had already finished.
# BUG-2: first_audio_gate_ms was violated because the code could wait the full
#        gate twice (wait_for_first_audio + wait_until_idle), doubling latency.
# BUG-3: recover_final_lane_response was ignored entirely, so callers could not
#        recover from final-lane failures/timeouts even though the API exposed it.
# SEC-1: Early-return, timeout, and interruption paths leaked background daemon
#        work. On-device this can continue tool/network activity after a turn is
#        cancelled, creating a practical CPU/network/token DoS and privacy leak.
# IMP-1: Added cooperative stop propagation, bounded cleanup joins, deadline
#        propagation into lane callables, and cleanup telemetry for stuck lanes.
# IMP-2: Added optional bridge-reply verification and deadline-bounded
#        speculative emission so low-value late fillers do not degrade UX.

from __future__ import annotations

from contextvars import copy_context
from dataclasses import dataclass, field
from inspect import Parameter, signature
from threading import Event, Lock, Thread
from typing import Callable, Generic, TypeVar, cast
import time

from twinr.agent.base_agent.contracts import FirstWordReply
from twinr.agent.tools.runtime.speech_lane import SpeechLaneDelta
from twinr.agent.tools.runtime.streaming_loop import StreamingToolLoopResult
from twinr.agent.workflows.forensics import capture_thread_snapshot


T = TypeVar("T")
_NS_PER_MS = 1_000_000
_NS_PER_S = 1_000_000_000


class FinalLaneTimeoutError(RuntimeError):
    """Raise when the dual-lane final path misses its hard deadline."""


@dataclass(frozen=True, slots=True)
class StreamingTurnTimeoutPolicy:
    """Define the bounded waiting policy for parallel streaming lanes.

    Attributes:
        bridge_reply_timeout_ms: Maximum time to wait for the bridge reply
            before falling back to a watchdog filler.
        final_lane_watchdog_timeout_ms: Soft deadline that emits a watchdog
            signal when the final lane still has not completed.
        final_lane_hard_timeout_ms: Hard deadline after which the turn either
            raises FinalLaneTimeoutError or uses recover_final_lane_response
            when such a recovery callback is explicitly provided.
        first_audio_gate_ms: Maximum total time to wait for the first bridge
            audio/idle transition before replacing it with the final lane text.
        poll_interval_ms: Internal polling interval for lane coordination.
        cleanup_join_timeout_ms: Bounded join time used when requesting lane
            shutdown during interruption, timeout, or early-return cleanup.
    """

    bridge_reply_timeout_ms: int
    final_lane_watchdog_timeout_ms: int
    final_lane_hard_timeout_ms: int
    first_audio_gate_ms: int
    poll_interval_ms: int = 25
    cleanup_join_timeout_ms: int = 10

    def __post_init__(self) -> None:
        ints = {
            "bridge_reply_timeout_ms": self.bridge_reply_timeout_ms,
            "final_lane_watchdog_timeout_ms": self.final_lane_watchdog_timeout_ms,
            "final_lane_hard_timeout_ms": self.final_lane_hard_timeout_ms,
            "first_audio_gate_ms": self.first_audio_gate_ms,
            "poll_interval_ms": self.poll_interval_ms,
            "cleanup_join_timeout_ms": self.cleanup_join_timeout_ms,
        }
        for name, value in ints.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.poll_interval_ms == 0:
            raise ValueError("poll_interval_ms must be > 0")
        if self.final_lane_hard_timeout_ms < self.final_lane_watchdog_timeout_ms:
            raise ValueError(
                "final_lane_hard_timeout_ms must be >= final_lane_watchdog_timeout_ms"
            )


@dataclass(frozen=True, slots=True)
class StreamingTurnLaneOutcome:
    """Capture the completed result of the coordinated streaming turn."""

    response: StreamingToolLoopResult
    first_word_reply: FirstWordReply | None
    first_word_source: str
    bridge_watchdog_triggered: bool = False
    final_lane_watchdog_triggered: bool = False
    final_lane_timed_out: bool = False
    final_lane_recovered: bool = False
    bridge_lane_elapsed_ms: float | None = None
    final_lane_elapsed_ms: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class _BackgroundLaneTask(Generic[T]):
    """Run one blocking lane call in a thread and retain its outcome."""

    def __init__(
        self,
        *,
        name: str,
        target: Callable[[], T],
        deadline_ns: int | None,
        daemon: bool = True,
    ) -> None:
        self.name = name
        self._target = target
        self._deadline_ns = deadline_ns
        self._done = Event()
        self._stop_requested = Event()
        self._lock = Lock()
        self._result: T | None = None
        self._error: BaseException | None = None
        self._stop_reason: str | None = None
        self._started_at_ns: int | None = None
        self._finished_at_ns: int | None = None
        self._context = copy_context()
        self._thread = Thread(
            target=self._run_in_context,
            name=f"twinr-{name}",
            daemon=daemon,
        )

    def start(self) -> None:
        """Start the background lane task exactly once."""

        if self._started_at_ns is not None:
            raise RuntimeError(f"background task '{self.name}' already started")
        self._started_at_ns = time.monotonic_ns()
        self._thread.start()

    @property
    def done(self) -> bool:
        """Return whether the background task has finished."""

        return self._done.is_set()

    @property
    def elapsed_ms(self) -> float | None:
        """Return elapsed task runtime in milliseconds."""

        start_ns = self._started_at_ns
        if start_ns is None:
            return None
        end_ns = self._finished_at_ns if self._finished_at_ns is not None else time.monotonic_ns()
        return round((end_ns - start_ns) / _NS_PER_MS, 3)

    def request_stop(self, reason: str) -> None:
        """Set the cooperative stop signal for the lane."""

        with self._lock:
            if self._stop_reason is None:
                self._stop_reason = reason
        self._stop_requested.set()

    def join(self, timeout_s: float | None) -> bool:
        """Join the worker thread for at most timeout_s seconds."""

        if self._started_at_ns is None:
            return True
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()

    def _run_in_context(self) -> None:
        self._context.run(self._run)

    def _run(self) -> None:
        try:
            result = self._invoke_target()
        except BaseException as exc:  # pragma: no cover - exercised via caller paths
            with self._lock:
                self._error = exc
        else:
            with self._lock:
                self._result = result
        finally:
            self._finished_at_ns = time.monotonic_ns()
            self._done.set()

    def _invoke_target(self) -> T:
        runtime_kwargs = {
            "stop_event": self._stop_requested,
            "cancel_event": self._stop_requested,
            "deadline_monotonic_ns": self._deadline_ns,
            "deadline_ns": self._deadline_ns,
            "deadline_s": None if self._deadline_ns is None else (self._deadline_ns / _NS_PER_S),
            "lane_name": self.name,
        }
        try:
            params = signature(self._target).parameters
        except (TypeError, ValueError):
            return self._target()
        accepts_kwargs = any(
            parameter.kind is Parameter.VAR_KEYWORD
            for parameter in params.values()
        )
        if accepts_kwargs:
            return self._target(**runtime_kwargs)
        filtered_kwargs = {
            key: value
            for key, value in runtime_kwargs.items()
            if key in params
        }
        if filtered_kwargs:
            return self._target(**filtered_kwargs)
        return self._target()

    def result(self) -> T:
        """Return the finished result or re-raise the stored exception."""

        if not self._done.is_set():
            raise RuntimeError(f"background task '{self.name}' is not finished")
        with self._lock:
            if self._error is not None:
                raise self._error
            return cast(T, self._result)

    def snapshot(self) -> dict[str, object]:
        """Describe the current task/thread state for bounded forensics."""

        with self._lock:
            error_type = type(self._error).__name__ if self._error is not None else None
            has_result = self._result is not None
            stop_reason = self._stop_reason
        return {
            "name": self.name,
            "done": self.done,
            "elapsed_ms": self.elapsed_ms,
            "error_type": error_type,
            "has_result": has_result,
            "stop_requested": self._stop_requested.is_set(),
            "stop_reason": stop_reason,
            "thread": capture_thread_snapshot(self._thread),
        }


class StreamingTurnOrchestrator:
    """Run the bridge and final lanes concurrently with bounded fallbacks.

    The orchestrator starts the final lane immediately, starts the bridge lane
    in parallel when available, and emits a fallback filler if the bridge lane
    misses its deadline. It preserves the speech-lane contract by emitting
    speculative bridge deltas first and replacing them atomically with the
    final lane answer once the slow path completes. On early exit, timeout, or
    interruption, it requests lane shutdown and performs a bounded cleanup join.
    """

    def __init__(
        self,
        *,
        timeout_policy: StreamingTurnTimeoutPolicy,
        queue_lane_delta: Callable[[SpeechLaneDelta], None],
        wait_for_first_audio: Callable[..., bool],
        wait_until_idle: Callable[..., bool] | None = None,
        is_output_idle: Callable[[], bool] | None = None,
        ensure_processing_feedback: Callable[[], None],
        resume_processing_after_bridge: Callable[[], None] | None = None,
        stop_final_lane_feedback: Callable[[], None] | None = None,
        emit: Callable[[str], None] | None = None,
        trace_event: Callable[..., None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        request_final_lane_stop: Callable[[str], None] | None = None,
        request_bridge_lane_stop: Callable[[str], None] | None = None,
        verify_bridge_reply: Callable[[FirstWordReply], bool] | None = None,
    ) -> None:
        self.timeout_policy = timeout_policy
        self.queue_lane_delta = queue_lane_delta
        self.wait_for_first_audio = wait_for_first_audio
        self.wait_until_idle = wait_until_idle
        self.is_output_idle = is_output_idle
        self.ensure_processing_feedback = ensure_processing_feedback
        self.resume_processing_after_bridge = resume_processing_after_bridge
        self.stop_final_lane_feedback = stop_final_lane_feedback
        self.emit = emit
        self.trace_event = trace_event
        self.should_stop = should_stop
        self.request_final_lane_stop = request_final_lane_stop
        self.request_bridge_lane_stop = request_bridge_lane_stop
        self.verify_bridge_reply = verify_bridge_reply

    def execute(
        self,
        *,
        prefetched_first_word: FirstWordReply | None,
        prefetched_first_word_source: str,
        generate_first_word: Callable[[], FirstWordReply | None] | None,
        bridge_fallback_reply: FirstWordReply | None,
        run_final_lane: Callable[[], StreamingToolLoopResult],
        recover_final_lane_response: Callable[[str], StreamingToolLoopResult] | None,
        should_recover_final_lane_error: Callable[[BaseException], bool] | None = None,
    ) -> StreamingTurnLaneOutcome:
        """Run one streaming turn with a parallel bridge and final lane."""

        started_at_ns = time.monotonic_ns()
        cleanup_reason = "execute_exit"
        bridge_deadline_ns = started_at_ns + _ms_to_ns(self.timeout_policy.bridge_reply_timeout_ms)
        final_watchdog_deadline_ns = (
            started_at_ns + _ms_to_ns(self.timeout_policy.final_lane_watchdog_timeout_ms)
        )
        final_hard_deadline_ns = (
            started_at_ns + _ms_to_ns(self.timeout_policy.final_lane_hard_timeout_ms)
        )
        poll_sleep_s = max(0.005, self.timeout_policy.poll_interval_ms / 1000.0)

        bridge_reply, bridge_source = self._normalize_bridge_reply(
            prefetched_first_word,
            source=prefetched_first_word_source if prefetched_first_word is not None else "none",
            deadline_expired=False,
        )
        bridge_emitted = False
        bridge_watchdog_triggered = False
        bridge_timeout_triggered = False
        final_lane_watchdog_triggered = False
        first_audio_gate_required = False
        bridge_wait_feedback_resumed = False
        bridge_resume_processing_required = False

        if bridge_reply is not None and bridge_reply.mode == "direct":
            cleanup_reason = "prefetched_direct_reply"
            self._stop_final_lane_feedback()
            self._emit_lane_delta(
                bridge_reply.spoken_text,
                lane="direct",
                replace_current=False,
                atomic=True,
            )
            return StreamingTurnLaneOutcome(
                response=_direct_reply_result(bridge_reply),
                first_word_reply=bridge_reply,
                first_word_source=bridge_source,
                bridge_lane_elapsed_ms=None,
                final_lane_elapsed_ms=None,
                metadata={"exit_reason": cleanup_reason},
            )

        final_task = _BackgroundLaneTask(
            name="final-lane",
            target=run_final_lane,
            deadline_ns=final_hard_deadline_ns,
        )
        final_task.start()

        bridge_task: _BackgroundLaneTask[FirstWordReply | None] | None = None
        if bridge_reply is None and generate_first_word is not None:
            bridge_task = _BackgroundLaneTask(
                name="bridge-lane",
                target=generate_first_word,
                deadline_ns=bridge_deadline_ns,
            )
            bridge_task.start()

        try:
            while True:
                self._raise_if_interrupted()
                now_ns = time.monotonic_ns()

                if bridge_task is not None and bridge_reply is None and bridge_task.done:
                    try:
                        candidate_reply = bridge_task.result()
                    except Exception as exc:  # pragma: no cover - covered through emit/error paths
                        self._emit(f"first_word_sync_failed={type(exc).__name__}")
                        bridge_reply = None
                        bridge_source = "none"
                    else:
                        bridge_reply, bridge_source = self._normalize_bridge_reply(
                            candidate_reply,
                            source="sync" if candidate_reply is not None else "none",
                            deadline_expired=bridge_timeout_triggered or now_ns >= bridge_deadline_ns,
                        )

                if bridge_reply is not None and not bridge_emitted:
                    lane = "direct" if bridge_reply.mode == "direct" else "filler"
                    self._emit_lane_delta(
                        bridge_reply.spoken_text,
                        lane=lane,
                        replace_current=False,
                        atomic=True,
                    )
                    bridge_emitted = True
                    first_audio_gate_required = bridge_reply.mode == "filler"
                    bridge_resume_processing_required = first_audio_gate_required and not final_task.done
                    if bridge_reply.mode == "direct":
                        cleanup_reason = "bridge_direct_reply"
                        self._stop_final_lane_feedback()
                        return StreamingTurnLaneOutcome(
                            response=_direct_reply_result(bridge_reply),
                            first_word_reply=bridge_reply,
                            first_word_source=bridge_source,
                            bridge_watchdog_triggered=bridge_watchdog_triggered,
                            bridge_lane_elapsed_ms=None if bridge_task is None else bridge_task.elapsed_ms,
                            final_lane_elapsed_ms=final_task.elapsed_ms,
                            metadata={"exit_reason": cleanup_reason},
                        )

                if (
                    bridge_emitted
                    and first_audio_gate_required
                    and bridge_resume_processing_required
                    and not bridge_wait_feedback_resumed
                    and self._is_output_idle()
                ):
                    bridge_wait_feedback_resumed = True
                    self._emit_bridge_ack_completed(
                        started_at_ns=started_at_ns,
                        bridge_source=bridge_source,
                        bridge_reply=bridge_reply,
                        final_task=final_task,
                        final_lane_done_now=final_task.done,
                    )

                if final_task.done:
                    try:
                        response = final_task.result()
                    except Exception as exc:
                        response = self._recover_or_raise_final_lane_exception(
                            exc=exc,
                            recover_final_lane_response=recover_final_lane_response,
                            should_recover_final_lane_error=should_recover_final_lane_error,
                        )
                        recovered = True
                        cleanup_reason = "final_recovered"
                    else:
                        recovered = False
                        cleanup_reason = "final_completed"
                    bridge_audio_drained = False
                    if (
                        bridge_emitted
                        and first_audio_gate_required
                        and bridge_resume_processing_required
                        and not bridge_wait_feedback_resumed
                    ):
                        bridge_audio_drained = self._wait_for_bridge_audio_to_finish()
                        if bridge_audio_drained:
                            bridge_wait_feedback_resumed = True
                            self._emit_bridge_ack_completed(
                                started_at_ns=started_at_ns,
                                bridge_source=bridge_source,
                                bridge_reply=bridge_reply,
                                final_task=final_task,
                                final_lane_done_now=True,
                            )
                    self._queue_final_response(
                        response,
                        bridge_reply=bridge_reply if bridge_emitted else None,
                        wait_for_bridge_audio=first_audio_gate_required and not bridge_audio_drained,
                    )
                    self._stop_final_lane_feedback()
                    return StreamingTurnLaneOutcome(
                        response=response,
                        first_word_reply=bridge_reply if bridge_emitted else None,
                        first_word_source=bridge_source,
                        bridge_watchdog_triggered=bridge_watchdog_triggered,
                        final_lane_watchdog_triggered=final_lane_watchdog_triggered,
                        final_lane_recovered=recovered,
                        bridge_lane_elapsed_ms=None if bridge_task is None else bridge_task.elapsed_ms,
                        final_lane_elapsed_ms=final_task.elapsed_ms,
                        metadata={"exit_reason": cleanup_reason},
                    )

                if (
                    not bridge_timeout_triggered
                    and not bridge_emitted
                    and bridge_reply is None
                    and now_ns >= bridge_deadline_ns
                ):
                    bridge_timeout_triggered = True
                    bridge_watchdog_triggered = True
                    self._emit("first_word_timeout=true")
                    bridge_reply, bridge_source = self._normalize_bridge_reply(
                        bridge_fallback_reply,
                        source="watchdog_fallback",
                        deadline_expired=False,
                    )
                    if bridge_reply is not None:
                        self._emit_lane_delta(
                            bridge_reply.spoken_text,
                            lane="direct" if bridge_reply.mode == "direct" else "filler",
                            replace_current=False,
                            atomic=True,
                        )
                        bridge_emitted = True
                        first_audio_gate_required = bridge_reply.mode == "filler"
                        bridge_resume_processing_required = first_audio_gate_required and not final_task.done
                        if bridge_reply.mode == "direct":
                            cleanup_reason = "bridge_watchdog_direct_reply"
                            self._stop_final_lane_feedback()
                            return StreamingTurnLaneOutcome(
                                response=_direct_reply_result(bridge_reply),
                                first_word_reply=bridge_reply,
                                first_word_source=bridge_source,
                                bridge_watchdog_triggered=True,
                                bridge_lane_elapsed_ms=None if bridge_task is None else bridge_task.elapsed_ms,
                                final_lane_elapsed_ms=final_task.elapsed_ms,
                                metadata={"exit_reason": cleanup_reason},
                            )
                    else:
                        self.ensure_processing_feedback()
                        bridge_source = "none"

                if not final_lane_watchdog_triggered and now_ns >= final_watchdog_deadline_ns:
                    final_lane_watchdog_triggered = True
                    self._emit("final_lane_watchdog=true")
                    self._trace_event(
                        "streaming_final_lane_watchdog_triggered",
                        kind="warning",
                        level="WARN",
                        details={
                            "elapsed_ms": _elapsed_ms(started_at_ns),
                            "bridge_emitted": bridge_emitted,
                            "bridge_source": bridge_source,
                            "final_lane": final_task.snapshot(),
                        },
                    )
                    if not bridge_emitted:
                        self.ensure_processing_feedback()

                if now_ns >= final_hard_deadline_ns:
                    cleanup_reason = "final_timeout"
                    self._emit("final_lane_timeout=true")
                    self._trace_event(
                        "streaming_final_lane_timeout",
                        kind="exception",
                        level="ERROR",
                        details={
                            "elapsed_ms": _elapsed_ms(started_at_ns),
                            "bridge_emitted": bridge_emitted,
                            "bridge_source": bridge_source,
                            "bridge_watchdog_triggered": bridge_watchdog_triggered,
                            "final_lane": final_task.snapshot(),
                        },
                    )
                    self._request_final_lane_stop("timeout")
                    self._stop_final_lane_feedback()
                    if recover_final_lane_response is not None:
                        response = self._call_recovery(
                            recover_final_lane_response,
                            reason="timeout",
                        )
                        cleanup_reason = "final_timeout_recovered"
                        self._queue_final_response(
                            response,
                            bridge_reply=bridge_reply if bridge_emitted else None,
                            wait_for_bridge_audio=first_audio_gate_required,
                        )
                        return StreamingTurnLaneOutcome(
                            response=response,
                            first_word_reply=bridge_reply if bridge_emitted else None,
                            first_word_source=bridge_source,
                            bridge_watchdog_triggered=bridge_watchdog_triggered,
                            final_lane_watchdog_triggered=final_lane_watchdog_triggered,
                            final_lane_timed_out=True,
                            final_lane_recovered=True,
                            bridge_lane_elapsed_ms=None if bridge_task is None else bridge_task.elapsed_ms,
                            final_lane_elapsed_ms=final_task.elapsed_ms,
                            metadata={"exit_reason": cleanup_reason},
                        )
                    raise FinalLaneTimeoutError("streaming final lane exceeded its hard deadline")

                time.sleep(poll_sleep_s)
        except InterruptedError:
            cleanup_reason = "interrupted"
            self._request_bridge_lane_stop("interrupted")
            self._request_final_lane_stop("interrupted")
            self._stop_final_lane_feedback()
            raise
        finally:
            self._cleanup_background_task(
                task=bridge_task,
                reason=cleanup_reason,
                external_stop=self._request_bridge_lane_stop,
            )
            self._cleanup_background_task(
                task=final_task,
                reason=cleanup_reason,
                external_stop=self._request_final_lane_stop,
            )

    def _recover_or_raise_final_lane_exception(
        self,
        *,
        exc: BaseException,
        recover_final_lane_response: Callable[[str], StreamingToolLoopResult] | None,
        should_recover_final_lane_error: Callable[[BaseException], bool] | None,
    ) -> StreamingToolLoopResult:
        should_recover = recover_final_lane_response is not None
        if should_recover and should_recover_final_lane_error is not None:
            try:
                should_recover = bool(should_recover_final_lane_error(exc))
            except Exception as callback_exc:
                self._emit(
                    f"should_recover_final_lane_error_failed={type(callback_exc).__name__}"
                )
                should_recover = False
        if should_recover and recover_final_lane_response is not None:
            self._emit(f"final_lane_recovered={type(exc).__name__}")
            self._trace_event(
                "streaming_final_lane_recovered",
                kind="warning",
                level="WARN",
                details={"error_type": type(exc).__name__},
            )
            return self._call_recovery(
                recover_final_lane_response,
                reason=f"error:{type(exc).__name__}",
            )
        self._emit(f"final_lane_failed={type(exc).__name__}")
        self._stop_final_lane_feedback()
        raise exc.with_traceback(exc.__traceback__)

    def _call_recovery(
        self,
        recover_final_lane_response: Callable[[str], StreamingToolLoopResult],
        *,
        reason: str,
    ) -> StreamingToolLoopResult:
        try:
            return recover_final_lane_response(reason)
        except Exception as recovery_exc:
            self._emit(f"final_lane_recovery_failed={type(recovery_exc).__name__}")
            raise

    def _cleanup_background_task(
        self,
        *,
        task: _BackgroundLaneTask[object] | None,
        reason: str,
        external_stop: Callable[[str], None] | None,
    ) -> None:
        if task is None or task.done:
            return
        task.request_stop(reason)
        if external_stop is not None:
            try:
                external_stop(reason)
            except Exception:
                pass
        timeout_s = max(0.0, self.timeout_policy.cleanup_join_timeout_ms / 1000.0)
        if task.join(timeout_s=timeout_s):
            return
        self._trace_event(
            "streaming_lane_cleanup_incomplete",
            kind="warning",
            level="WARN",
            details={
                "reason": reason,
                "cleanup_join_timeout_ms": self.timeout_policy.cleanup_join_timeout_ms,
                "lane": task.snapshot(),
            },
        )

    def _normalize_bridge_reply(
        self,
        reply: FirstWordReply | None,
        *,
        source: str,
        deadline_expired: bool,
    ) -> tuple[FirstWordReply | None, str]:
        if reply is None:
            return None, "none"
        spoken_text = str(getattr(reply, "spoken_text", "") or "").strip()
        if not spoken_text:
            self._emit("bridge_reply_empty_dropped=true")
            self._trace_event(
                "streaming_bridge_reply_dropped",
                kind="warning",
                level="WARN",
                details={"source": source, "reason": "empty_spoken_text"},
            )
            return None, "none"
        if deadline_expired and getattr(reply, "mode", None) != "direct":
            self._emit("bridge_reply_late_dropped=true")
            self._trace_event(
                "streaming_bridge_reply_dropped",
                kind="branch",
                details={"source": source, "reason": "late_non_direct_reply"},
            )
            return None, "none"
        if self.verify_bridge_reply is not None:
            try:
                approved = bool(self.verify_bridge_reply(reply))
            except Exception as exc:
                self._emit(f"bridge_reply_verifier_failed={type(exc).__name__}")
                approved = True
            if not approved:
                self._emit("bridge_reply_rejected=true")
                self._trace_event(
                    "streaming_bridge_reply_rejected",
                    kind="branch",
                    details={"source": source},
                )
                return None, "none"
        return reply, source

    def _queue_final_response(
        self,
        response: StreamingToolLoopResult,
        *,
        bridge_reply: FirstWordReply | None,
        wait_for_bridge_audio: bool,
    ) -> None:
        """Emit the final lane response, replacing any existing filler."""

        final_text = str(getattr(response, "text", "") or "").strip()
        if not final_text:
            self._trace_event(
                "streaming_final_lane_empty_text",
                kind="warning",
                level="WARN",
                details={"bridge_present": bridge_reply is not None},
            )
            return
        if wait_for_bridge_audio and bridge_reply is not None:
            self._wait_for_bridge_audio_to_finish()
        self._emit_lane_delta(
            final_text,
            lane="final" if bridge_reply is not None else "direct",
            replace_current=bridge_reply is not None,
            atomic=True,
        )

    def _wait_for_bridge_audio_to_finish(self) -> bool:
        gate_deadline_ns = time.monotonic_ns() + _ms_to_ns(self.timeout_policy.first_audio_gate_ms)
        remaining_s = _remaining_timeout_s(gate_deadline_ns)
        if remaining_s > 0.0:
            self.wait_for_first_audio(timeout_s=remaining_s)
        if self.wait_until_idle is not None:
            remaining_s = _remaining_timeout_s(gate_deadline_ns)
            if remaining_s > 0.0:
                self.wait_until_idle(timeout_s=remaining_s)
        return self._is_output_idle()

    def _emit_lane_delta(
        self,
        text: str,
        *,
        lane: str,
        replace_current: bool,
        atomic: bool,
    ) -> None:
        cleaned = str(text or "").strip()
        if not cleaned:
            return
        self.queue_lane_delta(
            SpeechLaneDelta(
                text=cleaned,
                lane=lane,
                replace_current=replace_current,
                atomic=atomic,
            )
        )

    def _emit(self, message: str) -> None:
        if self.emit is not None:
            self.emit(message)

    def _trace_event(
        self,
        name: str,
        *,
        kind: str,
        details: dict[str, object] | None = None,
        level: str | None = None,
        kpi: dict[str, object] | None = None,
    ) -> None:
        if self.trace_event is None:
            return
        try:
            self.trace_event(
                name,
                kind=kind,
                details=dict(details or {}),
                level=level,
                kpi=kpi,
            )
        except Exception:
            pass

    def _stop_final_lane_feedback(self) -> None:
        if self.stop_final_lane_feedback is None:
            return
        self.stop_final_lane_feedback()

    def _emit_bridge_ack_completed(
        self,
        *,
        started_at_ns: int,
        bridge_source: str,
        bridge_reply: FirstWordReply | None,
        final_task: _BackgroundLaneTask,
        final_lane_done_now: bool,
    ) -> None:
        self._emit("bridge_ack_completed_while_final_lane_running=true")
        self._trace_event(
            "streaming_final_lane_still_running_after_bridge",
            kind="branch",
            details={
                "elapsed_ms": _elapsed_ms(started_at_ns),
                "bridge_source": bridge_source,
                "bridge_mode": bridge_reply.mode if bridge_reply is not None else None,
                "final_lane_done_now": final_lane_done_now,
                "final_lane": final_task.snapshot(),
            },
        )
        self._resume_processing_after_bridge()

    def _resume_processing_after_bridge(self) -> None:
        if self.resume_processing_after_bridge is None:
            self.ensure_processing_feedback()
            return
        self.resume_processing_after_bridge()

    def _is_output_idle(self) -> bool:
        if self.is_output_idle is not None:
            return bool(self.is_output_idle())
        if self.wait_until_idle is None:
            return False
        return bool(self.wait_until_idle(timeout_s=0.0))

    def _raise_if_interrupted(self) -> None:
        if self.should_stop is None:
            return
        if self.should_stop():
            raise InterruptedError("streaming turn interrupted")

    def _request_final_lane_stop(self, reason: str) -> None:
        if self.request_final_lane_stop is None:
            return
        try:
            self.request_final_lane_stop(reason)
        except Exception:
            pass

    def _request_bridge_lane_stop(self, reason: str) -> None:
        if self.request_bridge_lane_stop is None:
            return
        try:
            self.request_bridge_lane_stop(reason)
        except Exception:
            pass


def _direct_reply_result(reply: FirstWordReply) -> StreamingToolLoopResult:
    """Build a synthetic completed result for a terminal direct reply."""

    return StreamingToolLoopResult(
        text=reply.spoken_text,
        rounds=0,
        tool_calls=(),
        tool_results=(),
        response_id=reply.response_id,
        request_id=reply.request_id,
        model=reply.model,
        token_usage=reply.token_usage,
        used_web_search=False,
    )


def _ms_to_ns(value_ms: int) -> int:
    return max(0, int(value_ms)) * _NS_PER_MS


def _remaining_timeout_s(deadline_ns: int) -> float:
    return max(0.0, (deadline_ns - time.monotonic_ns()) / _NS_PER_S)


def _elapsed_ms(started_at_ns: int, finished_at_ns: int | None = None) -> float:
    end_ns = time.monotonic_ns() if finished_at_ns is None else finished_at_ns
    return round((end_ns - started_at_ns) / _NS_PER_MS, 3)
