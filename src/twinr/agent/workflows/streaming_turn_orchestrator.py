"""Coordinate parallel bridge and final lanes for streaming turns.

This helper keeps the streaming hardware loop thin by owning the concurrency,
deadline handling, and bounded fallback behavior for the dual-lane speech path.
It runs the short bridge lane and the slower final tool/search lane in
parallel, emits a watchdog fallback when the bridge lane stalls, and returns a
bounded synthetic result when the final lane exceeds its hard deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Callable, Generic, TypeVar
import time

from twinr.agent.base_agent.contracts import FirstWordReply
from twinr.agent.tools.runtime.dual_lane_loop import SpeechLaneDelta
from twinr.agent.tools.runtime.streaming_loop import StreamingToolLoopResult


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StreamingTurnTimeoutPolicy:
    """Define the bounded waiting policy for parallel streaming lanes.

    Attributes:
        bridge_reply_timeout_ms: Maximum time to wait for the bridge reply
            before falling back to a watchdog filler.
        final_lane_watchdog_timeout_ms: Soft deadline that emits a watchdog
            signal when the final lane still has not completed.
        final_lane_hard_timeout_ms: Hard deadline after which the turn returns
            a synthetic fallback result instead of waiting indefinitely.
        first_audio_gate_ms: Maximum time to wait for the first bridge audio
            before replacing it with the final lane text.
        poll_interval_ms: Internal polling interval for lane coordination.
    """

    bridge_reply_timeout_ms: int
    final_lane_watchdog_timeout_ms: int
    final_lane_hard_timeout_ms: int
    first_audio_gate_ms: int
    poll_interval_ms: int = 25


@dataclass(frozen=True, slots=True)
class StreamingTurnLaneOutcome:
    """Capture the completed result of the coordinated streaming turn."""

    response: StreamingToolLoopResult
    first_word_reply: FirstWordReply | None
    first_word_source: str
    bridge_watchdog_triggered: bool = False
    final_lane_watchdog_triggered: bool = False
    final_lane_timed_out: bool = False


class _BackgroundLaneTask(Generic[T]):
    """Run one blocking lane call in a daemon thread and retain its outcome."""

    def __init__(self, *, name: str, target: Callable[[], T]) -> None:
        self.name = name
        self._target = target
        self._done = Event()
        self._lock = Lock()
        self._result: T | None = None
        self._error: BaseException | None = None
        self._thread = Thread(
            target=self._run,
            name=f"twinr-{name}",
            daemon=True,
        )

    def start(self) -> None:
        """Start the background lane task exactly once."""

        self._thread.start()

    @property
    def done(self) -> bool:
        """Return whether the background task has finished."""

        return self._done.is_set()

    def _run(self) -> None:
        try:
            result = self._target()
        except BaseException as exc:  # pragma: no cover - exercised via caller paths
            with self._lock:
                self._error = exc
        else:
            with self._lock:
                self._result = result
        finally:
            self._done.set()

    def result(self) -> T:
        """Return the finished result or re-raise the stored exception."""

        if not self._done.is_set():
            raise RuntimeError(f"background task '{self.name}' is not finished")
        with self._lock:
            if self._error is not None:
                raise self._error
            return self._result


class StreamingTurnOrchestrator:
    """Run the bridge and final lanes concurrently with bounded fallbacks.

    The orchestrator starts the final lane immediately, starts the bridge lane
    in parallel when available, and emits a fallback filler if the bridge lane
    misses its deadline. It preserves the existing speech-lane contract by
    emitting filler deltas first and replacing them atomically with the final
    lane answer or an explicit timeout reply.
    """

    def __init__(
        self,
        *,
        timeout_policy: StreamingTurnTimeoutPolicy,
        queue_lane_delta: Callable[[SpeechLaneDelta], None],
        wait_for_first_audio: Callable[..., bool],
        wait_until_idle: Callable[..., bool] | None = None,
        ensure_processing_feedback: Callable[[], None],
        emit: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.timeout_policy = timeout_policy
        self.queue_lane_delta = queue_lane_delta
        self.wait_for_first_audio = wait_for_first_audio
        self.wait_until_idle = wait_until_idle
        self.ensure_processing_feedback = ensure_processing_feedback
        self.emit = emit
        self.should_stop = should_stop

    def execute(
        self,
        *,
        prefetched_first_word: FirstWordReply | None,
        prefetched_first_word_source: str,
        generate_first_word: Callable[[], FirstWordReply | None] | None,
        bridge_fallback_reply: FirstWordReply | None,
        run_final_lane: Callable[[], StreamingToolLoopResult],
        final_timeout_reply: str,
        final_error_reply: str,
    ) -> StreamingTurnLaneOutcome:
        """Run one streaming turn with a parallel bridge and final lane.

        Args:
            prefetched_first_word: Already prefetched bridge reply, if any.
            prefetched_first_word_source: Source label for the prefetched
                bridge reply.
            generate_first_word: Optional synchronous bridge-reply callable to
                execute in a background task.
            bridge_fallback_reply: Fallback reply emitted when the bridge lane
                misses its deadline.
            run_final_lane: Callable that computes the final lane result.
            final_timeout_reply: User-facing reply for hard final-lane timeout.
            final_error_reply: User-facing reply when the final lane fails.

        Returns:
            The coordinated turn result, including the chosen bridge reply.

        Raises:
            InterruptedError: If the active turn is interrupted while waiting.
        """

        bridge_reply = prefetched_first_word
        bridge_source = prefetched_first_word_source if prefetched_first_word is not None else "none"
        bridge_emitted = False
        bridge_watchdog_triggered = False
        bridge_timeout_triggered = False
        final_lane_watchdog_triggered = False
        first_audio_gate_required = False

        if bridge_reply is not None and bridge_reply.mode == "direct":
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
            )

        final_task = _BackgroundLaneTask(name="final-lane", target=run_final_lane)
        final_task.start()

        bridge_task: _BackgroundLaneTask[FirstWordReply | None] | None = None
        if bridge_reply is None and generate_first_word is not None:
            bridge_task = _BackgroundLaneTask(name="bridge-lane", target=generate_first_word)
            bridge_task.start()

        started_at = time.monotonic()
        bridge_deadline = started_at + (self.timeout_policy.bridge_reply_timeout_ms / 1000.0)
        final_watchdog_deadline = started_at + (self.timeout_policy.final_lane_watchdog_timeout_ms / 1000.0)
        final_hard_deadline = started_at + (self.timeout_policy.final_lane_hard_timeout_ms / 1000.0)
        poll_sleep_s = max(0.005, self.timeout_policy.poll_interval_ms / 1000.0)

        while True:
            self._raise_if_interrupted()
            now = time.monotonic()

            if bridge_task is not None and bridge_reply is None and bridge_task.done:
                try:
                    bridge_reply = bridge_task.result()
                except Exception as exc:  # pragma: no cover - covered through emit/error paths
                    self._emit(f"first_word_sync_failed={type(exc).__name__}")
                    bridge_reply = None
                else:
                    bridge_source = "sync" if bridge_reply is not None else "none"

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
                if bridge_reply.mode == "direct":
                    return StreamingTurnLaneOutcome(
                        response=_direct_reply_result(bridge_reply),
                        first_word_reply=bridge_reply,
                        first_word_source=bridge_source,
                        bridge_watchdog_triggered=bridge_watchdog_triggered,
                    )

            if (
                not bridge_timeout_triggered
                and
                not bridge_emitted
                and bridge_reply is None
                and now >= bridge_deadline
            ):
                bridge_timeout_triggered = True
                bridge_watchdog_triggered = True
                self._emit("first_word_timeout=true")
                bridge_reply = bridge_fallback_reply
                if bridge_reply is not None:
                    bridge_source = "watchdog_fallback"
                    self._emit_lane_delta(
                        bridge_reply.spoken_text,
                        lane="direct" if bridge_reply.mode == "direct" else "filler",
                        replace_current=False,
                        atomic=True,
                    )
                    bridge_emitted = True
                    first_audio_gate_required = bridge_reply.mode == "filler"
                    if bridge_reply.mode == "direct":
                        return StreamingTurnLaneOutcome(
                            response=_direct_reply_result(bridge_reply),
                            first_word_reply=bridge_reply,
                            first_word_source=bridge_source,
                            bridge_watchdog_triggered=True,
                        )
                else:
                    self.ensure_processing_feedback()
                    bridge_source = "none"

            bridge_phase_resolved = (
                bridge_emitted
                or bridge_task is None
                or bridge_task.done
                or now >= bridge_deadline
            )

            if final_task.done and (
                bridge_phase_resolved
                or (
                    bridge_reply is None
                    and bridge_fallback_reply is None
                )
            ):
                try:
                    response = final_task.result()
                except Exception as exc:
                    self._emit(f"final_lane_failed={type(exc).__name__}")
                    response = _fallback_result(final_error_reply)
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
                )

            if not final_lane_watchdog_triggered and now >= final_watchdog_deadline:
                final_lane_watchdog_triggered = True
                self._emit("final_lane_watchdog=true")
                if not bridge_emitted:
                    self.ensure_processing_feedback()

            if now >= final_hard_deadline:
                self._emit("final_lane_timeout=true")
                timeout_response = _fallback_result(final_timeout_reply)
                self._queue_final_response(
                    timeout_response,
                    bridge_reply=bridge_reply if bridge_emitted else None,
                    wait_for_bridge_audio=first_audio_gate_required,
                )
                return StreamingTurnLaneOutcome(
                    response=timeout_response,
                    first_word_reply=bridge_reply if bridge_emitted else None,
                    first_word_source=bridge_source,
                    bridge_watchdog_triggered=bridge_watchdog_triggered,
                    final_lane_watchdog_triggered=final_lane_watchdog_triggered,
                    final_lane_timed_out=True,
                )

            time.sleep(poll_sleep_s)

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
            return
        if wait_for_bridge_audio and bridge_reply is not None:
            gate_timeout_s = max(0.0, self.timeout_policy.first_audio_gate_ms / 1000.0)
            self.wait_for_first_audio(
                timeout_s=gate_timeout_s,
            )
            if self.wait_until_idle is not None:
                self.wait_until_idle(timeout_s=gate_timeout_s)
        self._emit_lane_delta(
            final_text,
            lane="final" if bridge_reply is not None else "direct",
            replace_current=bridge_reply is not None,
            atomic=True,
        )

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

    def _raise_if_interrupted(self) -> None:
        if self.should_stop is None:
            return
        if self.should_stop():
            raise InterruptedError("streaming turn interrupted")


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


def _fallback_result(text: str) -> StreamingToolLoopResult:
    """Build a synthetic completed result for bounded timeout/error fallbacks."""

    return StreamingToolLoopResult(
        text=str(text or "").strip(),
        rounds=0,
        tool_calls=(),
        tool_results=(),
        response_id=None,
        request_id=None,
        model=None,
        token_usage=None,
        used_web_search=False,
    )
