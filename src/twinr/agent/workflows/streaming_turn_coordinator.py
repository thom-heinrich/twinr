"""Coordinate one streaming turn under a single authoritative state machine.

This module owns the full lifecycle of one streamed Twinr turn after a
transcript is available: transcript submission, bounded processing feedback,
bridge/final lane execution, spoken-output lifecycle, interruption handling,
runtime-state transitions, and final completion bookkeeping.

The streaming runner remains responsible for capture/transcription and for
building the lane plan, while this coordinator is the sole executor of the
turn once that plan exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Event, Thread
from typing import Callable, Protocol
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.closure import ConversationClosureEvaluation
from twinr.agent.base_agent.contracts import FirstWordReply
from twinr.agent.tools.runtime.streaming_loop import StreamingToolLoopResult
from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator
from twinr.agent.workflows.speech_output import InterruptibleSpeechOutput
from twinr.agent.workflows.streaming_turn_orchestrator import (
    StreamingTurnLaneOutcome,
    StreamingTurnOrchestrator,
    StreamingTurnTimeoutPolicy,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

_ANSWERING_SNAPSHOT_REFRESH_INTERVAL_S = 5.0


class RuntimeStatusLike(Protocol):
    """Describe the runtime status object accessed by the coordinator."""

    value: str


class StreamingTurnRuntimeLike(Protocol):
    """Describe the runtime methods required by the turn coordinator."""

    status: RuntimeStatusLike

    def submit_transcript(self, transcript: str) -> None:
        ...

    def begin_answering(self) -> None:
        ...

    def resume_processing(self) -> None:
        ...

    def resume_answering_after_print(self) -> None:
        ...

    def finalize_agent_turn(self, response_text: str) -> str:
        ...

    def finish_speaking(self) -> None:
        ...

    def rearm_follow_up(self, *, request_source: str = "follow_up") -> None:
        ...

    def refresh_snapshot_activity(self) -> None:
        ...


class StreamingTurnPhase(str, Enum):
    """Name the durable phases of one coordinated streaming turn."""

    READY = "ready"
    TRANSCRIPT_SUBMITTED = "transcript_submitted"
    PROCESSING = "processing"
    LANES_RUNNING = "lanes_running"
    LLM_COMPLETED = "llm_completed"
    ANSWER_FINALIZED = "answer_finalized"
    PLAYBACK_DRAINING = "playback_draining"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


_ALLOWED_PHASE_TRANSITIONS: dict[StreamingTurnPhase, set[StreamingTurnPhase]] = {
    StreamingTurnPhase.READY: {
        StreamingTurnPhase.TRANSCRIPT_SUBMITTED,
        StreamingTurnPhase.INTERRUPTED,
        StreamingTurnPhase.FAILED,
    },
    StreamingTurnPhase.TRANSCRIPT_SUBMITTED: {
        StreamingTurnPhase.PROCESSING,
        StreamingTurnPhase.LANES_RUNNING,
        StreamingTurnPhase.INTERRUPTED,
        StreamingTurnPhase.FAILED,
    },
    StreamingTurnPhase.PROCESSING: {
        StreamingTurnPhase.LANES_RUNNING,
        StreamingTurnPhase.LLM_COMPLETED,
        StreamingTurnPhase.INTERRUPTED,
        StreamingTurnPhase.FAILED,
    },
    StreamingTurnPhase.LANES_RUNNING: {
        StreamingTurnPhase.PROCESSING,
        StreamingTurnPhase.LLM_COMPLETED,
        StreamingTurnPhase.INTERRUPTED,
        StreamingTurnPhase.FAILED,
    },
    StreamingTurnPhase.LLM_COMPLETED: {
        StreamingTurnPhase.ANSWER_FINALIZED,
        StreamingTurnPhase.INTERRUPTED,
        StreamingTurnPhase.FAILED,
    },
    StreamingTurnPhase.ANSWER_FINALIZED: {
        StreamingTurnPhase.PLAYBACK_DRAINING,
        StreamingTurnPhase.INTERRUPTED,
        StreamingTurnPhase.FAILED,
    },
    StreamingTurnPhase.PLAYBACK_DRAINING: {
        StreamingTurnPhase.COMPLETED,
        StreamingTurnPhase.INTERRUPTED,
        StreamingTurnPhase.FAILED,
    },
    StreamingTurnPhase.COMPLETED: set(),
    StreamingTurnPhase.INTERRUPTED: set(),
    StreamingTurnPhase.FAILED: set(),
}


@dataclass(frozen=True, slots=True)
class StreamingTurnRequest:
    """Capture the immutable request data for one completed transcript turn."""

    transcript: str
    listen_source: str
    proactive_trigger: str | None
    turn_started: float
    capture_ms: int
    stt_ms: int
    allow_follow_up_rearm: bool = False


@dataclass(frozen=True, slots=True)
class StreamingTurnLanePlan:
    """Describe how the coordinator should execute the LLM lane work."""

    turn_instructions: str | None
    run_single_lane: Callable[[Callable[[str], None]], StreamingToolLoopResult] | None = None
    run_final_lane: Callable[[], StreamingToolLoopResult] | None = None
    prefetched_first_word: FirstWordReply | None = None
    prefetched_first_word_source: str = "none"
    generate_first_word: Callable[[], FirstWordReply | None] | None = None
    bridge_fallback_reply: FirstWordReply | None = None
    timeout_policy: StreamingTurnTimeoutPolicy | None = None
    recover_final_lane_response: Callable[[str], StreamingToolLoopResult] | None = None

    @property
    def is_dual_lane(self) -> bool:
        """Return whether the plan executes the parallel bridge/final path."""

        return callable(self.run_final_lane)


@dataclass(frozen=True, slots=True)
class StreamingTurnSpeechServices:
    """Bundle speech-output dependencies so the coordinator owns playback."""

    tts_provider: object
    player: object
    playback_coordinator: PlaybackCoordinator | None
    segment_boundary: Callable[[str], int | None]


@dataclass(frozen=True, slots=True)
class StreamingTurnCoordinatorHooks:
    """Bundle the runner-owned hooks consumed by the coordinator."""

    emit: Callable[[str], None]
    emit_status: Callable[[], None]
    trace_event: Callable[..., None]
    trace_decision: Callable[..., None]
    start_processing_feedback_loop: Callable[[str], Callable[[], None]]
    is_search_feedback_active: Callable[[], bool]
    stop_search_feedback: Callable[[], None]
    should_stop: Callable[[], bool]
    request_turn_stop: Callable[[str], None]
    cancel_interrupted_turn: Callable[[], None]
    record_usage: Callable[..., None]
    evaluate_follow_up_closure: Callable[..., ConversationClosureEvaluation]
    apply_follow_up_closure_evaluation: Callable[..., bool]
    follow_up_rearm_allowed_now: Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class StreamingTurnExecutionOutcome:
    """Return the completed turn result to the streaming runner."""

    keep_listening: bool
    response: StreamingToolLoopResult
    answer: str
    llm_ms: int
    first_audio_at: float | None
    first_word_reply: FirstWordReply | None
    first_word_source: str


@dataclass(slots=True)
class _StreamingTurnStateMachine:
    """Track legal coordinator state transitions and emit trace evidence."""

    trace_event: Callable[..., None]
    phase: StreamingTurnPhase = StreamingTurnPhase.READY
    history: list[str] = field(default_factory=lambda: [StreamingTurnPhase.READY.value])

    def transition(
        self,
        new_phase: StreamingTurnPhase,
        *,
        reason: str,
        details: dict[str, object] | None = None,
    ) -> None:
        """Move to the next phase and emit a bounded state-transition trace."""

        if new_phase == self.phase:
            return
        allowed = _ALLOWED_PHASE_TRANSITIONS[self.phase]
        if new_phase not in allowed:
            raise RuntimeError(
                f"invalid streaming turn phase transition: {self.phase.value} -> {new_phase.value}"
            )
        previous = self.phase
        self.phase = new_phase
        self.history.append(new_phase.value)
        payload = {
            "from": previous.value,
            "to": new_phase.value,
            "reason": reason,
        }
        if details:
            payload.update(details)
        self.trace_event(
            "streaming_turn_phase_changed",
            kind="mutation",
            details=payload,
        )


@dataclass(slots=True)
class _ProcessingFeedbackController:
    """Keep the processing tone lifecycle under one explicit owner."""

    start_loop: Callable[[str], Callable[[], None]]
    trace_event: Callable[..., None]
    state_machine: _StreamingTurnStateMachine
    _stop: Callable[[], None] = field(default=lambda: None)
    _started: bool = False

    def ensure_started(self) -> None:
        """Start the processing feedback loop exactly once."""

        if self._started:
            return
        self._stop = self.start_loop("processing")
        self._started = True
        self.state_machine.transition(
            StreamingTurnPhase.PROCESSING,
            reason="processing_feedback_started",
        )
        self.trace_event("streaming_processing_feedback_started", kind="io", details={})

    def stop(self) -> None:
        """Stop the processing feedback loop if it is active."""

        if not self._started:
            return
        stop = self._stop
        self._stop = lambda: None
        self._started = False
        stop()


@dataclass(slots=True)
class _SpeechLifecycle:
    """Record first-audio and answering lifecycle for one speech output."""

    runtime: StreamingTurnRuntimeLike
    emit_status: Callable[[], None]
    trace_event: Callable[..., None]
    processing_feedback: _ProcessingFeedbackController
    state_machine: _StreamingTurnStateMachine
    turn_started: float
    answer_started: bool = False
    first_audio_at: float | None = None
    snapshot_refresh_interval_s: float = _ANSWERING_SNAPSHOT_REFRESH_INTERVAL_S
    _snapshot_refresh_stop: Event = field(default_factory=Event)
    _snapshot_refresh_thread: Thread | None = None

    def on_speaking_started(self) -> None:
        """Start `answering` exactly when real audio begins."""

        if self.answer_started:
            return
        self.processing_feedback.stop()
        if self.runtime.status.value != "answering":
            self.runtime.begin_answering()
            self.emit_status()
        self.answer_started = True
        self._start_snapshot_heartbeat()
        self.trace_event("streaming_answering_started", kind="mutation", details={})

    def on_first_audio(self) -> None:
        """Capture the first audible audio timestamp exactly once."""

        if self.first_audio_at is not None:
            return
        self.first_audio_at = time.monotonic()
        self.trace_event(
            "streaming_first_audio_observed",
            kind="metric",
            kpi={"duration_ms": round((self.first_audio_at - self.turn_started) * 1000.0, 3)},
        )

    def ensure_answering_started(self) -> None:
        """Start `answering` late when speech finished before audio callback."""

        if self.answer_started:
            return
        self.processing_feedback.stop()
        self.runtime.begin_answering()
        self.emit_status()
        self.answer_started = True
        self._start_snapshot_heartbeat()
        self.trace_event("streaming_answering_started_late", kind="mutation", details={})

    def resume_processing(self) -> None:
        """Return from a spoken bridge acknowledgement back to processing."""

        if not self.answer_started:
            return
        self.stop_snapshot_heartbeat()
        if self.runtime.status.value != "processing":
            self.runtime.resume_processing()
            self.emit_status()
        self.answer_started = False
        self.trace_event("streaming_processing_resumed", kind="mutation", details={})

    def stop_snapshot_heartbeat(self) -> None:
        """Stop the bounded snapshot refresh loop for an active spoken reply."""

        thread = self._snapshot_refresh_thread
        if thread is None:
            return
        self._snapshot_refresh_stop.set()
        thread.join(timeout=max(0.1, min(0.5, self.snapshot_refresh_interval_s)))
        stopped = not thread.is_alive()
        if stopped:
            self._snapshot_refresh_thread = None
        self.trace_event(
            "streaming_answering_snapshot_refresh_stopped",
            kind="span_end",
            details={"stopped": stopped},
        )

    def _start_snapshot_heartbeat(self) -> None:
        """Keep the runtime snapshot fresh while long speech is still audible."""

        refresh_snapshot = getattr(self.runtime, "refresh_snapshot_activity", None)
        if not callable(refresh_snapshot):
            return
        thread = self._snapshot_refresh_thread
        if thread is not None and thread.is_alive():
            return
        stop = Event()
        self._snapshot_refresh_stop = stop

        def _worker() -> None:
            while not stop.wait(self.snapshot_refresh_interval_s):
                try:
                    refresh_snapshot()
                except Exception as exc:  # pragma: no cover - defensive runtime guard
                    self.trace_event(
                        "streaming_answering_snapshot_refresh_failed",
                        kind="warning",
                        level="WARN",
                        details={"error_type": type(exc).__name__},
                    )
                    return

        self._snapshot_refresh_thread = Thread(
            target=_worker,
            daemon=True,
            name="twinr-streaming-answering-snapshot-refresh",
        )
        self._snapshot_refresh_thread.start()
        self.trace_event(
            "streaming_answering_snapshot_refresh_started",
            kind="span_start",
            details={"interval_s": self.snapshot_refresh_interval_s},
        )


@dataclass(slots=True)
class _DeferredFollowUpClosureEvaluation:
    """Evaluate closure policy in parallel with speech playback."""

    evaluate: Callable[..., ConversationClosureEvaluation]
    trace_event: Callable[..., None]
    user_transcript: str
    assistant_response: str
    request_source: str
    proactive_trigger: str | None
    _ready: Event = field(default_factory=Event)
    _evaluation: ConversationClosureEvaluation | None = None

    def start(self) -> None:
        """Launch the closure evaluation on a daemon thread."""

        self.trace_event(
            "streaming_follow_up_closure_eval_started",
            kind="span_start",
            details={"request_source": self.request_source},
        )

        def _worker() -> None:
            started = time.monotonic()
            try:
                evaluation = self.evaluate(
                    user_transcript=self.user_transcript,
                    assistant_response=self.assistant_response,
                    request_source=self.request_source,
                    proactive_trigger=self.proactive_trigger,
                )
            except Exception as exc:  # pragma: no cover - guarded by realtime runner
                evaluation = ConversationClosureEvaluation(error_type=type(exc).__name__)
            self._evaluation = evaluation
            self._ready.set()
            self.trace_event(
                "streaming_follow_up_closure_eval_finished",
                kind="span_end",
                details={
                    "error_type": evaluation.error_type,
                    "has_decision": evaluation.decision is not None,
                },
                kpi={"duration_ms": round((time.monotonic() - started) * 1000.0, 3)},
            )

        Thread(
            target=_worker,
            daemon=True,
            name="twinr-streaming-follow-up-closure",
        ).start()

    def result(
        self,
        *,
        timeout_s: float | None = None,
        should_abort: Callable[[], bool] | None = None,
        poll_interval_s: float = 0.05,
    ) -> ConversationClosureEvaluation:
        """Return the finished closure evaluation without waiting indefinitely."""

        wait_started = time.monotonic()
        deadline_at = None if timeout_s is None else wait_started + max(0.0, timeout_s)
        interval_s = max(0.01, float(poll_interval_s))
        while True:
            remaining_s = None if deadline_at is None else max(0.0, deadline_at - time.monotonic())
            wait_s = interval_s if remaining_s is None else min(interval_s, remaining_s)
            if self._ready.wait(timeout=wait_s):
                break
            if callable(should_abort) and should_abort():
                wait_ms = round((time.monotonic() - wait_started) * 1000.0, 3)
                self.trace_event(
                    "streaming_follow_up_closure_eval_join_interrupted",
                    kind="branch",
                    details={"timeout_s": timeout_s},
                    kpi={"wait_ms": wait_ms},
                )
                return ConversationClosureEvaluation(error_type="closure_eval_interrupted")
            if deadline_at is not None and time.monotonic() >= deadline_at:
                wait_ms = round((time.monotonic() - wait_started) * 1000.0, 3)
                self.trace_event(
                    "streaming_follow_up_closure_eval_join_timeout",
                    kind="warning",
                    level="WARN",
                    details={"timeout_s": timeout_s},
                    kpi={"wait_ms": wait_ms},
                )
                return ConversationClosureEvaluation(error_type="closure_eval_timeout")
        wait_ms = round((time.monotonic() - wait_started) * 1000.0, 3)
        evaluation = self._evaluation or ConversationClosureEvaluation(error_type="closure_eval_missing")
        self.trace_event(
            "streaming_follow_up_closure_eval_joined",
            kind="metric",
            details={"error_type": evaluation.error_type},
            kpi={"wait_ms": wait_ms},
        )
        return evaluation


class StreamingTurnCoordinator:
    """Execute one streaming turn under a single stateful owner."""

    def __init__(
        self,
        *,
        config: TwinrConfig,
        runtime: StreamingTurnRuntimeLike,
        request: StreamingTurnRequest,
        lane_plan_factory: Callable[[], StreamingTurnLanePlan],
        speech_services: StreamingTurnSpeechServices,
        hooks: StreamingTurnCoordinatorHooks,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.request = request
        self.lane_plan_factory = lane_plan_factory
        self.speech_services = speech_services
        self.hooks = hooks
        self.state_machine = _StreamingTurnStateMachine(trace_event=self.hooks.trace_event)
        self.processing_feedback = _ProcessingFeedbackController(
            start_loop=self.hooks.start_processing_feedback_loop,
            trace_event=self.hooks.trace_event,
            state_machine=self.state_machine,
        )
        self.speech_lifecycle = _SpeechLifecycle(
            runtime=self.runtime,
            emit_status=self.hooks.emit_status,
            trace_event=self.hooks.trace_event,
            processing_feedback=self.processing_feedback,
            state_machine=self.state_machine,
            turn_started=self.request.turn_started,
        )
        self._lane_plan: StreamingTurnLanePlan | None = None
        self._speech_output: InterruptibleSpeechOutput | None = None

    def execute(self) -> StreamingTurnExecutionOutcome:
        """Run the full streaming turn until completion or interruption."""

        self.runtime.submit_transcript(self.request.transcript)
        self.hooks.emit_status()
        self._lane_plan = self.lane_plan_factory()
        self.state_machine.transition(
            StreamingTurnPhase.TRANSCRIPT_SUBMITTED,
            reason="runtime_submitted_transcript",
            details={"transcript_len": len(self.request.transcript)},
        )
        self._speech_output = self._build_speech_output()
        response: StreamingToolLoopResult | None = None
        answer = ""
        llm_ms = 0
        first_word_reply: FirstWordReply | None = None
        first_word_source = "none"
        deferred_closure_evaluation: _DeferredFollowUpClosureEvaluation | None = None
        try:
            response, llm_ms, first_word_reply, first_word_source = self._run_lane_plan()
            self.state_machine.transition(
                StreamingTurnPhase.LLM_COMPLETED,
                reason="llm_lane_completed",
                details={
                    "response_text_len": len(getattr(response, "text", "") or ""),
                    "tool_calls": len(getattr(response, "tool_calls", ()) or ()),
                },
            )
            self.hooks.trace_event(
                "streaming_llm_lane_completed",
                kind="llm_call",
                details={
                    "response_text_len": len(response.text),
                    "tool_calls": len(response.tool_calls),
                },
                kpi={"duration_ms": llm_ms},
            )
            self._raise_if_interrupted(phase_reason="after_llm")
            if not response.text.strip():
                self.hooks.trace_event(
                    "streaming_turn_empty_response",
                    kind="warning",
                    level="WARN",
                    details={},
                )
                raise RuntimeError("Streaming tool loop completed without text output")
            answer = self._finalize_answer(response)
            self.state_machine.transition(
                StreamingTurnPhase.ANSWER_FINALIZED,
                reason="runtime_agent_turn_finalized",
                details={"answer_len": len(answer)},
            )
            deferred_closure_evaluation = self._start_follow_up_closure_evaluation(
                response=response,
                answer=answer,
            )
            self._speech_output.flush()
        except InterruptedError:
            self._interrupt_turn("coordinator_interrupted")
            raise
        except Exception:
            self.state_machine.transition(
                StreamingTurnPhase.FAILED,
                reason="turn_execution_failed",
            )
            raise
        finally:
            self._close_speech_output()

        try:
            self._raise_if_interrupted(phase_reason="after_playback")
            self._speech_output.raise_if_error()
            self.state_machine.transition(
                StreamingTurnPhase.PLAYBACK_DRAINING,
                reason="speech_output_drained",
            )
            self.speech_lifecycle.ensure_answering_started()
            self._emit_completion_metrics(
                response=response,
                answer=answer,
                llm_ms=llm_ms,
                first_audio_at=self.speech_lifecycle.first_audio_at,
                first_word_reply=first_word_reply,
                first_word_source=first_word_source,
            )
            keep_listening = self._finish_turn(
                response=response,
                answer=answer,
                llm_ms=llm_ms,
                deferred_closure_evaluation=deferred_closure_evaluation,
            )
            self.state_machine.transition(
                StreamingTurnPhase.COMPLETED,
                reason="turn_completed",
                details={"keep_listening": keep_listening},
            )
        except InterruptedError:
            self._interrupt_turn("coordinator_interrupted")
            raise
        except Exception:
            self.state_machine.transition(
                StreamingTurnPhase.FAILED,
                reason="post_playback_completion_failed",
            )
            raise
        return StreamingTurnExecutionOutcome(
            keep_listening=keep_listening,
            response=response,
            answer=answer,
            llm_ms=llm_ms,
            first_audio_at=self.speech_lifecycle.first_audio_at,
            first_word_reply=first_word_reply,
            first_word_source=first_word_source,
        )

    def _build_speech_output(self) -> InterruptibleSpeechOutput:
        """Create the turn-local speech output bound to this coordinator."""

        chunk_size = max(512, int(self.config.openai_tts_stream_chunk_size))
        self.hooks.trace_event(
            "streaming_speech_output_created",
            kind="queue",
            details={"chunk_size": chunk_size},
        )
        return InterruptibleSpeechOutput(
            tts_provider=self.speech_services.tts_provider,
            player=self.speech_services.player,
            chunk_size=chunk_size,
            segment_boundary=self.speech_services.segment_boundary,
            on_speaking_started=self.speech_lifecycle.on_speaking_started,
            on_first_audio=self.speech_lifecycle.on_first_audio,
            on_preempt=lambda: self.hooks.emit("tts_lane_preempted=true"),
            playback_coordinator=self.speech_services.playback_coordinator,
            should_stop=self.hooks.should_stop,
            trace_event=lambda msg, details=None: self.hooks.trace_event(msg, details=details),
        )

    def _run_lane_plan(
        self,
    ) -> tuple[StreamingToolLoopResult, int, FirstWordReply | None, str]:
        """Execute either the single-lane or dual-lane response plan."""

        lane_plan = self._require_lane_plan()
        self.state_machine.transition(
            StreamingTurnPhase.LANES_RUNNING,
            reason="lane_execution_started",
            details={"dual_lane": lane_plan.is_dual_lane},
        )
        llm_started = time.monotonic()
        if lane_plan.is_dual_lane:
            lane_outcome = self._run_dual_lane()
            response = lane_outcome.response
            first_word_reply = lane_outcome.first_word_reply
            first_word_source = lane_outcome.first_word_source
            if first_word_reply is not None:
                self.hooks.emit(f"first_word_mode={first_word_reply.mode}")
                self.hooks.emit(f"first_word_source={first_word_source}")
                self.hooks.trace_event(
                    "streaming_first_word_selected",
                    kind="decision",
                    details={"mode": first_word_reply.mode, "source": first_word_source},
                )
        else:
            self.processing_feedback.ensure_started()
            if lane_plan.run_single_lane is None:
                raise RuntimeError("single-lane turn plan missing run_single_lane callback")
            self.hooks.trace_event(
                "streaming_single_lane_tool_loop_starting",
                kind="decision",
                details={},
            )
            response = lane_plan.run_single_lane(self._queue_ready_segments)
            first_word_reply = None
            first_word_source = "none"
        llm_ms = int((time.monotonic() - llm_started) * 1000)
        return response, llm_ms, first_word_reply, first_word_source

    def _run_dual_lane(self) -> StreamingTurnLaneOutcome:
        """Execute the coordinated bridge/final-lane path for this turn."""

        lane_plan = self._require_lane_plan()
        if lane_plan.timeout_policy is None:
            raise RuntimeError("dual-lane turn plan missing timeout policy")
        if lane_plan.run_final_lane is None:
            raise RuntimeError("dual-lane turn plan missing run_final_lane callback")
        orchestrator = StreamingTurnOrchestrator(
            timeout_policy=lane_plan.timeout_policy,
            queue_lane_delta=self._queue_lane_segments,
            wait_for_first_audio=self._speech_output.wait_for_first_audio,
            wait_until_idle=self._speech_output.wait_until_idle,
            is_output_idle=lambda: self._speech_output.wait_until_idle(timeout_s=0.0),
            ensure_processing_feedback=self.processing_feedback.ensure_started,
            resume_processing_after_bridge=self._resume_processing_after_bridge_wait,
            stop_final_lane_feedback=self.hooks.stop_search_feedback,
            emit=self.hooks.emit,
            trace_event=self.hooks.trace_event,
            should_stop=self.hooks.should_stop,
            request_final_lane_stop=self.hooks.request_turn_stop,
        )
        return orchestrator.execute(
            prefetched_first_word=lane_plan.prefetched_first_word,
            prefetched_first_word_source=lane_plan.prefetched_first_word_source,
            generate_first_word=lane_plan.generate_first_word,
            bridge_fallback_reply=lane_plan.bridge_fallback_reply,
            run_final_lane=lane_plan.run_final_lane,
            recover_final_lane_response=lane_plan.recover_final_lane_response,
            should_recover_final_lane_error=self._should_recover_final_lane_error,
        )

    def _resume_processing_after_bridge_wait(self) -> None:
        """Move the runtime back to processing once the bridge audio is done."""

        self.speech_lifecycle.resume_processing()
        if self.hooks.is_search_feedback_active():
            return
        self.processing_feedback.ensure_started()

    def _should_recover_final_lane_error(self, exc: BaseException) -> bool:
        """Classify fatal required-remote errors before propagating them.

        The streaming final lane now fails closed for every error. This helper
        only keeps the required-remote classification and telemetry so the
        runtime can distinguish explicit remote blockers from ordinary turn
        failures in the emitted evidence.
        """

        if not isinstance(exc, LongTermRemoteUnavailableError):
            return True
        remote_required = (
            getattr(self.config, "long_term_memory_enabled", False)
            and str(getattr(self.config, "long_term_memory_mode", "") or "").strip().lower()
            == "remote_primary"
            and getattr(self.config, "long_term_memory_remote_required", False)
        )
        if not remote_required:
            return True
        self.hooks.emit(f"final_lane_fatal={type(exc).__name__}")
        self.hooks.trace_event(
            "streaming_final_lane_fatal_remote_error",
            kind="exception",
            level="ERROR",
            details={"error_type": type(exc).__name__},
        )
        return False

    def _require_lane_plan(self) -> StreamingTurnLanePlan:
        """Return the submitted-turn lane plan once it has been built."""

        lane_plan = getattr(self, "_lane_plan", None)
        if lane_plan is None:
            raise RuntimeError("streaming turn lane plan is not available before transcript submission")
        return lane_plan

    def _queue_ready_segments(self, delta: str) -> None:
        """Stream plain text deltas into speech output under coordinator control."""

        self.hooks.trace_event(
            "streaming_text_delta_queued",
            kind="queue",
            details={"delta_len": len(delta)},
        )
        self._speech_output.submit_text_delta(delta)

    def _queue_lane_segments(self, delta) -> None:
        """Stream dual-lane deltas into speech output under coordinator control."""

        self.hooks.trace_event(
            "streaming_lane_delta_queued",
            kind="queue",
            details={
                "delta_len": len(getattr(delta, "text", "") or ""),
                "replace_current": getattr(delta, "replace_current", False),
                "atomic": getattr(delta, "atomic", False),
            },
        )
        self._speech_output.submit_lane_delta(delta)

    def _finalize_answer(self, response: StreamingToolLoopResult) -> str:
        """Finalize the runtime turn once the response text is available."""

        if self.runtime.status.value == "printing":
            self.runtime.resume_answering_after_print()
            self.hooks.emit_status()
            self.speech_lifecycle.answer_started = True
            self.hooks.trace_event("streaming_print_resume_answering", kind="mutation", details={})
        finalize_started = time.monotonic()
        answer = self.runtime.finalize_agent_turn(response.text)
        self.hooks.trace_event(
            "streaming_runtime_finalize_agent_turn_completed",
            kind="metric",
            details={"response_text_len": len(response.text)},
            kpi={"duration_ms": round((time.monotonic() - finalize_started) * 1000.0, 3)},
        )
        tool_calls = tuple(getattr(response, "tool_calls", ()) or ())
        tool_results = tuple(getattr(response, "tool_results", ()) or ())
        record_tool_history = getattr(self.runtime, "record_personality_tool_history", None)
        if callable(record_tool_history):
            tool_history_started = time.monotonic()
            record_tool_history(
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            self.hooks.trace_event(
                "streaming_runtime_tool_history_recorded",
                kind="metric",
                details={
                    "tool_calls": len(tool_calls),
                    "tool_results": len(tool_results),
                },
                kpi={"duration_ms": round((time.monotonic() - tool_history_started) * 1000.0, 3)},
            )
        self.hooks.trace_event(
            "streaming_agent_turn_finalized",
            kind="mutation",
            details={
                "answer_len": len(answer),
                "response_text_len": len(response.text),
                "tool_calls": len(tool_calls),
                "tool_results": len(tool_results),
            },
            kpi={"duration_ms": round((time.monotonic() - finalize_started) * 1000.0, 3)},
        )
        return answer

    def _close_speech_output(self) -> None:
        """Close or abort the speech worker and always stop processing feedback."""

        if self._speech_output is None:
            self.hooks.stop_search_feedback()
            self.processing_feedback.stop()
            return
        close_timeout_s = self._speech_output_close_timeout()
        interrupted = self.hooks.should_stop()
        try:
            if interrupted:
                stopped = self._speech_output.abort(timeout_s=min(0.25, close_timeout_s))
                if not stopped:
                    self.hooks.emit("speech_output_abort_timeout=true")
                self.hooks.trace_event(
                    "streaming_speech_output_abort_path",
                    kind="branch",
                    details={"stopped": stopped, "close_timeout_s": close_timeout_s},
                )
            else:
                self._speech_output.close(timeout_s=close_timeout_s)
                self.hooks.trace_event(
                    "streaming_speech_output_close_path",
                    kind="branch",
                    details={"close_timeout_s": close_timeout_s},
                )
        finally:
            self.hooks.stop_search_feedback()
            self.processing_feedback.stop()
            self.speech_lifecycle.stop_snapshot_heartbeat()

    def _speech_output_close_timeout(self) -> float:
        """Normalize the speech worker shutdown timeout from config."""

        try:
            return max(0.1, float(getattr(self.config, "tts_worker_join_timeout_s", 30.0)))
        except (TypeError, ValueError):
            return 30.0

    def _emit_completion_metrics(
        self,
        *,
        response: StreamingToolLoopResult,
        answer: str,
        llm_ms: int,
        first_audio_at: float | None,
        first_word_reply: FirstWordReply | None,
        first_word_source: str,
    ) -> None:
        """Emit user-visible response and timing telemetry for the turn."""

        del first_word_reply, first_word_source
        self.hooks.emit(f"response={answer}")
        if response.response_id:
            self.hooks.emit(f"agent_response_id={response.response_id}")
        if response.request_id:
            self.hooks.emit(f"agent_request_id={response.request_id}")
        self.hooks.emit(f"agent_tool_rounds={response.rounds}")
        self.hooks.emit(f"agent_tool_calls={len(response.tool_calls)}")
        self.hooks.emit(f"agent_used_web_search={str(response.used_web_search).lower()}")
        self.hooks.emit(f"timing_capture_ms={self.request.capture_ms}")
        self.hooks.emit(f"timing_stt_ms={self.request.stt_ms}")
        self.hooks.emit(f"timing_llm_ms={llm_ms}")
        self.hooks.emit("timing_playback_ms=streamed")
        if first_audio_at is not None:
            self.hooks.emit(
                f"timing_first_audio_ms={int((first_audio_at - self.request.turn_started) * 1000)}"
            )
        self.hooks.emit(
            f"timing_total_ms={int((time.monotonic() - self.request.turn_started) * 1000)}"
        )

    def _start_follow_up_closure_evaluation(
        self,
        *,
        response: StreamingToolLoopResult,
        answer: str,
    ) -> _DeferredFollowUpClosureEvaluation | None:
        """Kick off closure evaluation while audio is still draining."""

        if any(call.name == "end_conversation" for call in response.tool_calls):
            return None
        deferred = _DeferredFollowUpClosureEvaluation(
            evaluate=self.hooks.evaluate_follow_up_closure,
            trace_event=self.hooks.trace_event,
            user_transcript=self.request.transcript,
            assistant_response=answer,
            request_source=self.request.listen_source,
            proactive_trigger=self.request.proactive_trigger,
        )
        deferred.start()
        return deferred

    def _resolve_follow_up_closure_evaluation(
        self,
        *,
        end_conversation: bool,
        answer: str,
        deferred_closure_evaluation: _DeferredFollowUpClosureEvaluation | None,
    ) -> ConversationClosureEvaluation:
        """Collect the closure evaluation that gates follow-up reopening."""

        if end_conversation:
            return ConversationClosureEvaluation()
        if deferred_closure_evaluation is not None:
            return deferred_closure_evaluation.result(
                timeout_s=self._closure_evaluation_join_timeout_s(),
                should_abort=lambda: self.hooks.should_stop()
                or getattr(getattr(self.runtime, "status", None), "value", None) == "error",
            )
        return self.hooks.evaluate_follow_up_closure(
            user_transcript=self.request.transcript,
            assistant_response=answer,
            request_source=self.request.listen_source,
            proactive_trigger=self.request.proactive_trigger,
        )

    def _closure_evaluation_join_timeout_s(self) -> float:
        """Bound the post-playback closure join so turns cannot stall in `answering`."""

        try:
            provider_timeout_s = float(
                getattr(self.config, "conversation_closure_provider_timeout_seconds", 2.0)
            )
        except (TypeError, ValueError):
            provider_timeout_s = 2.0
        return min(15.0, max(0.25, provider_timeout_s) + 0.25)

    def _finish_turn(
        self,
        *,
        response: StreamingToolLoopResult,
        answer: str,
        llm_ms: int,
        deferred_closure_evaluation: _DeferredFollowUpClosureEvaluation | None,
    ) -> bool:
        """Record usage, finish runtime speaking state, and decide follow-up."""

        self.hooks.record_usage(
            request_kind="conversation",
            source="streaming_loop",
            model=response.model,
            response_id=response.response_id,
            request_id=response.request_id,
            used_web_search=response.used_web_search,
            token_usage=response.token_usage,
            transcript=self.request.transcript,
            request_source=self.request.listen_source,
            proactive_trigger=self.request.proactive_trigger,
            tool_rounds=response.rounds,
            tool_calls=len(response.tool_calls),
        )
        end_conversation = any(call.name == "end_conversation" for call in response.tool_calls)
        closure_evaluation = self._resolve_follow_up_closure_evaluation(
            end_conversation=end_conversation,
            answer=answer,
            deferred_closure_evaluation=deferred_closure_evaluation,
        )
        self._raise_if_interrupted(phase_reason="during_closure_eval")
        force_close = end_conversation or self.hooks.apply_follow_up_closure_evaluation(
            evaluation=closure_evaluation,
            request_source=self.request.listen_source,
            proactive_trigger=self.request.proactive_trigger,
        )
        remote_follow_up_rearm_allowed_now = False
        if self.request.allow_follow_up_rearm:
            try:
                remote_follow_up_rearm_allowed_now = bool(
                    self.hooks.follow_up_rearm_allowed_now(self.request.listen_source)
                )
            except Exception as exc:
                self.hooks.emit(f"streaming_follow_up_rearm_check_failed={type(exc).__name__}")
        rearm_follow_up = (
            not force_close
            and self.request.allow_follow_up_rearm
            and remote_follow_up_rearm_allowed_now
        )
        self.hooks.emit(
            f"streaming_follow_up_rearm_snapshot={str(self.request.allow_follow_up_rearm).lower()}"
        )
        self.hooks.emit(
            f"streaming_follow_up_rearm_allowed_now={str(remote_follow_up_rearm_allowed_now).lower()}"
        )
        self.speech_lifecycle.stop_snapshot_heartbeat()
        self._raise_if_interrupted(phase_reason="before_runtime_finish")
        if rearm_follow_up:
            self.hooks.emit("streaming_turn_finish_path=rearm_follow_up")
            self.runtime.rearm_follow_up(request_source="follow_up")
        else:
            self.hooks.emit("streaming_turn_finish_path=finish_speaking")
            self.runtime.finish_speaking()
        self.hooks.emit_status()
        if end_conversation:
            self.hooks.emit("conversation_ended=true")
        elif force_close:
            self.hooks.emit("conversation_follow_up_vetoed=closure")
        self.hooks.trace_decision(
            "streaming_follow_up_decision",
            question="Should the session remain open for a follow-up turn?",
            selected={
                "id": "close" if force_close else "continue",
                "summary": "close session" if force_close else "keep session open",
            },
            options=[
                {"id": "continue", "summary": "Allow follow-up listening"},
                {"id": "close", "summary": "Close session now"},
            ],
            context={
                "end_conversation_tool": end_conversation,
                "listen_source": self.request.listen_source,
                "proactive_trigger": self.request.proactive_trigger,
            },
            guardrails=["closure_evaluator", "end_conversation_tool"],
        )
        self.hooks.trace_event(
            "streaming_turn_completion_finished",
            kind="span_end",
            details={"force_close": force_close, "end_conversation": end_conversation},
            kpi={
                "capture_ms": self.request.capture_ms,
                "stt_ms": self.request.stt_ms,
                "llm_ms": llm_ms,
                "total_ms": round((time.monotonic() - self.request.turn_started) * 1000.0, 3),
            },
        )
        return not force_close

    def _raise_if_interrupted(self, *, phase_reason: str) -> None:
        """Abort the turn immediately when the active turn was interrupted."""

        runtime_status = getattr(getattr(self.runtime, "status", None), "value", None)
        if not self.hooks.should_stop() and runtime_status != "error":
            return
        self.hooks.trace_event(
            f"streaming_turn_interrupted_{phase_reason}",
            kind="branch",
            details={
                "runtime_status": runtime_status,
                "stop_requested": self.hooks.should_stop(),
            },
        )
        self._interrupt_turn(phase_reason)
        raise InterruptedError("streaming turn interrupted")

    def _interrupt_turn(self, reason: str) -> None:
        """Move the state machine into the interrupted terminal phase."""

        if self.state_machine.phase not in {StreamingTurnPhase.INTERRUPTED, StreamingTurnPhase.FAILED}:
            self.state_machine.transition(
                StreamingTurnPhase.INTERRUPTED,
                reason=reason,
            )
        self.hooks.cancel_interrupted_turn()
