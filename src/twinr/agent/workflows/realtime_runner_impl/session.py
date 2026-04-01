# CHANGELOG: 2026-03-27
# BUG-1: Ensure interrupted sessions always reset runtime/listening/orchestrator state across all exit paths.
# BUG-2: Prevent `_conversation_session_lock` leaks when session setup fails before entering the main turn loop.
# BUG-3: Bound housekeeping worker burst loops to stop CPU-hogging hot spins on Raspberry Pi 4 when work stays continuously due.
# SEC-1: Sanitize externally sourced strings before `emit()` and bound transcript sizes to reduce log/event injection and transcript-driven DoS from remote inputs.
# IMP-1: Coalesce required-remote refresh requests so degraded/offline periods do not create refresh storms.
# IMP-2: Add lightweight duplicate-trigger suppression for remote wake events to match 2026 voice-agent debouncing practice.
# IMP-3: Normalize session cancellation flow so direct-text wake turns obey the same interrupt semantics as audio turns.

"""Session and trigger orchestration for the realtime workflow loop."""

# mypy: ignore-errors

from __future__ import annotations

from contextlib import ExitStack
from threading import Event, Thread
import time

from twinr.agent.base_agent.conversation.closure import (
    ConversationClosureDecision,
    ConversationClosureEvaluation,
)
from twinr.agent.base_agent.conversation.turn_controller import StreamingTurnController
from twinr.agent.base_agent.contracts import SpeechToTextProvider
from twinr.agent.workflows import realtime_follow_up, voice_orchestrator_runtime
from twinr.agent.workflows.button_dispatch import ButtonPressDispatcher
from twinr.agent.workflows.remote_transcript_commit import (
    RemoteTranscriptCommit,
    RemoteTranscriptWaitHandle,
)
from twinr.agent.workflows.required_remote_watch import RequiredRemoteDependencyWatch
from twinr.agent.workflows.voice_turn_latency import (
    bind_voice_turn_trace,
    clear_voice_turn_latency,
    mark_voice_turn_commit,
    mark_voice_turn_wake_confirmed,
)
from twinr.hardware.audio import AmbientAudioSampler
from twinr.hardware.buttons import ButtonAction
from twinr.orchestrator.voice_activation import VoiceActivationMatch
from twinr.orchestrator.voice_runtime_intent import VoiceRuntimeIntentContext
from twinr.proactive import SocialTriggerDecision
from twinr.proactive.runtime.gesture_wakeup_lane import GestureWakeupDecision


class TwinrRealtimeSessionMixin:
    """Own the live run loop, session lifecycle, and wake entry points."""

    def _float_config(
        self,
        name: str,
        default: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        raw_value = getattr(getattr(self, "config", None), name, default)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = float(default)
        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        return value

    def _int_config(
        self,
        name: str,
        default: int,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        raw_value = getattr(getattr(self, "config", None), name, default)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = int(default)
        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        return value

    def _external_emit_max_chars(self) -> int:
        return self._int_config(
            "session_external_emit_max_chars",
            160,
            minimum=32,
            maximum=4096,
        )

    def _seed_transcript_max_chars(self) -> int:
        return self._int_config(
            "conversation_seed_transcript_max_chars",
            640,
            minimum=64,
            maximum=8192,
        )

    def _remote_transcript_max_chars(self) -> int:
        return self._int_config(
            "remote_transcript_commit_max_chars",
            1024,
            minimum=64,
            maximum=16384,
        )

    def _housekeeping_burst_limit(self) -> int:
        return self._int_config("housekeeping_burst_limit", 4, minimum=1, maximum=64)

    def _housekeeping_burst_yield_seconds(self) -> float:
        return self._float_config(
            "housekeeping_burst_yield_s",
            0.01,
            minimum=0.0,
            maximum=1.0,
        )

    def _required_remote_refresh_min_interval_seconds(self) -> float:
        return self._float_config(
            "required_remote_dependency_refresh_min_interval_s",
            0.75,
            minimum=0.05,
            maximum=30.0,
        )

    def _external_trigger_cooldown_seconds(self, trigger_kind: str) -> float:
        if trigger_kind == "voice_activation":
            return self._float_config(
                "voice_activation_cooldown_s",
                0.4,
                minimum=0.0,
                maximum=30.0,
            )
        if trigger_kind == "gesture_wakeup":
            return self._float_config(
                "gesture_wakeup_cooldown_s",
                0.4,
                minimum=0.0,
                maximum=30.0,
            )
        return self._float_config(
            "external_trigger_cooldown_s",
            0.4,
            minimum=0.0,
            maximum=30.0,
        )

    def _sanitize_external_text(
        self,
        value: object,
        *,
        max_chars: int | None = None,
        default: str = "",
    ) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
        text = "".join(character if character.isprintable() else " " for character in text)
        text = " ".join(text.split())
        if max_chars is not None and max_chars > 0 and len(text) > max_chars:
            text = text[: max(1, max_chars - 1)].rstrip() + "…"
        return text or default

    def _normalize_transcript(
        self,
        transcript: str | None,
        *,
        max_chars: int,
        trace_reason: str,
    ) -> str | None:
        normalized = self._sanitize_external_text(transcript, max_chars=None, default="")
        if not normalized:
            return None
        if len(normalized) > max_chars:
            self._trace_event(
                "external_transcript_truncated",
                kind="mutation",
                level="WARN",
                details={
                    "reason": trace_reason,
                    "original_length": len(normalized),
                    "max_chars": max_chars,
                },
            )
            normalized = normalized[:max_chars].rstrip()
        return normalized or None

    def _emit_kv(
        self,
        key: str,
        value: object,
        *,
        max_chars: int | None = None,
        default: str = "unknown",
    ) -> None:
        max_len = self._external_emit_max_chars() if max_chars is None else max(1, int(max_chars))
        self.emit(
            f"{key}="
            f"{self._sanitize_external_text(value, max_chars=max_len, default=default)}"
        )

    def _request_required_remote_dependency_refresh_coalesced(
        self,
        *,
        reason: str,
        force: bool = False,
    ) -> bool:
        now = time.monotonic()
        min_interval_s = self._required_remote_refresh_min_interval_seconds()
        with self._get_lock("_required_remote_dependency_refresh_gate_lock"):
            next_allowed_at = float(
                getattr(self, "_required_remote_dependency_refresh_next_allowed_at", 0.0) or 0.0
            )
            if not force and now < next_allowed_at:
                return False
            self._required_remote_dependency_refresh_next_allowed_at = now + min_interval_s
        self._request_required_remote_dependency_refresh()
        self._trace_event(
            "required_remote_refresh_requested",
            kind="io",
            details={
                "reason": reason,
                "force": force,
                "min_interval_s": min_interval_s,
            },
        )
        return True

    def _acquire_external_trigger_token(
        self,
        *,
        trigger_kind: str,
        fingerprint: str,
    ) -> bool:
        cooldown_s = self._external_trigger_cooldown_seconds(trigger_kind)
        if cooldown_s <= 0.0:
            return True
        now = time.monotonic()
        token = f"{trigger_kind}:{fingerprint or '*'}"
        with self._get_lock("_external_trigger_gate_lock"):
            deadlines = getattr(self, "_external_trigger_gate_deadlines", None)
            if not isinstance(deadlines, dict):
                deadlines = {}
                self._external_trigger_gate_deadlines = deadlines
            stale_tokens = [key for key, deadline in deadlines.items() if deadline <= now]
            for stale_token in stale_tokens[:64]:
                deadlines.pop(stale_token, None)
            if len(deadlines) > 1024:
                oldest_tokens = sorted(deadlines.items(), key=lambda item: item[1])[:256]
                for old_token, _ in oldest_tokens:
                    deadlines.pop(old_token, None)
            existing_deadline = float(deadlines.get(token, 0.0) or 0.0)
            if now < existing_deadline:
                return False
            deadlines[token] = now + cooldown_s
        return True

    def run(self, *, duration_s: float | None = None, poll_timeout: float = 0.25) -> int:
        started_at = time.monotonic()
        self._trace_event(
            "workflow_run_loop_entered",
            kind="run_start",
            details={"duration_s": duration_s, "poll_timeout": poll_timeout},
        )
        self._required_remote_dependency_error_active = bool(
            self._remote_dependency_is_required()
            and getattr(getattr(self.runtime, "status", None), "value", None) == "error"
        )
        self._required_remote_dependency_cached_ready = self._required_remote_dependency_current_ready()
        if self._remote_dependency_is_required() and self._required_remote_dependency_uses_watchdog_artifact():
            self._required_remote_dependency_cached_ready = False
        self._required_remote_dependency_watch = RequiredRemoteDependencyWatch(
            interval_s=self._required_remote_dependency_interval_seconds(),
            refresh=lambda force: self._refresh_required_remote_dependency(
                force=force,
                force_sync=False,
            ),
            emit=self.emit,
            trace_event=lambda msg, details=None: self._trace_event(msg, details=details),
        )
        self._required_remote_dependency_watch.start()
        self._trace_event(
            "required_remote_watch_started",
            kind="queue",
            details={"interval_s": self._required_remote_dependency_interval_seconds()},
        )
        self._emit_status(force=True)
        if getattr(getattr(self.runtime, "status", None), "value", None) != "error":
            self._start_startup_boot_sound()
        self._ensure_voice_activation_ack_prefetch_started()
        try:
            safe_poll_timeout = max(0.0, float(poll_timeout))
        except (TypeError, ValueError):
            safe_poll_timeout = 0.25
        with ExitStack() as stack:
            monitor = stack.enter_context(self.button_monitor)
            smart_home_sensor_worker = self._start_smart_home_sensor_worker()
            if self.voice_orchestrator is not None:
                self._prime_voice_orchestrator_waiting_state()
                stack.enter_context(self.voice_orchestrator)
            if self.proactive_monitor is not None:
                stack.enter_context(self.proactive_monitor)
            button_dispatcher = ButtonPressDispatcher(
                handle_press=self.handle_button_press,
                interrupt_current=self._request_active_turn_interrupt,
                emit=self.emit,
                trace_event=lambda msg, details=None: self._trace_event(msg, details=details),
            )
            housekeeping_stop, housekeeping_thread = self._start_idle_housekeeping_worker(
                poll_timeout=safe_poll_timeout
            )
            try:
                while True:
                    try:
                        if duration_s is not None and time.monotonic() - started_at >= duration_s:
                            self._trace_event(
                                "workflow_run_duration_elapsed",
                                kind="run_end",
                                details={"duration_s": duration_s},
                            )
                            return 0
                        if not self._required_remote_dependency_current_ready():
                            refresh_requested = self._request_required_remote_dependency_refresh_coalesced(
                                reason="run_loop_remote_not_ready"
                            )
                            self._trace_event(
                                "workflow_poll_skipped_remote_not_ready",
                                kind="branch",
                                details={
                                    "poll_timeout": safe_poll_timeout,
                                    "refresh_requested": refresh_requested,
                                },
                            )
                            self.sleep(min(0.25, safe_poll_timeout or 0.25))
                            continue
                        event = monitor.poll(timeout=safe_poll_timeout)
                        if event is None:
                            continue
                        if event.action != ButtonAction.PRESSED:
                            self._trace_event(
                                "button_event_ignored_non_press",
                                kind="branch",
                                details={
                                    "name": self._sanitize_external_text(
                                        event.name,
                                        max_chars=32,
                                        default="unknown",
                                    ),
                                    "action": self._sanitize_external_text(
                                        event.action,
                                        max_chars=32,
                                        default="unknown",
                                    ),
                                },
                            )
                            continue
                        button_name = self._sanitize_external_text(
                            event.name,
                            max_chars=32,
                            default="unknown",
                        )
                        self._emit_kv("button", button_name, max_chars=32)
                        self._trace_event(
                            "button_press_received",
                            kind="io",
                            details={"name": button_name, "line_offset": event.line_offset},
                        )
                        self._record_event(
                            "button_pressed",
                            f"Physical button `{button_name}` was pressed.",
                            button=button_name,
                            line_offset=event.line_offset,
                        )
                        button_dispatcher.submit(button_name)
                    except Exception as exc:
                        self._handle_error(exc)
            finally:
                if smart_home_sensor_worker is not None:
                    smart_home_sensor_worker.stop(timeout_s=max(0.5, safe_poll_timeout + 0.25))
                housekeeping_stop.set()
                housekeeping_thread.join(timeout=max(0.5, safe_poll_timeout + 0.25))
                button_dispatcher.close(timeout_s=max(0.5, safe_poll_timeout + 0.25))
                self._required_remote_dependency_watch.stop(
                    timeout_s=max(0.5, safe_poll_timeout + 0.25)
                )
                self._trace_event("workflow_run_loop_exiting", kind="run_end", details={})
                self.workflow_forensics.close()

    def _run_idle_housekeeping_cycle(self) -> bool:
        if not self._required_remote_dependency_current_ready():
            return False
        if self.print_lane.is_busy():
            return False
        if self._maybe_deliver_due_reminder():
            return True
        if self._maybe_run_due_automation():
            return True
        if self._maybe_run_sensor_automation():
            return True
        if self._maybe_run_long_term_memory_proactive():
            return True
        if self._maybe_run_display_reserve_nightly_maintenance():
            return True
        return False

    def _start_idle_housekeeping_worker(self, *, poll_timeout: float) -> tuple[Event, Thread]:
        stop_event = Event()
        idle_sleep_s = max(0.02, min(0.25, poll_timeout or 0.25))
        burst_limit = self._housekeeping_burst_limit()
        burst_yield_s = self._housekeeping_burst_yield_seconds()

        def worker() -> None:
            consecutive_work = 0
            while not stop_event.is_set():
                try:
                    did_work = self._run_idle_housekeeping_cycle()
                except Exception as exc:
                    self._handle_error(exc)
                    did_work = False
                if did_work:
                    consecutive_work += 1
                    if consecutive_work >= burst_limit:
                        consecutive_work = 0
                        stop_event.wait(burst_yield_s)
                    continue
                consecutive_work = 0
                stop_event.wait(idle_sleep_s)

        thread = Thread(target=worker, daemon=True, name="twinr-realtime-housekeeping")
        thread.start()
        return stop_event, thread

    def handle_button_press(self, button_name: str) -> None:
        try:
            button_name = self._sanitize_external_text(button_name, max_chars=32, default="unknown")
            self._trace_event(
                "handle_button_press_entered",
                kind="io",
                details={"button_name": button_name},
            )
            if not self._required_remote_dependency_current_ready():
                self._trace_event(
                    "button_press_blocked_remote_not_ready",
                    kind="invariant",
                    level="WARN",
                    details={"button_name": button_name},
                )
                self._request_required_remote_dependency_refresh_coalesced(
                    reason="button_press_remote_not_ready"
                )
                return
            if button_name == "green":
                self._trace_decision(
                    "button_press_routed",
                    question="Which button workflow path should run?",
                    selected={"id": "green", "summary": "Start or interrupt conversation"},
                    options=[
                        {"id": "green", "summary": "Conversation turn"},
                        {"id": "yellow", "summary": "Print lane"},
                    ],
                    context={"button_name": button_name},
                    guardrails=["remote_ready_required"],
                )
                self._handle_green_turn()
                return
            if button_name == "yellow":
                self._trace_decision(
                    "button_press_routed",
                    question="Which button workflow path should run?",
                    selected={"id": "yellow", "summary": "Print lane"},
                    options=[
                        {"id": "green", "summary": "Conversation turn"},
                        {"id": "yellow", "summary": "Print lane"},
                    ],
                    context={"button_name": button_name},
                    guardrails=["remote_ready_required"],
                )
                self._handle_print_turn()
                return
            raise ValueError(f"Unsupported button: {button_name}")
        except Exception as exc:
            self._handle_error(exc)

    def _voice_quiet_active(self) -> bool:
        voice_quiet_active = getattr(self.runtime, "voice_quiet_active", None)
        if not callable(voice_quiet_active):
            return False
        return bool(voice_quiet_active())

    def handle_voice_activation(self, match: VoiceActivationMatch) -> bool:
        try:
            if not self._required_remote_dependency_current_ready():
                self._request_required_remote_dependency_refresh_coalesced(
                    reason="voice_activation_remote_not_ready"
                )
                return False
            matched_phrase = self._sanitize_external_text(
                match.matched_phrase,
                max_chars=64,
                default="unknown",
            )
            transcript_preview = self._sanitize_external_text(
                match.transcript,
                max_chars=160,
                default="",
            )
            if self._voice_quiet_active():
                self._emit_kv("voice_activation_skipped", "voice_quiet", max_chars=32)
                if getattr(self.runtime.status, "value", None) == "waiting":
                    self._notify_voice_orchestrator_state("waiting", detail="voice_quiet_active")
                self._record_event(
                    "voice_activation_skipped",
                    "Remote voice activation was ignored because Twinr was explicitly staying quiet.",
                    skip_reason="voice_quiet",
                    matched_phrase=matched_phrase,
                )
                return False
            if not self._background_work_allowed():
                skip_reason = (
                    "busy" if self.runtime.status.value != "waiting" else "conversation_active"
                )
                self._emit_kv("voice_activation_skipped", skip_reason, max_chars=32)
                self._record_event(
                    "voice_activation_skipped",
                    "Remote voice activation was detected but Twinr was not idle enough to open a turn.",
                    skip_reason=skip_reason,
                    matched_phrase=matched_phrase,
                )
                return False
            trigger_fingerprint = self._sanitize_external_text(
                matched_phrase,
                max_chars=64,
                default="voice_activation",
            )
            if not self._acquire_external_trigger_token(
                trigger_kind="voice_activation",
                fingerprint=trigger_fingerprint,
            ):
                self._emit_kv("voice_activation_skipped", "duplicate", max_chars=32)
                self._record_event(
                    "voice_activation_skipped",
                    "Remote voice activation was ignored because an equivalent activation was already processed very recently.",
                    skip_reason="duplicate",
                    matched_phrase=matched_phrase,
                )
                return False
            self._emit_kv("voice_activation_phrase", matched_phrase, max_chars=64)
            seed_transcript = self._normalize_transcript(
                match.remaining_text,
                max_chars=self._seed_transcript_max_chars(),
                trace_reason="voice_activation_seed",
            )
            mark_voice_turn_wake_confirmed(self, source="voice_activation")
            if seed_transcript:
                mark_voice_turn_commit(self, source="voice_activation")
            if seed_transcript:
                self.emit("voice_activation_mode=direct_text")
            else:
                self.emit("voice_activation_mode=listen")
            self._record_event(
                "voice_activation_detected",
                "Remote voice activation matched while Twinr was attentive.",
                matched_phrase=matched_phrase,
                transcript_preview=transcript_preview or None,
                remaining_text=seed_transcript or None,
            )
            play_initial_beep = True
            if not seed_transcript:
                self._acknowledge_voice_activation()
                play_initial_beep = False
            result = self._run_conversation_session(
                initial_source="voice_activation",
                seed_transcript=seed_transcript or None,
                play_initial_beep=play_initial_beep,
            )
            if not result:
                clear_voice_turn_latency(self)
            return result
        except Exception as exc:
            clear_voice_turn_latency(self)
            self._handle_error(exc)
            return False

    def handle_gesture_wakeup(self, decision: GestureWakeupDecision) -> bool:
        try:
            if not self._required_remote_dependency_current_ready():
                self._request_required_remote_dependency_refresh_coalesced(
                    reason="gesture_wakeup_remote_not_ready"
                )
                return False
            gesture_name = self._sanitize_external_text(
                decision.trigger_gesture.value,
                max_chars=48,
                default="unknown",
            )
            if not self._background_work_allowed():
                skip_reason = (
                    "busy" if self.runtime.status.value != "waiting" else "conversation_active"
                )
                self._emit_kv("gesture_wakeup_skipped", skip_reason, max_chars=32)
                self._record_event(
                    "gesture_wakeup_skipped",
                    "A visual wake gesture was detected but Twinr was not idle enough to open a turn.",
                    skip_reason=skip_reason,
                    gesture=gesture_name,
                )
                return False
            intent_context = VoiceRuntimeIntentContext.from_sensor_facts(
                getattr(self, "_latest_sensor_observation_facts", None)
            )
            if not intent_context.audio_bias_allowed():
                self._emit_kv("gesture_wakeup_skipped", "context_blocked", max_chars=32)
                self._record_event(
                    "gesture_wakeup_skipped",
                    "A visual wake gesture was ignored because the live person-state did not support a speech turn.",
                    skip_reason="context_blocked",
                    gesture=gesture_name,
                )
                return False
            trigger_fingerprint = self._sanitize_external_text(
                gesture_name,
                max_chars=48,
                default="gesture_wakeup",
            )
            if not self._acquire_external_trigger_token(
                trigger_kind="gesture_wakeup",
                fingerprint=trigger_fingerprint,
            ):
                self._emit_kv("gesture_wakeup_skipped", "duplicate", max_chars=32)
                self._record_event(
                    "gesture_wakeup_skipped",
                    "A visual wake gesture was ignored because an equivalent trigger was already processed very recently.",
                    skip_reason="duplicate",
                    gesture=gesture_name,
                )
                return False
            self._emit_kv("gesture_wakeup_gesture", gesture_name, max_chars=48)
            self.emit("gesture_wakeup_mode=listen")
            request_source = self._sanitize_external_text(
                decision.request_source,
                max_chars=48,
                default="gesture",
            )
            self._record_event(
                "gesture_wakeup_detected",
                "A configured visual wake gesture opened a hands-free listening turn.",
                gesture=gesture_name,
                confidence=round(float(decision.confidence), 4),
                request_source=request_source,
            )
            return self._run_conversation_session(
                initial_source=request_source,
                play_initial_beep=True,
            )
        except Exception as exc:
            self._handle_error(exc)
            return False

    def _handle_green_turn(self) -> None:
        self._run_conversation_session(initial_source="button")

    def _run_proactive_follow_up(self, trigger: SocialTriggerDecision) -> bool:
        try:
            self.emit("proactive_listen=true")
            self._record_event(
                "proactive_listen_started",
                "Twinr opened a hands-free listening window after a proactive prompt.",
                trigger=trigger.trigger_id,
                timeout_s=self.config.conversation_follow_up_timeout_s,
            )
            return self._run_conversation_session(
                initial_source="proactive",
                proactive_trigger=trigger.trigger_id,
            )
        except Exception as exc:
            self._handle_error(exc)
            return False

    def _follow_up_allowed_for_source(self, *, initial_source: str) -> bool:
        return realtime_follow_up.follow_up_allowed_for_source(self, initial_source=initial_source)

    def _follow_up_vetoed_by_closure(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None,
    ) -> bool:
        return realtime_follow_up.follow_up_vetoed_by_closure(
            self,
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
        )

    def _evaluate_follow_up_closure(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None,
    ) -> ConversationClosureEvaluation:
        return realtime_follow_up.evaluate_follow_up_closure(
            self,
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
        )

    def _apply_follow_up_closure_evaluation(
        self,
        *,
        evaluation: ConversationClosureEvaluation,
        request_source: str,
        proactive_trigger: str | None,
    ) -> bool:
        return realtime_follow_up.apply_follow_up_closure_evaluation(
            self,
            evaluation=evaluation,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
        )

    def _emit_closure_decision(self, decision: ConversationClosureDecision) -> None:
        realtime_follow_up.emit_closure_decision(self, decision)

    def _run_conversation_session(
        self,
        *,
        initial_source: str,
        proactive_trigger: str | None = None,
        seed_transcript: str | None = None,
        play_initial_beep: bool = True,
    ) -> bool:
        if not self._required_remote_dependency_current_ready():
            self._trace_event(
                "conversation_session_blocked_remote_not_ready",
                kind="invariant",
                level="WARN",
                details={"initial_source": initial_source, "proactive_trigger": proactive_trigger},
            )
            self._request_required_remote_dependency_refresh_coalesced(
                reason="conversation_session_remote_not_ready"
            )
            return False
        if not self._conversation_session_lock.acquire(blocking=False):
            self._trace_event(
                "conversation_session_lock_busy",
                kind="queue",
                details={"initial_source": initial_source, "proactive_trigger": proactive_trigger},
            )
            self.emit("conversation_session_skipped=busy")
            self._record_event(
                "conversation_session_skipped",
                "A new conversation trigger arrived while another session was still active.",
                initial_source=initial_source,
                proactive_trigger=proactive_trigger,
            )
            return False
        stop_event = Event()
        trace_id = None
        session_active_marked = False
        try:
            self._ensure_workflow_trace_capacity_for_session(
                initial_source=initial_source,
                proactive_trigger=proactive_trigger,
                seed_present=bool(seed_transcript and str(seed_transcript).strip()),
            )
            trace_id = self._new_workflow_trace_id()
            self._workflow_trace_set_active(trace_id)
            bind_voice_turn_trace(self, trace_id=trace_id, initial_source=initial_source)
            self._set_active_turn_stop_event(stop_event)
            with self._get_lock("_background_delivery_transition_lock"):
                self._conversation_session_active = True
            session_active_marked = True
            self._trace_event(
                "conversation_session_started",
                kind="span_start",
                details={
                    "initial_source": initial_source,
                    "proactive_trigger": proactive_trigger,
                    "seed_present": bool(seed_transcript and str(seed_transcript).strip()),
                },
                trace_id=trace_id,
            )
            follow_up = False
            normalized_seed = self._normalize_transcript(
                seed_transcript,
                max_chars=self._seed_transcript_max_chars(),
                trace_reason="conversation_seed",
            )
            if self._active_turn_stop_requested():
                self._cancel_interrupted_turn()
                return False
            if normalized_seed:
                self._trace_event(
                    "conversation_seed_transcript_branch",
                    kind="branch",
                    details={"length": len(normalized_seed), "initial_source": initial_source},
                    trace_id=trace_id,
                )
                text_turn_completed = self._run_single_text_turn(
                    transcript=normalized_seed,
                    listen_source=initial_source,
                    proactive_trigger=proactive_trigger,
                )
                if self._active_turn_stop_requested():
                    self._cancel_interrupted_turn()
                    return False
                if text_turn_completed:
                    if self._voice_orchestrator_handles_follow_up(initial_source=initial_source):
                        return True
                    if self._follow_up_allowed_for_source(initial_source=initial_source):
                        follow_up = True
                    else:
                        return True
                else:
                    return True
            while True:
                if self._active_turn_stop_requested():
                    self._cancel_interrupted_turn()
                    return False
                listening_window = self.runtime.listening_window(
                    initial_source=initial_source,
                    follow_up=follow_up,
                )
                self._trace_event(
                    "conversation_listening_window_selected",
                    kind="decision",
                    details={
                        "initial_source": initial_source,
                        "follow_up": follow_up,
                        "pause_ms": listening_window.speech_pause_ms,
                        "start_timeout_s": listening_window.start_timeout_s,
                        "pause_grace_ms": listening_window.pause_grace_ms,
                    },
                    trace_id=trace_id,
                )
                audio_turn_completed = self._run_single_audio_turn(
                    initial_source=initial_source,
                    follow_up=follow_up,
                    listening_window=listening_window,
                    listen_source="follow_up" if follow_up else initial_source,
                    proactive_trigger=None if follow_up else proactive_trigger,
                    speech_start_chunks=self._listening_window_speech_start_chunks(
                        initial_source=initial_source,
                        follow_up=follow_up,
                    ),
                    ignore_initial_ms=self._listening_window_ignore_initial_ms(
                        initial_source=initial_source,
                        follow_up=follow_up,
                    ),
                    timeout_emit_key=self._listening_timeout_emit_key(
                        initial_source=initial_source,
                        follow_up=follow_up,
                    ),
                    timeout_message=self._listening_timeout_message(
                        initial_source=initial_source,
                        follow_up=follow_up,
                    ),
                    play_initial_beep=True if follow_up else play_initial_beep,
                )
                if self._active_turn_stop_requested():
                    self._cancel_interrupted_turn()
                    return False
                if audio_turn_completed:
                    if self._voice_orchestrator_handles_follow_up(initial_source=initial_source):
                        return True
                    if self._follow_up_allowed_for_source(initial_source=initial_source):
                        self._trace_event(
                            "conversation_follow_up_continues",
                            kind="branch",
                            details={"initial_source": initial_source},
                            trace_id=trace_id,
                        )
                        follow_up = True
                        continue
                return True
        finally:
            if session_active_marked:
                with self._get_lock("_background_delivery_transition_lock"):
                    self._conversation_session_active = False
            self._clear_active_turn_stop_event(stop_event)
            self._conversation_session_lock.release()
            try:
                self._trace_event(
                    "conversation_session_finished",
                    kind="span_end",
                    details={"initial_source": initial_source},
                    trace_id=trace_id,
                )
            finally:
                self._workflow_trace_set_active(None)

    def _set_active_turn_stop_event(self, stop_event: Event) -> None:
        with self._active_turn_stop_lock:
            self._active_turn_stop_event = stop_event
            self._active_turn_stop_reason = None
        self._trace_event(
            "active_turn_stop_event_set",
            kind="mutation",
            details={"stop_event_id": id(stop_event)},
        )

    def _clear_active_turn_stop_event(self, stop_event: Event) -> None:
        with self._active_turn_stop_lock:
            if self._active_turn_stop_event is stop_event:
                self._active_turn_stop_event = None
                self._active_turn_stop_reason = None
        self._trace_event(
            "active_turn_stop_event_cleared",
            kind="mutation",
            details={"stop_event_id": id(stop_event)},
        )

    def _active_turn_stop_requested(self) -> bool:
        with self._active_turn_stop_lock:
            stop_event = self._active_turn_stop_event
        return bool(stop_event is not None and stop_event.is_set())

    def _signal_active_turn_stop(self, reason: str) -> None:
        normalized_reason = self._sanitize_external_text(reason, max_chars=48, default="unknown")
        with self._active_turn_stop_lock:
            stop_event = self._active_turn_stop_event
        if stop_event is None or stop_event.is_set():
            self._trace_event(
                "active_turn_stop_signal_ignored",
                kind="branch",
                details={"reason": normalized_reason, "has_stop_event": stop_event is not None},
            )
            return
        with self._active_turn_stop_lock:
            if self._active_turn_stop_event is stop_event:
                self._active_turn_stop_reason = normalized_reason
        stop_event.set()
        self._trace_event(
            "active_turn_stop_signaled",
            kind="mutation",
            details={"reason": normalized_reason, "stop_event_id": id(stop_event)},
        )

    def _request_active_turn_interrupt(self, source: str = "button") -> bool:
        normalized_source = self._sanitize_external_text(source, max_chars=48, default="button")
        with self._active_turn_stop_lock:
            stop_event = self._active_turn_stop_event
        if stop_event is None or stop_event.is_set():
            self._trace_event(
                "turn_interrupt_request_ignored",
                kind="branch",
                details={"source": normalized_source, "has_stop_event": stop_event is not None},
            )
            return False
        with self._active_turn_stop_lock:
            if self._active_turn_stop_event is stop_event:
                self._active_turn_stop_reason = normalized_source
        stop_event.set()
        self._best_effort_stop_player()
        self._stop_working_feedback()
        self._emit_kv("turn_interrupt_requested", normalized_source, max_chars=48)
        self._trace_event(
            "turn_interrupt_requested",
            kind="mutation",
            details={"source": normalized_source, "stop_event_id": id(stop_event)},
        )
        return True

    def _set_answer_interrupt_event(self, interrupt_event: Event) -> None:
        with self._answer_interrupt_lock:
            self._answer_interrupt_event = interrupt_event
        self._trace_event(
            "answer_interrupt_event_set",
            kind="mutation",
            details={"interrupt_event_id": id(interrupt_event)},
        )

    def _clear_answer_interrupt_event(self, interrupt_event: Event) -> None:
        with self._answer_interrupt_lock:
            if self._answer_interrupt_event is interrupt_event:
                self._answer_interrupt_event = None
        self._trace_event(
            "answer_interrupt_event_cleared",
            kind="mutation",
            details={"interrupt_event_id": id(interrupt_event)},
        )

    def _request_answer_interrupt(self, source: str) -> bool:
        with self._answer_interrupt_lock:
            interrupt_event = self._answer_interrupt_event
        normalized_source = self._sanitize_external_text(source, max_chars=48, default="unknown")
        if interrupt_event is None or interrupt_event.is_set():
            return self._request_active_turn_interrupt(normalized_source)
        interrupt_event.set()
        self._best_effort_stop_player()
        self._emit_kv("answer_interrupt_requested", normalized_source, max_chars=48)
        self._trace_event(
            "answer_interrupt_requested",
            kind="mutation",
            details={"source": normalized_source, "interrupt_event_id": id(interrupt_event)},
        )
        return True

    def _cancel_interrupted_turn(self) -> None:
        with self._active_turn_stop_lock:
            reason = self._active_turn_stop_reason
        runtime_status = getattr(getattr(self, "runtime", None), "status", None)
        runtime_status_value = getattr(runtime_status, "value", None)
        if reason == "required_remote" or runtime_status_value == "error":
            self._emit_status(force=True)
            self._trace_event(
                "turn_interrupt_preserved_runtime_error",
                kind="branch",
                details={
                    "reason": reason or "unknown",
                    "runtime_status": runtime_status_value,
                },
            )
            return
        self.runtime.cancel_listening()
        self._emit_status(force=True)
        self.emit("turn_interrupted=true")
        self._notify_voice_orchestrator_state("waiting")
        self._trace_event("turn_interrupt_canceled_current_turn", kind="branch", details={})

    def _voice_orchestrator_handles_follow_up(self, *, initial_source: str) -> bool:
        return voice_orchestrator_runtime.voice_orchestrator_handles_follow_up(
            self,
            initial_source=initial_source,
        )

    def _voice_orchestrator_follow_up_mode(self, *, initial_source: str) -> str:
        return voice_orchestrator_runtime.voice_orchestrator_follow_up_mode(
            self,
            initial_source=initial_source,
        )

    def _notify_voice_orchestrator_state(
        self,
        state: str,
        *,
        detail: str | None = None,
        follow_up_allowed: bool = False,
    ) -> None:
        voice_orchestrator_runtime.notify_voice_orchestrator_state(
            self,
            state,
            detail=detail,
            follow_up_allowed=follow_up_allowed,
        )

    def _prime_voice_orchestrator_waiting_state(self) -> None:
        voice_orchestrator_runtime.prime_voice_orchestrator_waiting_state(self)

    def _refresh_voice_orchestrator_sensor_context(self) -> None:
        voice_orchestrator_runtime.refresh_voice_orchestrator_sensor_context(self)

    def handle_remote_transcript_committed(self, transcript: str, source: str) -> bool:
        normalized_transcript = self._normalize_transcript(
            transcript,
            max_chars=self._remote_transcript_max_chars(),
            trace_reason="remote_transcript_commit",
        )
        return voice_orchestrator_runtime.handle_remote_transcript_committed(
            self,
            normalized_transcript or "",
            self._sanitize_external_text(source, max_chars=48, default="remote"),
        )

    def handle_remote_follow_up_closed(self, reason: str) -> None:
        voice_orchestrator_runtime.handle_remote_follow_up_closed(
            self,
            self._sanitize_external_text(reason, max_chars=48, default="unknown"),
        )

    def _voice_orchestrator_owns_live_listening(self) -> bool:
        return voice_orchestrator_runtime.voice_orchestrator_owns_live_listening(self)

    def _pause_voice_orchestrator_capture(self, *, reason: str) -> None:
        voice_orchestrator_runtime.pause_voice_orchestrator_capture(self, reason=reason)

    def _resume_voice_orchestrator_capture(self, *, reason: str) -> None:
        voice_orchestrator_runtime.resume_voice_orchestrator_capture(self, reason=reason)

    def _begin_remote_transcript_wait(self, *, source: str) -> RemoteTranscriptWaitHandle | None:
        return voice_orchestrator_runtime.begin_remote_transcript_wait(
            self,
            source=self._sanitize_external_text(source, max_chars=48, default="remote"),
        )

    def _wait_for_remote_transcript_commit(
        self,
        *,
        wait_handle: RemoteTranscriptWaitHandle,
        timeout_s: float,
        initial_source: str,
        follow_up: bool,
        listen_source: str,
        timeout_emit_key: str,
        timeout_message: str,
    ) -> RemoteTranscriptCommit | None:
        return voice_orchestrator_runtime.wait_for_remote_transcript_commit(
            self,
            wait_handle=wait_handle,
            timeout_s=timeout_s,
            initial_source=initial_source,
            follow_up=follow_up,
            listen_source=listen_source,
            timeout_emit_key=timeout_emit_key,
            timeout_message=timeout_message,
        )

    def _recorder_sample_rate(self) -> int:
        return int(
            getattr(
                self.recorder,
                "sample_rate",
                self.config.openai_realtime_input_sample_rate,
            )
        )

    def _turn_controller_conversation(self) -> tuple[tuple[str, str], ...]:
        return self.turn_guidance_runtime.controller_conversation()

    def _build_streaming_turn_controller(self) -> StreamingTurnController | None:
        return self.turn_guidance_runtime.build_streaming_turn_controller()

    def _turn_guidance_messages(self, turn_label: str | None) -> tuple[tuple[str, str], ...]:
        return self.turn_guidance_runtime.guidance_messages(turn_label)

    def _conversation_context_for_turn_label(
        self, turn_label: str | None
    ) -> tuple[tuple[str, str], ...]:
        return self.turn_guidance_runtime.conversation_context_for_turn_label(turn_label)

    def _interrupt_stt_provider(self) -> SpeechToTextProvider | None:
        if self._voice_orchestrator_owns_live_listening():
            return None
        provider = self.turn_stt_provider or self.stt_provider
        if provider is None or not callable(getattr(provider, "transcribe", None)):
            return None
        return provider

    def _ambient_sampler(self) -> AmbientAudioSampler:
        sampler = self._ambient_audio_sampler
        if sampler is None:
            sampler = AmbientAudioSampler.from_config(self.config)
            self._ambient_audio_sampler = sampler
        return sampler
