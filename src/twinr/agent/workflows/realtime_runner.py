from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Callable
import time
from zoneinfo import ZoneInfo

from twinr.automations import (
    AutomationAction,
    AutomationDefinition,
    TimeAutomationTrigger,
    build_sensor_trigger,
    describe_sensor_trigger,
    supported_sensor_trigger_kinds,
)
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.hardware.audio import SilenceDetectedRecorder, WaveAudioPlayer
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.buttons import ButtonAction, configured_button_monitor
from twinr.memory.context_store import PromptContextStore
from twinr.memory.reminders import format_due_label
from twinr.hardware.printer import RawReceiptPrinter
from twinr.hardware.voice_profile import VoiceProfileMonitor
from twinr.ops import TwinrUsageStore
from twinr.proactive import SocialTriggerDecision, build_default_proactive_monitor
from twinr.providers.openai.backend import (
    REMINDER_DELIVERY_INSTRUCTIONS,
    OpenAIBackend,
    OpenAIImageInput,
)
from twinr.providers.openai.realtime import OpenAIRealtimeSession

_SEARCH_FEEDBACK_TONE_PATTERN: tuple[tuple[int, int], ...] = (
    (784, 90),
    (1175, 70),
    (988, 80),
    (1318, 60),
)


def _default_emit(line: str) -> None:
    print(line, flush=True)


class TwinrRealtimeHardwareLoop:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        runtime: TwinrRuntime | None = None,
        realtime_session: OpenAIRealtimeSession | None = None,
        print_backend: OpenAIBackend | None = None,
        button_monitor=None,
        recorder: SilenceDetectedRecorder | None = None,
        player: WaveAudioPlayer | None = None,
        printer: RawReceiptPrinter | None = None,
        camera: V4L2StillCamera | None = None,
        usage_store: TwinrUsageStore | None = None,
        voice_profile_monitor: VoiceProfileMonitor | None = None,
        proactive_monitor=None,
        emit: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        error_reset_seconds: float = 1.0,
    ) -> None:
        self.config = config
        self.runtime = runtime or TwinrRuntime(config=config)
        self.print_backend = print_backend or OpenAIBackend(config=config)
        self.button_monitor = button_monitor or configured_button_monitor(config)
        self.recorder = recorder or SilenceDetectedRecorder(
            device=config.audio_input_device,
            sample_rate=config.openai_realtime_input_sample_rate,
            channels=config.audio_channels,
            chunk_ms=config.audio_chunk_ms,
            preroll_ms=config.audio_preroll_ms,
            speech_threshold=config.audio_speech_threshold,
            speech_start_chunks=config.audio_speech_start_chunks,
            start_timeout_s=config.audio_start_timeout_s,
            max_record_seconds=config.audio_max_record_seconds,
        )
        self.player = player or WaveAudioPlayer.from_config(config)
        self.printer = printer or RawReceiptPrinter.from_config(config)
        self.camera = camera or V4L2StillCamera.from_config(config)
        self.context_store = PromptContextStore.from_config(config)
        self.usage_store = usage_store or TwinrUsageStore.from_config(config)
        self.voice_profile_monitor = voice_profile_monitor or VoiceProfileMonitor.from_config(config)
        self._camera_lock = Lock()
        self._audio_lock = Lock()
        self._current_turn_audio_pcm: bytes | None = None
        self.realtime_session = realtime_session or OpenAIRealtimeSession(
            config=config,
            tool_handlers={
                "print_receipt": self._handle_print_tool_call,
                "search_live_info": self._handle_search_tool_call,
                "schedule_reminder": self._handle_schedule_reminder_tool_call,
                "list_automations": self._handle_list_automations_tool_call,
                "create_time_automation": self._handle_create_time_automation_tool_call,
                "create_sensor_automation": self._handle_create_sensor_automation_tool_call,
                "update_time_automation": self._handle_update_time_automation_tool_call,
                "update_sensor_automation": self._handle_update_sensor_automation_tool_call,
                "delete_automation": self._handle_delete_automation_tool_call,
                "remember_memory": self._handle_remember_memory_tool_call,
                "update_user_profile": self._handle_update_user_profile_tool_call,
                "update_personality": self._handle_update_personality_tool_call,
                "enroll_voice_profile": self._handle_enroll_voice_profile_tool_call,
                "get_voice_profile_status": self._handle_get_voice_profile_status_tool_call,
                "reset_voice_profile": self._handle_reset_voice_profile_tool_call,
                "inspect_camera": self._handle_inspect_camera_tool_call,
                "end_conversation": self._handle_end_conversation_tool_call,
            },
        )
        self.emit = emit or _default_emit
        self.sleep = sleep
        self.error_reset_seconds = error_reset_seconds
        self._last_status: str | None = None
        self._last_print_request_at: float | None = None
        self._next_reminder_check_at: float = 0.0
        self._next_automation_check_at: float = 0.0
        self._conversation_session_active = False
        self._sensor_observation_queue: Queue[tuple[dict[str, object], tuple[str, ...]]] = Queue()
        self.proactive_monitor = proactive_monitor or build_default_proactive_monitor(
            config=config,
            runtime=self.runtime,
            backend=self.print_backend,
            camera=self.camera,
            camera_lock=self._camera_lock,
            audio_lock=self._audio_lock,
            trigger_handler=self.handle_social_trigger,
            idle_predicate=self._background_work_allowed,
            observation_handler=self.handle_sensor_observation,
            emit=self.emit,
        )

    def run(self, *, duration_s: float | None = None, poll_timeout: float = 0.25) -> int:
        started_at = time.monotonic()
        self._emit_status(force=True)
        with ExitStack() as stack:
            monitor = stack.enter_context(self.button_monitor)
            if self.proactive_monitor is not None:
                stack.enter_context(self.proactive_monitor)
            while True:
                if duration_s is not None and time.monotonic() - started_at >= duration_s:
                    return 0
                event = monitor.poll(timeout=poll_timeout)
                if event is None:
                    if self._maybe_deliver_due_reminder():
                        continue
                    if self._maybe_run_due_automation():
                        continue
                    if self._maybe_run_sensor_automation():
                        continue
                    continue
                if event.action != ButtonAction.PRESSED:
                    continue
                self.emit(f"button={event.name}")
                self._record_event(
                    "button_pressed",
                    f"Physical button `{event.name}` was pressed.",
                    button=event.name,
                    line_offset=event.line_offset,
                )
                self.handle_button_press(event.name)

    def handle_button_press(self, button_name: str) -> None:
        try:
            if button_name == "green":
                self._handle_green_turn()
                return
            if button_name == "yellow":
                self._handle_print_turn()
                return
            raise ValueError(f"Unsupported button: {button_name}")
        except Exception as exc:
            self._handle_error(exc)

    def handle_social_trigger(self, trigger: SocialTriggerDecision) -> bool:
        if not self._background_work_allowed():
            skip_reason = "busy" if self.runtime.status.value != "waiting" else "conversation_active"
            self.emit(f"social_trigger_skipped={skip_reason}")
            self._record_event(
                "social_trigger_skipped",
                "Social trigger prompt was skipped because Twinr was not idle for background prompts.",
                trigger=trigger.trigger_id,
                reason=trigger.reason,
                prompt=trigger.prompt,
                priority=int(trigger.priority),
                skip_reason=skip_reason,
            )
            return False

        phrase_response = None
        prompt_text = trigger.prompt
        try:
            phrase_response = self.print_backend.phrase_proactive_prompt_with_metadata(
                trigger_id=trigger.trigger_id,
                reason=trigger.reason,
                default_prompt=trigger.prompt,
                priority=int(trigger.priority),
                conversation=self.runtime.conversation_context(),
                recent_prompts=self._recent_proactive_prompts(trigger_id=trigger.trigger_id),
            )
            candidate_prompt = phrase_response.text.strip()
            if candidate_prompt:
                prompt_text = candidate_prompt
            else:
                self.emit("social_prompt_fallback=empty_phrase")
        except Exception as exc:
            self.emit("social_prompt_fallback=default")
            self.emit(f"social_prompt_phrase_error={exc}")
            self._record_event(
                "social_trigger_phrase_fallback",
                "Twinr fell back to the default proactive prompt after proactive phrasing failed.",
                level="warning",
                trigger=trigger.trigger_id,
                error=str(exc),
            )

        prompt = self.runtime.begin_proactive_prompt(prompt_text)
        self._emit_status(force=True)
        tts_started = time.monotonic()
        first_audio_at: list[float | None] = [None]

        def mark_first_chunk():
            for chunk in self.print_backend.synthesize_stream(prompt):
                if first_audio_at[0] is None:
                    first_audio_at[0] = time.monotonic()
                yield chunk

        with self._audio_lock:
            self.player.play_wav_chunks(mark_first_chunk())
        tts_ms = int((time.monotonic() - tts_started) * 1000)
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"social_trigger={trigger.trigger_id}")
        self.emit(f"social_trigger_priority={int(trigger.priority)}")
        self.emit(f"social_prompt={prompt}")
        if phrase_response is not None:
            if phrase_response.response_id:
                self.emit(f"social_response_id={phrase_response.response_id}")
            if phrase_response.request_id:
                self.emit(f"social_request_id={phrase_response.request_id}")
            self._record_usage(
                request_kind="proactive_prompt",
                source="realtime_loop",
                model=phrase_response.model,
                response_id=phrase_response.response_id,
                request_id=phrase_response.request_id,
                used_web_search=False,
                token_usage=phrase_response.token_usage,
                proactive_trigger=trigger.trigger_id,
            )
        self.emit(f"timing_social_tts_ms={tts_ms}")
        if first_audio_at[0] is not None:
            self.emit(f"timing_social_first_audio_ms={int((first_audio_at[0] - tts_started) * 1000)}")
        self._record_event(
            "social_trigger_prompted",
            "Twinr spoke a proactive social prompt.",
            trigger=trigger.trigger_id,
            reason=trigger.reason,
            priority=int(trigger.priority),
            prompt=prompt,
            default_prompt=trigger.prompt,
        )
        self._run_proactive_follow_up(trigger)
        return True

    def handle_sensor_observation(self, facts: dict[str, object], event_names: tuple[str, ...]) -> None:
        self._sensor_observation_queue.put((dict(facts), tuple(event_names)))

    def _handle_green_turn(self) -> None:
        self._run_conversation_session(initial_source="button")

    def _run_proactive_follow_up(self, trigger: SocialTriggerDecision) -> None:
        self.emit("proactive_listen=true")
        self._record_event(
            "proactive_listen_started",
            "Twinr opened a hands-free listening window after a proactive prompt.",
            trigger=trigger.trigger_id,
            timeout_s=self.config.conversation_follow_up_timeout_s,
        )
        self._run_conversation_session(
            initial_source="proactive",
            proactive_trigger=trigger.trigger_id,
        )

    def _run_conversation_session(
        self,
        *,
        initial_source: str,
        proactive_trigger: str | None = None,
    ) -> None:
        previous_session_state = self._conversation_session_active
        self._conversation_session_active = True
        try:
            follow_up = False
            while True:
                if self._run_single_audio_turn(
                    listen_source="follow_up" if follow_up else initial_source,
                    proactive_trigger=None if follow_up else proactive_trigger,
                    start_timeout_s=self._listening_window_timeout_s(initial_source=initial_source, follow_up=follow_up),
                    speech_start_chunks=self._listening_window_speech_start_chunks(
                        initial_source=initial_source,
                        follow_up=follow_up,
                    ),
                    ignore_initial_ms=self._listening_window_ignore_initial_ms(
                        initial_source=initial_source,
                        follow_up=follow_up,
                    ),
                    timeout_emit_key=self._listening_timeout_emit_key(initial_source=initial_source, follow_up=follow_up),
                    timeout_message=self._listening_timeout_message(initial_source=initial_source, follow_up=follow_up),
                ):
                    if self.config.conversation_follow_up_enabled:
                        follow_up = True
                        continue
                return
        finally:
            self._conversation_session_active = previous_session_state

    def _run_single_audio_turn(
        self,
        *,
        listen_source: str,
        proactive_trigger: str | None,
        start_timeout_s: float | None,
        speech_start_chunks: int | None,
        ignore_initial_ms: int,
        timeout_emit_key: str,
        timeout_message: str,
    ) -> bool:
        turn_started = time.monotonic()
        self._play_listen_beep()
        if listen_source == "button":
            self.runtime.press_green_button()
        else:
            self.runtime.begin_listening(
                request_source=listen_source,
                proactive_trigger=proactive_trigger,
            )
        self._emit_status(force=True)

        capture_started = time.monotonic()
        try:
            with self._audio_lock:
                audio_pcm = self.recorder.record_pcm_until_pause_with_options(
                    pause_ms=self.config.speech_pause_ms,
                    start_timeout_s=start_timeout_s,
                    speech_start_chunks=speech_start_chunks,
                    ignore_initial_ms=ignore_initial_ms,
                )
        except RuntimeError as exc:
            if not self._is_no_speech_timeout(exc):
                raise
            self.runtime.cancel_listening()
            self._emit_status(force=True)
            self.emit(f"{timeout_emit_key}=true")
            self._record_event("listen_timeout", timeout_message, request_source=listen_source)
            return False
        capture_ms = int((time.monotonic() - capture_started) * 1000)
        self._update_voice_assessment_from_pcm(audio_pcm)

        self.runtime.submit_transcript("[voice input]")
        self._emit_status(force=True)

        audio_chunks: Queue[bytes | None] = Queue()
        playback_error: list[Exception] = []
        first_audio_at: list[float | None] = [None]
        answer_started = False

        def begin_answering() -> None:
            nonlocal answer_started
            if answer_started:
                return
            self.runtime.begin_answering()
            self._emit_status(force=True)
            answer_started = True

        def playback_generator():
            while True:
                chunk = audio_chunks.get()
                if chunk is None:
                    return
                yield chunk

        def playback_worker() -> None:
            try:
                self.player.play_pcm16_chunks(
                    playback_generator(),
                    sample_rate=self.config.openai_realtime_input_sample_rate,
                    channels=self.config.audio_channels,
                )
            except Exception as exc:
                playback_error.append(exc)

        worker = Thread(target=playback_worker, daemon=True)
        worker.start()

        def on_audio_chunk(chunk: bytes) -> None:
            begin_answering()
            if first_audio_at[0] is None:
                first_audio_at[0] = time.monotonic()
            audio_chunks.put(chunk)

        def on_output_text_delta(_delta: str) -> None:
            begin_answering()

        realtime_started = time.monotonic()
        self._current_turn_audio_pcm = audio_pcm
        try:
            with self.realtime_session:
                turn = self.realtime_session.run_audio_turn(
                    audio_pcm,
                    conversation=self.runtime.provider_conversation_context(),
                    on_audio_chunk=on_audio_chunk,
                    on_output_text_delta=on_output_text_delta,
                )
        finally:
            self._current_turn_audio_pcm = None
            audio_chunks.put(None)
        worker.join()
        realtime_ms = int((time.monotonic() - realtime_started) * 1000)
        if playback_error:
            raise playback_error[0]

        self.runtime.last_transcript = turn.transcript
        self.emit(f"transcript={turn.transcript}")
        if not answer_started:
            begin_answering()
        answer = self.runtime.finalize_agent_turn(turn.response_text)
        self.emit(f"response={answer}")
        if turn.response_id:
            self.emit(f"openai_response_id={turn.response_id}")
        self._record_usage(
            request_kind="realtime_conversation",
            source="realtime_loop",
            model=turn.model or self.config.openai_realtime_model,
            response_id=turn.response_id,
            request_id=None,
            used_web_search=None,
            token_usage=turn.token_usage,
            transcript=turn.transcript,
        )
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        if turn.end_conversation:
            self.emit("conversation_ended=true")
        self.emit(f"timing_capture_ms={capture_ms}")
        self.emit(f"timing_realtime_ms={realtime_ms}")
        self.emit("timing_playback_ms=streamed")
        if first_audio_at[0] is not None:
            self.emit(f"timing_first_audio_ms={int((first_audio_at[0] - turn_started) * 1000)}")
        self.emit(f"timing_total_ms={int((time.monotonic() - turn_started) * 1000)}")
        return not turn.end_conversation

    def _listening_window_timeout_s(self, *, initial_source: str, follow_up: bool) -> float | None:
        if initial_source == "button" and not follow_up:
            return None
        return self.config.conversation_follow_up_timeout_s

    def _listening_window_speech_start_chunks(self, *, initial_source: str, follow_up: bool) -> int | None:
        if initial_source == "button" and not follow_up:
            return None
        return self.config.audio_follow_up_speech_start_chunks

    def _listening_window_ignore_initial_ms(self, *, initial_source: str, follow_up: bool) -> int:
        if initial_source == "button" and not follow_up:
            return 0
        return self.config.audio_follow_up_ignore_ms

    def _listening_timeout_emit_key(self, *, initial_source: str, follow_up: bool) -> str:
        if follow_up:
            return "follow_up_timeout"
        if initial_source == "proactive":
            return "proactive_listen_timeout"
        return "listen_timeout"

    def _listening_timeout_message(self, *, initial_source: str, follow_up: bool) -> str:
        if follow_up:
            return "Follow-up listening window expired."
        if initial_source == "proactive":
            return "Hands-free listening window after a proactive prompt expired."
        return "Listening timed out before speech started."

    def _handle_print_turn(self) -> None:
        if self._is_print_cooldown_active():
            self.emit("print_skipped=cooldown")
            self._record_event("print_skipped", "Print request ignored because cooldown is active.")
            return
        response_to_print = self.runtime.press_yellow_button()
        self._emit_status(force=True)

        composed = self.print_backend.compose_print_job_with_metadata(
            conversation=self.runtime.provider_conversation_context(),
            focus_hint=self.runtime.last_transcript,
            direct_text=response_to_print,
            request_source="button",
        )
        print_job = self.printer.print_text(composed.text)
        self.emit(f"print_text={composed.text}")
        if composed.response_id:
            self.emit(f"print_response_id={composed.response_id}")
        self._record_usage(
            request_kind="print",
            source="realtime_loop",
            model=composed.model,
            response_id=composed.response_id,
            request_id=composed.request_id,
            used_web_search=False,
            token_usage=composed.token_usage,
            request_source="button",
        )
        if print_job:
            self.emit(f"print_job={print_job}")
        self._record_event(
            "print_job_sent",
            "Print job was sent to the configured printer.",
            queue=self.config.printer_queue,
            job=print_job,
        )

        self.runtime.finish_printing()
        self._emit_status(force=True)
        self._last_print_request_at = time.monotonic()

    def _handle_print_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        focus_hint = str(arguments.get("focus_hint", "")).strip()
        direct_text = str(arguments.get("text", "")).strip()
        if not focus_hint and not direct_text:
            raise RuntimeError("print_receipt requires `focus_hint` or `text`")

        self.runtime.maybe_begin_tool_print()
        self._emit_status(force=True)

        composed = self.print_backend.compose_print_job_with_metadata(
            conversation=self.runtime.provider_conversation_context(),
            focus_hint=focus_hint or None,
            direct_text=direct_text or None,
            request_source="tool",
        )
        print_job = self.printer.print_text(composed.text)
        self.emit("print_tool_call=true")
        self.emit(f"print_text={composed.text}")
        self._record_usage(
            request_kind="print",
            source="realtime_tool",
            model=composed.model,
            response_id=composed.response_id,
            request_id=composed.request_id,
            used_web_search=False,
            token_usage=composed.token_usage,
            request_source="tool",
        )
        if print_job:
            self.emit(f"print_job={print_job}")
        self._last_print_request_at = time.monotonic()
        return {
            "status": "printed",
            "text": composed.text,
            "job": print_job,
        }

    def _handle_end_conversation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        reason = str(arguments.get("reason", "")).strip()
        if reason:
            self.emit(f"end_conversation_reason={reason}")
        return {
            "status": "ending",
            "reason": reason or "user_requested_stop",
        }

    def _handle_schedule_reminder_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        due_at = str(arguments.get("due_at", "")).strip()
        summary = str(arguments.get("summary", "")).strip()
        details = str(arguments.get("details", "")).strip()
        kind = str(arguments.get("kind", "")).strip() or "reminder"
        original_request = str(arguments.get("original_request", "")).strip()
        if not due_at or not summary:
            raise RuntimeError("schedule_reminder requires `due_at` and `summary`")

        entry = self.runtime.schedule_reminder(
            due_at=due_at,
            summary=summary,
            details=details or None,
            kind=kind,
            source="schedule_reminder",
            original_request=original_request or None,
        )
        self.emit("reminder_tool_call=true")
        self.emit(f"reminder_scheduled={entry.summary}")
        self.emit(f"reminder_due_at={entry.due_at.isoformat()}")
        self._record_event(
            "reminder_tool_scheduled",
            "Realtime tool scheduled a reminder or timer.",
            reminder_id=entry.reminder_id,
            kind=entry.kind,
            due_at=entry.due_at.isoformat(),
        )
        return {
            "status": "scheduled",
            "reminder_id": entry.reminder_id,
            "kind": entry.kind,
            "summary": entry.summary,
            "due_at": entry.due_at.isoformat(),
        }

    def _handle_list_automations_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        include_disabled = bool(arguments.get("include_disabled", False))
        records = list(self.runtime.list_automation_records())
        if not include_disabled:
            records = [record for record in records if bool(record.get("enabled", True))]
        serialised = [self._serialize_automation_record(record) for record in records]
        self.emit(f"automation_list_count={len(serialised)}")
        self._record_event(
            "automation_listed",
            "Realtime tool listed configured automations.",
            count=len(serialised),
            include_disabled=include_disabled,
        )
        return {
            "status": "ok",
            "count": len(serialised),
            "automations": serialised,
        }

    def _handle_create_time_automation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        self._require_sensitive_voice_confirmation(arguments, action_label="create or change scheduled automations")
        name = str(arguments.get("name", "")).strip()
        schedule = str(arguments.get("schedule", "")).strip().lower()
        if not name or not schedule:
            raise RuntimeError("create_time_automation requires `name` and `schedule`")
        action = self._build_automation_action(arguments)
        weekdays = self._parse_weekdays(arguments.get("weekdays"))
        entry = self.runtime.create_time_automation(
            name=name,
            description=str(arguments.get("description", "")).strip() or None,
            enabled=self._optional_bool(arguments, "enabled", default=True),
            schedule=schedule,
            due_at=str(arguments.get("due_at", "")).strip() or None,
            time_of_day=str(arguments.get("time_of_day", "")).strip() or None,
            weekdays=weekdays,
            timezone_name=str(arguments.get("timezone_name", "")).strip() or self.config.local_timezone_name,
            actions=(action,),
            source="create_time_automation",
            tags=self._parse_tags(arguments.get("tags")),
        )
        self.emit("automation_tool_call=true")
        self.emit(f"automation_created={entry.name}")
        self.emit(f"automation_id={entry.automation_id}")
        self._record_event(
            "automation_tool_created",
            "Realtime tool created a time-based automation.",
            automation_id=entry.automation_id,
            name=entry.name,
            schedule=schedule,
        )
        return {
            "status": "created",
            "automation": self._serialize_automation_record(
                self.runtime.automation_store.engine.tool_record(entry)
            ),
        }

    def _handle_create_sensor_automation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        self._require_sensitive_voice_confirmation(arguments, action_label="create or change sensor automations")
        name = str(arguments.get("name", "")).strip()
        trigger_kind = str(arguments.get("trigger_kind", "")).strip().lower()
        if not name or not trigger_kind:
            raise RuntimeError("create_sensor_automation requires `name` and `trigger_kind`")
        trigger = self._build_sensor_trigger(arguments)
        action = self._build_automation_action(arguments)
        entry = self.runtime.create_if_then_automation(
            name=name,
            description=str(arguments.get("description", "")).strip() or None,
            enabled=self._optional_bool(arguments, "enabled", default=True),
            event_name=trigger.event_name,
            all_conditions=trigger.all_conditions,
            any_conditions=trigger.any_conditions,
            cooldown_seconds=trigger.cooldown_seconds,
            actions=(action,),
            source="create_sensor_automation",
            tags=self._build_sensor_automation_tags(trigger_kind, self._parse_tags(arguments.get("tags"))),
        )
        self.emit("automation_tool_call=true")
        self.emit(f"automation_created={entry.name}")
        self.emit(f"automation_id={entry.automation_id}")
        self.emit(f"automation_sensor_trigger={trigger_kind}")
        self._record_event(
            "automation_tool_created",
            "Realtime tool created a sensor-triggered automation.",
            automation_id=entry.automation_id,
            name=entry.name,
            trigger_kind=trigger_kind,
        )
        return {
            "status": "created",
            "automation": self._serialize_automation_record(
                self.runtime.automation_store.engine.tool_record(entry)
            ),
        }

    def _handle_update_time_automation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        self._require_sensitive_voice_confirmation(arguments, action_label="change scheduled automations")
        automation_ref = str(arguments.get("automation_ref", "")).strip()
        if not automation_ref:
            raise RuntimeError("update_time_automation requires `automation_ref`")
        entry = self._resolve_automation_reference(automation_ref)
        if not isinstance(entry.trigger, TimeAutomationTrigger):
            raise RuntimeError("Only time-based automations can be updated with update_time_automation")
        trigger = self._build_updated_time_trigger(entry, arguments)
        actions = self._build_updated_automation_actions(entry, arguments)
        updated = self.runtime.update_automation(
            entry.automation_id,
            name=(str(arguments.get("name", "")).strip() or None) if "name" in arguments else None,
            description=(str(arguments.get("description", "")).strip() or None)
            if "description" in arguments
            else None,
            enabled=self._optional_bool(arguments, "enabled", default=None),
            trigger=trigger,
            actions=actions,
            source="update_time_automation",
            tags=self._parse_tags(arguments.get("tags")) if "tags" in arguments else None,
        )
        self.emit("automation_tool_call=true")
        self.emit(f"automation_updated={updated.name}")
        self.emit(f"automation_id={updated.automation_id}")
        self._record_event(
            "automation_tool_updated",
            "Realtime tool updated a time-based automation.",
            automation_id=updated.automation_id,
            name=updated.name,
        )
        return {
            "status": "updated",
            "automation": self._serialize_automation_record(
                self.runtime.automation_store.engine.tool_record(updated)
            ),
        }

    def _handle_update_sensor_automation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        self._require_sensitive_voice_confirmation(arguments, action_label="change sensor automations")
        automation_ref = str(arguments.get("automation_ref", "")).strip()
        if not automation_ref:
            raise RuntimeError("update_sensor_automation requires `automation_ref`")
        entry = self._resolve_automation_reference(automation_ref)
        sensor_spec = describe_sensor_trigger(entry.trigger)
        if sensor_spec is None:
            raise RuntimeError("Only supported sensor-triggered automations can be updated with update_sensor_automation")
        trigger = self._build_updated_sensor_trigger(entry, arguments)
        actions = self._build_updated_automation_actions(entry, arguments)
        updated = self.runtime.update_automation(
            entry.automation_id,
            name=(str(arguments.get("name", "")).strip() or None) if "name" in arguments else None,
            description=(str(arguments.get("description", "")).strip() or None)
            if "description" in arguments
            else None,
            enabled=self._optional_bool(arguments, "enabled", default=None),
            trigger=trigger,
            actions=actions,
            source="update_sensor_automation",
            tags=self._build_updated_sensor_tags(entry, arguments),
        )
        updated_spec = describe_sensor_trigger(updated.trigger)
        self.emit("automation_tool_call=true")
        self.emit(f"automation_updated={updated.name}")
        self.emit(f"automation_id={updated.automation_id}")
        if updated_spec is not None:
            self.emit(f"automation_sensor_trigger={updated_spec.trigger_kind}")
        self._record_event(
            "automation_tool_updated",
            "Realtime tool updated a sensor-triggered automation.",
            automation_id=updated.automation_id,
            name=updated.name,
            trigger_kind=updated_spec.trigger_kind if updated_spec is not None else None,
        )
        return {
            "status": "updated",
            "automation": self._serialize_automation_record(
                self.runtime.automation_store.engine.tool_record(updated)
            ),
        }

    def _handle_delete_automation_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        self._require_sensitive_voice_confirmation(arguments, action_label="delete automations")
        automation_ref = str(arguments.get("automation_ref", "")).strip()
        if not automation_ref:
            raise RuntimeError("delete_automation requires `automation_ref`")
        entry = self._resolve_automation_reference(automation_ref)
        removed = self.runtime.delete_automation(entry.automation_id, source="delete_automation")
        self.emit("automation_tool_call=true")
        self.emit(f"automation_deleted={removed.name}")
        self.emit(f"automation_id={removed.automation_id}")
        self._record_event(
            "automation_tool_deleted",
            "Realtime tool deleted an automation.",
            automation_id=removed.automation_id,
            name=removed.name,
        )
        return {
            "status": "deleted",
            "automation_id": removed.automation_id,
            "name": removed.name,
        }

    def _handle_remember_memory_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        self._require_sensitive_voice_confirmation(arguments, action_label="save durable memory")
        kind = str(arguments.get("kind", "")).strip() or "memory"
        summary = str(arguments.get("summary", "")).strip()
        details = str(arguments.get("details", "")).strip()
        if not summary:
            raise RuntimeError("remember_memory requires `summary`")

        entry = self.context_store.memory_store.remember(
            kind=kind,
            summary=summary,
            details=details or None,
        )
        self.runtime.remember_note(
            kind="fact",
            content=f"Saved memory: {entry.summary}",
            source="remember_memory",
            metadata={"memory_kind": entry.kind, "memory_id": entry.entry_id},
        )
        self.emit("memory_tool_call=true")
        self.emit(f"memory_saved={entry.summary}")
        self._record_event(
            "memory_saved",
            "Important user-requested memory was stored in MEMORY.md.",
            kind=entry.kind,
            summary=entry.summary,
        )
        return {
            "status": "saved",
            "kind": entry.kind,
            "summary": entry.summary,
            "memory_id": entry.entry_id,
        }

    def _handle_update_user_profile_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        self._require_sensitive_voice_confirmation(arguments, action_label="change the saved user profile")
        category = str(arguments.get("category", "")).strip()
        instruction = str(arguments.get("instruction", "")).strip()
        if not category or not instruction:
            raise RuntimeError("update_user_profile requires `category` and `instruction`")

        entry = self.context_store.user_store.upsert(
            category=category,
            instruction=instruction,
        )
        self.runtime.remember_note(
            kind="preference",
            content=f"User profile update ({entry.key}): {entry.instruction}",
            source="update_user_profile",
            metadata={"category": entry.key},
        )
        self.emit("user_profile_tool_call=true")
        self.emit(f"user_profile_update={entry.key}")
        self._record_event(
            "user_profile_updated",
            "Stable user profile context was updated from an explicit user request.",
            category=entry.key,
        )
        return {
            "status": "updated",
            "category": entry.key,
            "instruction": entry.instruction,
        }

    def _handle_update_personality_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        self._require_sensitive_voice_confirmation(arguments, action_label="change Twinr's future behavior")
        category = str(arguments.get("category", "")).strip()
        instruction = str(arguments.get("instruction", "")).strip()
        if not category or not instruction:
            raise RuntimeError("update_personality requires `category` and `instruction`")

        entry = self.context_store.personality_store.upsert(
            category=category,
            instruction=instruction,
        )
        self.runtime.remember_note(
            kind="preference",
            content=f"Behavior update ({entry.key}): {entry.instruction}",
            source="update_personality",
            metadata={"category": entry.key},
        )
        self.emit("personality_tool_call=true")
        self.emit(f"personality_update={entry.key}")
        self._record_event(
            "personality_updated",
            "Twinr personality context was updated from an explicit user request.",
            category=entry.key,
        )
        return {
            "status": "updated",
            "category": entry.key,
            "instruction": entry.instruction,
        }

    def _handle_enroll_voice_profile_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        summary = self.voice_profile_monitor.summary()
        if summary.enrolled:
            self._require_sensitive_voice_confirmation(arguments, action_label="replace the saved voice profile")
        audio_pcm = self._require_current_turn_audio()
        template = self.voice_profile_monitor.enroll_pcm16(
            audio_pcm,
            sample_rate=self.config.openai_realtime_input_sample_rate,
            channels=self.config.audio_channels,
        )
        assessment = self.voice_profile_monitor.assess_pcm16(
            audio_pcm,
            sample_rate=self.config.openai_realtime_input_sample_rate,
            channels=self.config.audio_channels,
        )
        if assessment.should_persist:
            self.runtime.update_user_voice_assessment(
                status=assessment.status,
                confidence=assessment.confidence,
                checked_at=assessment.checked_at,
            )
        self.emit("voice_profile_tool_call=true")
        self.emit(f"voice_profile_samples={template.sample_count}")
        self._record_event(
            "voice_profile_enrolled",
            "Realtime tool stored or refreshed the local voice profile.",
            sample_count=template.sample_count,
            average_duration_ms=template.average_duration_ms,
        )
        return {
            "status": "enrolled",
            "sample_count": template.sample_count,
            "average_duration_ms": template.average_duration_ms,
            "detail": "Local voice profile stored from the current spoken turn.",
        }

    def _handle_get_voice_profile_status_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        del arguments
        summary = self.voice_profile_monitor.summary()
        self.emit("voice_profile_tool_call=true")
        self._record_event(
            "voice_profile_status_read",
            "Realtime tool read the local voice-profile status.",
            enrolled=summary.enrolled,
            sample_count=summary.sample_count,
        )
        return {
            "status": "ok",
            "enrolled": summary.enrolled,
            "sample_count": summary.sample_count,
            "updated_at": summary.updated_at,
            "average_duration_ms": summary.average_duration_ms,
            "current_signal": self.runtime.user_voice_status,
            "current_confidence": self.runtime.user_voice_confidence,
        }

    def _handle_reset_voice_profile_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        self._require_sensitive_voice_confirmation(arguments, action_label="delete the saved voice profile")
        summary = self.voice_profile_monitor.reset()
        self.runtime.update_user_voice_assessment(
            status=None,
            confidence=None,
            checked_at=None,
        )
        self.emit("voice_profile_tool_call=true")
        self._record_event(
            "voice_profile_reset",
            "Realtime tool deleted the local voice profile.",
            enrolled=summary.enrolled,
        )
        return {
            "status": "reset",
            "enrolled": summary.enrolled,
            "sample_count": summary.sample_count,
        }

    def _handle_search_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        question = str(arguments.get("question", "")).strip()
        location_hint = str(arguments.get("location_hint", "")).strip()
        date_context = str(arguments.get("date_context", "")).strip()
        if not question:
            raise RuntimeError("search_live_info requires `question`")

        self.emit("search_tool_call=true")
        self.emit(f"search_question={question}")
        self._record_event("search_started", "Live web search tool was invoked.", question=question)
        if location_hint:
            self.emit(f"search_location_hint={location_hint}")
        if date_context:
            self.emit(f"search_date_context={date_context}")

        stop_feedback = self._start_search_feedback_loop()
        try:
            result = self.print_backend.search_live_info_with_metadata(
                question,
                conversation=self.runtime.provider_conversation_context(),
                location_hint=location_hint or None,
                date_context=date_context or None,
            )
        finally:
            stop_feedback()

        self.emit(f"search_used_web_search={str(result.used_web_search).lower()}")
        if result.response_id:
            self.emit(f"search_response_id={result.response_id}")
        if result.request_id:
            self.emit(f"search_request_id={result.request_id}")
        self._record_usage(
            request_kind="search",
            source="realtime_tool",
            model=result.model,
            response_id=result.response_id,
            request_id=result.request_id,
            used_web_search=result.used_web_search,
            token_usage=result.token_usage,
            question=question,
        )
        for index, source in enumerate(result.sources, start=1):
            self.emit(f"search_source_{index}={source}")
        self._record_event(
            "search_finished",
            "Live web search completed.",
            sources=len(result.sources),
            used_web_search=result.used_web_search,
        )
        self.runtime.remember_search_result(
            question=question,
            answer=result.answer,
            sources=result.sources,
            location_hint=location_hint or None,
            date_context=date_context or None,
        )
        return {
            "status": "ok",
            "answer": result.answer,
            "sources": list(result.sources),
        }

    def _handle_inspect_camera_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        question = str(arguments.get("question", "")).strip()
        if not question:
            raise RuntimeError("inspect_camera requires `question`")

        self.emit("camera_tool_call=true")
        self.emit(f"camera_question={question}")
        images = self._build_vision_images()
        response = self.print_backend.respond_to_images_with_metadata(
            self._build_vision_prompt(question, include_reference=len(images) > 1),
            images=images,
            conversation=self.runtime.provider_conversation_context(),
            allow_web_search=False,
        )
        self.emit(f"vision_image_count={len(images)}")
        if response.response_id:
            self.emit(f"camera_response_id={response.response_id}")
        if response.request_id:
            self.emit(f"camera_request_id={response.request_id}")
        self._record_usage(
            request_kind="vision",
            source="realtime_tool",
            model=response.model,
            response_id=response.response_id,
            request_id=response.request_id,
            used_web_search=False,
            token_usage=response.token_usage,
            question=question,
            vision_image_count=len(images),
        )
        return {
            "status": "ok",
            "answer": response.text,
        }

    def _handle_error(self, exc: Exception) -> None:
        self.runtime.fail(str(exc))
        self._emit_status(force=True)
        self.emit(f"error={exc}")
        if self.error_reset_seconds > 0:
            self.sleep(self.error_reset_seconds)
        self.runtime.reset_error()
        self._emit_status(force=True)

    def _require_current_turn_audio(self) -> bytes:
        if self._current_turn_audio_pcm:
            return self._current_turn_audio_pcm
        raise RuntimeError("Voice profile enrollment needs the current live spoken turn.")

    def _require_sensitive_voice_confirmation(self, arguments: dict[str, object], *, action_label: str) -> None:
        status = (self.runtime.user_voice_status or "").strip().lower()
        if status not in {"uncertain", "unknown_voice"}:
            return
        if self._optional_bool(arguments, "confirmed", default=False):
            return
        raise RuntimeError(
            f"The current speaker signal is {status.replace('_', ' ')}. "
            f"Please ask for clear confirmation before you {action_label}, then call the tool again with confirmed=true."
        )

    def _emit_status(self, *, force: bool = False) -> None:
        status = self.runtime.status.value
        if force or status != self._last_status:
            self.emit(f"status={status}")
            self._last_status = status

    def _maybe_deliver_due_reminder(self) -> bool:
        now_monotonic = time.monotonic()
        if now_monotonic < self._next_reminder_check_at:
            return False
        self._next_reminder_check_at = now_monotonic + self.config.reminder_poll_interval_s
        if not self._background_work_allowed():
            return False
        due_entries = self.runtime.reserve_due_reminders(limit=1)
        if not due_entries:
            return False
        return self._deliver_due_reminder(due_entries[0])

    def _maybe_run_due_automation(self) -> bool:
        now_monotonic = time.monotonic()
        if now_monotonic < self._next_automation_check_at:
            return False
        self._next_automation_check_at = now_monotonic + self.config.automation_poll_interval_s
        if not self._background_work_allowed():
            return False
        due_entries = self.runtime.due_time_automations()
        if not due_entries:
            return False
        return self._run_automation_entry(due_entries[0], trigger_source="time_schedule")

    def _maybe_run_sensor_automation(self) -> bool:
        if not self._background_work_allowed():
            return False
        while True:
            try:
                facts, event_names = self._sensor_observation_queue.get_nowait()
            except Empty:
                return False
            if self._run_matching_sensor_automations(facts=facts, event_names=event_names):
                return True

    def _background_work_allowed(self) -> bool:
        return self.runtime.status.value == "waiting" and not self._conversation_session_active

    def _recent_proactive_prompts(
        self,
        *,
        trigger_id: str | None = None,
        limit: int = 3,
    ) -> tuple[str, ...]:
        prompts: list[str] = []
        for entry in reversed(self.runtime.ops_events.tail(limit=100)):
            if entry.get("event") != "social_trigger_prompted":
                continue
            data = entry.get("data") or {}
            if trigger_id is not None and data.get("trigger") != trigger_id:
                continue
            prompt = str(data.get("prompt", "")).strip()
            if not prompt:
                continue
            prompts.append(prompt)
            if len(prompts) >= limit:
                break
        prompts.reverse()
        return tuple(prompts)

    def _run_matching_sensor_automations(
        self,
        *,
        facts: dict[str, object],
        event_names: tuple[str, ...],
    ) -> bool:
        matched: dict[str, AutomationDefinition] = {}
        for entry in self.runtime.matching_if_then_automations(facts=facts, event_name=None):
            matched[entry.automation_id] = entry
        for event_name in event_names:
            for entry in self.runtime.matching_if_then_automations(facts=facts, event_name=event_name):
                matched[entry.automation_id] = entry
        if not matched:
            return False
        executed_any = False
        for entry in sorted(matched.values(), key=lambda item: item.name.lower()):
            event_label = ",".join(event_names) or "sensor.state"
            executed = self._run_automation_entry(
                entry,
                trigger_source="sensor",
                event_name=event_label,
                facts=facts,
            )
            executed_any = executed or executed_any
        return executed_any

    def _deliver_due_reminder(self, reminder) -> bool:
        response = None
        spoken_prompt = ""
        try:
            response = self._phrase_due_reminder_with_fallback(reminder)
            spoken_prompt = self.runtime.begin_reminder_prompt(response.text)
            self._emit_status(force=True)
            tts_started = time.monotonic()
            first_audio_at: list[float | None] = [None]

            def mark_first_chunk():
                for chunk in self.print_backend.synthesize_stream(spoken_prompt):
                    if first_audio_at[0] is None:
                        first_audio_at[0] = time.monotonic()
                    yield chunk

            with self._audio_lock:
                self.player.play_wav_chunks(mark_first_chunk())
            tts_ms = int((time.monotonic() - tts_started) * 1000)
            self.runtime.finish_speaking()
            self._emit_status(force=True)
            delivered = self.runtime.mark_reminder_delivered(reminder.reminder_id)
            self.emit("reminder_delivered=true")
            self.emit(f"reminder_due_at={delivered.due_at.isoformat()}")
            self.emit(f"reminder_text={spoken_prompt}")
            if response.response_id:
                self.emit(f"reminder_response_id={response.response_id}")
            if response.request_id:
                self.emit(f"reminder_request_id={response.request_id}")
            self.emit(f"timing_reminder_tts_ms={tts_ms}")
            if first_audio_at[0] is not None:
                self.emit(
                    f"timing_reminder_first_audio_ms={int((first_audio_at[0] - tts_started) * 1000)}"
                )
            self._record_usage(
                request_kind="reminder_delivery",
                source="realtime_loop",
                model=response.model,
                response_id=response.response_id,
                request_id=response.request_id,
                used_web_search=False,
                token_usage=response.token_usage,
                reminder_id=delivered.reminder_id,
                reminder_kind=delivered.kind,
            )
            return True
        except Exception as exc:
            if self.runtime.status.value == "answering":
                self.runtime.finish_speaking()
                self._emit_status(force=True)
            self.runtime.mark_reminder_failed(reminder.reminder_id, error=str(exc))
            self.emit(f"reminder_error={exc}")
            return False

    def _phrase_due_reminder_with_fallback(self, reminder):
        helper = getattr(self.print_backend, "phrase_due_reminder_with_metadata", None)
        if callable(helper):
            return helper(reminder)

        generic = getattr(self.print_backend, "respond_with_metadata", None)
        if not callable(generic):
            raise RuntimeError("The configured backend cannot phrase due reminders")

        current_time = datetime.now(ZoneInfo(self.config.local_timezone_name))
        prompt_parts = [
            "A stored Twinr reminder is due now.",
            f"Current local time: {format_due_label(current_time, timezone_name=self.config.local_timezone_name)}",
            f"Scheduled reminder time: {format_due_label(reminder.due_at, timezone_name=self.config.local_timezone_name)}",
            f"Reminder kind: {reminder.kind}",
            f"Reminder summary: {reminder.summary}",
        ]
        if reminder.details:
            prompt_parts.append(f"Reminder details: {reminder.details}")
        if reminder.original_request:
            prompt_parts.append(f"Original user request: {reminder.original_request}")
        prompt_parts.append("Speak the reminder now.")
        self.emit("reminder_backend_fallback=generic")
        return generic(
            "\n".join(prompt_parts),
            instructions=REMINDER_DELIVERY_INSTRUCTIONS,
            allow_web_search=False,
        )

    def _run_automation_entry(
        self,
        entry: AutomationDefinition,
        *,
        trigger_source: str,
        event_name: str | None = None,
        facts: dict[str, object] | None = None,
    ) -> bool:
        executed = False
        try:
            for action in entry.actions:
                if not action.enabled:
                    continue
                self._execute_automation_action(entry, action)
                executed = True
            if not executed:
                raise RuntimeError("Automation has no enabled actions")
            self.runtime.mark_automation_triggered(entry.automation_id)
            self.emit("automation_executed=true")
            self.emit(f"automation_name={entry.name}")
            self.emit(f"automation_id={entry.automation_id}")
            self.emit(f"automation_trigger_source={trigger_source}")
            if event_name:
                self.emit(f"automation_event_name={event_name}")
            self._record_event(
                "automation_executed",
                "An automation was executed.",
                automation_id=entry.automation_id,
                name=entry.name,
                trigger_source=trigger_source,
                event_name=event_name,
                facts=facts,
            )
            return True
        except Exception as exc:
            self._recover_automation_output_state()
            self.runtime.mark_automation_triggered(entry.automation_id, source="automation_failed")
            self.emit(f"automation_error={exc}")
            self._record_event(
                "automation_execution_failed",
                "An automation failed during execution.",
                level="error",
                automation_id=entry.automation_id,
                name=entry.name,
                trigger_source=trigger_source,
                event_name=event_name,
                error=str(exc),
            )
            return False

    def _execute_automation_action(self, entry: AutomationDefinition, action: AutomationAction) -> None:
        if action.kind == "say":
            self._speak_automation_text(entry, action.text or "")
            return
        if action.kind == "print":
            self._print_automation_text(entry, action.text or "", request_source="automation_static")
            return
        if action.kind == "llm_prompt":
            delivery = self._normalize_delivery(action.payload.get("delivery"))
            allow_web_search = bool(action.payload.get("allow_web_search", False))
            response = self.print_backend.fulfill_automation_prompt_with_metadata(
                action.text or "",
                allow_web_search=allow_web_search,
                delivery=delivery,
            )
            self._record_usage(
                request_kind="automation_generation",
                source="automation",
                model=response.model,
                response_id=response.response_id,
                request_id=response.request_id,
                used_web_search=response.used_web_search,
                token_usage=response.token_usage,
                automation_id=entry.automation_id,
                automation_name=entry.name,
                delivery=delivery,
            )
            if delivery == "printed":
                composed = self.print_backend.compose_print_job_with_metadata(
                    focus_hint=f"{entry.name}: {action.text or ''}".strip(": "),
                    direct_text=response.text,
                    request_source="automation",
                )
                self._record_usage(
                    request_kind="automation_print_compose",
                    source="automation",
                    model=composed.model,
                    response_id=composed.response_id,
                    request_id=composed.request_id,
                    used_web_search=False,
                    token_usage=composed.token_usage,
                    automation_id=entry.automation_id,
                    automation_name=entry.name,
                )
                self._print_automation_text(entry, composed.text, request_source="automation")
                return
            self._speak_automation_text(entry, response.text)
            return
        raise RuntimeError(f"Unsupported automation action kind during execution: {action.kind}")

    def _speak_automation_text(self, entry: AutomationDefinition, text: str) -> None:
        spoken_prompt = self.runtime.begin_automation_prompt(text)
        self._emit_status(force=True)
        tts_started = time.monotonic()
        first_audio_at: list[float | None] = [None]

        def mark_first_chunk():
            for chunk in self.print_backend.synthesize_stream(spoken_prompt):
                if first_audio_at[0] is None:
                    first_audio_at[0] = time.monotonic()
                yield chunk

        with self._audio_lock:
            self.player.play_wav_chunks(mark_first_chunk())
        tts_ms = int((time.monotonic() - tts_started) * 1000)
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"automation_spoken={spoken_prompt}")
        self.emit(f"automation_name={entry.name}")
        self.emit(f"automation_id={entry.automation_id}")
        self.emit(f"timing_automation_tts_ms={tts_ms}")
        if first_audio_at[0] is not None:
            self.emit(
                f"timing_automation_first_audio_ms={int((first_audio_at[0] - tts_started) * 1000)}"
            )

    def _print_automation_text(self, entry: AutomationDefinition, text: str, *, request_source: str) -> None:
        self.runtime.maybe_begin_automation_print()
        self._emit_status(force=True)
        print_job = self.printer.print_text(text)
        self.emit(f"automation_print_text={text}")
        self.emit(f"automation_name={entry.name}")
        self.emit(f"automation_id={entry.automation_id}")
        if print_job:
            self.emit(f"automation_print_job={print_job}")
        self._record_event(
            "automation_print_sent",
            "Scheduled automation sent content to the printer.",
            automation_id=entry.automation_id,
            name=entry.name,
            request_source=request_source,
            queue=self.config.printer_queue,
            job=print_job,
        )
        self.runtime.finish_printing()
        self._emit_status(force=True)

    def _recover_automation_output_state(self) -> None:
        if self.runtime.status.value == "answering":
            self.runtime.finish_speaking()
            self._emit_status(force=True)
            return
        if self.runtime.status.value == "printing":
            self.runtime.finish_printing()
            self._emit_status(force=True)

    def _build_automation_action(
        self,
        arguments: dict[str, object],
        *,
        fallback: AutomationAction | None = None,
    ) -> AutomationAction:
        fallback_delivery = "spoken"
        if fallback is not None:
            if fallback.kind == "print":
                fallback_delivery = "printed"
            elif fallback.kind == "llm_prompt":
                fallback_delivery = self._normalize_delivery(fallback.payload.get("delivery"))
        allow_web_search = self._optional_bool(
            arguments,
            "allow_web_search",
            default=bool(fallback.payload.get("allow_web_search")) if fallback is not None else False,
        )
        delivery = self._normalize_delivery(
            arguments.get("delivery")
            if "delivery" in arguments
            else fallback_delivery
        )
        raw_mode = (
            str(arguments.get("content_mode", "")).strip().lower()
            if "content_mode" in arguments
            else ("llm_prompt" if fallback is None or fallback.kind == "llm_prompt" else "static_text")
        )
        content_mode = "static_text" if raw_mode == "static_text" else "llm_prompt"
        content = (
            str(arguments.get("content", "")).strip()
            if "content" in arguments
            else (fallback.text or "" if fallback is not None else "")
        )
        if not content:
            raise RuntimeError("Automation content must not be empty")
        if content_mode == "llm_prompt":
            return AutomationAction(
                kind="llm_prompt",
                text=content,
                payload={
                    "delivery": delivery,
                    "allow_web_search": allow_web_search,
                },
            )
        return AutomationAction(
            kind="print" if delivery == "printed" else "say",
            text=content,
        )

    def _build_updated_time_trigger(
        self,
        entry: AutomationDefinition,
        arguments: dict[str, object],
    ) -> TimeAutomationTrigger | None:
        existing = entry.trigger
        if not isinstance(existing, TimeAutomationTrigger):
            raise RuntimeError("Only time-based automations can be updated")
        trigger_fields = {"schedule", "due_at", "time_of_day", "weekdays", "timezone_name"}
        if not any(field in arguments for field in trigger_fields):
            return None
        schedule = str(arguments.get("schedule", existing.schedule)).strip().lower() or existing.schedule
        weekdays = (
            self._parse_weekdays(arguments.get("weekdays"))
            if "weekdays" in arguments
            else existing.weekdays
        )
        timezone_name = str(arguments.get("timezone_name", existing.timezone_name or "")).strip() or (
            existing.timezone_name or self.config.local_timezone_name
        )
        due_at = str(arguments.get("due_at", existing.due_at or "")).strip() or None
        time_of_day = str(arguments.get("time_of_day", existing.time_of_day or "")).strip() or None
        return TimeAutomationTrigger(
            schedule=schedule,
            due_at=due_at,
            time_of_day=time_of_day,
            weekdays=weekdays,
            timezone_name=timezone_name,
        )

    def _build_updated_automation_actions(
        self,
        entry: AutomationDefinition,
        arguments: dict[str, object],
    ) -> tuple[AutomationAction, ...] | None:
        action_fields = {"delivery", "content_mode", "content", "allow_web_search"}
        if not any(field in arguments for field in action_fields):
            return None
        if not entry.actions:
            raise RuntimeError("Automation has no actions to update")
        primary = entry.actions[0]
        return (self._build_automation_action(arguments, fallback=primary),)

    def _build_sensor_trigger(self, arguments: dict[str, object], *, fallback: AutomationDefinition | None = None):
        fallback_spec = describe_sensor_trigger(fallback.trigger) if fallback is not None else None
        trigger_kind = (
            str(arguments.get("trigger_kind", "")).strip().lower()
            if "trigger_kind" in arguments
            else (fallback_spec.trigger_kind if fallback_spec is not None else "")
        )
        if not trigger_kind:
            raise RuntimeError("Sensor automation trigger_kind must not be empty")
        hold_default = fallback_spec.hold_seconds if fallback_spec is not None else 0.0
        cooldown_default = fallback_spec.cooldown_seconds if fallback_spec is not None else 0.0
        hold_seconds = self._optional_float(arguments, "hold_seconds", default=hold_default)
        cooldown_seconds = self._optional_float(arguments, "cooldown_seconds", default=cooldown_default)
        try:
            return build_sensor_trigger(
                trigger_kind,
                hold_seconds=hold_seconds,
                cooldown_seconds=cooldown_seconds,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def _build_updated_sensor_trigger(
        self,
        entry: AutomationDefinition,
        arguments: dict[str, object],
    ):
        sensor_fields = {"trigger_kind", "hold_seconds", "cooldown_seconds"}
        if not any(field in arguments for field in sensor_fields):
            return None
        return self._build_sensor_trigger(arguments, fallback=entry)

    def _build_sensor_automation_tags(
        self,
        trigger_kind: str,
        tags: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized_tags = list(tags)
        normalized_tags.extend(("sensor", trigger_kind))
        return tuple(dict.fromkeys(tag for tag in normalized_tags if tag))

    def _build_updated_sensor_tags(
        self,
        entry: AutomationDefinition,
        arguments: dict[str, object],
    ) -> tuple[str, ...] | None:
        sensor_spec = describe_sensor_trigger(entry.trigger)
        if sensor_spec is None:
            return None
        if "tags" in arguments:
            base_tags = self._parse_tags(arguments.get("tags"))
        elif "trigger_kind" not in arguments:
            return None
        else:
            base_tags = tuple(tag for tag in entry.tags if tag not in {"sensor", sensor_spec.trigger_kind})
        next_kind = (
            str(arguments.get("trigger_kind", "")).strip().lower()
            if "trigger_kind" in arguments
            else sensor_spec.trigger_kind
        )
        return self._build_sensor_automation_tags(next_kind, tuple(base_tags))

    def _resolve_automation_reference(self, automation_ref: str) -> AutomationDefinition:
        exact = self.runtime.automation_store.get(automation_ref)
        if exact is not None:
            return exact
        normalized_ref = automation_ref.strip().lower()
        if not normalized_ref:
            raise RuntimeError("Automation reference must not be empty")
        entries = self.runtime.automation_store.load_entries()
        exact_name_matches = [entry for entry in entries if entry.name.strip().lower() == normalized_ref]
        if len(exact_name_matches) == 1:
            return exact_name_matches[0]
        partial_matches = [entry for entry in entries if normalized_ref in entry.name.strip().lower()]
        if len(partial_matches) == 1:
            return partial_matches[0]
        if len(exact_name_matches) > 1 or len(partial_matches) > 1:
            raise RuntimeError("Automation reference is ambiguous; call list_automations first.")
        raise RuntimeError("No automation matched that reference.")

    def _serialize_automation_record(self, record: dict[str, object]) -> dict[str, object]:
        actions = tuple(record.get("actions", ()) or ())
        primary_action = actions[0] if actions else {}
        if not isinstance(primary_action, dict):
            primary_action = {}
        payload = primary_action.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        content_preview = str(primary_action.get("text", "")).strip()
        serialised = {
            "automation_id": record.get("automation_id"),
            "name": record.get("name"),
            "description": record.get("description"),
            "enabled": record.get("enabled"),
            "trigger_kind": record.get("trigger_kind"),
            "schedule": (record.get("trigger") or {}).get("schedule")
            if isinstance(record.get("trigger"), dict)
            else None,
            "due_at": (record.get("trigger") or {}).get("due_at")
            if isinstance(record.get("trigger"), dict)
            else None,
            "time_of_day": (record.get("trigger") or {}).get("time_of_day")
            if isinstance(record.get("trigger"), dict)
            else None,
            "weekdays": (record.get("trigger") or {}).get("weekdays")
            if isinstance(record.get("trigger"), dict)
            else None,
            "timezone_name": (record.get("trigger") or {}).get("timezone_name")
            if isinstance(record.get("trigger"), dict)
            else None,
            "next_run_at": record.get("next_run_at"),
            "due_now": record.get("due_now"),
            "delivery": payload.get("delivery", "printed" if primary_action.get("kind") == "print" else "spoken"),
            "content_mode": "llm_prompt" if primary_action.get("kind") == "llm_prompt" else "static_text",
            "content_preview": content_preview[:120] + ("…" if len(content_preview) > 120 else ""),
        }
        if record.get("trigger_kind") == "if_then":
            serialised.update(
                {
                    "event_name": record.get("event_name"),
                    "cooldown_seconds": record.get("cooldown_seconds"),
                    "sensor_trigger_kind": record.get("sensor_trigger_kind"),
                    "sensor_hold_seconds": record.get("sensor_hold_seconds"),
                    "sensor_label": record.get("sensor_label"),
                }
            )
        return serialised

    def _normalize_delivery(self, raw_value: object) -> str:
        normalized = str(raw_value or "").strip().lower()
        if normalized in {"print", "printed", "printer"}:
            return "printed"
        return "spoken"

    def _parse_weekdays(self, raw_value: object) -> tuple[int, ...]:
        if raw_value is None or raw_value == "":
            return ()
        if not isinstance(raw_value, (list, tuple)):
            raise RuntimeError("weekdays must be an array of weekday numbers 0-6")
        weekdays: list[int] = []
        for item in raw_value:
            try:
                weekday = int(item)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("weekdays must be integers 0-6") from exc
            if weekday < 0 or weekday > 6:
                raise RuntimeError("weekdays must use integers 0-6")
            weekdays.append(weekday)
        return tuple(sorted(set(weekdays)))

    def _parse_tags(self, raw_value: object) -> tuple[str, ...]:
        if raw_value is None or raw_value == "":
            return ()
        if not isinstance(raw_value, (list, tuple)):
            raise RuntimeError("tags must be an array of short strings")
        return tuple(str(item).strip() for item in raw_value if str(item).strip())

    def _optional_float(self, arguments: dict[str, object], key: str, *, default: float) -> float:
        if key not in arguments:
            return default
        raw_value = arguments.get(key)
        if raw_value in {None, ""}:
            return default
        try:
            return float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{key} must be a number") from exc

    def _optional_bool(
        self,
        arguments: dict[str, object],
        key: str,
        *,
        default: bool | None,
    ) -> bool | None:
        if key not in arguments:
            return default
        value = arguments.get(key)
        if value is None:
            return default
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        return bool(value)

    def _record_event(self, event: str, message: str, *, level: str = "info", **data: object) -> None:
        self.runtime.ops_events.append(event=event, message=message, level=level, data=data)

    def _update_voice_assessment_from_pcm(self, audio_pcm: bytes) -> None:
        try:
            assessment = self.voice_profile_monitor.assess_pcm16(
                audio_pcm,
                sample_rate=self.config.openai_realtime_input_sample_rate,
                channels=self.config.audio_channels,
            )
        except Exception as exc:
            self.emit(f"voice_profile_error={exc}")
            return
        if not assessment.should_persist:
            return
        self.runtime.update_user_voice_assessment(
            status=assessment.status,
            confidence=assessment.confidence,
            checked_at=assessment.checked_at,
        )
        self.emit(f"voice_profile_status={assessment.status}")
        if assessment.confidence is not None:
            self.emit(f"voice_profile_confidence={assessment.confidence:.2f}")

    def _record_usage(
        self,
        *,
        request_kind: str,
        source: str,
        model: str | None,
        response_id: str | None,
        request_id: str | None,
        used_web_search: bool | None,
        token_usage,
        **metadata: object,
    ) -> None:
        self.usage_store.append(
            source=source,
            request_kind=request_kind,
            model=model,
            response_id=response_id,
            request_id=request_id,
            used_web_search=used_web_search,
            token_usage=token_usage,
            metadata=metadata,
        )

    def _play_listen_beep(self) -> None:
        try:
            self.player.play_tone(
                frequency_hz=self.config.audio_beep_frequency_hz,
                duration_ms=self.config.audio_beep_duration_ms,
                volume=self.config.audio_beep_volume,
                sample_rate=self.config.openai_realtime_input_sample_rate,
            )
        except Exception as exc:
            self.emit(f"beep_error={exc}")
            return
        if self.config.audio_beep_settle_ms > 0:
            self.sleep(self.config.audio_beep_settle_ms / 1000.0)

    def _start_search_feedback_loop(self) -> Callable[[], None]:
        if not self.config.search_feedback_tones_enabled:
            return lambda: None

        stop_event = Event()
        delay_seconds = max(0.0, self.config.search_feedback_delay_ms / 1000.0)
        pause_seconds = max(0.12, self.config.search_feedback_pause_ms / 1000.0)

        def worker() -> None:
            if stop_event.wait(delay_seconds):
                return
            while not stop_event.is_set():
                try:
                    for frequency_hz, duration_ms in _SEARCH_FEEDBACK_TONE_PATTERN:
                        if stop_event.is_set():
                            return
                        self.player.play_tone(
                            frequency_hz=frequency_hz,
                            duration_ms=duration_ms,
                            volume=self.config.search_feedback_volume,
                            sample_rate=self.config.openai_realtime_input_sample_rate,
                        )
                        if stop_event.wait(0.03):
                            return
                except Exception as exc:
                    self.emit(f"search_feedback_error={exc}")
                    return
                if stop_event.wait(pause_seconds):
                    return

        thread = Thread(target=worker, daemon=True)
        thread.start()

        def stop() -> None:
            stop_event.set()
            thread.join()

        return stop

    def _build_vision_images(self) -> list[OpenAIImageInput]:
        with self._camera_lock:
            capture = self.camera.capture_photo(filename="camera-capture.png")
        self.emit(f"camera_device={capture.source_device}")
        self.emit(f"camera_input_format={capture.input_format or 'default'}")
        self.emit(f"camera_capture_bytes={len(capture.data)}")
        images = [
            OpenAIImageInput(
                data=capture.data,
                content_type=capture.content_type,
                filename=capture.filename,
                label="Image 1: live camera frame from the device.",
            )
        ]
        reference_image = self._load_reference_image()
        if reference_image is not None:
            images.append(reference_image)
        return images

    def _load_reference_image(self) -> OpenAIImageInput | None:
        raw_path = (self.config.vision_reference_image_path or "").strip()
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.exists():
            self.emit(f"vision_reference_missing={path}")
            return None
        self.emit(f"vision_reference_image={path}")
        return OpenAIImageInput.from_path(
            path,
            label="Image 2: stored reference image of the main user. Use it only for person or identity comparison.",
        )

    def _build_vision_prompt(self, question: str, *, include_reference: bool) -> str:
        if include_reference:
            return (
                "This request includes camera input. "
                "Image 1 is the current live camera frame from the device. "
                "Image 2 is a stored reference image of the main user. "
                "Use the reference image only when the user's question depends on whether the live image shows that user. "
                "If identity is uncertain, say that clearly. "
                "If the camera view is too unclear, tell the user how to position themselves or the object.\n\n"
                f"User request: {question}"
            )
        return (
            "This request includes camera input. "
            "Image 1 is the current live camera frame from the device. "
            "Answer from what is actually visible. "
            "If the view is too unclear, tell the user how to position themselves or the object in front of the camera.\n\n"
            f"User request: {question}"
        )

    def _is_no_speech_timeout(self, exc: Exception) -> bool:
        return "No speech detected before timeout" in str(exc)

    def _is_print_cooldown_active(self) -> bool:
        if self._last_print_request_at is None:
            return False
        return (time.monotonic() - self._last_print_request_at) < self.config.print_button_cooldown_s
