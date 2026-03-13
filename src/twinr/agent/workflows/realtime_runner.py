from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.hardware.audio import SilenceDetectedRecorder, WaveAudioPlayer
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.buttons import ButtonAction, configured_button_monitor
from twinr.memory.context_store import PromptContextStore
from twinr.hardware.printer import RawReceiptPrinter
from twinr.ops import TwinrUsageStore
from twinr.proactive import SocialTriggerDecision, build_default_proactive_monitor
from twinr.providers.openai.backend import OpenAIBackend, OpenAIImageInput
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
        self._camera_lock = Lock()
        self._audio_lock = Lock()
        self.realtime_session = realtime_session or OpenAIRealtimeSession(
            config=config,
            tool_handlers={
                "print_receipt": self._handle_print_tool_call,
                "search_live_info": self._handle_search_tool_call,
                "schedule_reminder": self._handle_schedule_reminder_tool_call,
                "remember_memory": self._handle_remember_memory_tool_call,
                "update_user_profile": self._handle_update_user_profile_tool_call,
                "update_personality": self._handle_update_personality_tool_call,
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
        self.proactive_monitor = proactive_monitor or build_default_proactive_monitor(
            config=config,
            runtime=self.runtime,
            backend=self.print_backend,
            camera=self.camera,
            camera_lock=self._camera_lock,
            audio_lock=self._audio_lock,
            trigger_handler=self.handle_social_trigger,
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
                    self._maybe_deliver_due_reminder()
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
        if self.runtime.status.value != "waiting":
            self.emit("social_trigger_skipped=busy")
            self._record_event(
                "social_trigger_skipped",
                "Social trigger prompt was skipped because Twinr was not idle.",
                trigger=trigger.trigger_id,
                reason=trigger.reason,
                prompt=trigger.prompt,
                priority=int(trigger.priority),
            )
            return False

        prompt = self.runtime.begin_proactive_prompt(trigger.prompt)
        self._emit_status(force=True)
        tts_started = time.monotonic()
        first_audio_at: list[float | None] = [None]

        def mark_first_chunk():
            for chunk in self.print_backend.synthesize_stream(prompt):
                if first_audio_at[0] is None:
                    first_audio_at[0] = time.monotonic()
                yield chunk

        self.player.play_wav_chunks(mark_first_chunk())
        tts_ms = int((time.monotonic() - tts_started) * 1000)
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"social_trigger={trigger.trigger_id}")
        self.emit(f"social_trigger_priority={int(trigger.priority)}")
        self.emit(f"social_prompt={prompt}")
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
        )
        return True

    def _handle_green_turn(self) -> None:
        follow_up = False
        while True:
            if self._run_single_green_turn(follow_up=follow_up):
                if self.config.conversation_follow_up_enabled:
                    follow_up = True
                    continue
            return

    def _run_single_green_turn(self, *, follow_up: bool) -> bool:
        turn_started = time.monotonic()
        self._play_listen_beep()
        self.runtime.press_green_button()
        self._emit_status(force=True)

        capture_started = time.monotonic()
        try:
            with self._audio_lock:
                audio_pcm = self.recorder.record_pcm_until_pause_with_options(
                    pause_ms=self.config.speech_pause_ms,
                    start_timeout_s=(
                        self.config.conversation_follow_up_timeout_s if follow_up else None
                    ),
                    speech_start_chunks=(
                        self.config.audio_follow_up_speech_start_chunks if follow_up else None
                    ),
                    ignore_initial_ms=(
                        self.config.audio_follow_up_ignore_ms if follow_up else 0
                    ),
                )
        except RuntimeError as exc:
            if not self._is_no_speech_timeout(exc):
                raise
            self.runtime.cancel_listening()
            self._emit_status(force=True)
            if follow_up:
                self.emit("follow_up_timeout=true")
                self._record_event("listen_timeout", "Follow-up listening window expired.")
            else:
                self.emit("listen_timeout=true")
                self._record_event("listen_timeout", "Listening timed out before speech started.")
            return False
        capture_ms = int((time.monotonic() - capture_started) * 1000)

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
        try:
            with self.realtime_session:
                turn = self.realtime_session.run_audio_turn(
                    audio_pcm,
                    conversation=self.runtime.conversation_context(),
                    on_audio_chunk=on_audio_chunk,
                    on_output_text_delta=on_output_text_delta,
                )
        finally:
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

    def _handle_print_turn(self) -> None:
        if self._is_print_cooldown_active():
            self.emit("print_skipped=cooldown")
            self._record_event("print_skipped", "Print request ignored because cooldown is active.")
            return
        response_to_print = self.runtime.press_yellow_button()
        self._emit_status(force=True)

        composed = self.print_backend.compose_print_job_with_metadata(
            conversation=self.runtime.conversation_context(),
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
            conversation=self.runtime.conversation_context(),
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

    def _handle_remember_memory_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
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
                conversation=self.runtime.conversation_context(),
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
            conversation=self.runtime.conversation_context(),
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
        if self.runtime.status.value != "waiting":
            return False
        due_entries = self.runtime.reserve_due_reminders(limit=1)
        if not due_entries:
            return False
        return self._deliver_due_reminder(due_entries[0])

    def _deliver_due_reminder(self, reminder) -> bool:
        response = None
        spoken_prompt = ""
        try:
            response = self.print_backend.phrase_due_reminder_with_metadata(reminder)
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

    def _record_event(self, event: str, message: str, *, level: str = "info", **data: object) -> None:
        self.runtime.ops_events.append(event=event, message=message, level=level, data=data)

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
