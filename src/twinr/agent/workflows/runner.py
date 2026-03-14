from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Callable
import time

from twinr.agent.base_agent.contracts import (
    AgentTextProvider,
    CombinedSpeechAgentProvider,
    CompositeSpeechAgentProvider,
    SpeechToTextProvider,
    TextToSpeechProvider,
)
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.hardware.audio import SilenceDetectedRecorder, WaveAudioPlayer, pcm16_to_wav_bytes
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.buttons import ButtonAction, configured_button_monitor
from twinr.hardware.printer import RawReceiptPrinter
from twinr.hardware.voice_profile import VoiceProfileMonitor
from twinr.ops import TwinrUsageStore
from twinr.proactive import (
    SocialTriggerDecision,
    build_default_proactive_monitor,
    proactive_observation_facts,
    proactive_prompt_mode,
)
from twinr.providers.openai import OpenAIImageInput, OpenAIProviderBundle

_WEB_SEARCH_KEYWORDS = (
    "latest",
    "current",
    "today",
    "tonight",
    "tomorrow",
    "yesterday",
    "news",
    "weather",
    "forecast",
    "temperature",
    "price",
    "prices",
    "stock",
    "stocks",
    "score",
    "scores",
    "schedule",
    "schedules",
    "standings",
    "traffic",
    "version",
    "release",
    "update",
    "updates",
    "recent",
    "recently",
    "breaking",
    "ceo",
    "president",
    "election",
    "exchange rate",
    "market",
)

_CAMERA_TRIGGER_PHRASES = (
    "schau mich an",
    "schau mich mal an",
    "guck mich an",
    "guck mich mal an",
    "sieh mich an",
    "sieh mich mal an",
    "schau dir mal an was ich zeige",
    "schau dir an was ich zeige",
    "guck dir mal an was ich zeige",
    "sieh dir mal an was ich zeige",
    "schau dir das mal an",
    "guck dir das mal an",
    "sieh dir das mal an",
    "was zeige ich",
    "was halte ich",
    "was habe ich hier",
    "was ist das hier",
    "erkennst du das",
    "kannst du das sehen",
    "kannst du mich sehen",
    "siehst du das",
    "siehst du mich",
    "wie sehe ich aus",
    "wie seh ich aus",
    "wie sehe ich heute aus",
    "wie seh ich heute aus",
    "look at me",
    "look at this",
    "can you see me",
    "can you see this",
    "what am i showing you",
    "how do i look",
    "how do i look today",
)


def _default_emit(line: str) -> None:
    print(line, flush=True)


class TwinrHardwareLoop:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        runtime: TwinrRuntime | None = None,
        backend: CombinedSpeechAgentProvider | None = None,
        stt_provider: SpeechToTextProvider | None = None,
        agent_provider: AgentTextProvider | None = None,
        tts_provider: TextToSpeechProvider | None = None,
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
        openai_bundle: OpenAIProviderBundle | None = None
        if backend is None and (stt_provider is None or agent_provider is None or tts_provider is None):
            openai_bundle = OpenAIProviderBundle.from_config(config)
        self.stt_provider = stt_provider or backend or (openai_bundle.stt if openai_bundle is not None else None)
        self.agent_provider = agent_provider or backend or (openai_bundle.agent if openai_bundle is not None else None)
        self.tts_provider = tts_provider or backend or (openai_bundle.tts if openai_bundle is not None else None)
        if self.stt_provider is None or self.agent_provider is None or self.tts_provider is None:
            raise ValueError("TwinrHardwareLoop requires STT, agent, and TTS providers")
        self.backend = backend or (
            openai_bundle.combined
            if openai_bundle is not None
            else CompositeSpeechAgentProvider(
                stt=self.stt_provider,
                agent=self.agent_provider,
                tts=self.tts_provider,
            )
        )
        self.button_monitor = button_monitor or configured_button_monitor(config)
        self.recorder = recorder or SilenceDetectedRecorder.from_config(config)
        self.player = player or WaveAudioPlayer.from_config(config)
        self.printer = printer or RawReceiptPrinter.from_config(config)
        self.camera = camera or V4L2StillCamera.from_config(config)
        self.usage_store = usage_store or TwinrUsageStore.from_config(config)
        self.voice_profile_monitor = voice_profile_monitor or VoiceProfileMonitor.from_config(config)
        self._camera_lock = Lock()
        self._audio_lock = Lock()
        self.emit = emit or _default_emit
        self.sleep = sleep
        self.error_reset_seconds = error_reset_seconds
        self._last_status: str | None = None
        self._last_print_request_at: float | None = None
        self._next_reminder_check_at: float = 0.0
        self.proactive_monitor = proactive_monitor or build_default_proactive_monitor(
            config=config,
            runtime=self.runtime,
            backend=self.backend,
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

        phrase_response = None
        prompt_mode = proactive_prompt_mode(trigger)
        prompt_text = trigger.prompt
        if prompt_mode == "llm":
            try:
                phrase_response = self.agent_provider.phrase_proactive_prompt_with_metadata(
                    trigger_id=trigger.trigger_id,
                    reason=trigger.reason,
                    default_prompt=trigger.prompt,
                    priority=int(trigger.priority),
                    conversation=self.runtime.provider_conversation_context(),
                    recent_prompts=self._recent_proactive_prompts(trigger_id=trigger.trigger_id),
                    observation_facts=proactive_observation_facts(trigger),
                )
                candidate_prompt = phrase_response.text.strip()
                if candidate_prompt:
                    prompt_text = candidate_prompt
                else:
                    prompt_mode = "default_fallback"
                    self.emit("social_prompt_fallback=empty_phrase")
            except Exception as exc:
                prompt_mode = "default_fallback"
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
        turn_started = time.monotonic()
        tts_ms, first_audio_ms = self._speak_full_answer(prompt, turn_started=turn_started)
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"social_trigger={trigger.trigger_id}")
        self.emit(f"social_trigger_priority={int(trigger.priority)}")
        self.emit(f"social_prompt_mode={prompt_mode}")
        self.emit(f"social_prompt={prompt}")
        if phrase_response is not None:
            if phrase_response.response_id:
                self.emit(f"social_response_id={phrase_response.response_id}")
            if phrase_response.request_id:
                self.emit(f"social_request_id={phrase_response.request_id}")
            self._record_usage(
                request_kind="proactive_prompt",
                source="hardware_loop",
                model=phrase_response.model,
                response_id=phrase_response.response_id,
                request_id=phrase_response.request_id,
                used_web_search=False,
                token_usage=phrase_response.token_usage,
                proactive_trigger=trigger.trigger_id,
            )
        self.emit(f"timing_social_tts_ms={tts_ms}")
        if first_audio_ms is not None:
            self.emit(f"timing_social_first_audio_ms={first_audio_ms}")
        self._record_event(
            "social_trigger_prompted",
            "Twinr spoke a proactive social prompt.",
            trigger=trigger.trigger_id,
            reason=trigger.reason,
            priority=int(trigger.priority),
            prompt=prompt,
            default_prompt=trigger.prompt,
            prompt_mode=prompt_mode,
        )
        return True

    def _recent_proactive_prompts(
        self,
        *,
        trigger_id: str,
        limit: int = 3,
    ) -> tuple[str, ...]:
        prompts: list[str] = []
        for entry in reversed(self.runtime.ops_events.tail(limit=50)):
            if entry.get("event") != "social_trigger_prompted":
                continue
            data = entry.get("data", {})
            if data.get("trigger") != trigger_id:
                continue
            prompt = str(data.get("prompt", "")).strip()
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= limit:
                break
        return tuple(prompts)

    def _handle_green_turn(self) -> None:
        turn_started = time.monotonic()
        self.runtime.press_green_button()
        self._emit_status(force=True)

        listening_window = self.runtime.listening_window(initial_source="button", follow_up=False)
        capture_started = time.monotonic()
        try:
            with self._audio_lock:
                capture_result = self.recorder.capture_pcm_until_pause_with_options(
                    pause_ms=listening_window.speech_pause_ms,
                    start_timeout_s=listening_window.start_timeout_s,
                    pause_grace_ms=listening_window.pause_grace_ms,
                )
        except RuntimeError as exc:
            if not self._is_no_speech_timeout(exc):
                raise
            self.runtime.remember_listen_timeout(initial_source="button", follow_up=False)
            self.runtime.cancel_listening()
            self._emit_status(force=True)
            self.emit("listen_timeout=true")
            self._record_event(
                "listen_timeout",
                "Listening timed out before speech started.",
                request_source="button",
            )
            return
        self.runtime.remember_listen_capture(
            initial_source="button",
            follow_up=False,
            speech_started_after_ms=capture_result.speech_started_after_ms,
            resumed_after_pause_count=capture_result.resumed_after_pause_count,
        )
        audio_bytes = pcm16_to_wav_bytes(
            capture_result.pcm_bytes,
            sample_rate=self.config.audio_sample_rate,
            channels=self.config.audio_channels,
        )
        capture_ms = int((time.monotonic() - capture_started) * 1000)
        self._update_voice_assessment_from_wav(audio_bytes)

        stt_started = time.monotonic()
        try:
            transcript = self.stt_provider.transcribe(
                audio_bytes,
                filename="twinr-listen.wav",
                content_type="audio/wav",
            ).strip()
        except Exception as exc:
            self._record_event("stt_failed", "Speech-to-text failed.", level="error", error=str(exc))
            raise
        stt_ms = int((time.monotonic() - stt_started) * 1000)
        if not transcript:
            self._record_event("stt_failed", "Speech-to-text returned an empty transcript.", level="error")
            raise RuntimeError("Speech-to-text returned an empty transcript")

        self.emit(f"transcript={transcript}")
        self.runtime.submit_transcript(transcript)
        self._emit_status(force=True)

        allow_web_search = self._should_use_web_search(transcript)
        if self._should_use_camera(transcript):
            self._handle_green_vision_turn(
                transcript,
                turn_started=turn_started,
                capture_ms=capture_ms,
                stt_ms=stt_ms,
                allow_web_search=allow_web_search,
            )
            return

        llm_started = time.monotonic()
        spoken_segments: Queue[str | None] = Queue()
        tts_error: list[Exception] = []
        first_audio_at: list[float | None] = [None]
        answer_started = False
        pending_segment = ""
        worker_started = False

        def tts_worker() -> None:
            while True:
                segment = spoken_segments.get()
                if segment is None:
                    return
                try:
                    def mark_first_chunk() -> object:
                        for chunk in self.tts_provider.synthesize_stream(segment):
                            if first_audio_at[0] is None:
                                first_audio_at[0] = time.monotonic()
                            yield chunk

                    self.player.play_wav_chunks(mark_first_chunk())
                except Exception as exc:
                    tts_error.append(exc)
                    return

        worker = Thread(target=tts_worker, daemon=True)
        worker.start()
        worker_started = True

        def queue_ready_segments(delta: str) -> None:
            nonlocal answer_started, pending_segment
            pending_segment += delta
            while True:
                boundary = self._segment_boundary(pending_segment)
                if boundary is None:
                    return
                segment = pending_segment[:boundary].strip()
                pending_segment = pending_segment[boundary:].lstrip()
                if not segment:
                    continue
                if not answer_started:
                    self.runtime.begin_answering()
                    self._emit_status(force=True)
                    answer_started = True
                spoken_segments.put(segment)

        try:
            response = self.agent_provider.respond_streaming(
                transcript,
                conversation=self.runtime.provider_conversation_context(),
                allow_web_search=allow_web_search,
                on_text_delta=queue_ready_segments,
            )
            llm_ms = int((time.monotonic() - llm_started) * 1000)
            if response.used_web_search:
                self.runtime.remember_search_result(
                    question=transcript,
                    answer=response.text,
                )
            answer = self.runtime.finalize_agent_turn(response.text)
            if pending_segment.strip():
                if not answer_started:
                    self.runtime.begin_answering()
                    self._emit_status(force=True)
                    answer_started = True
                spoken_segments.put(pending_segment.strip())
        finally:
            if worker_started:
                spoken_segments.put(None)
        tts_started = time.monotonic()
        if worker_started:
            worker.join()
        tts_ms = int((time.monotonic() - tts_started) * 1000)
        if tts_error:
            raise tts_error[0]
        if not answer_started:
            self.runtime.begin_answering()
            self._emit_status(force=True)
        self.emit(f"response={answer}")
        if response.response_id:
            self.emit(f"openai_response_id={response.response_id}")
        if response.request_id:
            self.emit(f"openai_request_id={response.request_id}")
        self.emit(f"openai_allow_web_search={str(allow_web_search).lower()}")
        self.emit(f"openai_used_web_search={str(response.used_web_search).lower()}")
        self._record_usage(
            request_kind="conversation",
            source="hardware_loop",
            model=response.model,
            response_id=response.response_id,
            request_id=response.request_id,
            used_web_search=response.used_web_search,
            token_usage=response.token_usage,
            transcript=transcript,
        )
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"timing_capture_ms={capture_ms}")
        self.emit(f"timing_stt_ms={stt_ms}")
        self.emit(f"timing_llm_ms={llm_ms}")
        self.emit(f"timing_tts_ms={tts_ms}")
        self.emit("timing_playback_ms=streamed")
        if first_audio_at[0] is not None:
            self.emit(f"timing_first_audio_ms={int((first_audio_at[0] - turn_started) * 1000)}")
        self.emit(f"timing_total_ms={int((time.monotonic() - turn_started) * 1000)}")

    def _handle_green_vision_turn(
        self,
        transcript: str,
        *,
        turn_started: float,
        capture_ms: int,
        stt_ms: int,
        allow_web_search: bool,
    ) -> None:
        self.emit("camera_used=true")
        camera_capture_started = time.monotonic()
        images = self._build_vision_images()
        camera_capture_ms = int((time.monotonic() - camera_capture_started) * 1000)

        llm_started = time.monotonic()
        response = self.agent_provider.respond_to_images_with_metadata(
            self._build_vision_prompt(transcript, include_reference=len(images) > 1),
            images=images,
            conversation=self.runtime.provider_conversation_context(),
            allow_web_search=allow_web_search,
        )
        llm_ms = int((time.monotonic() - llm_started) * 1000)
        if response.used_web_search:
            self.runtime.remember_search_result(
                question=transcript,
                answer=response.text,
            )

        answer = self.runtime.complete_agent_turn(response.text)
        self._emit_status(force=True)
        self.emit(f"vision_image_count={len(images)}")
        self.emit(f"response={answer}")
        if response.response_id:
            self.emit(f"openai_response_id={response.response_id}")
        if response.request_id:
            self.emit(f"openai_request_id={response.request_id}")
        self.emit(f"openai_allow_web_search={str(allow_web_search).lower()}")
        self.emit(f"openai_used_web_search={str(response.used_web_search).lower()}")
        self._record_usage(
            request_kind="vision",
            source="hardware_loop",
            model=response.model,
            response_id=response.response_id,
            request_id=response.request_id,
            used_web_search=response.used_web_search,
            token_usage=response.token_usage,
            transcript=transcript,
            vision_image_count=len(images),
        )

        tts_ms, first_audio_ms = self._speak_full_answer(answer, turn_started=turn_started)
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"timing_capture_ms={capture_ms}")
        self.emit(f"timing_stt_ms={stt_ms}")
        self.emit(f"timing_camera_capture_ms={camera_capture_ms}")
        self.emit(f"timing_llm_ms={llm_ms}")
        self.emit(f"timing_tts_ms={tts_ms}")
        self.emit("timing_playback_ms=streamed")
        if first_audio_ms is not None:
            self.emit(f"timing_first_audio_ms={first_audio_ms}")
        self.emit(f"timing_total_ms={int((time.monotonic() - turn_started) * 1000)}")

    def _handle_print_turn(self) -> None:
        if self._is_print_cooldown_active():
            self.emit("print_skipped=cooldown")
            self._record_event("print_skipped", "Print request ignored because cooldown is active.")
            return
        response_to_print = self.runtime.press_yellow_button()
        self._emit_status(force=True)

        composed = self.agent_provider.compose_print_job_with_metadata(
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
            source="hardware_loop",
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

    def _update_voice_assessment_from_wav(self, audio_bytes: bytes) -> None:
        try:
            assessment = self.voice_profile_monitor.assess_wav_bytes(audio_bytes)
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
            response = self.agent_provider.phrase_due_reminder_with_metadata(reminder)
            spoken_prompt = self.runtime.begin_reminder_prompt(response.text)
            self._emit_status(force=True)
            tts_ms, first_audio_ms = self._speak_full_answer(spoken_prompt, turn_started=time.monotonic())
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
            if first_audio_ms is not None:
                self.emit(f"timing_reminder_first_audio_ms={first_audio_ms}")
            self._record_usage(
                request_kind="reminder_delivery",
                source="hardware_loop",
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

    def _should_use_web_search(self, transcript: str) -> bool:
        mode = self.config.conversation_web_search.strip().lower()
        if mode == "always":
            return True
        if mode == "never":
            return False
        normalized = transcript.lower()
        return any(keyword in normalized for keyword in _WEB_SEARCH_KEYWORDS)

    def _should_use_camera(self, transcript: str) -> bool:
        normalized = " ".join(transcript.lower().split())
        return any(phrase in normalized for phrase in _CAMERA_TRIGGER_PHRASES)

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

    def _build_vision_prompt(self, transcript: str, *, include_reference: bool) -> str:
        if include_reference:
            return (
                "This request includes camera input. "
                "Image 1 is the current live camera frame from the device. "
                "Image 2 is a stored reference image of the main user. "
                "Use the reference image only when the user's question depends on whether the live image shows that user. "
                "If identity is uncertain, say that clearly. "
                "If the camera view is too unclear, tell the user how to position themselves or the object.\n\n"
                f"User request: {transcript.strip()}"
            )
        return (
            "This request includes camera input. "
            "Image 1 is the current live camera frame from the device. "
            "Answer from what is actually visible. "
            "If the view is too unclear, tell the user how to position themselves or the object in front of the camera.\n\n"
            f"User request: {transcript.strip()}"
        )

    def _speak_full_answer(self, text: str, *, turn_started: float) -> tuple[int, int | None]:
        first_audio_at: list[float | None] = [None]

        def mark_first_chunk():
            for chunk in self.tts_provider.synthesize_stream(text):
                if first_audio_at[0] is None:
                    first_audio_at[0] = time.monotonic()
                yield chunk

        tts_started = time.monotonic()
        self.player.play_wav_chunks(mark_first_chunk())
        tts_ms = int((time.monotonic() - tts_started) * 1000)
        if first_audio_at[0] is None:
            return tts_ms, None
        return tts_ms, int((first_audio_at[0] - turn_started) * 1000)

    def _segment_boundary(self, text: str) -> int | None:
        for index, character in enumerate(text):
            if character in ".?!":
                return index + 1
        if len(text) >= 140:
            return len(text)
        return None

    def _is_no_speech_timeout(self, exc: Exception) -> bool:
        return "No speech detected before timeout" in str(exc)

    def _is_print_cooldown_active(self) -> bool:
        if self._last_print_request_at is None:
            return False
        return (time.monotonic() - self._last_print_request_at) < self.config.print_button_cooldown_s
