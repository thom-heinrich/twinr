from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Callable
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.hardware.audio import SilenceDetectedRecorder, WaveAudioPlayer
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.buttons import ButtonAction, configured_button_monitor
from twinr.hardware.printer import RawReceiptPrinter
from twinr.ops import TwinrUsageStore
from twinr.proactive import SocialTriggerDecision, build_default_proactive_monitor
from twinr.provider.openai.backend import OpenAIBackend, OpenAIImageInput

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
        backend: OpenAIBackend | None = None,
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
        self.backend = backend or OpenAIBackend(config=config)
        self.button_monitor = button_monitor or configured_button_monitor(config)
        self.recorder = recorder or SilenceDetectedRecorder.from_config(config)
        self.player = player or WaveAudioPlayer.from_config(config)
        self.printer = printer or RawReceiptPrinter.from_config(config)
        self.camera = camera or V4L2StillCamera.from_config(config)
        self.usage_store = usage_store or TwinrUsageStore.from_config(config)
        self._camera_lock = Lock()
        self._audio_lock = Lock()
        self.emit = emit or _default_emit
        self.sleep = sleep
        self.error_reset_seconds = error_reset_seconds
        self._last_status: str | None = None
        self._last_print_request_at: float | None = None
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
                if event is None or event.action != ButtonAction.PRESSED:
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
            )
            return False

        prompt = self.runtime.begin_proactive_prompt(trigger.prompt)
        self._emit_status(force=True)
        turn_started = time.monotonic()
        tts_ms, first_audio_ms = self._speak_full_answer(prompt, turn_started=turn_started)
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"social_trigger={trigger.trigger_id}")
        self.emit(f"social_trigger_priority={int(trigger.priority)}")
        self.emit(f"social_prompt={prompt}")
        self.emit(f"timing_social_tts_ms={tts_ms}")
        if first_audio_ms is not None:
            self.emit(f"timing_social_first_audio_ms={first_audio_ms}")
        self._record_event(
            "social_trigger_prompted",
            "Twinr spoke a proactive social prompt.",
            trigger=trigger.trigger_id,
            reason=trigger.reason,
            priority=int(trigger.priority),
        )
        return True

    def _handle_green_turn(self) -> None:
        turn_started = time.monotonic()
        self.runtime.press_green_button()
        self._emit_status(force=True)

        capture_started = time.monotonic()
        with self._audio_lock:
            audio_bytes = self.recorder.record_until_pause(pause_ms=self.config.speech_pause_ms)
        capture_ms = int((time.monotonic() - capture_started) * 1000)

        stt_started = time.monotonic()
        try:
            transcript = self.backend.transcribe(
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
                        for chunk in self.backend.synthesize_stream(segment):
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
            response = self.backend.respond_streaming(
                transcript,
                conversation=self.runtime.conversation_context(),
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
        response = self.backend.respond_to_images_with_metadata(
            self._build_vision_prompt(transcript, include_reference=len(images) > 1),
            images=images,
            conversation=self.runtime.conversation_context(),
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

        composed = self.backend.compose_print_job_with_metadata(
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
            for chunk in self.backend.synthesize_stream(text):
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

    def _is_print_cooldown_active(self) -> bool:
        if self._last_print_request_at is None:
            return False
        return (time.monotonic() - self._last_print_request_at) < self.config.print_button_cooldown_s
